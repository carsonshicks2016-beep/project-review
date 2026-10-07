"""
Track generation, collision detection, clipping zones, and LIDAR raycasting for drifting.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ClippingZone:
    """Designated apex or outer clipping zone for drift judging."""
    x: float
    y: float
    radius: float = 3.0             # Radius for maximum score
    zone_type: str = "inside"       # 'inside' (ICP), 'outside' (OCZ), 'wall_tap', 'donut'
    angle_target: float = 0.0       # Target drift heading angle at clip (rad)
    score_value: float = 1000.0     # Base trick points
    hit: bool = False


@dataclass
class Track:
    """Represents a drifting track with boundaries, centerline, and clipping zones."""
    name: str
    centerline: np.ndarray          # Shape (N, 2)
    inner_boundary: np.ndarray      # Shape (M, 2)
    outer_boundary: np.ndarray      # Shape (K, 2)
    obstacles: List[np.ndarray] = field(default_factory=list) # List of polygon obstacles / pillars
    clipping_zones: List[ClippingZone] = field(default_factory=list)
    spawn_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    spawn_yaw: float = 0.0
    spawn_speed: float = 8.0        # Initial speed m/s so car is already moving

    def __post_init__(self):
        # Precompute segment vectors for fast raycast and distance checks
        self._build_wall_segments()

    def _build_wall_segments(self):
        """Construct list of 2D line segments [(p1, p2), ...] for all boundaries and obstacles."""
        segments = []

        def add_poly_segments(poly: np.ndarray, closed: bool = True):
            if len(poly) < 2:
                return
            n = len(poly)
            limit = n if closed else n - 1
            for i in range(limit):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                segments.append((p1, p2))

        add_poly_segments(self.inner_boundary, closed=True)
        add_poly_segments(self.outer_boundary, closed=True)
        for obs in self.obstacles:
            add_poly_segments(obs, closed=True)

        self.segments = segments

    def check_collision(self, car_corners: np.ndarray) -> bool:
        """
        Check if any car corner penetrates outside the track boundaries or into obstacles.
        car_corners: shape (4, 2)
        """
        # Ray casting / point in polygon or segment intersection
        # For efficiency, check if car corners cross any wall segment
        center = np.mean(car_corners, axis=0)
        for corner in car_corners:
            # Segment from center to corner
            p1 = center
            p2 = corner
            for w1, w2 in self.segments:
                if line_segments_intersect(p1, p2, w1, w2):
                    return True
        return False

    def raycast_lidar(self, origin: np.ndarray, yaw: float, num_rays: int = 16, max_range: float = 35.0) -> np.ndarray:
        """
        Cast distance rays from origin in vehicle body frame to nearest boundary/obstacle.
        Returns array of distances shape (num_rays,).
        """
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False) + yaw
        ray_dirs = np.column_stack([np.cos(angles), np.sin(angles)]) # shape (num_rays, 2)
        distances = np.full(num_rays, max_range, dtype=np.float32)

        # Vectorized or segment check
        ox, oy = origin[0], origin[1]
        for w1, w2 in self.segments:
            x1, y1 = w1[0], w1[1]
            x2, y2 = w2[0], w2[1]
            dx_w = x2 - x1
            dy_w = y2 - y1

            # Check each ray
            for i in range(num_rays):
                rdx = ray_dirs[i, 0]
                rdy = ray_dirs[i, 1]

                # Denominator for 2D line intersection: (dx_w * rdy - dy_w * rdx)
                denom = dx_w * rdy - dy_w * rdx
                if abs(denom) < 1e-7:
                    continue

                t_seg = ((ox - x1) * rdy - (oy - y1) * rdx) / denom
                t_ray = ((ox - x1) * dy_w - (oy - y1) * dx_w) / denom

                if 0.0 <= t_seg <= 1.0 and 0.0 < t_ray < distances[i]:
                    distances[i] = t_ray

        return distances

    def find_nearest_waypoint_index(self, pos: np.ndarray) -> int:
        """Find the index of the closest waypoint along the centerline."""
        diffs = self.centerline - pos
        sq_dist = np.sum(diffs ** 2, axis=1)
        return int(np.argmin(sq_dist))

    def get_target_waypoints(self, pos: np.ndarray, count: int = 3, step_stride: int = 5) -> np.ndarray:
        """
        Get the next `count` upcoming waypoints relative to the current position.
        Returns shape (count, 2).
        """
        curr_idx = self.find_nearest_waypoint_index(pos)
        n = len(self.centerline)
        targets = []
        for k in range(1, count + 1):
            idx = (curr_idx + k * step_stride) % n
            targets.append(self.centerline[idx])
        return np.array(targets, dtype=np.float32)

    def reset_clipping_zones(self):
        """Reset hit flags for all clipping zones."""
        for zone in self.clipping_zones:
            zone.hit = False


def line_segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


# -------------------------------------------------------------
# Track Builders
# -------------------------------------------------------------

def build_touge_track() -> Track:
    """
    Touge Circuit: Mountain pass with sweeping turns, sharp hairpins,
    and multiple clipping points ideal for continuous drift combos.
    """
    # Generate continuous smooth spline/parametric track
    t = np.linspace(0, 2 * np.pi, 250, endpoint=False)
    # Figure-8 / peanut shaped circuit with elevation feel
    r = 45.0 + 20.0 * np.cos(2 * t) + 12.0 * np.sin(3 * t)
    x = r * np.cos(t) * 1.5
    y = r * np.sin(t) * 1.1

    centerline = np.column_stack([x, y])
    # Compute normals to build inner and outer barriers (track width = 14m)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.hypot(dx, dy)
    nx = -dy / norm
    ny = dx / norm

    half_width = 7.0
    inner_boundary = np.column_stack([x - nx * half_width, y - ny * half_width])
    outer_boundary = np.column_stack([x + nx * half_width, y + ny * half_width])

    # Add clipping zones at corner apexes
    clipping_zones = [
        ClippingZone(x=centerline[30, 0], y=centerline[30, 1] + 4.0, radius=3.5, zone_type="inside", score_value=1200.0),
        ClippingZone(x=centerline[85, 0] - 4.0, y=centerline[85, 1], radius=3.5, zone_type="outside", score_value=1500.0),
        ClippingZone(x=centerline[150, 0], y=centerline[150, 1] - 4.0, radius=3.5, zone_type="inside", score_value=1200.0),
        ClippingZone(x=centerline[210, 0] + 4.0, y=centerline[210, 1], radius=3.5, zone_type="outside", score_value=1500.0),
    ]

    # Spawn position along straight
    spawn_pos = centerline[0].copy()
    spawn_dir = centerline[2] - centerline[0]
    spawn_yaw = float(np.arctan2(spawn_dir[1], spawn_dir[0]))

    return Track(
        name="Touge Pass",
        centerline=centerline,
        inner_boundary=inner_boundary,
        outer_boundary=outer_boundary,
        obstacles=[],
        clipping_zones=clipping_zones,
        spawn_pos=spawn_pos,
        spawn_yaw=spawn_yaw,
        spawn_speed=12.0,
    )


def build_gymkhana_arena() -> Track:
    """
    Gymkhana Drift Arena: Open skidpad with central donut pillars,
    slalom barriers, and outer curved wall-tap barriers.
    """
    # Outer rectangular boundary: 100m x 80m
    w, h = 50.0, 40.0
    outer_boundary = np.array([
        [-w, -h], [w, -h], [w, h], [-w, h]
    ], dtype=np.float64)

    # Centerline oval loop for guidance
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    x = 32.0 * np.cos(t)
    y = 24.0 * np.sin(t)
    centerline = np.column_stack([x, y])

    # Inner boundary dummy (none for open arena, or small center island)
    inner_boundary = np.array([
        [-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]
    ], dtype=np.float64)

    # Obstacles: 2 Donut pillars + 2 slalom blocks
    def make_box(cx, cy, size=2.5):
        s = size * 0.5
        return np.array([
            [cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]
        ], dtype=np.float64)

    obstacles = [
        make_box(-18.0, 0.0, 4.0),  # Left Donut Pillar
        make_box(18.0, 0.0, 4.0),   # Right Donut Pillar
        make_box(0.0, 15.0, 3.0),   # North Slalom
        make_box(0.0, -15.0, 3.0),  # South Slalom
    ]

    clipping_zones = [
        ClippingZone(x=-18.0, y=5.0, radius=3.5, zone_type="donut", score_value=2000.0),
        ClippingZone(x=18.0, y=-5.0, radius=3.5, zone_type="donut", score_value=2000.0),
        ClippingZone(x=0.0, y=20.0, radius=3.0, zone_type="inside", score_value=1000.0),
        ClippingZone(x=46.0, y=0.0, radius=4.0, zone_type="wall_tap", score_value=3000.0),
        ClippingZone(x=-46.0, y=0.0, radius=4.0, zone_type="wall_tap", score_value=3000.0),
    ]

    return Track(
        name="Gymkhana Playground",
        centerline=centerline,
        inner_boundary=inner_boundary,
        outer_boundary=outer_boundary,
        obstacles=obstacles,
        clipping_zones=clipping_zones,
        spawn_pos=np.array([-25.0, -20.0]),
        spawn_yaw=0.0,
        spawn_speed=10.0,
    )


def build_formula_stadium() -> Track:
    """
    Formula Drift Stadium: High-speed initiation sweeper into tight switchback.
    """
    # Kidney bean shape with high speed entry curve
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    x = 60.0 * np.cos(t)
    y = 35.0 * np.sin(t) * (1.0 + 0.3 * np.cos(t))

    centerline = np.column_stack([x, y])
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.hypot(dx, dy)
    nx = -dy / norm
    ny = dx / norm

    half_width = 8.5
    inner_boundary = np.column_stack([x - nx * half_width, y - ny * half_width])
    outer_boundary = np.column_stack([x + nx * half_width, y + ny * half_width])

    clipping_zones = [
        ClippingZone(x=centerline[40, 0] + 6.0, y=centerline[40, 1], radius=4.0, zone_type="wall_tap", score_value=2500.0),
        ClippingZone(x=centerline[100, 0] - 5.0, y=centerline[100, 1], radius=3.5, zone_type="inside", score_value=1500.0),
        ClippingZone(x=centerline[160, 0] + 6.0, y=centerline[160, 1], radius=4.0, zone_type="wall_tap", score_value=2500.0),
    ]

    return Track(
        name="Formula Stadium",
        centerline=centerline,
        inner_boundary=inner_boundary,
        outer_boundary=outer_boundary,
        obstacles=[],
        clipping_zones=clipping_zones,
        spawn_pos=centerline[0].copy(),
        spawn_yaw=float(np.arctan2(dy[0], dx[0])),
        spawn_speed=16.0,
    )


def get_track(name: str = "touge") -> Track:
    """Track factory function."""
    name_clean = name.lower()
    if "gymkhana" in name_clean or "arena" in name_clean:
        return build_gymkhana_arena()
    elif "stadium" in name_clean or "formula" in name_clean:
        return build_formula_stadium()
    else:
        return build_touge_track()

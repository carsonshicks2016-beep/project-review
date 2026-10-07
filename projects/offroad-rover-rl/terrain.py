"""
Procedural 3D Off-Road Terrain Engine with Multi-Octave Elevations,
Cross-Axial Moguls, Rocky Crawls, and Boulder Field Collisions.
Includes analytical surface normals and 3D raycast collision queries.
"""
import numpy as np
from config import TERRAIN_LENGTH, TERRAIN_WIDTH, TERRAIN_RESOLUTION

class Boulder:
    def __init__(self, x, y, rx, ry, height):
        self.x = float(x)
        self.y = float(y)
        self.rx = float(rx)       # semi-axis along x
        self.ry = float(ry)       # semi-axis along y
        self.height = float(height)

    def get_elevation_and_normal(self, x, y, base_z):
        dx = (x - self.x) / self.rx
        dy = (y - self.y) / self.ry
        d2 = dx * dx + dy * dy
        if d2 >= 1.0:
            return None, None
        
        # Smooth semi-ellipsoid dome shape
        dome = np.sqrt(max(0.0, 1.0 - d2))
        rock_z = base_z + self.height * dome
        
        # Analytic gradient on the dome:
        # dz/dx = - (height / rx^2) * (x - self.x) / dome
        eps = 1e-4
        denom = max(eps, dome)
        dz_dx = -(self.height / (self.rx**2)) * (x - self.x) / denom
        dz_dy = -(self.height / (self.ry**2)) * (y - self.y) / denom
        normal = np.array([-dz_dx, -dz_dy, 1.0])
        normal = normal / np.linalg.norm(normal)
        return rock_z, normal


class ProceduralTerrain:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.boulders = []
        self._generate_boulder_field()

    def _generate_boulder_field(self):
        """Scatters boulders realistically across the course sections."""
        self.boulders.clear()
        
        # Zone 2: Bouldering Alley (X = 22m to 52m)
        # Moderate to large rocks creating challenging lines
        for _ in range(32):
            bx = self.rng.uniform(22.0, 52.0)
            by = self.rng.uniform(-7.0, 7.0)
            rx = self.rng.uniform(0.45, 1.1)
            ry = self.rng.uniform(0.45, 1.1)
            h = self.rng.uniform(0.35, 0.75)
            self.boulders.append(Boulder(bx, by, rx, ry, h))

        # Zone 3: Mogul Ridge (X = 55m to 88m)
        # Strategic rocks placed in articulation troughs
        for _ in range(24):
            bx = self.rng.uniform(55.0, 88.0)
            by = self.rng.uniform(-6.0, 6.0)
            rx = self.rng.uniform(0.5, 0.95)
            ry = self.rng.uniform(0.5, 0.95)
            h = self.rng.uniform(0.30, 0.65)
            self.boulders.append(Boulder(bx, by, rx, ry, h))

        # Zone 4: The Steep Rocky Step-Up (X = 92m to 126m)
        # Rocky ledges and stair-step boulders on the 30% incline
        for _ in range(28):
            bx = self.rng.uniform(92.0, 126.0)
            by = self.rng.uniform(-6.5, 6.5)
            rx = self.rng.uniform(0.6, 1.3)
            ry = self.rng.uniform(0.6, 1.3)
            h = self.rng.uniform(0.4, 0.9)
            self.boulders.append(Boulder(bx, by, rx, ry, h))

    def _base_elevation(self, x, y):
        """
        Computes continuous base terrain height and analytical gradient.
        Returns: (z, dz_dx, dz_dy)
        """
        # Bank slope barriers on outer edges (keep rover inside track width)
        edge_dist = abs(y)
        bank_z = 0.0
        dbank_dy = 0.0
        if edge_dist > 8.0:
            overshoot = edge_dist - 8.0
            bank_z = 0.45 * (overshoot ** 2)
            dbank_dy = 0.9 * overshoot * np.sign(y)

        # Zone 1: Staging (0 to 20m) - gentle rolling washboard
        # Zone 2: Whoops (20 to 50m) - undulating mounds
        # Zone 3: Moguls (50 to 90m) - cross-axial humps
        # Zone 4: Hill Climb (90 to 130m) - steep incline rising ~12 meters
        # Zone 5: Summit (130 to 150m) - high plateau

        # General upward grade
        ramp = 0.0
        dramp_dx = 0.0
        if x > 15.0 and x <= 130.0:
            # S-curve ramp from 0 to 12 meters
            progress = (x - 15.0) / 115.0
            ramp = 12.0 * (progress ** 1.6)
            dramp_dx = 12.0 * 1.6 * (progress ** 0.6) / 115.0
        elif x > 130.0:
            ramp = 12.0

        # Undulating ripples / washboard
        ripple = 0.25 * np.sin(0.8 * x) * np.cos(0.4 * y)
        drip_dx = 0.25 * 0.8 * np.cos(0.8 * x) * np.cos(0.4 * y)
        drip_dy = -0.25 * 0.4 * np.sin(0.8 * x) * np.sin(0.4 * y)

        # Cross-axial moguls in Zone 3 (50m to 90m)
        mogul = 0.0
        dmog_dx = 0.0
        dmog_dy = 0.0
        if 48.0 < x < 92.0:
            fade = np.sin((x - 48.0) / 44.0 * np.pi)
            dfade = (np.pi / 44.0) * np.cos((x - 48.0) / 44.0 * np.pi)
            raw_mogul = 0.75 * np.sin(0.45 * x + 0.9 * y) * np.cos(0.3 * x - 0.7 * y)
            mogul = fade * raw_mogul
            # Approximate derivative for normal
            dmog_dx = dfade * raw_mogul + fade * (
                0.75 * 0.45 * np.cos(0.45 * x + 0.9 * y) * np.cos(0.3 * x - 0.7 * y) -
                0.75 * 0.3 * np.sin(0.45 * x + 0.9 * y) * np.sin(0.3 * x - 0.7 * y)
            )
            dmog_dy = fade * (
                0.75 * 0.9 * np.cos(0.45 * x + 0.9 * y) * np.cos(0.3 * x - 0.7 * y) +
                0.75 * 0.7 * np.sin(0.45 * x + 0.9 * y) * np.sin(0.3 * x - 0.7 * y)
            )

        z = ramp + ripple + mogul + bank_z
        dz_dx = dramp_dx + drip_dx + dmog_dx
        dz_dy = drip_dy + dmog_dy + dbank_dy

        return z, dz_dx, dz_dy

    def get_elevation_and_normal(self, x, y):
        """
        Returns the combined ground elevation z and unit surface normal vector [nx, ny, nz].
        Considers both continuous landscape and local boulder domes.
        """
        base_z, dz_dx, dz_dy = self._base_elevation(x, y)
        normal = np.array([-dz_dx, -dz_dy, 1.0])
        normal = normal / np.linalg.norm(normal)
        max_z = base_z
        best_normal = normal

        # Check nearby boulders
        for b in self.boulders:
            if abs(x - b.x) <= b.rx and abs(y - b.y) <= b.ry:
                rock_z, rock_norm = b.get_elevation_and_normal(x, y, base_z)
                if rock_z is not None and rock_z > max_z:
                    max_z = rock_z
                    best_normal = rock_norm

        return max_z, best_normal

    def get_height(self, x, y):
        """Fast scalar query for ground elevation at (x, y)."""
        z, _ = self.get_elevation_and_normal(x, y)
        return z

    def raycast(self, origin, direction, max_dist=10.0, num_steps=32):
        """
        Performs 3D ray-marching against the procedural terrain and boulders.
        Returns: (hit_pos, distance, normal, hit_detected)
        """
        dir_unit = direction / np.linalg.norm(direction)
        step_size = max_dist / float(num_steps)
        curr_pos = origin.copy()
        traveled = 0.0

        for _ in range(num_steps):
            traveled += step_size
            curr_pos = origin + dir_unit * traveled
            ground_z, ground_norm = self.get_elevation_and_normal(curr_pos[0], curr_pos[1])
            
            # Intersection detected if ray goes below terrain surface
            if curr_pos[2] <= ground_z:
                # Refine intersection by linear interpolation
                prev_pos = curr_pos - dir_unit * step_size
                prev_ground_z, _ = self.get_elevation_and_normal(prev_pos[0], prev_pos[1])
                diff_prev = prev_pos[2] - prev_ground_z
                diff_curr = ground_z - curr_pos[2]
                total_diff = diff_prev + diff_curr
                alpha = diff_prev / max(1e-6, total_diff)
                hit_pos = prev_pos + dir_unit * (step_size * alpha)
                actual_dist = np.linalg.norm(hit_pos - origin)
                return hit_pos, actual_dist, ground_norm, True

        # Ray did not hit terrain within max_dist
        miss_pos = origin + dir_unit * max_dist
        return miss_pos, max_dist, np.array([0.0, 0.0, 1.0]), False

    def export_terrain_data(self, nx=150, ny=50):
        """
        Exports the terrain elevation grid and boulder list for the Three.js WebGL visualizer.
        """
        xs = np.linspace(-5.0, TERRAIN_LENGTH + 5.0, nx)
        ys = np.linspace(-TERRAIN_WIDTH / 2.0, TERRAIN_WIDTH / 2.0, ny)
        grid = []

        for y in ys:
            row = []
            for x in xs:
                row.append(float(self.get_height(x, y)))
            grid.append(row)

        boulder_data = []
        for b in self.boulders:
            base_z, _, _ = self._base_elevation(b.x, b.y)
            boulder_data.append({
                "x": round(b.x, 2),
                "y": round(b.y, 2),
                "z": round(base_z, 2),
                "rx": round(b.rx, 2),
                "ry": round(b.ry, 2),
                "height": round(b.height, 2)
            })

        return {
            "xs": [round(float(x), 2) for x in xs],
            "ys": [round(float(y), 2) for y in ys],
            "grid": grid,
            "boulders": boulder_data
        }

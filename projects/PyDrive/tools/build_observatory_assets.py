#!/usr/bin/env python3
"""Build clean-room, deterministic assets for the browser Observatory.

This generator intentionally depends on the live Track API, the checked-in
Nordschleife data, and the separately authored Porsche 919 Evo package only.
It never imports or reads any legacy three-dimensional viewer package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "assets" / "vehicles"))
from cleanroom_mesh import shade_scene  # noqa: E402
from supra.track import configure_hills, named_track  # noqa: E402


SCHEMA_WORLD = "supra-observatory-world-v1"
SCHEMA_VEHICLE = "supra-observatory-vehicle-v2"
TRACK_SAMPLES = 6944
MAZDA_SOURCE = REPO / "assets" / "vehicles" / "mazda787b"
PUBLIC = REPO / "observatory" / "public" / "assets" / "observatory"
PORSCHE_SOURCE = REPO / "assets" / "vehicles" / "porsche_919evo"
DGM_CACHE = REPO / ".cache" / "nordschleife_dgm1"
DGM_META = DGM_CACHE / "dgm1_tif_07.meta4"

# WGS84 to UTM zone 32N, kept local so the offline builder does not require
# pyproj or any service call.
_UTM_A = 6378137.0
_UTM_F = 1.0 / 298.257223563
_UTM_K0 = 0.9996
_UTM_E = math.sqrt(_UTM_F * (2.0 - _UTM_F))
_UTM_EP2 = _UTM_E * _UTM_E / (1.0 - _UTM_E * _UTM_E)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_payload(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def file_record(path: str, payload: bytes, **extra) -> dict:
    return {"path": path, "sha256": sha256_bytes(payload), "bytes": len(payload), **extra}


def utm32(lat: float, lon: float) -> tuple[float, float]:
    latr = math.radians(lat)
    lonr = math.radians(lon)
    lon0 = math.radians(9.0)
    sin_lat = math.sin(latr)
    cos_lat = math.cos(latr)
    n = _UTM_A / math.sqrt(1.0 - _UTM_E * _UTM_E * sin_lat * sin_lat)
    t = math.tan(latr) ** 2
    c = _UTM_EP2 * cos_lat * cos_lat
    a = cos_lat * (lonr - lon0)
    m = _UTM_A * (
        (1 - _UTM_E**2 / 4 - 3 * _UTM_E**4 / 64 - 5 * _UTM_E**6 / 256) * latr
        - (3 * _UTM_E**2 / 8 + 3 * _UTM_E**4 / 32 + 45 * _UTM_E**6 / 1024) * math.sin(2 * latr)
        + (15 * _UTM_E**4 / 256 + 45 * _UTM_E**6 / 1024) * math.sin(4 * latr)
        - (35 * _UTM_E**6 / 3072) * math.sin(6 * latr)
    )
    x = _UTM_K0 * n * (
        a + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t * t + 72 * c - 58 * _UTM_EP2) * a**5 / 120
    ) + 500000.0
    y = _UTM_K0 * (
        m + n * math.tan(latr) * (
            a**2 / 2 + (5 - t + 9 * c + 4 * c * c) * a**4 / 24
            + (61 - 58 * t + t * t + 600 * c - 330 * _UTM_EP2) * a**6 / 720
        )
    )
    return x, y


class OfflineDgmTerrain:
    """Validated, local-only DGM1 tile sampler and LOD mesh source."""

    tile_pattern = re.compile(r"dgm1_32_(\d+)_(\d+)_1_rp_(\d{4})\.tif$")

    def __init__(self, track_asset: dict, runtime_track):
        if not DGM_CACHE.is_dir():
            raise RuntimeError(f"required offline DGM cache is missing: {DGM_CACHE}")
        if not DGM_META.is_file():
            raise RuntimeError(f"required offline DGM manifest is missing: {DGM_META}")
        latlon = np.asarray(track_asset.get("latlon"), dtype=float)
        if latlon.shape != (TRACK_SAMPLES, 2):
            raise RuntimeError("Nordschleife source does not contain the expected lat/lon samples")
        self.utm_centerline = np.asarray([utm32(float(lat), float(lon)) for lat, lon in latlon], dtype=float)
        self.scale = float(track_asset["length_scale"]) * float(track_asset["final_chord_scale"])
        self.origin_utm = self.utm_centerline.mean(axis=0)
        aligned = (self.utm_centerline - self.origin_utm) * self.scale
        alignment_error = np.linalg.norm(aligned - runtime_track.center, axis=1)
        self.max_alignment_error_m = float(alignment_error.max())
        if self.max_alignment_error_m > 0.15:
            raise RuntimeError(f"DGM/simulator coordinate alignment drifted by {self.max_alignment_error_m:.4f} m")
        self.required_keys = sorted({(int(x // 1000), int(y // 1000)) for x, y in self.utm_centerline})
        discovered: dict[tuple[int, int], Path] = {}
        for path in sorted(DGM_CACHE.glob("*.tif")):
            match = self.tile_pattern.fullmatch(path.name)
            if not match:
                continue
            key = (int(match.group(1)), int(match.group(2)))
            if key in discovered:
                raise RuntimeError(f"duplicate cached DGM tile for {key}: {path.name}")
            discovered[key] = path
        missing = [key for key in self.required_keys if key not in discovered]
        if missing:
            raise RuntimeError(f"required offline DGM tiles are missing: {missing}")
        self.paths = {key: discovered[key] for key in self.required_keys}
        self.tiles: dict[tuple[int, int], dict] = {}

    def sim_to_utm(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=float) / self.scale + self.origin_utm

    def utm_to_sim(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, dtype=float) - self.origin_utm) * self.scale

    def load(self, key: tuple[int, int]) -> dict:
        if key in self.tiles:
            return self.tiles[key]
        path = self.paths[key]
        with Image.open(path) as image:
            array = np.asarray(image, dtype=np.float32)
            tie = tuple(float(value) for value in image.tag_v2[33922])
            pixel = tuple(float(value) for value in image.tag_v2[33550])
            nodata = float(image.tag_v2.get(42113, -9999))
            crs = str(image.tag_v2.get(34737, ""))
        if array.shape != (1000, 1000) or pixel[:2] != (1.0, 1.0):
            raise RuntimeError(f"unexpected DGM raster contract for {path.name}: {array.shape} {pixel[:2]}")
        expected_origin = (key[0] * 1000.0, (key[1] + 1) * 1000.0)
        if not np.allclose((tie[3], tie[4]), expected_origin, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"DGM georeference drift for {path.name}: {(tie[3], tie[4])}")
        if "UTM_Zone_32N" not in crs and "UTM zone 32N" not in crs:
            raise RuntimeError(f"DGM CRS is not ETRS89 / UTM zone 32N: {path.name}")
        item = {
            "array": array, "ox": tie[3], "oy": tie[4],
            "px": pixel[0], "py": pixel[1], "nodata": nodata,
            "path": path,
        }
        self.tiles[key] = item
        return item

    def sample_utm(self, x: float, y: float) -> float:
        key = (int(x // 1000), int(y // 1000))
        if key not in self.paths:
            return float("nan")
        tile = self.load(key)
        col = (x - tile["ox"]) / tile["px"]
        row = (tile["oy"] - y) / tile["py"]
        c0, r0 = int(math.floor(col)), int(math.floor(row))
        if c0 < 0 or r0 < 0 or c0 >= 999 or r0 >= 999:
            return float("nan")
        dc, dr = col - c0, row - r0
        values = tile["array"][r0:r0 + 2, c0:c0 + 2].astype(float).reshape(-1)
        if np.any(values == tile["nodata"]) or not np.all(np.isfinite(values)):
            return float("nan")
        return float(
            (1 - dc) * (1 - dr) * values[0] + dc * (1 - dr) * values[1]
            + (1 - dc) * dr * values[2] + dc * dr * values[3]
        )

    def sample_sim(self, x: float, y: float) -> float:
        utm = self.sim_to_utm(np.asarray((x, y)))
        return self.sample_utm(float(utm[0]), float(utm[1]))

    def normal_sim(self, x: float, y: float, step_m: float = 1.5) -> np.ndarray:
        """Return a local DGM normal in simulator x/y/z coordinates.

        This is visual placement data only.  It lets scenery follow the same
        surveyed surface the browser renders without ever affecting physics.
        """
        center = self.sample_sim(x, y)
        xp, xm = self.sample_sim(x + step_m, y), self.sample_sim(x - step_m, y)
        yp, ym = self.sample_sim(x, y + step_m), self.sample_sim(x, y - step_m)
        if not all(math.isfinite(value) for value in (center, xp, xm, yp, ym)):
            return np.asarray((0.0, 0.0, 1.0), dtype=float)
        normal = np.asarray((-(xp - xm) / (2 * step_m), -(yp - ym) / (2 * step_m), 1.0), dtype=float)
        return normal / max(float(np.linalg.norm(normal)), 1e-9)

    def input_records(self) -> list[dict]:
        records = []
        for key in self.required_keys:
            tile = self.load(key)
            path = tile["path"]
            records.append({
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
                "tile": [key[0], key[1]],
                "utm_bounds_m": [tile["ox"], tile["oy"] - 1000.0, tile["ox"] + 1000.0, tile["oy"]],
                "pixel_size_m": 1.0,
            })
        return records

    def lod_mesh(self, key: tuple[int, int], stride: int, runtime_track, exclusion_m: float) -> tuple[trimesh.Trimesh, dict]:
        tile = self.load(key)
        columns = np.unique(np.append(np.arange(0, 1000, stride, dtype=int), 999))
        rows = np.unique(np.append(np.arange(0, 1000, stride, dtype=int), 999))
        grid_col, grid_row = np.meshgrid(columns, rows)
        utm_xy = np.column_stack((
            tile["ox"] + grid_col.reshape(-1) * tile["px"],
            tile["oy"] - grid_row.reshape(-1) * tile["py"],
        ))
        sim_xy = self.utm_to_sim(utm_xy)
        elevations = tile["array"][np.ix_(rows, columns)].astype(float).reshape(-1)
        valid = np.isfinite(elevations) & (elevations != tile["nodata"])
        nearest_distance, nearest_index = cKDTree(runtime_track.center).query(sim_xy, k=1)
        road_clearance = runtime_track.half_width[nearest_index] + exclusion_m
        signed_clearance = nearest_distance - road_clearance
        source_vertices = np.column_stack((sim_xy, elevations))
        row_count, column_count = len(rows), len(columns)
        vertices: list[np.ndarray] = []
        faces = []

        def append_vertex(vertex: np.ndarray) -> int:
            vertices.append(vertex)
            return len(vertices) - 1

        def clipped_triangle(indices: tuple[int, int, int]) -> None:
            # Clip every grid triangle against the road/shoulder envelope.  The
            # former all-four-corners test was conservative by up to a tile
            # diagonal, producing visible white cracks between the DGM and the
            # authored verge.  This keeps the same non-overlap policy while
            # making the shared boundary continuous.
            if not valid[list(indices)].all():
                return
            polygon = [(source_vertices[index], float(signed_clearance[index])) for index in indices]
            clipped: list[tuple[np.ndarray, float]] = []
            for current, following in zip(polygon, polygon[1:] + polygon[:1]):
                current_inside, following_inside = current[1] >= 0.0, following[1] >= 0.0
                if current_inside:
                    clipped.append(current)
                if current_inside != following_inside:
                    fraction = current[1] / (current[1] - following[1])
                    clipped.append((current[0] + (following[0] - current[0]) * fraction, 0.0))
            if len(clipped) < 3:
                return
            first = append_vertex(clipped[0][0])
            previous = append_vertex(clipped[1][0])
            for entry in clipped[2:]:
                current = append_vertex(entry[0])
                faces.append((first, previous, current))
                previous = current

        for row in range(row_count - 1):
            for column in range(column_count - 1):
                a = row * column_count + column
                b, c, d = a + 1, a + column_count, a + column_count + 1
                clipped_triangle((a, c, b))
                clipped_triangle((b, c, d))
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)
        # `append_vertex` emits a fresh vertex per triangle corner (it has to,
        # for the clip path), which leaves the grid fully unwelded — 3x the
        # vertex data, and smooth normals that cannot span a shared edge.
        # Welding is deterministic and leaves the triangle set untouched.
        mesh.merge_vertices()
        mesh = paint(mesh, WORLD_MATERIALS["grass" if stride <= 25 else "forest_floor"])
        metadata = {
            "tile": [key[0], key[1]], "sample_stride_m": stride,
            "grid": [column_count, row_count], "triangle_count": len(faces),
            "road_exclusion_m_beyond_half_width": exclusion_m,
            "source_sha256": sha256_path(tile["path"]),
            "sim_bounds_xy_m": [
                round(float(sim_xy[:, 0].min()), 4), round(float(sim_xy[:, 1].min()), 4),
                round(float(sim_xy[:, 0].max()), 4), round(float(sim_xy[:, 1].max()), 4),
            ],
        }
        return mesh, metadata


def material(name: str, rgba: tuple[int, int, int, int], metallic=0.0, roughness=0.7):
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=np.asarray(rgba, dtype=np.uint8),
        metallicFactor=float(metallic),
        roughnessFactor=float(roughness),
    )


WORLD_MATERIALS = {
    "road": material("truth_asphalt", (38, 43, 45, 255), 0.0, 0.96),
    "shoulder": material("truth_shoulders", (83, 87, 82, 255), 0.0, 0.92),
    "grass": material("visual_grass", (61, 91, 52, 255), 0.0, 1.0),
    "forest_floor": material("visual_forest_floor", (35, 62, 37, 255), 0.0, 1.0),
    "rail": material("visual_guardrail", (141, 149, 147, 255), 0.7, 0.33),
    "fence": material("visual_fence", (69, 77, 75, 255), 0.55, 0.58),
    "landmark": material("visual_landmark", (222, 226, 215, 255), 0.1, 0.72),
    "accent": material("visual_landmark_accent", (224, 102, 40, 255), 0.05, 0.5),
}

def paint(mesh: trimesh.Trimesh, mat) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


def box(extents, center, mat) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, mat)


def strip_mesh(a: np.ndarray, b: np.ndarray, mat, longitudinal_uv: np.ndarray | None = None) -> trimesh.Trimesh:
    count = len(a)
    vertices = np.empty((count * 2, 3), dtype=np.float64)
    vertices[0::2] = a
    vertices[1::2] = b
    indices = np.arange(count, dtype=np.int64)
    nxt = (indices + 1) % count
    faces = np.column_stack((indices * 2, nxt * 2, indices * 2 + 1))
    faces2 = np.column_stack((indices * 2 + 1, nxt * 2, nxt * 2 + 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack((faces, faces2)), process=False)
    if longitudinal_uv is None:
        return paint(mesh, mat)
    u = np.asarray(longitudinal_uv, dtype=float)
    if u.shape != (count,):
        raise ValueError(f"longitudinal UV must contain {count} samples, received {u.shape}")
    uv = np.empty((count * 2, 2), dtype=float)
    uv[0::2, 0] = u
    uv[1::2, 0] = u
    uv[0::2, 1] = 0.0
    uv[1::2, 1] = 1.0
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return mesh


def segmented_strip_mesh(a: np.ndarray, b: np.ndarray, parity: int, mat, segment_samples: int = 8) -> trimesh.Trimesh:
    count = len(a)
    vertices = np.empty((count * 2, 3), dtype=np.float64)
    vertices[0::2], vertices[1::2] = a, b
    faces = []
    for index in range(count):
        if (index // segment_samples) % 2 != parity:
            continue
        nxt = (index + 1) % count
        faces.extend(((index * 2, nxt * 2, index * 2 + 1), (index * 2 + 1, nxt * 2, nxt * 2 + 1)))
    return paint(trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False), mat)


def vertical_ribbon(points: np.ndarray, height: float, mat) -> trimesh.Trimesh:
    lower = points.copy()
    upper = points.copy()
    upper[:, 2] += height
    return strip_mesh(lower, upper, mat)


def terrain_skirt(points: np.ndarray, depth: float, mat) -> trimesh.Trimesh:
    """A downward seam curtain that hides a DGM/road-envelope gap without
    creating a second coplanar ground surface."""
    upper = points.copy()
    lower = points.copy()
    lower[:, 2] -= depth
    return strip_mesh(lower, upper, mat)


def containment_terrain_mesh(dgm: OfflineDgmTerrain, runtime_track, exclusion_m: float, depth_m: float = 14.0, padding_m: float = 1800.0, stride_m: float = 120.0) -> trimesh.Trimesh:
    """Build a non-coplanar visual catch surface for gaps beyond cached tiles.

    This is not the retired near-road underlay: it is clipped out of the same
    road envelope, sits materially below the surveyed DGM, and only appears
    through areas where no primary terrain is present.  It keeps atmosphere
    from leaking through sparse source-tile coverage at broadcast camera
    distances without participating in any simulation surface.
    """
    bounds = []
    for key in dgm.required_keys:
        tile = dgm.load(key)
        corners = dgm.utm_to_sim(np.asarray((
            (tile["ox"], tile["oy"]),
            (tile["ox"] + 1000.0, tile["oy"] - 1000.0),
        )))
        bounds.append((corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()))
    minimum_x = min(item[0] for item in bounds) - padding_m
    minimum_y = min(item[1] for item in bounds) - padding_m
    maximum_x = max(item[2] for item in bounds) + padding_m
    maximum_y = max(item[3] for item in bounds) + padding_m
    xs = np.unique(np.append(np.arange(minimum_x, maximum_x, stride_m), maximum_x))
    ys = np.unique(np.append(np.arange(minimum_y, maximum_y, stride_m), maximum_y))
    grid_x, grid_y = np.meshgrid(xs, ys)
    sim_xy = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    tile_centers = []
    for key in dgm.required_keys:
        tile = dgm.load(key)
        center = dgm.utm_to_sim(np.asarray(((tile["ox"] + 500.0, tile["oy"] - 500.0),)))[0]
        tile_centers.append((key, center))

    def closed_height(x: float, y: float) -> float:
        value = dgm.sample_sim(x, y)
        if math.isfinite(value):
            return value - depth_m
        # Outside the sparse cache, deterministically drape the shell from the
        # nearest cached tile edge rather than using a flat horizon plane.
        key, _ = min(tile_centers, key=lambda item: float(np.sum((item[1] - np.asarray((x, y))) ** 2)))
        tile = dgm.load(key)
        utm = dgm.sim_to_utm(np.asarray((x, y)))
        column = int(np.clip(round((utm[0] - tile["ox"]) / tile["px"]), 0, 999))
        row = int(np.clip(round((tile["oy"] - utm[1]) / tile["py"]), 0, 999))
        return float(tile["array"][row, column]) - depth_m

    elevations = np.asarray([closed_height(float(x), float(y)) for x, y in sim_xy])
    nearest_distance, nearest_index = cKDTree(runtime_track.center).query(sim_xy, k=1)
    signed_clearance = nearest_distance - (runtime_track.half_width[nearest_index] + exclusion_m)
    source_vertices = np.column_stack((sim_xy, elevations))
    row_count, column_count = len(ys), len(xs)
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    def append_vertex(vertex: np.ndarray) -> int:
        vertices.append(vertex)
        return len(vertices) - 1

    def clip_triangle(indices: tuple[int, int, int]) -> None:
        polygon = [(source_vertices[index], float(signed_clearance[index])) for index in indices]
        clipped: list[tuple[np.ndarray, float]] = []
        for current, following in zip(polygon, polygon[1:] + polygon[:1]):
            current_inside, following_inside = current[1] >= 0.0, following[1] >= 0.0
            if current_inside:
                clipped.append(current)
            if current_inside != following_inside:
                fraction = current[1] / (current[1] - following[1])
                clipped.append((current[0] + (following[0] - current[0]) * fraction, 0.0))
        if len(clipped) < 3:
            return
        first, previous = append_vertex(clipped[0][0]), append_vertex(clipped[1][0])
        for entry in clipped[2:]:
            current = append_vertex(entry[0])
            faces.append((first, previous, current))
            previous = current

    for row in range(row_count - 1):
        for column in range(column_count - 1):
            a = row * column_count + column
            b, c, d = a + 1, a + column_count, a + column_count + 1
            clip_triangle((a, c, b))
            clip_triangle((b, c, d))
    return paint(trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False), WORLD_MATERIALS["forest_floor"])


def track_hash(track) -> str:
    digest = hashlib.sha256()
    digest.update(b"fable-observatory-track-v1\0")
    for label in ("center", "arc", "z", "half_width", "bank"):
        values = getattr(track, label)
        array = np.asarray(values, dtype="<f8")
        digest.update(label.encode("ascii") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_world() -> dict[str, bytes]:
    configure_hills(enabled=True, scale=1.0, force_flat=False)
    track = named_track("nordschleife")
    if len(track.center) != TRACK_SAMPLES:
        raise RuntimeError(f"expected {TRACK_SAMPLES} runtime samples, got {len(track.center)}")

    track_source_path = REPO / "supra" / "data" / "tracks" / "nordschleife.json"
    track_asset = json.loads(track_source_path.read_text())
    dgm = OfflineDgmTerrain(track_asset, track)
    center = np.column_stack((track.center, track.z))
    bank_rise = np.tan(track.bank) * track.half_width
    left = np.column_stack((track.left, track.z + bank_rise))
    right = np.column_stack((track.right, track.z - bank_rise))
    shoulder_width = 2.75
    outer_left_xy = track.center + track.normal * (track.half_width + shoulder_width)[:, None]
    outer_right_xy = track.center - track.normal * (track.half_width + shoulder_width)[:, None]
    outer_left = np.column_stack((outer_left_xy, track.z + np.tan(track.bank) * (track.half_width + shoulder_width)))
    outer_right = np.column_stack((outer_right_xy, track.z - np.tan(track.bank) * (track.half_width + shoulder_width)))

    truth = trimesh.Scene(base_frame="nordschleife_truth_origin")
    road_uv = np.asarray(track.arc, dtype=float) / 18.0
    road_mesh = strip_mesh(left, right, WORLD_MATERIALS["road"], road_uv)
    truth.add_geometry(road_mesh, node_name="truth_road", geom_name="truth_road")
    truth.add_geometry(strip_mesh(left, outer_left, WORLD_MATERIALS["shoulder"], road_uv), node_name="truth_shoulder_left", geom_name="truth_shoulder_left")
    truth.add_geometry(strip_mesh(outer_right, right, WORLD_MATERIALS["shoulder"], road_uv), node_name="truth_shoulder_right", geom_name="truth_shoulder_right")
    truth.metadata.update({
        "schema": SCHEMA_WORLD,
        "layer": "simulation_truth",
        "sample_count": TRACK_SAMPLES,
        "note": "Exact runtime-smoothed Track road and shoulder samples; visualization only.",
    })
    # Smooth-only: the exported road is validated against an exact vertex
    # layout below, so splitting at creases would break that check.
    shade_scene(truth, allow_split=False)
    truth_glb = bytes(truth.export(file_type="glb"))
    exported_truth = trimesh.load(io.BytesIO(truth_glb), file_type="glb", force="scene")
    expected_road_vertices = np.empty((TRACK_SAMPLES * 2, 3), dtype=float)
    expected_road_vertices[0::2], expected_road_vertices[1::2] = left, right
    exported_road_vertices = np.asarray(exported_truth.geometry["truth_road"].vertices, dtype=float)
    forward_error = cKDTree(exported_road_vertices).query(expected_road_vertices, k=1)[0]
    reverse_error = cKDTree(expected_road_vertices).query(exported_road_vertices, k=1)[0]
    road_vertex_max_error_m = float(max(forward_error.max(), reverse_error.max()))
    if road_vertex_max_error_m > 0.02:
        raise RuntimeError(f"exported truth road drifted {road_vertex_max_error_m:.6f} m from runtime Track")

    # Close verges bridge the smoothed truth road to the local DGM surface.
    # Every outer elevation is sampled from a cached GeoTIFF; the centerline
    # source elevation (itself DGM-derived) is the deterministic nodata fallback.
    raw_center_elevation = np.asarray(track_asset["elevation"], dtype=float)
    # This is also the terrain cut-out boundary below.  Keeping one shared
    # envelope means a verge never overlaps a terrain triangle in depth.
    terrain_envelope_m = 18.0
    verge_distance = track.half_width + terrain_envelope_m
    verge_left_xy = track.center + track.normal * verge_distance[:, None]
    verge_right_xy = track.center - track.normal * verge_distance[:, None]
    verge_fallbacks = 0

    def verge_points(xy_values: np.ndarray) -> np.ndarray:
        nonlocal verge_fallbacks
        z_values = []
        for index, xy in enumerate(xy_values):
            value = dgm.sample_sim(float(xy[0]), float(xy[1]))
            if not math.isfinite(value):
                value = float(raw_center_elevation[index])
                verge_fallbacks += 1
            z_values.append(value)
        return np.column_stack((xy_values, np.asarray(z_values)))

    verge_left = verge_points(verge_left_xy)
    verge_right = verge_points(verge_right_xy)
    visual = trimesh.Scene(base_frame="nordschleife_visual_origin")
    visual.add_geometry(strip_mesh(outer_left, verge_left, WORLD_MATERIALS["grass"]), node_name="visual_dgm_verge_left", geom_name="visual_dgm_verge_left")
    visual.add_geometry(strip_mesh(verge_right, outer_right, WORLD_MATERIALS["forest_floor"]), node_name="visual_dgm_verge_right", geom_name="visual_dgm_verge_right")
    containment_depth_m = 14.0
    visual.add_geometry(terrain_skirt(verge_left, containment_depth_m + 1.0, WORLD_MATERIALS["grass"]), node_name="visual_dgm_seam_skirt_left", geom_name="visual_dgm_seam_skirt_left")
    visual.add_geometry(terrain_skirt(verge_right, containment_depth_m + 1.0, WORLD_MATERIALS["forest_floor"]), node_name="visual_dgm_seam_skirt_right", geom_name="visual_dgm_seam_skirt_right")
    containment = containment_terrain_mesh(dgm, track, terrain_envelope_m)
    visual.add_geometry(containment, node_name="visual_terrain_containment_shell", geom_name="visual_terrain_containment_shell")

    curb_width = 0.45
    curb_left_outer_xy = track.center + track.normal * (track.half_width + curb_width)[:, None]
    curb_right_outer_xy = track.center - track.normal * (track.half_width + curb_width)[:, None]
    curb_left_inner = left.copy(); curb_left_inner[:, 2] += 0.025
    curb_right_inner = right.copy(); curb_right_inner[:, 2] += 0.025
    curb_left_outer = np.column_stack((curb_left_outer_xy, track.z + np.tan(track.bank) * (track.half_width + curb_width) + 0.025))
    curb_right_outer = np.column_stack((curb_right_outer_xy, track.z - np.tan(track.bank) * (track.half_width + curb_width) + 0.025))
    for side, inner, outer in (("left", curb_left_inner, curb_left_outer), ("right", curb_right_inner, curb_right_outer)):
        visual.add_geometry(segmented_strip_mesh(inner, outer, 0, WORLD_MATERIALS["landmark"]), node_name=f"visual_curb_{side}_light", geom_name=f"visual_curb_{side}_light")
        visual.add_geometry(segmented_strip_mesh(inner, outer, 1, WORLD_MATERIALS["accent"]), node_name=f"visual_curb_{side}_accent", geom_name=f"visual_curb_{side}_accent")

    sampled = np.arange(0, TRACK_SAMPLES, 4)
    for side_name, sign in (("left", 1.0), ("right", -1.0)):
        distance = track.half_width[sampled] + 4.25
        xy = track.center[sampled] + track.normal[sampled] * (sign * distance[:, None])
        z = np.asarray([dgm.sample_sim(float(x), float(y)) for x, y in xy])
        z = np.where(np.isfinite(z), z, track.z[sampled]) + 0.42
        rail = np.column_stack((xy, z))
        visual.add_geometry(vertical_ribbon(rail, 0.34, WORLD_MATERIALS["rail"]), node_name=f"visual_guardrail_{side_name}", geom_name=f"visual_guardrail_{side_name}")

    # Narrow DGM-stamped drainage runs add readable road-edge construction
    # without becoming a collision or simulation surface.
    drainage_sampled = np.arange(0, TRACK_SAMPLES, 5)
    for side_name, sign in (("left", 1.0), ("right", -1.0)):
        distance = track.half_width[drainage_sampled] + 2.92
        xy = track.center[drainage_sampled] + track.normal[drainage_sampled] * (sign * distance[:, None])
        z = np.asarray([dgm.sample_sim(float(x), float(y)) for x, y in xy])
        z = np.where(np.isfinite(z), z, track.z[drainage_sampled]) + 0.035
        inner = np.column_stack((xy, z))
        outer_xy = xy + track.normal[drainage_sampled] * (sign * 0.28)
        outer = np.column_stack((outer_xy, z + 0.012))
        visual.add_geometry(strip_mesh(inner, outer, WORLD_MATERIALS["fence"]), node_name=f"visual_drainage_{side_name}", geom_name=f"visual_drainage_{side_name}")

    fence_sampled = np.arange(0, TRACK_SAMPLES, 8)
    for side_name, sign in (("left", 1.0), ("right", -1.0)):
        distance = track.half_width[fence_sampled] + 8.5
        xy = track.center[fence_sampled] + track.normal[fence_sampled] * (sign * distance[:, None])
        z = np.asarray([dgm.sample_sim(float(x), float(y)) for x, y in xy])
        z = np.where(np.isfinite(z), z, track.z[fence_sampled]) + 0.1
        fence = np.column_stack((xy, z))
        visual.add_geometry(vertical_ribbon(fence, 1.5, WORLD_MATERIALS["fence"]), node_name=f"visual_safety_fence_{side_name}", geom_name=f"visual_safety_fence_{side_name}")

    landmark_data = []
    for landmark_index, landmark in enumerate(track.landmarks):
        arc_mid = 0.5 * (float(landmark["start_arc"]) + float(landmark["end_arc"]))
        index = int(np.argmin(np.abs(track.arc - arc_mid)))
        sign = 1 if landmark_index % 2 == 0 else -1
        distance = float(track.half_width[index] + 7.0)
        xy = track.center[index] + track.normal[index] * sign * distance
        terrain_z = dgm.sample_sim(float(xy[0]), float(xy[1]))
        if not math.isfinite(terrain_z):
            terrain_z = float(track.z[index])
        position = np.array((xy[0], xy[1], terrain_z + 1.25))
        safe_name = re.sub(r"[^a-z0-9]+", "_", landmark["name"].lower()).strip("_")
        visual.add_geometry(
            box((0.32, 0.32, 2.5), position, WORLD_MATERIALS["landmark"]),
            node_name=f"visual_landmark_{landmark_index:02d}_{safe_name}",
            geom_name=f"visual_landmark_{landmark_index:02d}_{safe_name}",
        )
        landmark_data.append({
            "name": landmark["name"], "sample_index": index,
            "position": np.round(position, 5).tolist(),
            "normal": np.round(dgm.normal_sim(float(xy[0]), float(xy[1])), 6).tolist(),
            "source": "track landmark arc; visual marker is project-authored and landmark-inspired",
        })

    # A deliberately generic timing gantry marks the runtime start/finish sample.
    # It needs uprights and a sign face, not just a deck: a bare beam floating
    # 4 m over the road smears across the whole frame from a chase camera, which
    # is exactly where the car sits at the start of every episode.
    start = center[0]
    heading = math.atan2(track.tangent[0, 1], track.tangent[0, 0])
    span = float(track.width_profile[0] + 3.0)
    leg_y = span * 0.5
    beam_z = start[2] + 5.4
    gantry_parts = [
        # Cross beam, then the sign board hanging just under it.
        box((0.30, span, 0.34), (start[0], start[1], beam_z), WORLD_MATERIALS["accent"]),
        box((0.12, span * 0.42, 0.86), (start[0], start[1], beam_z - 0.62), WORLD_MATERIALS["accent"]),
    ]
    for side in (-1.0, 1.0):
        leg_center_z = start[2] + (beam_z - start[2]) * 0.5
        gantry_parts.append(
            box((0.34, 0.34, beam_z - start[2]), (start[0], start[1] + side * leg_y, leg_center_z), WORLD_MATERIALS["accent"])
        )
        # Small diagonal-ish brace block where each leg meets the beam.
        gantry_parts.append(
            box((0.24, 0.9, 0.24), (start[0], start[1] + side * (leg_y - 0.55), beam_z - 0.30), WORLD_MATERIALS["accent"])
        )
    gantry = trimesh.util.concatenate(gantry_parts)
    gantry.apply_transform(trimesh.transformations.rotation_matrix(heading, (0, 0, 1), point=start))
    visual.add_geometry(gantry, node_name="visual_landmark_start_gantry", geom_name="visual_landmark_start_gantry")
    visual.metadata.update({
        "schema": SCHEMA_WORLD,
        "layer": "visual_only",
        "collision_authority": False,
        "note": "Offline DGM-derived visual verges, drainage, rails, fences, and landmark-inspired markers; never collision authority.",
    })
    shade_scene(visual)
    visual_glb = bytes(visual.export(file_type="glb"))

    reflector_data = []
    guardrail_post_data = []
    drainage_data = []
    marshal_data = []
    for index in range(0, TRACK_SAMPLES, 84):
        point, next_point = track.center[index], track.center[(index + 3) % TRACK_SAMPLES]
        tangent = next_point - point
        length = max(0.001, float(np.linalg.norm(tangent)))
        normal_xy = np.asarray((-tangent[1] / length, tangent[0] / length))
        offset = (float(track.half_width[index]) + 3.15)
        for sign in (-1.0, 1.0):
            xy = point + normal_xy * offset * sign
            z = dgm.sample_sim(float(xy[0]), float(xy[1]))
            if not math.isfinite(z):
                z = float(track.z[index])
            reflector_data.append({
                "position": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(z), 4)],
                "yaw": round(float(math.atan2(tangent[1], tangent[0])), 7),
                "normal": np.round(dgm.normal_sim(float(xy[0]), float(xy[1])), 6).tolist(),
            })

    for index in range(0, TRACK_SAMPLES, 22):
        point, next_point = track.center[index], track.center[(index + 3) % TRACK_SAMPLES]
        tangent = next_point - point
        length = max(0.001, float(np.linalg.norm(tangent)))
        normal_xy = np.asarray((-tangent[1] / length, tangent[0] / length))
        yaw = round(float(math.atan2(tangent[1], tangent[0])), 7)
        for sign in (-1.0, 1.0):
            rail_xy = point + normal_xy * sign * (float(track.half_width[index]) + 4.25)
            drain_xy = point + normal_xy * sign * (float(track.half_width[index]) + 3.05)
            rail_z = dgm.sample_sim(float(rail_xy[0]), float(rail_xy[1]))
            drain_z = dgm.sample_sim(float(drain_xy[0]), float(drain_xy[1]))
            guardrail_post_data.append({
                "position": [round(float(rail_xy[0]), 4), round(float(rail_xy[1]), 4), round(float(rail_z if math.isfinite(rail_z) else track.z[index]), 4)],
                "yaw": yaw,
            })
            drainage_data.append({
                "position": [round(float(drain_xy[0]), 4), round(float(drain_xy[1]), 4), round(float(drain_z if math.isfinite(drain_z) else track.z[index]), 4)],
                "yaw": yaw,
            })
        if index % 484 == 0:
            sign = -1.0 if (index // 484) % 2 else 1.0
            xy = point + normal_xy * sign * (float(track.half_width[index]) + 10.5)
            z = dgm.sample_sim(float(xy[0]), float(xy[1]))
            marshal_data.append({
                "position": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(z if math.isfinite(z) else track.z[index]), 4)],
                "yaw": yaw,
            })

    terrain_files: dict[str, bytes] = {}
    terrain_levels = []
    # The DGM source is 1 m LiDAR, so a 25 m stride threw away almost all of it
    # and left the near hills reading as big flat facets. 12 m is ~4x the
    # triangles for the near band while the coarse level still carries the
    # hinterland cheaply.
    for level, stride, display_distance in ((0, 12, 900), (1, 100, 4000)):
        records = []
        for key in dgm.required_keys:
            mesh, metadata = dgm.lod_mesh(key, stride, track, exclusion_m=terrain_envelope_m)
            tile_scene = trimesh.Scene(base_frame="nordschleife_dgm_visual_origin")
            node_name = f"visual_dgm_tile_{key[0]}_{key[1]}_lod{level}"
            tile_scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)
            tile_scene.metadata.update({
                "schema": SCHEMA_WORLD, "layer": "visual_only_dgm_terrain",
                "lod": level, "collision_authority": False,
                "source_sha256": metadata["source_sha256"],
            })
            # Heightfield: no real creases, and smooth-only keeps the
            # tile vertex count (and download) where it is.
            shade_scene(tile_scene, allow_split=False)
            payload = bytes(tile_scene.export(file_type="glb"))
            relative = f"terrain/lod{level}/dgm1_{key[0]}_{key[1]}_lod{level}.glb"
            terrain_files[relative] = payload
            records.append(file_record(relative, payload, node=node_name, **metadata))
        terrain_levels.append({
            "level": level, "sample_stride_m": stride,
            "display_distance_m": display_distance, "files": records,
        })

    runtime_hash = track_hash(track)
    rng = np.random.default_rng(int(runtime_hash[:16], 16))
    forest = []
    for index in range(0, TRACK_SAMPLES, 7):
        for sign in (-1, 1):
            if rng.random() < 0.12:
                continue
            hero = rng.random() < 0.16
            distance = float(rng.uniform(21.0, 52.0) if hero else rng.uniform(48.0, 118.0))
            xy = track.center[index] + track.normal[index] * sign * distance
            z = dgm.sample_sim(float(xy[0]), float(xy[1]))
            if not math.isfinite(z):
                continue
            forest.append({
                "position": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(z, 4)],
                "normal": np.round(dgm.normal_sim(float(xy[0]), float(xy[1])), 6).tolist(),
                "scale": round(float(rng.uniform(0.88, 1.32) if hero else rng.uniform(0.58, 1.06)), 4),
                "rotation": round(float(rng.uniform(0.0, math.tau)), 5),
                "species": int(rng.integers(0, 3)),
                "tier": "hero" if hero else "mid",
            })
    for index in range(0, TRACK_SAMPLES, 13):
        for sign in (-1, 1):
            if rng.random() < 0.08:
                continue
            distance = float(rng.uniform(122.0, 265.0))
            xy = track.center[index] + track.normal[index] * sign * distance
            z = dgm.sample_sim(float(xy[0]), float(xy[1]))
            if not math.isfinite(z):
                continue
            forest.append({
                "position": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(z, 4)],
                "normal": np.round(dgm.normal_sim(float(xy[0]), float(xy[1])), 6).tolist(),
                "scale": round(float(rng.uniform(0.52, 0.92)), 4),
                "rotation": round(float(rng.uniform(0.0, math.tau)), 5),
                "species": int(rng.integers(0, 3)),
                "tier": "horizon",
            })

    world_data = {
        "schema": SCHEMA_WORLD,
        "coordinate_system": {"units": "m", "forward_plane": "xy", "up_axis": "+z"},
        "track": {
            "id": "nordschleife_runtime_smoothed",
            "sample_count": TRACK_SAMPLES,
            "length_m": round(float(track.length), 8),
            "centerline": np.round(center, 5).tolist(),
            "heading": np.round(np.arctan2(track.tangent[:, 1], track.tangent[:, 0]), 8).tolist(),
            "width_m": np.round(track.width_profile, 5).tolist(),
            "bank_rad": np.round(track.bank, 8).tolist(),
        },
        "forest": forest,
        "landmarks": landmark_data,
        "trackside": {
            "reflectors": reflector_data,
            "guardrail_posts": guardrail_post_data,
            "drainage_markers": drainage_data,
            "marshal_stations": marshal_data,
        },
        "camera_anchor_stride_samples": 180,
        "terrain": {
            "source": "Rheinland-Pfalz DGM1 GeoTIFF cache",
            "visual_only": True,
            "collision_authority": False,
            "lod_levels": [{"level": item["level"], "sample_stride_m": item["sample_stride_m"]} for item in terrain_levels],
            "surface_stack": {
                "terrain_cutout_from_road_edge_m": terrain_envelope_m,
                "shoulder_width_m": shoulder_width,
                "seam_skirt_depth_m": containment_depth_m + 1.0,
                "containment_shell_depth_m": containment_depth_m,
                "overlap_policy": "terrain triangles are excluded inside the verge envelope; the containment shell is separately clipped and 14 m below DGM, so road, shoulders, curbs, primary terrain, and closure geometry are never coplanar",
            },
        },
    }
    data_bytes = json_payload(world_data)
    builder_rel = str(Path(__file__).resolve().relative_to(REPO))
    source_records = {}
    for relative in ("supra/data/tracks/nordschleife.json", "supra/track.py", builder_rel):
        path = REPO / relative
        source_records[relative] = {"sha256": sha256_path(path), "bytes": path.stat().st_size}
    source_records[str(DGM_META.relative_to(REPO))] = {
        "sha256": sha256_path(DGM_META), "bytes": DGM_META.stat().st_size,
    }

    manifest = {
        "schema": SCHEMA_WORLD,
        "asset_id": "nordschleife_observatory_world_v1",
        "generator": {"path": builder_rel, "deterministic": True, "network_required": False},
        "track": {
            "id": "nordschleife", "sample_count": TRACK_SAMPLES,
            "length_m": round(float(track.length), 8), "hash": runtime_hash,
            "elevation_min_m": round(float(track.z.min()), 5),
            "elevation_max_m": round(float(track.z.max()), 5),
            "runtime_pipeline": "configure_hills(enabled=True, scale=1.0, force_flat=False); named_track('nordschleife')",
            "truth_road_vertex_max_error_m": round(road_vertex_max_error_m, 9),
            "truth_road_vertex_error_limit_m": 0.02,
        },
        "layers": {
            "truth": {"simulation_authority": "displayed runtime geometry only", "sample_count": TRACK_SAMPLES, "nodes": ["truth_road", "truth_shoulder_left", "truth_shoulder_right"]},
            "visual": {"simulation_authority": "none", "collision_authority": False, "nodes": ["DGM terrain LOD tiles", "terrain containment shell", "DGM verges", "drainage strips", "visual curb strips", "guardrails", "safety fencing", "reflector and marshal furniture", "landmark-inspired markers", "three-tier forest instances"]},
        },
        "curbs": {
            "visual_only": True, "collision_authority": False,
            "sample_count": TRACK_SAMPLES, "width_m": curb_width,
            "nodes": ["visual_curb_left_light", "visual_curb_left_accent", "visual_curb_right_light", "visual_curb_right_accent"],
        },
        "files": {
            "truth": file_record("world_truth.glb", truth_glb, triangle_count=int(sum(len(g.faces) for g in truth.geometry.values()))),
            "visual": file_record("world_visual.glb", visual_glb, triangle_count=int(sum(len(g.faces) for g in visual.geometry.values()))),
            "data": file_record("world_data.json", data_bytes),
        },
        "terrain": {
            "schema": "supra-observatory-dgm-terrain-v1",
            "source": "Rheinland-Pfalz DGM1, ETRS89 / UTM zone 32N + DHHN2016 height",
            "attribution": "Rheinland-Pfalz DGM1 terrain data (dl-de/by-2-0)",
            "cache": str(DGM_CACHE.relative_to(REPO)),
            "network_required": False,
            "collision_authority": False,
            "coordinate_alignment_max_error_m": round(dgm.max_alignment_error_m, 8),
            "verge_nodata_fallback_samples": verge_fallbacks,
            "surface_stack": {
                "terrain_cutout_from_road_edge_m": terrain_envelope_m,
                "seam_skirt_depth_m": containment_depth_m + 1.0,
                "containment_shell_depth_m": containment_depth_m,
                "containment_shell_padding_m": 1800.0,
                "lod_transition": {"near_m": 820, "far_m": 980, "mode": "hysteretic_single_surface"},
            },
            "inputs": dgm.input_records(),
            "levels": terrain_levels,
        },
        "sources": source_records,
    }
    return {
        "world_truth.glb": truth_glb,
        "world_visual.glb": visual_glb,
        "world_data.json": data_bytes,
        "world.manifest.json": json_payload(manifest),
        **terrain_files,
    }


def _load_vehicle_builder(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load vehicle builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_mazda() -> dict[str, bytes]:
    builder = _load_vehicle_builder(MAZDA_SOURCE / "build_asset.py", "observatory_mazda787b_builder")
    return builder.build_outputs()


def build_porsche_wrapper() -> dict[str, bytes]:
    source_manifest_path = PORSCHE_SOURCE / "asset_manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    glb_name = source_manifest["files"]["glb"]["path"]
    glb = (PORSCHE_SOURCE / glb_name).read_bytes()
    expected = source_manifest["files"]["glb"]["sha256"]
    if sha256_bytes(glb) != expected:
        raise RuntimeError("project-authored Porsche 919 Evo GLB does not match its source manifest")
    texture_record = source_manifest.get("files", {}).get("texture")
    texture_name = str(texture_record.get("path")) if isinstance(texture_record, dict) else ""
    texture = (PORSCHE_SOURCE / texture_name).read_bytes() if texture_name else b""
    if texture_name and sha256_bytes(texture) != texture_record.get("sha256"):
        raise RuntimeError("project-authored Porsche 919 Evo texture does not match its source manifest")
    named_nodes = source_manifest["canonical_nodes"]
    wrapper = {
        "schema": SCHEMA_VEHICLE,
        "asset_id": source_manifest["asset_id"],
        "vehicle_id": "porsche_919evo",
        "entry_glb": glb_name,
        "classification": source_manifest["status"],
        "display_name": source_manifest.get("display_name", "Porsche 919 Hybrid Evo"),
        "detail_level": source_manifest.get("detail_level", "observatory-hero"),
        "runtime_detail_kit": bool(source_manifest.get("runtime_detail_kit", False)),
        "coordinate_system": source_manifest["coordinate_system"],
        "visual_ground_contact": source_manifest["visual_ground_contact"],
        "dimensions_m": source_manifest["canonical_dimensions_m"],
        "budgets": source_manifest.get("budgets", {}),
        "files": {
            "glb": file_record(glb_name, glb),
            "source_manifest": file_record("source_asset_manifest.json", source_manifest_bytes),
            **({"texture": file_record(texture_name, texture, role="external copy of the atlas embedded in the GLB")} if texture_name else {}),
        },
        "named_nodes": named_nodes,
        "source_package": "assets/vehicles/porsche_919evo",
        "binding_policy": source_manifest["binding_policy"],
    }
    return {
        glb_name: glb,
        **({texture_name: texture} if texture_name else {}),
        "source_asset_manifest.json": source_manifest_bytes,
        "asset_manifest.json": json_payload(wrapper),
    }


def expected_outputs() -> list[tuple[Path, dict[str, bytes]]]:
    mazda = build_mazda()
    return [
        (MAZDA_SOURCE, mazda),
        (PUBLIC / "vehicles" / "mazda787b", mazda),
        (PUBLIC / "vehicles" / "porsche_919evo", build_porsche_wrapper()),
        (PUBLIC / "world", build_world()),
    ]


def write_or_check(check: bool) -> None:
    failures = []
    for directory, files in expected_outputs():
        for name, payload in files.items():
            path = directory / name
            if check:
                if not path.is_file():
                    failures.append(f"missing {path.relative_to(REPO)}")
                elif path.read_bytes() != payload:
                    failures.append(
                        f"stale {path.relative_to(REPO)}: stored={sha256_path(path)} expected={sha256_bytes(payload)}"
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                print(f"wrote {path.relative_to(REPO)} ({len(payload)} bytes, sha256={sha256_bytes(payload)[:16]})")
    if failures:
        raise SystemExit("Observatory asset validation failed:\n- " + "\n- ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="regenerate in memory and byte-compare every checked-in artifact")
    args = parser.parse_args()
    write_or_check(args.check)
    print("Observatory assets are deterministic and current." if args.check else "Observatory assets built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

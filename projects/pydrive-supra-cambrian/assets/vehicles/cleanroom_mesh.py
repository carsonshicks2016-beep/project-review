"""Deterministic clean-room mesh helpers for Observatory vehicle assets.

The functions in this module create geometry from authored measurements and
section profiles.  They never ingest an existing mesh, CAD file, scan, or
reference image.  Reference photographs are used only by the human author to
choose the profile data consumed by these routines.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import trimesh
from PIL import Image


RGBA = tuple[int, int, int, int]


def pbr(
    name: str,
    rgba: RGBA,
    *,
    metallic: float = 0.0,
    roughness: float = 0.5,
    image: Image.Image | None = None,
) -> trimesh.visual.material.PBRMaterial:
    kwargs: dict[str, object] = {
        "name": name,
        "baseColorFactor": np.asarray(rgba, dtype=np.uint8),
        "metallicFactor": metallic,
        "roughnessFactor": roughness,
    }
    if image is not None:
        kwargs["baseColorTexture"] = image
    return trimesh.visual.material.PBRMaterial(**kwargs)


def paint(mesh: trimesh.Trimesh, material: trimesh.visual.material.PBRMaterial) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def box(
    extents: Sequence[float],
    center: Sequence[float],
    material: trimesh.visual.material.PBRMaterial,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, material)


def ellipsoid(
    scale: Sequence[float],
    center: Sequence[float],
    material: trimesh.visual.material.PBRMaterial,
    subdivisions: int = 3,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(np.asarray(scale, dtype=float))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, material)


def cylinder(
    radius: float,
    height: float,
    center: Sequence[float],
    material: trimesh.visual.material.PBRMaterial,
    *,
    axis: str = "z",
    sections: int = 48,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if axis == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, (1.0, 0.0, 0.0)))
    elif axis == "x":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, (0.0, 1.0, 0.0)))
    elif axis != "z":
        raise ValueError(f"unsupported cylinder axis: {axis}")
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, material)


def oriented_cylinder(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    material: trimesh.visual.material.PBRMaterial,
    *,
    sections: int = 16,
) -> trimesh.Trimesh:
    start_v = np.asarray(start, dtype=float)
    end_v = np.asarray(end, dtype=float)
    vector = end_v - start_v
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise ValueError("oriented cylinder endpoints must differ")
    transform = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), vector / length)
    transform[:3, 3] = (start_v + end_v) * 0.5
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections, transform=transform)
    return paint(mesh, material)


def add_local(
    scene: trimesh.Scene,
    name: str,
    mesh: trimesh.Trimesh,
    *,
    pivot: Sequence[float] | None = None,
) -> None:
    """Add a named mesh with a stable articulation pivot."""
    pivot_v = np.asarray(pivot, dtype=float) if pivot is not None else np.asarray(mesh.bounds, dtype=float).mean(axis=0)
    mesh.apply_translation(-pivot_v)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = pivot_v
    scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)


def add_anchor(scene: trimesh.Scene, name: str, point: Sequence[float]) -> None:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(point, dtype=float)
    scene.graph.update(frame_from=scene.graph.base_frame, frame_to=name, matrix=transform)


def body_shell(
    stations: Sequence[Sequence[float]],
    material: trimesh.visual.material.PBRMaterial,
    *,
    lateral_segments: int = 32,
) -> trimesh.Trimesh:
    """Create a smooth closed body from x/width/floor/edge/crown stations.

    Each station is ``(x, half_width, floor_z, edge_z, crown_z, crown_power)``.
    This is intentionally section-authored rather than an inflated primitive.
    """
    station_data = np.asarray(stations, dtype=float)
    if station_data.ndim != 2 or station_data.shape[1] != 6:
        raise ValueError("body shell stations must have six values")
    lateral = np.linspace(-1.0, 1.0, lateral_segments)
    vertices: list[tuple[float, float, float]] = []
    uv: list[tuple[float, float]] = []
    station_count = len(station_data)
    for station_index, (x, half_width, floor_z, edge_z, crown_z, power) in enumerate(station_data):
        for lateral_index, t in enumerate(lateral):
            shoulder = max(0.0, 1.0 - abs(float(t)) ** max(0.5, power)) ** 0.62
            top_z = edge_z + (crown_z - edge_z) * shoulder
            bottom_z = floor_z + (edge_z - floor_z) * 0.08 * abs(float(t)) ** 3
            vertices.append((x, half_width * t, top_z))
            uv.append((station_index / (station_count - 1), lateral_index / (lateral_segments - 1)))
            vertices.append((x, half_width * t, bottom_z))
            uv.append((station_index / (station_count - 1), 1.0 - lateral_index / (lateral_segments - 1)))
    faces: list[tuple[int, int, int]] = []
    stride = lateral_segments * 2
    for station_index in range(station_count - 1):
        for lateral_index in range(lateral_segments - 1):
            top_a = station_index * stride + lateral_index * 2
            top_b = top_a + 2
            top_c = top_a + stride
            top_d = top_c + 2
            faces.extend(((top_a, top_c, top_b), (top_b, top_c, top_d)))
            bottom_a = top_a + 1
            bottom_b = top_b + 1
            bottom_c = top_c + 1
            bottom_d = top_d + 1
            faces.extend(((bottom_a, bottom_b, bottom_c), (bottom_b, bottom_d, bottom_c)))
        left_top = station_index * stride
        left_bottom = left_top + 1
        next_left_top = left_top + stride
        next_left_bottom = next_left_top + 1
        faces.extend(((left_top, left_bottom, next_left_top), (left_bottom, next_left_bottom, next_left_top)))
        right_top = station_index * stride + (lateral_segments - 1) * 2
        right_bottom = right_top + 1
        next_right_top = right_top + stride
        next_right_bottom = next_right_top + 1
        faces.extend(((right_top, next_right_top, right_bottom), (right_bottom, next_right_top, next_right_bottom)))
    for station_index, reverse in ((0, True), (station_count - 1, False)):
        base = station_index * stride
        for lateral_index in range(lateral_segments - 1):
            top_a = base + lateral_index * 2
            bottom_a = top_a + 1
            top_b = top_a + 2
            bottom_b = top_b + 1
            if reverse:
                faces.extend(((top_a, top_b, bottom_a), (top_b, bottom_b, bottom_a)))
            else:
                faces.extend(((top_a, bottom_a, top_b), (top_b, bottom_a, bottom_b)))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=material)
    return mesh


def arch_shell(
    center: Sequence[float],
    length: float,
    half_width: float,
    base_z: float,
    crown_z: float,
    material: trimesh.visual.material.PBRMaterial,
    *,
    longitudinal_segments: int = 42,
    arc_segments: int = 22,
    end_scale: float = 0.12,
) -> trimesh.Trimesh:
    """Create an open-bottom wheel fender with a real arch silhouette."""
    cx, cy, _ = np.asarray(center, dtype=float)
    longitudinal = np.linspace(-1.0, 1.0, longitudinal_segments)
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, arc_segments)
    vertices: list[tuple[float, float, float]] = []
    uv: list[tuple[float, float]] = []
    for ui, u in enumerate(longitudinal):
        roundness = math.sqrt(max(0.0, 1.0 - float(u) ** 2))
        local_half_width = half_width * (end_scale + (1.0 - end_scale) * roundness ** 0.72)
        local_height = (crown_z - base_z) * (0.38 + 0.62 * roundness ** 0.66)
        x = cx + u * length * 0.5
        for vi, angle in enumerate(angles):
            y = cy + local_half_width * math.sin(float(angle))
            z = base_z + local_height * max(0.0, math.cos(float(angle))) ** 0.78
            vertices.append((x, y, z))
            uv.append((ui / (longitudinal_segments - 1), vi / (arc_segments - 1)))
    faces: list[tuple[int, int, int]] = []
    for ui in range(longitudinal_segments - 1):
        for vi in range(arc_segments - 1):
            a = ui * arc_segments + vi
            b = a + 1
            c = a + arc_segments
            d = c + 1
            faces.extend(((a, c, b), (b, c, d)))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=material)
    return mesh


def canopy_shell(
    stations: Sequence[Sequence[float]],
    material: trimesh.visual.material.PBRMaterial,
    *,
    lateral_segments: int = 32,
) -> trimesh.Trimesh:
    """Create a teardrop canopy from x/half-width/base/roof stations."""
    station_data = np.asarray(stations, dtype=float)
    lateral = np.linspace(-1.0, 1.0, lateral_segments)
    vertices: list[tuple[float, float, float]] = []
    uv: list[tuple[float, float]] = []
    for ui, (x, width, base_z, roof_z) in enumerate(station_data):
        for vi, t in enumerate(lateral):
            dome = max(0.0, 1.0 - abs(float(t)) ** 1.72) ** 0.58
            vertices.append((x, width * t, base_z + (roof_z - base_z) * dome))
            uv.append((ui / (len(station_data) - 1), vi / (lateral_segments - 1)))
    faces: list[tuple[int, int, int]] = []
    for ui in range(len(station_data) - 1):
        for vi in range(lateral_segments - 1):
            a = ui * lateral_segments + vi
            b = a + 1
            c = a + lateral_segments
            d = c + 1
            faces.extend(((a, c, b), (b, c, d)))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=material)
    return mesh


def airfoil(
    center: Sequence[float],
    chord: float,
    width: float,
    thickness: float,
    material: trimesh.visual.material.PBRMaterial,
    *,
    incidence_deg: float = 0.0,
) -> trimesh.Trimesh:
    """Create a symmetric, softly cambered wing section extruded across y."""
    profile = np.asarray([
        (0.50, 0.00), (0.35, 0.44), (0.08, 0.62), (-0.30, 0.38), (-0.50, 0.00),
        (-0.30, -0.20), (0.08, -0.24), (0.35, -0.14),
    ], dtype=float)
    vertices: list[tuple[float, float, float]] = []
    for y in (-width * 0.5, width * 0.5):
        for px, pz in profile:
            vertices.append((px * chord, y, pz * thickness))
    count = len(profile)
    faces: list[tuple[int, int, int]] = []
    # The profile is convex enough for a deterministic fan cap.
    for side, reverse in ((0, True), (count, False)):
        for i in range(1, count - 1):
            face = (side, side + i, side + i + 1)
            faces.append(tuple(reversed(face)) if reverse else face)
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, count + i, j), (j, count + i, count + j)))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int64), process=False)
    if incidence_deg:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(incidence_deg), (0.0, 1.0, 0.0)))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, material)


def quad(
    points: Sequence[Sequence[float]],
    material: trimesh.visual.material.PBRMaterial,
    *,
    uv_rect: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
    double_sided: bool = True,
    flip_u: bool = False,
) -> trimesh.Trimesh:
    if len(points) != 4:
        raise ValueError("quad needs four points")
    u0, v0, u1, v1 = (float(value) for value in uv_rect)
    faces = [(0, 1, 2), (0, 2, 3)]
    if double_sided:
        faces.extend(((2, 1, 0), (3, 2, 0)))
    mesh = trimesh.Trimesh(vertices=np.asarray(points, dtype=float), faces=np.asarray(faces), process=False)
    uv_values = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    if flip_u:
        uv_values = tuple((u1 - (u - u0), v) for u, v in uv_values)
    mesh.visual = trimesh.visual.TextureVisuals(
        # Surface points use the exterior-view convention: lower-left,
        # lower-right, upper-right, upper-left.  The atlas rectangles have
        # already been converted from Pillow's upper-left origin.
        uv=np.asarray(uv_values, dtype=float),
        material=material,
    )
    return mesh


def rim_ring(
    outer_radius: float,
    inner_radius: float,
    depth: float,
    center: Sequence[float],
    material: trimesh.visual.material.PBRMaterial,
) -> trimesh.Trimesh:
    """Create an open wheel rim rather than a solid disc."""
    if not 0.0 < inner_radius < outer_radius:
        raise ValueError("rim radii must satisfy 0 < inner < outer")
    major = 0.5 * (outer_radius + inner_radius)
    minor = 0.5 * (outer_radius - inner_radius)
    mesh = trimesh.creation.torus(
        major_radius=major,
        minor_radius=minor,
        major_sections=64,
        minor_sections=12,
    )
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, (1.0, 0.0, 0.0)))
    mesh.apply_scale((1.0, depth / max(2.0 * minor, 1e-6), 1.0))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, material)


def tyre(
    outer_radius: float,
    rim_radius: float,
    width: float,
    center: Sequence[float],
    material: trimesh.visual.material.PBRMaterial,
) -> trimesh.Trimesh:
    major = 0.5 * (outer_radius + rim_radius)
    minor = 0.5 * (outer_radius - rim_radius)
    mesh = trimesh.creation.torus(major_radius=major, minor_radius=minor, major_sections=56, minor_sections=14)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, (1.0, 0.0, 0.0)))
    mesh.apply_scale((1.0, width / max(2.0 * minor, 1e-6), 1.0))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return paint(mesh, material)


def radial_spoke(
    center: Sequence[float],
    radius: float,
    angle: float,
    depth: float,
    thickness: float,
    material: trimesh.visual.material.PBRMaterial,
) -> trimesh.Trimesh:
    center_v = np.asarray(center, dtype=float)
    local_center = center_v + np.asarray((math.cos(angle) * radius * 0.5, 0.0, math.sin(angle) * radius * 0.5))
    mesh = box((radius, depth, thickness), local_center, material)
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(
            -angle,
            (0.0, 1.0, 0.0),
            point=center_v,
        )
    )
    return mesh


def arch_lip(
    center: Sequence[float],
    radius: float,
    material: trimesh.visual.material.PBRMaterial,
    *,
    start_deg: float = 8.0,
    end_deg: float = 172.0,
    segments: int = 18,
    tube_radius: float = 0.012,
) -> trimesh.Trimesh:
    cx, cy, cz = np.asarray(center, dtype=float)
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), segments)
    pieces = []
    for first, second in zip(angles[:-1], angles[1:]):
        a = (cx + radius * math.cos(first), cy, cz + radius * math.sin(first))
        b = (cx + radius * math.cos(second), cy, cz + radius * math.sin(second))
        pieces.append(oriented_cylinder(a, b, tube_radius, material, sections=8))
    return paint(trimesh.util.concatenate(pieces), material)


def mesh_triangle_count(scene: trimesh.Scene) -> int:
    return int(sum(len(geometry.faces) for geometry in scene.geometry.values()))


def scene_nodes(scene: trimesh.Scene) -> set[str]:
    return set(scene.graph.nodes)


def bounds_extents(scene: trimesh.Scene) -> np.ndarray:
    bounds = np.asarray(scene.bounds, dtype=float)
    return bounds[1] - bounds[0]


def add_many(scene: trimesh.Scene, entries: Iterable[tuple[str, trimesh.Trimesh]]) -> None:
    for name, mesh in entries:
        add_local(scene, name, mesh)

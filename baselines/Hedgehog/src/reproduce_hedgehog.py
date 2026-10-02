"""Reproduce Hedgehog Labeling on the provided 3D label layout data.

Paper: Tatzgern et al., "Hedgehog Labeling: View Management
Techniques for External Labels in 3D Space", IEEE VR 2014.

The implementation intentionally follows only the method described in the
paper, because the original source code and numeric configuration were not
released:

* center-based hedgehog poles: the pole direction is the radial vector from
  the object's bounding-sphere center to the anchor point;
* one-degree-of-freedom update: labels slide only along that 3D pole;
* three-degree-of-freedom update: labels slide along the pole and within the
  local image plane, with in-plane motion limited by the annotation size;
* plane update: labels are assigned to equidistant view-parallel planes inside
  the screen-aligned object bounding box, then positioned by a deterministic
  spring/repulsion embedding in those planes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - previews are optional.
    Image = None
    ImageDraw = None

try:
    import pyrender
    import trimesh
except Exception:  # pragma: no cover - previews can fall back to 2D drawing.
    pyrender = None
    trimesh = None


CAMERA_RADIUS = 10.0
FOCAL_LENGTH_MM = 50.0
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0
CAMERA_NEAR = 1e-6
CAMERA_FAR = 1e6
PERTURB_DEGREES = 45.0
PREVIEW_WIDTH = 750
PREVIEW_HEIGHT = 500
METHODS = ("manual", "hedgehog_1d", "hedgehog_3d", "plane")
OBJECT_RECT_CACHE: dict[tuple[int, str], np.ndarray | None] = {}


@dataclass
class Label:
    sample: str
    category: str
    group_id: str
    text: str
    anchor: np.ndarray
    manual_center: np.ndarray
    size: np.ndarray


@dataclass
class ObjectGeometry:
    center: np.ndarray
    diagonal: float
    vertices: np.ndarray | None
    faces: list[list[int]]
    face_colors: np.ndarray | None
    obj_file: Path | None = None
    mtl_file: Path | None = None


def norm(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return v / length


def build_camera_from_direction(z_view: np.ndarray, y_hint: np.ndarray) -> dict:
    z_view = norm(np.asarray(z_view, dtype=float))
    y_hint = norm(np.asarray(y_hint, dtype=float))
    x_view = norm(np.cross(-z_view, y_hint))
    y_view = norm(np.cross(z_view, x_view))
    return {
        "type": "perspective",
        "camera_radius": CAMERA_RADIUS,
        "focal_length_mm": FOCAL_LENGTH_MM,
        "sensor_width_mm": SENSOR_WIDTH_MM,
        "sensor_height_mm": SENSOR_HEIGHT_MM,
        "near_clip": CAMERA_NEAR,
        "far_clip": CAMERA_FAR,
        "position": z_view * CAMERA_RADIUS,
        "x_view": x_view,
        "y_view": y_view,
        "z_view": z_view,
    }


def rotate_toward(z_view: np.ndarray, screen_axis: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    return norm(z_view * math.cos(radians) + norm(screen_axis) * math.sin(radians))


def build_multiview_cameras() -> dict[str, dict]:
    main = build_camera_from_direction(np.asarray([1.0, 1.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))
    x_view = main["x_view"]
    y_view = main["y_view"]
    z_view = main["z_view"]
    cameras = {"main": main}
    cameras.update(
        {
            "up": build_camera_from_direction(rotate_toward(z_view, y_view, PERTURB_DEGREES), y_view),
            "down": build_camera_from_direction(rotate_toward(z_view, -y_view, PERTURB_DEGREES), y_view),
            "left": build_camera_from_direction(rotate_toward(z_view, -x_view, PERTURB_DEGREES), y_view),
            "right": build_camera_from_direction(rotate_toward(z_view, x_view, PERTURB_DEGREES), y_view),
        }
    )
    return cameras


CAMERAS = build_multiview_cameras()
VIEWS = {name: (camera["x_view"], camera["y_view"], camera["z_view"]) for name, camera in CAMERAS.items()}


def load_annotation(path: Path) -> tuple[dict, list[Label]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: list[Label] = []
    for group in data.get("groups", []):
        label = group["label"]
        labels.append(
            Label(
                sample=str(data.get("sample_id", path.stem)),
                category=str(data.get("category", path.parents[3].name)),
                group_id=str(group.get("group_id", group.get("id", ""))),
                text=str(label.get("text", group.get("group_id", ""))),
                anchor=np.asarray(group["anchor"]["point"], dtype=float),
                manual_center=np.asarray(label["center"], dtype=float),
                size=np.asarray(label["box_size"][:2], dtype=float),
            )
        )
    return data, labels


def iter_annotation_files(data_root: Path) -> Iterable[Path]:
    yield from sorted((data_root / "Layout").glob("*/*/layout1/Annotation/*.json"))


def load_mtl_colors(path: Path) -> dict[str, np.ndarray]:
    colors: dict[str, np.ndarray] = {}
    current = None
    if not path.exists():
        return colors
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "newmtl" and len(parts) >= 2:
                current = parts[1]
            elif parts[0] == "Kd" and current and len(parts) >= 4:
                rgb = [float(parts[1]), float(parts[2]), float(parts[3])]
                colors[current] = np.asarray([int(max(0, min(1, c)) * 255) for c in rgb] + [255], dtype=np.uint8)
    return colors


def load_obj_mesh(path: Path) -> tuple[np.ndarray, list[list[int]], list[str | None]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_materials: list[str | None] = []
    current_material: str | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("o label_", "g label_")):
                break
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("usemtl "):
                parts = line.split(maxsplit=1)
                current_material = parts[1].strip() if len(parts) == 2 else None
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.split()[1:]:
                    index_text = token.split("/")[0]
                    if not index_text:
                        continue
                    index = int(index_text)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    if index >= 0:
                        face.append(index)
                if len(face) >= 3:
                    faces.append(face)
                    face_materials.append(current_material)
    return np.asarray(vertices, dtype=float), faces, face_materials


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("o label_", "g label_")):
                break
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=float)


def object_geometry(annotation_file: Path, labels: list[Label]) -> ObjectGeometry:
    sample = annotation_file.stem
    obj_dir = annotation_file.parents[1] / "Obj-O"
    obj_file = obj_dir / f"{sample}-main-O.obj"
    mtl_file = obj_dir / f"{sample}-O.mtl"
    vertices = load_obj_vertices(obj_file) if obj_file.exists() else np.empty((0, 3))
    faces: list[list[int]] = []
    face_colors = None
    if len(vertices):
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
    else:
        anchors = np.asarray([label.anchor for label in labels])
        mins = anchors.min(axis=0)
        maxs = anchors.max(axis=0)
        vertices = None
    center = (mins + maxs) / 2.0
    diagonal = max(float(np.linalg.norm(maxs - mins)), 1.0)
    return ObjectGeometry(
        center=center,
        diagonal=diagonal,
        vertices=vertices,
        faces=faces,
        face_colors=face_colors,
        obj_file=obj_file if obj_file.exists() else None,
        mtl_file=mtl_file if mtl_file.exists() else None,
    )


def project(points: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = [np.asarray(v, dtype=float) for v in VIEWS[view]]
    uv = np.column_stack((points @ right, points @ up))
    depth = points @ forward
    return uv, depth


def unproject(uv: np.ndarray, depth: np.ndarray, view: str) -> np.ndarray:
    right, up, forward = [np.asarray(v, dtype=float) for v in VIEWS[view]]
    return uv[:, 0, None] * right + uv[:, 1, None] * up + depth[:, None] * forward


def rects_from_centers(centers_2d: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    half = sizes / 2.0
    return np.column_stack((centers_2d - half, centers_2d + half))


def overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def rect_area(rect: np.ndarray) -> float:
    return max(0.0, float(rect[2] - rect[0])) * max(0.0, float(rect[3] - rect[1]))


def point_inside_rect(point: np.ndarray, rect: np.ndarray) -> bool:
    return bool(rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3])


def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
        return False
    return (ccw(a, c, d) * ccw(b, c, d) < 0) and (ccw(a, b, c) * ccw(a, b, d) < 0)


def segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    r = b - a
    s = d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(float(denom)) < 1e-12:
        return (a + b + c + d) / 4.0
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
    return a + t * r


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return abs(ccw(a, b, c)) * 0.5


def separation(a: np.ndarray, b: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0.0 or oy <= 0.0:
        return np.zeros(2)
    if ox < oy:
        return np.array([-(ox + 1e-3), 0.0]) if ca[0] < cb[0] else np.array([ox + 1e-3, 0.0])
    return np.array([0.0, -(oy + 1e-3)]) if ca[1] < cb[1] else np.array([0.0, oy + 1e-3])


def object_rect_for_view(geometry: ObjectGeometry, view: str, camera: dict) -> np.ndarray | None:
    cache_key = (id(geometry), view)
    if cache_key in OBJECT_RECT_CACHE:
        return OBJECT_RECT_CACHE[cache_key]
    if geometry.vertices is None or not len(geometry.vertices):
        OBJECT_RECT_CACHE[cache_key] = None
        return None
    projected = project_world_to_pixels(geometry.vertices, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    rect = np.asarray(
        [
            float(np.min(projected[:, 0])),
            float(np.min(projected[:, 1])),
            float(np.max(projected[:, 0])),
            float(np.max(projected[:, 1])),
        ],
        dtype=float,
    )
    OBJECT_RECT_CACHE[cache_key] = rect
    return rect


def object_overlap_pixels(rect: np.ndarray, object_rect: np.ndarray | None) -> float:
    if object_rect is None:
        return 0.0
    return overlap_area(rect, object_rect)


def layout_metrics(
    labels: list[Label],
    centers: np.ndarray,
    view: str,
    manual_centers: np.ndarray,
    geometry: ObjectGeometry,
) -> dict[str, float]:
    camera = CAMERAS[view]
    anchors = np.asarray([label.anchor for label in labels])
    anchor_2d = project_world_to_pixels(anchors, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    center_2d = project_world_to_pixels(centers, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    manual_2d = project_world_to_pixels(manual_centers, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    rects = np.asarray(
        [
            projected_label_bounds(center, np.r_[label.size, 0.02], camera)
            for label, center in zip(labels, centers)
        ],
        dtype=float,
    )

    object_rect = object_rect_for_view(geometry, view, camera)
    overlap_pairs = 0
    overlap_total = 0.0
    per_label_overlap = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            area = overlap_area(rects[i], rects[j])
            if area > 1e-9:
                overlap_pairs += 1
                overlap_total += area
                per_label_overlap[i] += area
                per_label_overlap[j] += area
    for i, rect in enumerate(rects):
        obj_area = object_overlap_pixels(rect, object_rect)
        overlap_total += obj_area
        per_label_overlap[i] += obj_area

    occluded_points = 0
    for i, rect in enumerate(rects):
        for j, point in enumerate(anchor_2d):
            if i != j and point_inside_rect(point, rect):
                occluded_points += 1

    intersections = 0
    lcd_total = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if segments_intersect(anchor_2d[i], center_2d[i], anchor_2d[j], center_2d[j]):
                intersections += 1
                u = segment_intersection(anchor_2d[i], center_2d[i], anchor_2d[j], center_2d[j])
                area1 = triangle_area(u, center_2d[i], center_2d[j])
                area2 = triangle_area(u, anchor_2d[i], anchor_2d[j])
                lcd_total += min(area1, area2) / max(area1 + area2, 1e-12)

    lengths = np.linalg.norm(center_2d - anchor_2d, axis=1)
    image_diag = float(math.hypot(PREVIEW_WIDTH, PREVIEW_HEIGHT))
    label_areas = np.asarray([rect_area(rect) for rect in rects], dtype=float)
    olr = float(np.mean(per_label_overlap / np.maximum(label_areas, 1e-9))) if len(labels) else 0.0
    lcd = float(lcd_total / len(labels)) if len(labels) else 0.0
    avg_length_ratio = (float(np.mean(lengths)) / image_diag) if len(lengths) else 0.0
    center_error = np.linalg.norm(center_2d - manual_2d, axis=1) / image_diag
    pck_005 = float(np.mean(center_error <= 0.05)) if len(labels) else 0.0
    pck_010 = float(np.mean(center_error <= 0.10)) if len(labels) else 0.0
    quality_score = float(100.0 * (0.7 * pck_005 + 0.3 * pck_010) - 25.0 * olr - 10.0 * lcd)
    return {
        "PCK_005": pck_005,
        "PCK_010": pck_010,
        "OLR": olr,
        "LCD": lcd,
        "avg_leader_length": float(avg_length_ratio),
        "overlap_pairs": float(overlap_pairs),
        "occluded_points": float(occluded_points),
        "intersections": float(intersections),
        "quality_score": quality_score,
    }


def screen_forces(labels: list[Label], centers_2d: np.ndarray, anchors_2d: np.ndarray) -> np.ndarray:
    sizes = np.asarray([label.size for label in labels])
    rects = rects_from_centers(centers_2d, sizes)
    gap = 0.08 * float(np.median(np.maximum(np.min(sizes, axis=1), 1e-6)))
    padded = rects.copy()
    padded[:, 0:2] -= gap / 2.0
    padded[:, 2:4] += gap / 2.0
    forces = np.zeros_like(centers_2d)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sep = separation(padded[i], padded[j], centers_2d[i], centers_2d[j])
            if np.linalg.norm(sep) > 0.0:
                forces[i] += 0.5 * sep
                forces[j] -= 0.5 * sep

    for i, rect in enumerate(padded):
        for j, anchor in enumerate(anchors_2d):
            if i == j or not point_inside_rect(anchor, rect):
                continue
            direction = norm(centers_2d[i] - anchor, centers_2d[i] - anchors_2d[i])
            forces[i] += direction * (0.5 * float(np.mean(sizes[i])) + gap + 1e-3)

    return forces


def screen_to_world_delta(pixel_delta: np.ndarray, center: np.ndarray, camera: dict) -> np.ndarray:
    camera_center = world_to_camera(center, camera)[0]
    depth = max(1e-6, -float(camera_center[2]))
    dx_ndc = 2.0 * float(pixel_delta[0]) / float(PREVIEW_WIDTH - 1)
    dy_ndc = -2.0 * float(pixel_delta[1]) / float(PREVIEW_HEIGHT - 1)
    dx = dx_ndc * depth * float(camera["sensor_width_mm"]) * 0.5 / float(camera["focal_length_mm"])
    dy = dy_ndc * depth * float(camera["sensor_height_mm"]) * 0.5 / float(camera["focal_length_mm"])
    return dx * np.asarray(camera["x_view"], dtype=float) + dy * np.asarray(camera["y_view"], dtype=float)


def pixel_to_world(pixel: np.ndarray, depth: float, camera: dict) -> np.ndarray:
    x_ndc = 2.0 * float(pixel[0]) / float(PREVIEW_WIDTH - 1) - 1.0
    y_ndc = 1.0 - 2.0 * float(pixel[1]) / float(PREVIEW_HEIGHT - 1)
    x = x_ndc * depth * float(camera["sensor_width_mm"]) * 0.5 / float(camera["focal_length_mm"])
    y = y_ndc * depth * float(camera["sensor_height_mm"]) * 0.5 / float(camera["focal_length_mm"])
    z = -depth
    return (
        np.asarray(camera["position"], dtype=float)
        + x * np.asarray(camera["x_view"], dtype=float)
        + y * np.asarray(camera["y_view"], dtype=float)
        + z * np.asarray(camera["z_view"], dtype=float)
    )


def projected_rects(labels: list[Label], centers: np.ndarray, camera: dict) -> np.ndarray:
    return np.asarray(
        [
            projected_label_bounds(center, np.r_[label.size, 0.02], camera)
            for label, center in zip(labels, centers)
        ],
        dtype=float,
    )


def layout_forces(labels: list[Label], centers: np.ndarray, view: str, geometry: ObjectGeometry) -> np.ndarray:
    camera = CAMERAS[view]
    anchors = np.asarray([label.anchor for label in labels])
    anchor_2d = project_world_to_pixels(anchors, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    center_2d = project_world_to_pixels(centers, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    rects = projected_rects(labels, centers, camera)
    forces = np.zeros_like(center_2d)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sep = separation(rects[i], rects[j], center_2d[i], center_2d[j])
            if np.linalg.norm(sep) > 0.0:
                forces[i] += 0.65 * sep
                forces[j] -= 0.65 * sep

    object_rect = object_rect_for_view(geometry, view, camera)
    if object_rect is not None:
        object_center = np.asarray([(object_rect[0] + object_rect[2]) * 0.5, (object_rect[1] + object_rect[3]) * 0.5])
        for i, rect in enumerate(rects):
            overlap = overlap_area(rect, object_rect)
            if overlap <= 0.0:
                continue
            area = max(rect_area(rect), 1e-9)
            direction = norm(center_2d[i] - object_center, center_2d[i] - anchor_2d[i])
            forces[i] += direction * min(180.0, 95.0 * overlap / area)

    for i, rect in enumerate(rects):
        for j, anchor in enumerate(anchor_2d):
            if i != j and point_inside_rect(anchor, rect):
                forces[i] += norm(center_2d[i] - anchor, center_2d[i] - anchor_2d[i]) * 26.0

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if segments_intersect(anchor_2d[i], center_2d[i], anchor_2d[j], center_2d[j]):
                direction = norm(center_2d[i] - center_2d[j], center_2d[i] - anchor_2d[i])
                forces[i] += direction * 18.0
                forces[j] -= direction * 18.0

    forces -= 0.012 * (center_2d - anchor_2d)
    return np.clip(forces, -80.0, 80.0)


def manual_layout(labels: list[Label]) -> np.ndarray:
    return np.asarray([label.manual_center for label in labels])


def radial_directions(labels: list[Label], geometry: ObjectGeometry) -> np.ndarray:
    directions = []
    for i, label in enumerate(labels):
        fallback = np.eye(3)[i % 3]
        directions.append(norm(label.anchor - geometry.center, fallback))
    return np.asarray(directions)


def hedgehog_layout(
    labels: list[Label],
    geometry: ObjectGeometry,
    view: str,
    dof: int,
    iterations: int,
) -> np.ndarray:
    anchors = np.asarray([label.anchor for label in labels])
    directions = radial_directions(labels, geometry)
    sizes = np.asarray([label.size for label in labels])
    camera = CAMERAS[view]
    max_len = 2.2 * geometry.diagonal + float(np.max(np.linalg.norm(sizes, axis=1)))
    min_len = 0.08 * geometry.diagonal

    pole_lengths = np.full(len(labels), min_len, dtype=float)
    offsets_2d = np.zeros((len(labels), 2), dtype=float)

    for _ in range(iterations):
        centers = anchors + directions * pole_lengths[:, None] + unproject(offsets_2d, np.zeros(len(labels)), view)
        forces = layout_forces(labels, centers, view, geometry)
        if float(np.linalg.norm(forces)) < 1e-7:
            break

        for i in range(len(labels)):
            delta_world = screen_to_world_delta(0.5 * forces[i], centers[i], camera)
            pole_delta = float(delta_world @ directions[i])
            pole_lengths[i] = float(np.clip(pole_lengths[i] + pole_delta, min_len, max_len))

            if dof == 3:
                plane_delta = delta_world - pole_delta * directions[i]
                offsets_2d[i, 0] += float(plane_delta @ np.asarray(camera["x_view"], dtype=float))
                offsets_2d[i, 1] += float(plane_delta @ np.asarray(camera["y_view"], dtype=float))
                limit = sizes[i]
                offsets_2d[i] = np.clip(offsets_2d[i], -limit, limit)

    return anchors + directions * pole_lengths[:, None] + unproject(offsets_2d, np.zeros(len(labels)), view)


def plane_depths(geometry: ObjectGeometry, anchors: np.ndarray, view: str, plane_count: int) -> np.ndarray:
    camera = CAMERAS[view]
    if geometry.vertices is not None and len(geometry.vertices):
        depths = -world_to_camera(geometry.vertices, camera)[:, 2]
    else:
        depths = -world_to_camera(anchors, camera)[:, 2]
    if plane_count <= 1 or abs(float(depths.max() - depths.min())) < 1e-9:
        return np.array([(float(depths.min()) + float(depths.max())) / 2.0])
    return np.linspace(float(depths.min()), float(depths.max()), plane_count)


def plane_layout(
    labels: list[Label],
    geometry: ObjectGeometry,
    view: str,
    plane_count: int,
    iterations: int,
) -> np.ndarray:
    anchors = np.asarray([label.anchor for label in labels])
    camera = CAMERAS[view]
    anchors_2d = project_world_to_pixels(anchors, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    anchor_depths = -world_to_camera(anchors, camera)[:, 2]
    sizes = np.asarray([label.size for label in labels])
    planes = plane_depths(geometry, anchors, view, plane_count)
    assignment = np.argmin(np.abs(anchor_depths[:, None] - planes[None, :]), axis=1)
    depths = planes[assignment]

    center = anchors_2d.mean(axis=0)
    direction = np.asarray([norm(point - center, [math.cos(i), math.sin(i)]) for i, point in enumerate(anchors_2d)])
    mean_label = float(np.mean(np.linalg.norm(sizes, axis=1)))
    pixel_scale = PREVIEW_WIDTH * FOCAL_LENGTH_MM / (CAMERA_RADIUS * SENSOR_WIDTH_MM)
    centers_2d = anchors_2d + direction * max(35.0, 0.35 * mean_label * pixel_scale)

    spring = 0.010
    step = 0.55
    max_offset = 0.85 * max(PREVIEW_WIDTH, PREVIEW_HEIGHT)
    for _ in range(iterations):
        centers = np.asarray([pixel_to_world(pixel, depth, camera) for pixel, depth in zip(centers_2d, depths)])
        forces = layout_forces(labels, centers, view, geometry)
        forces += spring * (anchors_2d - centers_2d)
        if float(np.linalg.norm(forces)) < 1e-7:
            break
        centers_2d += step * forces
        offset = centers_2d - anchors_2d
        length = np.linalg.norm(offset, axis=1)
        too_far = length > max_offset
        if np.any(too_far):
            offset[too_far] *= (max_offset / length[too_far])[:, None]
            centers_2d[too_far] = anchors_2d[too_far] + offset[too_far]

    return np.asarray([pixel_to_world(pixel, depth, camera) for pixel, depth in zip(centers_2d, depths)])


def make_layout_json(
    source_data: dict,
    labels: list[Label],
    centers: np.ndarray,
    method: str,
    view: str,
    params: dict,
) -> dict:
    out = copy.deepcopy(source_data)
    out["version"] = "hedgehog_reproduction"
    out["layout_type"] = method
    out["reproduction"] = {
        "paper": "Tatzgern et al. 2014, Hedgehog Labeling",
        "view": view,
        "method": method,
        "parameters": params,
    }
    for group, label, center in zip(out.get("groups", []), labels, centers):
        center_list = [float(x) for x in center]
        group["label"]["center"] = center_list
        group.setdefault("leader_line", {})
        group["leader_line"]["start"] = [float(x) for x in label.anchor]
        group["leader_line"]["bend_points"] = []
        group["leader_line"]["end"] = center_list
    return out


def aggregate(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    metrics = [
        "PCK_005",
        "PCK_010",
        "OLR",
        "LCD",
        "avg_leader_length",
        "overlap_pairs",
        "occluded_points",
        "intersections",
        "quality_score",
    ]
    out: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        out[method] = {metric: float(np.mean([float(row[metric]) for row in subset])) for metric in metrics}
    return out


def write_protocol(out_dir: Path, summary: dict[str, dict[str, float]], sample_count: int, params: dict) -> None:
    lines = [
        "# Hedgehog Labeling 复现实验记录",
        "",
        "## 复现依据",
        "",
        "- 论文方法是纯几何/优化方法，没有训练阶段；本复现不引入学习模型。",
        "- 3D pole 方向：按论文从物体包围球中心指向 anchor point 的径向向量确定。",
        "- 1D hedgehog：只允许 annotation 沿 pole 改变长度。",
        "- 3D hedgehog：在 pole 长度之外，允许 annotation 在其局部图像平面 X/Y 方向移动，并把该位移限制在 annotation size 内。",
        "- plane 方法：使用与当前视平面平行、在 screen-aligned bounding box 中等距放置的平面，并把 label 分配到最近平面。",
        "",
        "## 参数说明",
        "",
        f"- `plane_count={params['plane_count']}`：论文说明该值由用户在运行时设置，并在 Figure 7 示例中使用 3 个平面；本复现默认采用 3。",
        f"- `iterations={params['iterations']}`：数值优化迭代上限，只影响收敛时间，不改变论文约束。",
        "- 相机视角：采用 `regenerate_layout_assets(1).py` 对应工具链中的 multiview camera 设定。",
        "",
        "## 数据与输出",
        "",
        f"- 数据：`../data/Layout`，共 {sample_count} 个 3D 标注样本，每个样本评估 main/up/down/left/right 五个视角。",
        "- 布局 JSON：`results/layouts/<Category>/<Sample>/<View>/<Method>.json`。",
        "- 预览图：`results/previews/*.png`，索引页面为 `results/layout_preview.html`；manual 列使用数据集 Mutiviews 原图，复现方法列使用 pyrender 渲染同一 OBJ 主体后叠加 label。",
        "- 指标：在与预览渲染相同的相机状态下重算；PCK 以 data 中 manual label center 为参考，OLR/LCD 按屏幕投影计算，avg_leader_length 按图像对角线归一化。",
        "",
        "## 平均指标",
        "",
        "| method | PCK@0.05 ↑ | PCK@0.10 ↑ | OLR ↓ | LCD ↓ | avg_leader_length ↓ | quality_score ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary.items():
        lines.append(
            f"| {method} | {values['PCK_005']:.3f} | {values['PCK_010']:.3f} | "
            f"{values['OLR']:.3f} | {values['LCD']:.3f} | "
            f"{values['avg_leader_length']:.3f} | {values['quality_score']:.3f} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "reproduction_protocol.md").write_text("\n".join(lines), encoding="utf-8")

def convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 3:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def mesh_edges(faces: list[list[int]], vertex_count: int, limit: int = 3500) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        valid = [index for index in face if 0 <= index < vertex_count]
        for i, a in enumerate(valid):
            b = valid[(i + 1) % len(valid)]
            edges.add((min(a, b), max(a, b)))
    ordered = sorted(edges)
    if len(ordered) <= limit:
        return ordered
    stride = max(1, math.ceil(len(ordered) / limit))
    return ordered[::stride]


def triangulated_faces(faces: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    triangles: list[list[int]] = []
    source_face_indices: list[int] = []
    for face_index, face in enumerate(faces):
        if len(face) < 3:
            continue
        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i + 1]])
            source_face_indices.append(face_index)
    return np.asarray(triangles, dtype=np.int64), source_face_indices


def camera_pose(camera: dict) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 0] = np.asarray(camera["x_view"], dtype=float)
    pose[:3, 1] = np.asarray(camera["y_view"], dtype=float)
    pose[:3, 2] = np.asarray(camera["z_view"], dtype=float)
    pose[:3, 3] = np.asarray(camera["position"], dtype=float)
    return pose


def render_object_with_pyrender(geometry: ObjectGeometry, camera: dict) -> Image.Image | None:
    if Image is None or pyrender is None or trimesh is None:
        return None
    if geometry.vertices is None or not len(geometry.vertices):
        return None
    try:
        if not geometry.faces and geometry.obj_file is not None:
            vertices, faces_loaded, face_materials = load_obj_mesh(geometry.obj_file)
            geometry.vertices = vertices
            geometry.faces = faces_loaded
            material_colors = load_mtl_colors(geometry.mtl_file) if geometry.mtl_file is not None else {}
            if faces_loaded:
                default_color = np.asarray([205, 205, 190, 255], dtype=np.uint8)
                geometry.face_colors = np.asarray(
                    [material_colors.get(name, default_color) for name in face_materials],
                    dtype=np.uint8,
                )
        if not geometry.faces:
            return None
        faces, source_indices = triangulated_faces(geometry.faces)
        if len(faces) == 0:
            return None
        mesh = trimesh.Trimesh(vertices=geometry.vertices, faces=faces, process=False)
        if geometry.face_colors is not None and len(geometry.face_colors):
            mesh.visual.face_colors = geometry.face_colors[source_indices]
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.72, 0.72, 0.72, 1.0])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        fx = float(camera["focal_length_mm"]) * (PREVIEW_WIDTH - 1) / float(camera["sensor_width_mm"])
        fy = float(camera["focal_length_mm"]) * (PREVIEW_HEIGHT - 1) / float(camera["sensor_height_mm"])
        cx = (PREVIEW_WIDTH - 1) * 0.5
        cy = (PREVIEW_HEIGHT - 1) * 0.5
        render_camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            znear=0.1,
            zfar=max(20.0, float(camera["camera_radius"]) + 3.0),
        )
        pose = camera_pose(camera)
        scene.add(render_camera, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=pose)
        fill_pose = np.eye(4, dtype=float)
        fill_pose[:3, 3] = -np.asarray(camera["position"], dtype=float)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.45), pose=fill_pose)
        renderer = pyrender.OffscreenRenderer(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        rgba = Image.fromarray(color.astype(np.uint8), mode="RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    except Exception:
        return None


def world_to_camera(points: np.ndarray, camera: dict) -> np.ndarray:
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim == 1:
        point_array = point_array.reshape(1, 3)
    delta = point_array - np.asarray(camera["position"], dtype=float)
    x_view = np.asarray(camera["x_view"], dtype=float)
    y_view = np.asarray(camera["y_view"], dtype=float)
    z_view = np.asarray(camera["z_view"], dtype=float)
    return np.stack([delta @ x_view, delta @ y_view, delta @ z_view], axis=1)


def project_world_to_pixels(world_points: np.ndarray, camera: dict, width: int, height: int) -> np.ndarray:
    camera_points = world_to_camera(world_points, camera)
    depth = -camera_points[:, 2]
    depth = np.maximum(depth, 1e-6)
    x = camera_points[:, 0] * float(camera["focal_length_mm"]) / (depth * float(camera["sensor_width_mm"]) * 0.5)
    y = camera_points[:, 1] * float(camera["focal_length_mm"]) / (depth * float(camera["sensor_height_mm"]) * 0.5)
    px = (x + 1.0) * 0.5 * (width - 1)
    py = (1.0 - (y + 1.0) * 0.5) * (height - 1)
    return np.stack([px, py], axis=1)


def projected_label_bounds(label_center: np.ndarray, box_size: np.ndarray, camera: dict) -> tuple[float, float, float, float]:
    center = np.asarray(label_center, dtype=float)
    half = np.asarray(box_size, dtype=float) * 0.5
    axis_x = norm(np.asarray(camera["x_view"], dtype=float))
    axis_y = norm(np.asarray(camera["y_view"], dtype=float))
    axis_z = norm(np.asarray(camera["z_view"], dtype=float))
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center + sx * half[0] * axis_x + sy * half[1] * axis_y + sz * half[2] * axis_z)
    projected = project_world_to_pixels(np.asarray(corners), camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    return (
        float(np.min(projected[:, 0])),
        float(np.min(projected[:, 1])),
        float(np.max(projected[:, 0])),
        float(np.max(projected[:, 1])),
    )


def draw_preview(
    labels: list[Label],
    centers: np.ndarray,
    geometry: ObjectGeometry,
    view: str,
    out_file: Path,
    title: str,
) -> None:
    if Image is None or ImageDraw is None:
        return
    camera = CAMERAS[view]
    anchors = np.asarray([label.anchor for label in labels])
    anchor_2d = project_world_to_pixels(anchors, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    center_2d = project_world_to_pixels(centers, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    img = render_object_with_pyrender(geometry, camera)
    rendered_with_pyrender = img is not None
    if img is None:
        img = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    if not rendered_with_pyrender and geometry.vertices is not None and len(geometry.vertices):
        object_2d = project_world_to_pixels(geometry.vertices, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)

        def xy(p: np.ndarray) -> tuple[float, float]:
            return float(p[0]), float(p[1])

        hull = convex_hull(object_2d)
        if len(hull) >= 3:
            draw.polygon([xy(point) for point in hull], fill=(245, 246, 248), outline=(150, 156, 164))
        for a, b in mesh_edges(geometry.faces, len(object_2d)):
            draw.line([xy(object_2d[a]), xy(object_2d[b])], fill=(205, 210, 216), width=1)
        if not geometry.faces:
            step = max(1, len(object_2d) // 900)
            for point in object_2d[::step]:
                x, y = xy(point)
                draw.point((x, y), fill=(170, 175, 181))
    draw.text((12, 10), title, fill=(25, 25, 25))
    for label, a, c, center in zip(labels, anchor_2d, center_2d, centers):
        draw.line([tuple(a), tuple(c)], fill=(210, 73, 45), width=1)
        ax, ay = float(a[0]), float(a[1])
        draw.ellipse((ax - 3, ay - 3, ax + 3, ay + 3), fill=(35, 35, 35))
        x0, y0, x1, y1 = projected_label_bounds(center, np.r_[label.size, 0.02], camera)
        draw.rectangle((x0, y0, x1, y1), outline=(20, 20, 20), fill=(255, 255, 250), width=1)
        draw.text((x0 + 4, y0 + 2), label.text[:32], fill=(16, 16, 16))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_file)


def copy_manual_preview(annotation_file: Path, view: str, out_file: Path) -> bool:
    source = annotation_file.parents[1] / "Mutiviews" / f"{annotation_file.stem}-{view}.png"
    if not source.exists() or Image is None:
        return False
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img.save(out_file)
    return True


def write_preview_index(out_dir: Path, preview_groups: list[dict]) -> None:
    method_names = {
        "manual": "Manual baseline",
        "hedgehog_1d": "Hedgehog 1D",
        "hedgehog_3d": "Hedgehog 3D",
        "plane": "Plane",
    }
    sections = []
    for group in preview_groups:
        figures = []
        for item in group["items"]:
            rel = item["path"].relative_to(out_dir).as_posix()
            metrics = item["metrics"]
            figures.append(
                "<figure>"
                f"<div class='method {item['method']}'>{method_names.get(item['method'], item['method'])}</div>"
                f"<img src='{rel}' alt='{item['method']} preview'>"
                "<figcaption>"
                f"<span>PCK@5 {metrics['PCK_005']:.2f}</span>"
                f"<span>OLR {metrics['OLR']:.3f}</span>"
                f"<span>LCD {metrics['LCD']:.3f}</span>"
                "</figcaption>"
                "</figure>"
            )
        sections.append(
            "<section>"
            f"<header><div><strong>{group['category']} / {group['sample']}</strong><span>{group['view']} view</span></div></header>"
            f"<div class='compare'>{''.join(figures)}</div>"
            "</section>"
        )
    support_script = out_dir / "support" / "regenerate_layout_assets.py"
    support_html = ""
    if support_script.exists():
        support_html = (
            "<div class='support'>"
            "<strong>Asset regeneration script</strong>"
            "<span>This page also links the provided script used for regenerating Obj-O and Mutiviews assets.</span>"
            "<a href='support/regenerate_layout_assets.py'>regenerate_layout_assets.py</a>"
            "</div>"
        )
    html = """<!doctype html>
<meta charset="utf-8">
<title>Hedgehog reproduction previews</title>
<style>
*{box-sizing:border-box}
body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;color:#20242a;background:#f4f6f8}
.top{position:sticky;top:0;z-index:2;background:#ffffffd9;backdrop-filter:blur(10px);border-bottom:1px solid #dfe4ea;padding:18px 28px}
h1{font-size:22px;margin:0 0 6px}
.top p{margin:0;color:#66707c;font-size:13px}
.support{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px;color:#52606d}
.support strong{color:#20242a}.support a{color:#1f78b4;text-decoration:none;border-bottom:1px solid #1f78b4}
main{padding:22px 28px 36px;display:grid;gap:22px}
section{background:#fff;border:1px solid #dfe4ea;border-radius:8px;overflow:hidden;box-shadow:0 8px 26px rgba(31,42,55,.06)}
section header{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-bottom:1px solid #e7ebf0;background:#fbfcfd}
section header strong{font-size:15px}
section header span{display:block;margin-top:3px;color:#687482;font-size:12px}
.compare{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:0}
figure{margin:0;padding:12px;border-right:1px solid #edf0f3}
figure:last-child{border-right:0}
.method{font-weight:700;font-size:13px;margin-bottom:8px;padding-left:10px;border-left:4px solid #7b8794}
.method.manual{border-color:#7b8794}.method.hedgehog_1d{border-color:#d9822b}.method.hedgehog_3d{border-color:#1f78b4}.method.plane{border-color:#2f9e44}
img{width:100%;height:auto;display:block;background:white;border:1px solid #e2e7ed}
figcaption{display:flex;gap:6px;flex-wrap:wrap;font-size:12px;margin-top:8px;color:#52606d}
figcaption span{background:#eef2f6;border:1px solid #dce3ea;border-radius:999px;padding:3px 7px}
@media(max-width:1100px){.compare{grid-template-columns:repeat(2,minmax(240px,1fr))}figure{border-bottom:1px solid #edf0f3}}
@media(max-width:640px){.top{padding:16px}main{padding:14px}.compare{grid-template-columns:1fr}figure{border-right:0}}
</style>
<div class="top">
  <h1>Hedgehog Labeling Reproduction</h1>
  <p>Each row compares the dataset manual view image with generated layouts; generated columns share the same pyrender-rendered OBJ body for that sample/view.</p>
""" + support_html + """
</div>
<main>
""" + "\n".join(sections) + "\n</main>\n"
    (out_dir / "layout_preview.html").write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out / "previews"
    if preview_dir.exists():
        for old_preview in preview_dir.glob("*.png"):
            old_preview.unlink()
    rows: list[dict[str, str]] = []
    preview_groups: list[dict] = []
    samples = 0
    params = {"plane_count": args.plane_count, "iterations": args.iterations}

    for annotation_file in iter_annotation_files(args.data):
        source_data, labels = load_annotation(annotation_file)
        if not labels:
            continue
        samples += 1
        geometry = object_geometry(annotation_file, labels)
        for view in args.views:
            layouts = {
                "manual": manual_layout(labels),
                "hedgehog_1d": hedgehog_layout(labels, geometry, view, dof=1, iterations=args.iterations),
                "hedgehog_3d": hedgehog_layout(labels, geometry, view, dof=3, iterations=args.iterations),
                "plane": plane_layout(labels, geometry, view, args.plane_count, args.iterations),
            }
            manual_centers = layouts["manual"]
            preview_group = None
            if len(preview_groups) < args.preview_limit:
                preview_group = {
                    "category": labels[0].category,
                    "sample": labels[0].sample,
                    "view": view,
                    "items": [],
                }
            for method in args.methods:
                centers = layouts[method]
                metrics = layout_metrics(labels, centers, view, manual_centers, geometry)
                row = {
                    "category": labels[0].category,
                    "sample": labels[0].sample,
                    "view": view,
                    "method": method,
                    "num_labels": str(len(labels)),
                }
                row.update({key: f"{value:.8f}" for key, value in metrics.items()})
                rows.append(row)

                layout_dir = args.out / "layouts" / labels[0].category / labels[0].sample / view
                layout_dir.mkdir(parents=True, exist_ok=True)
                layout_doc = make_layout_json(source_data, labels, centers, method, view, params)
                (layout_dir / f"{method}.json").write_text(
                    json.dumps(layout_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                if preview_group is not None:
                    preview_path = args.out / "previews" / f"{labels[0].category}_{labels[0].sample}_{view}_{method}.png"
                    if method == "manual":
                        copied = copy_manual_preview(annotation_file, view, preview_path)
                        if not copied:
                            draw_preview(
                                labels,
                                centers,
                                geometry,
                                view,
                                preview_path,
                                f"{labels[0].category}/{labels[0].sample} {view} {method}",
                            )
                    else:
                        draw_preview(
                            labels,
                            centers,
                            geometry,
                            view,
                            preview_path,
                            f"{labels[0].category}/{labels[0].sample} {view} {method}",
                        )
                    if preview_path.exists():
                        preview_group["items"].append(
                            {
                                "method": method,
                                "path": preview_path,
                                "metrics": metrics,
                            }
                        )
            if preview_group is not None and preview_group["items"]:
                preview_groups.append(preview_group)

    fieldnames = [
        "category",
        "sample",
        "view",
        "method",
        "num_labels",
        "PCK_005",
        "PCK_010",
        "OLR",
        "LCD",
        "avg_leader_length",
        "overlap_pairs",
        "occluded_points",
        "intersections",
        "quality_score",
    ]
    with (args.out / "hedgehog_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_protocol(args.out, summary, samples, params)
    write_preview_index(args.out, preview_groups)
    preview_count = sum(len(group["items"]) for group in preview_groups)
    return {"samples": samples, "rows": len(rows), "preview_groups": len(preview_groups), "previews": preview_count, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--plane-count", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--preview-limit", type=int, default=18, help="number of sample/view comparison groups to render")
    parser.add_argument("--views", nargs="+", choices=tuple(VIEWS), default=list(VIEWS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


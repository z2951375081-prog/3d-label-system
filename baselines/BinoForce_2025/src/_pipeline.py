"""Reproduce the computable BinoForce 2025 method on the local Layout data.

The paper does not publish source code, Unity scenes, or camera trajectories. This
script therefore keeps the paper-defined method parameters explicit, uses the local
annotation/OBJ/manual-render data as the experiment substrate, and writes paper-shaped
tables for DBV and OLR/LCD.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IPD_METERS = 0.064
OVERLAP_M = 4.0
W_REPULSE = 0.02
W_ATTRACT = 0.09
W_OVERLAP = 0.06
W_LINE = 0.8
W_CIRC = 0.03
DISPLACEMENT_SCALE = 0.01
FOV_DEGREES = 60.0
CAMERA_RADIUS = 10.0
FOCAL_LENGTH_MM = 50.0
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0
PREVIEW_WIDTH = 750
PREVIEW_HEIGHT = 500
PERTURB_DEGREES = 45.0

# View order is taken from C:/Users/chenyv/Desktop/project/3D_baselines/regenerate_layout_assets(1).py.
# The camera basis matches the existing Hedgehog reproduction so all methods are
# evaluated and previewed in the same camera state.
VIEW_ORDER = ("main", "up", "down", "left", "right")
VIEW_CAMERA_SOURCE = "regenerate_layout_assets(1).py::VIEW_ORDER plus Hedgehog-compatible multiview camera parameters"
DEFAULT_WARMUP_FRAMES = 500
DEFAULT_EVAL_FRAMES = 300
DEFAULT_SWEEP_CANDIDATES = (300, 500, 800)
STABILITY_TOLERANCE = 0.08
DEFAULT_FPS = 30
METRIC_FIELDS = (
    "PCK_005",
    "PCK_010",
    "OLR",
    "LCD",
    "DBV",
    "avg_leader_length",
    "overlap_pairs",
    "occluded_points",
    "intersections",
    "quality_score",
)
RESULT_FIELDNAMES = ["category", "sample", "view", "method", "num_labels", *METRIC_FIELDS]

CAMERAS: dict[str, dict[str, np.ndarray | float | str]] = {}
VIEWS: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

PAPER_DBD = {
    "Fruit": {"with_bopt": 0.00937, "without_bopt": 0.01112},
    "Virus": {"with_bopt": 0.05151, "without_bopt": 0.05800},
    "Helicopter": {"with_bopt": 0.02530, "without_bopt": 0.02921},
}

PAPER_OLR_LCD = {
    "Fruit": {
        "Baseline": {"OLR": 0.13079, "LCD": 0.00004},
        "Hedgehog": {"OLR": 0.08397, "LCD": 0.01998},
        "BinoForce": {"OLR": 0.02005, "LCD": 0.00285},
    },
    "Virus": {
        "Baseline": {"OLR": 0.40784, "LCD": 0.00118},
        "Hedgehog": {"OLR": 0.32541, "LCD": 0.02048},
        "BinoForce": {"OLR": 0.20177, "LCD": 0.01747},
    },
    "Helicopter": {
        "Baseline": {"OLR": 0.69942, "LCD": 0.00020},
        "Hedgehog": {"OLR": 0.32311, "LCD": 0.02748},
        "BinoForce": {"OLR": 0.15847, "LCD": 0.01287},
    },
}


@dataclass
class Scene:
    category: str
    sample: str
    annotation_path: Path
    obj_dir: Path
    mutiviews_dir: Path
    anchors: np.ndarray
    manual_labels: np.ndarray
    sizes: np.ndarray
    texts: list[str]
    object_centers: np.ndarray
    object_corners: list[np.ndarray]
    object_vertices: np.ndarray
    group_ids: list[str]
    scene_center: np.ndarray
    scene_radius: float
    camera_distance: float
    circular_radius: float


@dataclass
class Camera:
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    center: np.ndarray
    distance: float
    matrix: np.ndarray | None = None
    offset: np.ndarray | None = None
    depth_origin: float = 0.0
    focal_length_mm: float = FOCAL_LENGTH_MM
    sensor_width_mm: float = SENSOR_WIDTH_MM
    sensor_height_mm: float = SENSOR_HEIGHT_MM

    @property
    def user_position(self) -> np.ndarray:
        return self.center - self.forward * self.distance


def as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def norm(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return np.array([1.0, 0.0, 0.0]) if fallback is None else fallback.copy()
    return v / length


def build_camera_from_direction(z_view: np.ndarray, y_hint: np.ndarray) -> dict[str, np.ndarray | float | str]:
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
        "position": z_view * CAMERA_RADIUS,
        "x_view": x_view,
        "y_view": y_view,
        "z_view": z_view,
    }


def rotate_toward(z_view: np.ndarray, screen_axis: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    return norm(z_view * math.cos(radians) + norm(screen_axis) * math.sin(radians))


def build_multiview_cameras() -> dict[str, dict[str, np.ndarray | float | str]]:
    main = build_camera_from_direction(np.asarray([1.0, 1.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))
    x_view = np.asarray(main["x_view"], dtype=float)
    y_view = np.asarray(main["y_view"], dtype=float)
    z_view = np.asarray(main["z_view"], dtype=float)
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
VIEWS = {
    name: (
        np.asarray(camera["x_view"], dtype=float),
        np.asarray(camera["y_view"], dtype=float),
        -np.asarray(camera["z_view"], dtype=float),
    )
    for name, camera in CAMERAS.items()
}

def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / denom)))))


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return abs(cross2(b - a, c - a)) * 0.5


def bbox_corners(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((8, 3), dtype=float)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return np.asarray([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], dtype=float)


def parse_obj_body_vertices(obj_path: Path, max_points: int = 30000) -> np.ndarray:
    vertices: list[list[float]] = []
    if not obj_path.exists():
        return np.empty((0, 3), dtype=float)
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("o label_", "g label_", "o leader_", "g leader_")):
                break
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.asarray(vertices, dtype=float)
    if len(arr) > max_points:
        arr = arr[:: max(1, len(arr) // max_points)]
    return arr

def iter_annotations(data_root: Path) -> Iterable[Path]:
    yield from sorted((data_root / "Layout").glob("*/*/layout1/Annotation/*.json"))


def load_scene(path: Path) -> Scene:
    data = json.loads(path.read_text(encoding="utf-8"))
    sample_id = str(data.get("sample_id", path.stem))
    layout_dir = path.parents[1]
    anchors: list[list[float]] = []
    labels: list[list[float]] = []
    sizes: list[list[float]] = []
    texts: list[str] = []
    group_ids: list[str] = []

    for group in data.get("groups", []):
        anchor = as_array(group["anchor"]["point"])
        label = as_array(group["label"]["center"])
        size = as_array(group["label"]["box_size"][:2])
        anchors.append(anchor.tolist())
        labels.append(label.tolist())
        sizes.append(size.tolist())
        texts.append(str(group["label"].get("text", group.get("group_id", "label"))))
        group_ids.append(str(group.get("group_id", "")))

    anchor_arr = np.asarray(anchors, dtype=float)
    label_arr = np.asarray(labels, dtype=float)
    size_arr = np.asarray(sizes, dtype=float)
    object_vertices = parse_obj_body_vertices(layout_dir / "Obj-O" / f"{sample_id}-main-O.obj")
    # Layout forces use local object proxies around anchors. Evaluations and
    # previews use the clean OBJ body vertices parsed above, with label/leader
    # groups skipped so the model itself remains unlabeled.
    object_centers = [anchor for anchor in anchor_arr]
    object_corners = []
    for anchor, size in zip(anchor_arr, size_arr):
        eps = max(float(np.linalg.norm(size)) * 0.08, 0.035)
        object_corners.append(bbox_corners(anchor + np.asarray([[-eps, -eps, -eps], [eps, eps, eps]])))
    obj_center_arr = np.asarray(object_centers, dtype=float) if len(object_centers) else np.zeros((0, 3), dtype=float)
    all_parts = [arr for arr in (anchor_arr, label_arr, obj_center_arr) if len(arr)]
    all_points = np.vstack(all_parts) if all_parts else np.zeros((1, 3))
    scene_center = anchor_arr.mean(axis=0) if len(anchor_arr) else np.zeros(3)
    scene_radius = max(float(np.max(np.linalg.norm(all_points - scene_center, axis=1))), 0.5)
    leader_lengths = np.linalg.norm(label_arr - anchor_arr, axis=1) if len(anchor_arr) else np.asarray([1.0])
    circular_radius = max(float(np.median(leader_lengths)), scene_radius * 0.85)

    return Scene(
        category=str(data.get("category", path.parents[3].name)),
        sample=sample_id,
        annotation_path=path,
        obj_dir=layout_dir / "Obj-O",
        mutiviews_dir=layout_dir / "Mutiviews",
        anchors=anchor_arr,
        manual_labels=label_arr,
        sizes=np.maximum(size_arr, 1e-4),
        texts=texts,
        object_centers=obj_center_arr,
        object_corners=object_corners,
        object_vertices=object_vertices,
        group_ids=group_ids,
        scene_center=scene_center,
        scene_radius=scene_radius,
        camera_distance=CAMERA_RADIUS,
        circular_radius=circular_radius,
    )


def camera_for(scene: Scene, view: str) -> Camera:
    camera = CAMERAS[view]
    right = np.asarray(camera["x_view"], dtype=float)
    up = np.asarray(camera["y_view"], dtype=float)
    forward = -np.asarray(camera["z_view"], dtype=float)
    return Camera(
        right=right,
        up=up,
        forward=forward,
        center=np.zeros(3, dtype=float),
        distance=float(camera["camera_radius"]),
        focal_length_mm=float(camera["focal_length_mm"]),
        sensor_width_mm=float(camera["sensor_width_mm"]),
        sensor_height_mm=float(camera["sensor_height_mm"]),
    )


def camera_from_position(scene: Scene, position: np.ndarray) -> Camera:
    z_view = norm(position - scene.scene_center, np.asarray([1.0, 1.0, 1.0]))
    y_hint = np.asarray([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(z_view, y_hint))) > 0.96:
        y_hint = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = norm(np.cross(-z_view, y_hint))
    up = norm(np.cross(z_view, right))
    return Camera(right=right, up=up, forward=-z_view, center=scene.scene_center, distance=float(np.linalg.norm(position - scene.scene_center)))


def trajectory_camera(scene: Scene, frame: int, fps: int = DEFAULT_FPS) -> Camera:
    segment = max(1, int(fps) * 5)
    segment_index = frame // segment
    local_t = (frame % segment) / float(segment)
    rng = np.random.default_rng(sum(ord(ch) for ch in f"{scene.category}/{scene.sample}"))
    angle = 0.0
    lateral = np.zeros(3, dtype=float)
    radius = CAMERA_RADIUS
    for idx in range(segment_index + 1):
        paused = bool(rng.random() < 0.5)
        direction = rng.uniform(0.0, 2.0 * math.pi)
        duration = local_t if idx == segment_index else 1.0
        if paused:
            continue
        angle += duration * (2.0 * math.pi / 12.0)
        lateral += duration * 0.18 * scene.scene_radius * np.asarray([math.cos(direction), 0.0, math.sin(direction)], dtype=float)
    position = scene.scene_center + np.asarray([math.cos(angle) * radius, 0.0, math.sin(angle) * radius], dtype=float) + lateral
    return camera_from_position(scene, position)

def project(points: np.ndarray, cam: Camera, eye_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    eye_pos = cam.user_position + cam.right * eye_offset
    rel = points - eye_pos
    depth = np.maximum(rel @ cam.forward, 1e-6)
    fx = cam.focal_length_mm / (cam.sensor_width_mm * 0.5)
    fy = cam.focal_length_mm / (cam.sensor_height_mm * 0.5)
    x = fx * (rel @ cam.right) / depth
    y = fy * (rel @ cam.up) / depth
    return np.column_stack((x, y)), depth


def parse_obj_group_centers(obj_path: Path) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    current: str | None = None
    if not obj_path.exists():
        return {}
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("g "):
                current = line.strip().split(maxsplit=1)[1]
                groups.setdefault(current, [])
            elif current is not None and line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    groups[current].append([float(parts[1]), float(parts[2]), float(parts[3])])
    return {name: np.asarray(verts, dtype=float).mean(axis=0) for name, verts in groups.items() if verts}


_VIEW_TRANSFORM_CACHE: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None, float]] = {}


def view_transform(scene: Scene, view: str) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    key = (str(scene.annotation_path), view)
    if key in _VIEW_TRANSFORM_CACHE:
        return _VIEW_TRANSFORM_CACHE[key]
    centers = parse_obj_group_centers(obj_path_for(scene, view))
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    for group_id, label in zip(scene.group_ids, scene.manual_labels):
        center = centers.get(f"label_{group_id}")
        if center is not None:
            src.append(label)
            dst.append(center)
    if len(src) >= 4:
        x = np.column_stack([np.asarray(src, dtype=float), np.ones(len(src))])
        y = np.asarray(dst, dtype=float)
        fit, *_ = np.linalg.lstsq(x, y, rcond=None)
        matrix = fit[:3, :]
        offset = fit[3, :]
        all_view = np.vstack([scene.anchors, scene.manual_labels, scene.object_centers]) @ matrix + offset
        result = (matrix, offset, float(all_view[:, 2].min()))
    else:
        result = (None, None, 0.0)
    _VIEW_TRANSFORM_CACHE[key] = result
    return result


def to_view_coords(scene: Scene, view: str, points: np.ndarray) -> np.ndarray:
    cam = camera_for(scene, view)
    xy, depth = project(points, cam, 0.0)
    return np.column_stack([xy, depth])


def label_rects(scene: Scene, labels: np.ndarray, cam: Camera, eye_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, depth = project(labels, cam, eye_offset)
    fx = cam.focal_length_mm / (cam.sensor_width_mm * 0.5)
    fy = cam.focal_length_mm / (cam.sensor_height_mm * 0.5)
    projected_sizes = np.column_stack((fx * scene.sizes[:, 0] / depth, fy * scene.sizes[:, 1] / depth))
    half = projected_sizes * 0.5
    return np.column_stack((centers - half, centers + half)), centers, depth


def object_rects(scene: Scene, cam: Camera, eye_offset: float) -> np.ndarray:
    rects = []
    for corners in scene.object_corners:
        pts, _ = project(corners, cam, eye_offset)
        rects.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return np.asarray(rects, dtype=float)


def overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )


def rect_area(a: np.ndarray) -> float:
    return max(1e-9, max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1])))


def point_inside_rect(point: np.ndarray, rect: np.ndarray) -> bool:
    return bool(rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3])


def world_to_camera(points: np.ndarray, camera: dict[str, np.ndarray | float | str]) -> np.ndarray:
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim == 1:
        point_array = point_array.reshape(1, 3)
    delta = point_array - np.asarray(camera["position"], dtype=float)
    x_view = np.asarray(camera["x_view"], dtype=float)
    y_view = np.asarray(camera["y_view"], dtype=float)
    z_view = np.asarray(camera["z_view"], dtype=float)
    return np.stack([delta @ x_view, delta @ y_view, delta @ z_view], axis=1)


def project_world_to_pixels(world_points: np.ndarray, camera: dict[str, np.ndarray | float | str], width: int, height: int) -> np.ndarray:
    camera_points = world_to_camera(world_points, camera)
    depth = np.maximum(-camera_points[:, 2], 1e-6)
    x = camera_points[:, 0] * float(camera["focal_length_mm"]) / (depth * float(camera["sensor_width_mm"]) * 0.5)
    y = camera_points[:, 1] * float(camera["focal_length_mm"]) / (depth * float(camera["sensor_height_mm"]) * 0.5)
    px = (x + 1.0) * 0.5 * (width - 1)
    py = (1.0 - (y + 1.0) * 0.5) * (height - 1)
    return np.stack([px, py], axis=1)


def projected_label_bounds(label_center: np.ndarray, box_size: np.ndarray, camera: dict[str, np.ndarray | float | str]) -> tuple[float, float, float, float]:
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


def object_rect_for_scene(scene: Scene, view: str, camera: dict[str, np.ndarray | float | str]) -> np.ndarray | None:
    if len(scene.object_vertices):
        projected = project_world_to_pixels(scene.object_vertices, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    elif scene.object_corners:
        projected = project_world_to_pixels(np.vstack(scene.object_corners), camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    else:
        return None
    return np.asarray(
        [
            float(np.min(projected[:, 0])),
            float(np.min(projected[:, 1])),
            float(np.max(projected[:, 0])),
            float(np.max(projected[:, 1])),
        ],
        dtype=float,
    )

def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
        return False
    return (ccw(a, c, d) * ccw(b, c, d) < 0) and (ccw(a, b, c) * ccw(a, b, d) < 0)


def segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    r = b - a
    s = d - c
    denom = cross2(r, s)
    if abs(denom) < 1e-12:
        return (a + b + c + d) * 0.25
    return a + (cross2(c - a, s) / denom) * r


def project_onto_user_plane(direction: np.ndarray, reference: np.ndarray, user_position: np.ndarray) -> np.ndarray:
    normal = norm(reference - user_position)
    projected = direction - np.dot(direction, normal) * normal
    return norm(projected, norm(direction))


def repulsive_force(labels: np.ndarray, i: int) -> np.ndarray:
    if len(labels) <= 1:
        return np.zeros(3)
    force = np.zeros(3)
    for j in range(len(labels)):
        if i == j:
            continue
        delta = labels[i] - labels[j]
        force += norm(delta) / max(float(np.linalg.norm(delta)), 1e-9)
    return force / (len(labels) - 1)


def attractive_force(scene: Scene, labels: np.ndarray, i: int) -> np.ndarray:
    leader = labels[i] - scene.anchors[i]
    current = float(np.linalg.norm(leader))
    direction = norm(leader, norm(scene.anchors[i] - scene.scene_center))
    target = max(scene.circular_radius, 1e-6)
    return (target - current) * direction


def circular_force(scene: Scene, labels: np.ndarray, i: int) -> np.ndarray:
    radial = norm(labels[i] - scene.scene_center, norm(scene.anchors[i] - scene.scene_center))
    target = scene.scene_center + radial * scene.circular_radius
    return target - labels[i]


def line_crossing_forces(scene: Scene, labels: np.ndarray, cam: Camera) -> np.ndarray:
    result = np.zeros_like(labels)
    label_2d = project(labels, cam, 0.0)[0]
    anchor_2d = project(scene.anchors, cam, 0.0)[0]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if not segments_intersect(anchor_2d[i], label_2d[i], anchor_2d[j], label_2d[j]):
                continue
            u = segment_intersection(anchor_2d[i], label_2d[i], anchor_2d[j], label_2d[j])
            area1 = triangle_area(u, label_2d[i], label_2d[j])
            area2 = triangle_area(u, anchor_2d[i], anchor_2d[j])
            degree = min(area1, area2) / max(area1 + area2, 1e-12)
            ni = norm(np.cross(scene.anchors[j] - scene.anchors[i], labels[i] - scene.anchors[i]))
            nj = norm(np.cross(scene.anchors[i] - scene.anchors[j], labels[j] - scene.anchors[j]))
            diri = -ni if angle_between(ni, -nj) > angle_between(ni, nj) else ni
            result[i] += degree * diri
            result[j] -= degree * diri
    return result


def force_layout(
    scene: Scene,
    view: str,
    binocular: bool,
    iterations: int,
    initial_labels: np.ndarray | None = None,
    camera_override: Camera | None = None,
) -> np.ndarray:
    labels = scene.anchors.copy() if initial_labels is None else initial_labels.copy()
    cam = camera_override if camera_override is not None else camera_for(scene, view)
    user_position = cam.user_position
    eyes = (-IPD_METERS / 2, IPD_METERS / 2) if binocular else (0.0,)

    for _ in range(iterations):
        line_forces = line_crossing_forces(scene, labels, cam)
        label_rect_cache = {eye: label_rects(scene, labels, cam, eye)[0] for eye in eyes}
        object_rect_cache = {eye: object_rects(scene, cam, eye) for eye in eyes}
        forces = np.zeros_like(labels)

        for i in range(len(labels)):
            forces[i] += W_REPULSE * repulsive_force(labels, i)
            forces[i] += W_ATTRACT * attractive_force(scene, labels, i)
            forces[i] += W_CIRC * circular_force(scene, labels, i)

            for j in range(len(labels)):
                if i == j:
                    continue
                ratio = max(overlap_area(label_rect_cache[eye][i], label_rect_cache[eye][j]) / rect_area(label_rect_cache[eye][i]) for eye in eyes)
                if ratio > 0:
                    ref = (labels[i] + labels[j]) * 0.5
                    direction = project_onto_user_plane(norm(labels[i] - labels[j]), ref, user_position)
                    forces[i] += W_OVERLAP * OVERLAP_M * ratio * direction

            for j in range(len(scene.object_centers)):
                ratio = 0.0
                for eye in eyes:
                    lrect = label_rect_cache[eye][i]
                    ratio = max(ratio, overlap_area(lrect, object_rect_cache[eye][j]) / rect_area(lrect))
                if ratio > 0:
                    ref = (labels[i] + scene.object_centers[j]) * 0.5
                    direction = project_onto_user_plane(norm(labels[i] - scene.object_centers[j]), ref, user_position)
                    forces[i] += W_OVERLAP * OVERLAP_M * ratio * direction

            forces[i] += W_LINE * line_forces[i]
        labels = labels + forces * DISPLACEMENT_SCALE
    return labels


def mean_metric_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {field: 0.0 for field in METRIC_FIELDS}
    return {field: float(np.mean([item[field] for item in values])) for field in METRIC_FIELDS}


def dynamic_binoforce_run(
    scene: Scene,
    warmup_frames: int,
    eval_frames: int,
    fps: int = DEFAULT_FPS,
    binocular: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    labels = scene.anchors.copy()
    total_frames = max(0, warmup_frames) + max(1, eval_frames)
    final_layouts: dict[str, np.ndarray] = {}
    metric_history: dict[str, list[dict[str, float]]] = {view: [] for view in VIEW_ORDER}

    for frame in range(total_frames):
        cam = trajectory_camera(scene, frame, fps)
        labels = force_layout(scene, "main", binocular=binocular, iterations=1, initial_labels=labels, camera_override=cam)
        if frame >= warmup_frames:
            for view in VIEW_ORDER:
                final_layouts[view] = labels.copy()
                metric_history[view].append(metric_values(scene, labels, view))

    for view in VIEW_ORDER:
        final_layouts.setdefault(view, labels.copy())
    return final_layouts, {view: mean_metric_dict(metric_history[view]) for view in VIEW_ORDER}


def baseline_layout(scene: Scene) -> np.ndarray:
    labels = np.zeros_like(scene.anchors)
    for i, anchor in enumerate(scene.anchors):
        labels[i] = anchor + norm(anchor - scene.scene_center) * scene.circular_radius
    return labels


def manual_layout(scene: Scene) -> np.ndarray:
    return scene.manual_labels.copy()


def per_label_overlap_areas(scene: Scene, labels: np.ndarray, cam: Camera, eye: float) -> np.ndarray:
    lrects, _, _ = label_rects(scene, labels, cam, eye)
    orects = object_rects(scene, cam, eye)
    overlaps = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        total = 0.0
        for j in range(len(labels)):
            if i != j:
                total += overlap_area(lrects[i], lrects[j])
        for orect in orects:
            total += overlap_area(lrects[i], orect)
        overlaps[i] = total
    return overlaps


def olr(scene: Scene, labels: np.ndarray, view: str, binocular: bool = True) -> float:
    cam = camera_for(scene, view)
    eyes = (-IPD_METERS / 2, IPD_METERS / 2) if binocular else (0.0,)
    ratios = []
    for eye in eyes:
        lrects, _, _ = label_rects(scene, labels, cam, eye)
        areas = np.asarray([rect_area(r) for r in lrects])
        ratios.append(per_label_overlap_areas(scene, labels, cam, eye) / np.maximum(areas, 1e-9))
    return float(np.mean(np.max(np.vstack(ratios), axis=0))) if len(ratios) else 0.0


def dbv(scene: Scene, labels: np.ndarray, view: str) -> float:
    cam = camera_for(scene, view)
    lrects, _, _ = label_rects(scene, labels, cam, 0.0)
    areas = np.asarray([rect_area(r) for r in lrects])
    left = per_label_overlap_areas(scene, labels, cam, -IPD_METERS / 2)
    right = per_label_overlap_areas(scene, labels, cam, IPD_METERS / 2)
    return float(np.mean(np.abs(left - right) / np.maximum(areas, 1e-9))) if len(labels) else 0.0


def lcd(scene: Scene, labels: np.ndarray, view: str) -> float:
    cam = camera_for(scene, view)
    label_2d = project(labels, cam, 0.0)[0]
    anchor_2d = project(scene.anchors, cam, 0.0)[0]
    if len(labels) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if segments_intersect(anchor_2d[i], label_2d[i], anchor_2d[j], label_2d[j]):
                u = segment_intersection(anchor_2d[i], label_2d[i], anchor_2d[j], label_2d[j])
                area1 = triangle_area(u, label_2d[i], label_2d[j])
                area2 = triangle_area(u, anchor_2d[i], anchor_2d[j])
                total += min(area1, area2) / max(area1 + area2, 1e-12)
    return float(total / len(labels))


def leader_length(scene: Scene, labels: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(labels - scene.anchors, axis=1))) if len(labels) else 0.0


def metric_values(scene: Scene, labels: np.ndarray, view: str) -> dict[str, float]:
    camera = CAMERAS[view]
    anchors = scene.anchors
    anchor_2d = project_world_to_pixels(anchors, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    center_2d = project_world_to_pixels(labels, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    manual_2d = project_world_to_pixels(scene.manual_labels, camera, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    rects = np.asarray(
        [
            projected_label_bounds(center, np.r_[size, 0.02], camera)
            for center, size in zip(labels, scene.sizes)
        ],
        dtype=float,
    )

    object_rect = object_rect_for_scene(scene, view, camera)
    overlap_pairs = 0
    per_label_overlap = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            area = overlap_area(rects[i], rects[j])
            if area > 1e-9:
                overlap_pairs += 1
                per_label_overlap[i] += area
                per_label_overlap[j] += area
    if object_rect is not None:
        for i, rect in enumerate(rects):
            per_label_overlap[i] += overlap_area(rect, object_rect)

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
    olr_value = float(np.mean(per_label_overlap / np.maximum(label_areas, 1e-9))) if len(labels) else 0.0
    lcd_value = float(lcd_total / len(labels)) if len(labels) else 0.0
    avg_length_ratio = (float(np.mean(lengths)) / image_diag) if len(lengths) else 0.0
    center_error = np.linalg.norm(center_2d - manual_2d, axis=1) / image_diag
    pck_005 = float(np.mean(center_error <= 0.05)) if len(labels) else 0.0
    pck_010 = float(np.mean(center_error <= 0.10)) if len(labels) else 0.0
    quality_score = float(100.0 * (0.7 * pck_005 + 0.3 * pck_010) - 25.0 * olr_value - 10.0 * lcd_value)
    return {
        "PCK_005": pck_005,
        "PCK_010": pck_010,
        "OLR": olr_value,
        "LCD": lcd_value,
        "DBV": dbv(scene, labels, view),
        "avg_leader_length": avg_length_ratio,
        "overlap_pairs": float(overlap_pairs),
        "occluded_points": float(occluded_points),
        "intersections": float(intersections),
        "quality_score": quality_score,
    }


def run_scene_experiment(
    scene: Scene,
    warmup_frames: int,
    eval_frames: int,
    fps: int = DEFAULT_FPS,
    include_monocular: bool = True,
) -> tuple[list[dict[str, str]], list[tuple[str, str, str, str, np.ndarray]]]:
    rows: list[dict[str, str]] = []
    layouts: list[tuple[str, str, str, str, np.ndarray]] = []
    baseline = baseline_layout(scene)
    binoforce_layouts, binoforce_metrics = dynamic_binoforce_run(scene, warmup_frames, eval_frames, fps, binocular=True)
    monocular_layouts: dict[str, np.ndarray] = {}
    monocular_metrics: dict[str, dict[str, float]] = {}
    if include_monocular:
        monocular_layouts, monocular_metrics = dynamic_binoforce_run(scene, warmup_frames, eval_frames, fps, binocular=False)
    for view in VIEW_ORDER:
        methods = {"Baseline": baseline}
        if include_monocular:
            methods["MonocularForce"] = monocular_layouts[view]
        methods["BinoForce"] = binoforce_layouts[view]
        layouts.extend((scene.category, scene.sample, view, name, arr) for name, arr in methods.items())
        for method, label_centers in methods.items():
            if method == "BinoForce":
                metrics = binoforce_metrics[view]
            elif method == "MonocularForce":
                metrics = monocular_metrics[view]
            else:
                metrics = metric_values(scene, label_centers, view)
            row = {"category": scene.category, "sample": scene.sample, "view": view, "method": method, "num_labels": str(len(scene.anchors))}
            row.update({key: f"{value:.8f}" for key, value in metrics.items()})
            rows.append(row)
    return rows, layouts

def aggregate(rows: list[dict[str, str]], group_fields: tuple[str, ...]) -> list[dict[str, object]]:
    out = []
    keys = sorted({tuple(row[field] for field in group_fields) for row in rows})
    for key in keys:
        subset = [row for row in rows if tuple(row[field] for field in group_fields) == key]
        item: dict[str, object] = dict(zip(group_fields, key))
        for metric in METRIC_FIELDS:
            vals = [float(row[metric]) for row in subset if metric in row and row[metric] != ""]
            item[metric] = float(np.mean(vals)) if vals else 0.0
        out.append(item)
    return out


def rows_for_method(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return [row for row in rows if row["method"] == method]


def category_method_metric(rows: list[dict[str, str]], category: str, method: str, metric: str) -> float:
    vals = [float(row[metric]) for row in rows if row["category"] == category and row["method"] == method]
    return float(np.mean(vals)) if vals else 0.0


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_layout_jsonl(out_dir: Path, layout_cache: dict[tuple[str, str, str, str], np.ndarray], scenes: list[Scene]) -> Path:
    scene_map = {(s.category, s.sample): s for s in scenes}
    path = out_dir / "binoforce_layouts.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for (category, sample, view, method), labels in sorted(layout_cache.items()):
            if method != "BinoForce":
                continue
            scene = scene_map[(category, sample)]
            record = {
                "category": category,
                "sample": sample,
                "view": view,
                "method": method,
                "camera_source": VIEW_CAMERA_SOURCE,
                "label_centers": [
                    {"group_id": group_id, "text": text, "center": [float(v) for v in center]}
                    for group_id, text, center in zip(scene.group_ids, scene.texts, labels)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_metric_tables(out_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    method_summary = aggregate(rows, ("method",))
    category_summary = aggregate(rows, ("category", "method"))

    method_csv = out_dir / "method_summary.csv"
    category_csv = out_dir / "category_method_summary.csv"
    write_csv(method_csv, method_summary, ["method", *METRIC_FIELDS])
    write_csv(category_csv, category_summary, ["category", "method", *METRIC_FIELDS])

    display_metrics = ("PCK_005", "PCK_010", "OLR", "LCD", "DBV", "avg_leader_length", "quality_score")
    imported_methods = {"manual", "hedgehog_1d", "hedgehog_3d", "plane"}

    def display_value(row: dict[str, object], metric: str) -> str:
        if metric == "DBV" and str(row.get("method", "")) in imported_methods:
            return "n/a"
        return f"{float(row[metric]):.5f}"

    method_lines = [
        "# Method summary using Hedgehog metrics",
        "",
        "| Method | " + " | ".join(display_metrics) + " |",
        "|---|" + "---:|" * len(display_metrics),
    ]
    method_lines += [
        "| " + str(row["method"]) + " | " + " | ".join(display_value(row, m) for m in display_metrics) + " |"
        for row in method_summary
    ]
    method_md = out_dir / "method_summary.md"
    method_md.write_text("\n".join(method_lines), encoding="utf-8")

    category_lines = [
        "# Category/method summary using Hedgehog metrics",
        "",
        "| Category | Method | " + " | ".join(display_metrics) + " |",
        "|---|---|" + "---:|" * len(display_metrics),
    ]
    category_lines += [
        "| "
        + str(row["category"])
        + " | "
        + str(row["method"])
        + " | "
        + " | ".join(display_value(row, m) for m in display_metrics)
        + " |"
        for row in category_summary
    ]
    category_md = out_dir / "category_method_summary.md"
    category_md.write_text("\n".join(category_lines), encoding="utf-8")
    return method_md, category_md

def write_dbv_comparison(out_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    categories = sorted({row["category"] for row in rows})
    comparison_rows: list[dict[str, object]] = []
    for category in categories:
        mono = category_method_metric(rows, category, "MonocularForce", "DBV")
        bino = category_method_metric(rows, category, "BinoForce", "DBV")
        reduction = (1.0 - bino / mono) * 100.0 if mono > 1e-12 else 0.0
        comparison_rows.append(
            {
                "category": category,
                "DBV MonocularForce": f"{mono:.5f}",
                "DBV BinoForce": f"{bino:.5f}",
                "Reduction": f"{reduction:.2f}%",
            }
        )
    mono_all = float(np.mean([float(row["DBV"]) for row in rows if row["method"] == "MonocularForce"])) if any(row["method"] == "MonocularForce" for row in rows) else 0.0
    bino_all = float(np.mean([float(row["DBV"]) for row in rows if row["method"] == "BinoForce"])) if any(row["method"] == "BinoForce" for row in rows) else 0.0
    reduction_all = (1.0 - bino_all / mono_all) * 100.0 if mono_all > 1e-12 else 0.0
    comparison_rows.append(
        {
            "category": "All",
            "DBV MonocularForce": f"{mono_all:.5f}",
            "DBV BinoForce": f"{bino_all:.5f}",
            "Reduction": f"{reduction_all:.2f}%",
        }
    )
    csv_path = out_dir / "dbv_comparison.csv"
    md_path = out_dir / "dbv_comparison.md"
    write_csv(csv_path, comparison_rows, ["category", "DBV MonocularForce", "DBV BinoForce", "Reduction"])
    lines = [
        "# DBV comparison: MonocularForce vs BinoForce",
        "",
        "| Category | DBV MonocularForce | DBV BinoForce | Reduction |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {row['category']} | {row['DBV MonocularForce']} | {row['DBV BinoForce']} | {row['Reduction']} |"
        for row in comparison_rows
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path

def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def shorten(text: str, max_chars: int = 21) -> str:
    text = text.replace("_", " ")
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


def parse_obj_vertices(obj_path: Path, max_points: int = 22000) -> np.ndarray:
    vertices: list[list[float]] = []
    if not obj_path.exists():
        return np.empty((0, 3), dtype=float)
    keep_group = True
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("g "):
                name = line.strip().split(maxsplit=1)[1].lower() if len(line.split(maxsplit=1)) > 1 else ""
                keep_group = not (name.startswith("label") or name.startswith("leader"))
            elif keep_group and line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.asarray(vertices, dtype=float)
    if len(arr) > max_points:
        arr = arr[:: max(1, len(arr) // max_points)]
    return arr


MESH_CACHE: dict[Path, tuple[np.ndarray, list[list[int]], list[tuple[int, int, int]]]] = {}
MTL_CACHE: dict[Path, dict[str, tuple[int, int, int]]] = {}


def load_mtl_diffuse(path: Path) -> dict[str, tuple[int, int, int]]:
    if path in MTL_CACHE:
        return MTL_CACHE[path]
    colors: dict[str, tuple[int, int, int]] = {}
    current: str | None = None
    if path.exists():
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "newmtl" and len(parts) >= 2:
                    current = parts[1]
                elif parts[0] == "Kd" and current and len(parts) >= 4:
                    rgb = tuple(int(max(0.0, min(1.0, float(value))) * 255) for value in parts[1:4])
                    colors[current] = rgb
    MTL_CACHE[path] = colors
    return colors


def parse_clean_obj_mesh(obj_path: Path) -> tuple[np.ndarray, list[list[int]], list[tuple[int, int, int]]]:
    if obj_path in MESH_CACHE:
        return MESH_CACHE[obj_path]
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_colors: list[tuple[int, int, int]] = []
    material_colors = load_mtl_diffuse(obj_path.with_name(f"{obj_path.stem.split('-')[0]}-O.mtl"))
    current_color = material_colors.get("object_default", (184, 184, 184))
    keep_group = True
    if obj_path.exists():
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("o label_", "g label_", "o leader_", "g leader_")):
                    keep_group = False
                    continue
                if line.startswith(("o ", "g ")) and not line.startswith(("o label_", "g label_", "o leader_", "g leader_")):
                    keep_group = True
                if line.startswith("usemtl "):
                    material = line.split(maxsplit=1)[1].strip()
                    current_color = material_colors.get(material, current_color)
                elif keep_group and line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif keep_group and line.startswith("f "):
                    face: list[int] = []
                    for token in line.split()[1:]:
                        index_text = token.split("/")[0]
                        if not index_text:
                            continue
                        index = int(index_text)
                        index = len(vertices) + index if index < 0 else index - 1
                        if 0 <= index < len(vertices):
                            face.append(index)
                    if len(face) >= 3:
                        faces.append(face)
                        face_colors.append(current_color)
    result = (np.asarray(vertices, dtype=float), faces, face_colors)
    MESH_CACHE[obj_path] = result
    return result


def projected_label_bounds_pixels(label_center: np.ndarray, box_size: np.ndarray, camera: dict[str, np.ndarray | float | str], width: int, height: int) -> tuple[float, float, float, float]:
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
    projected = project_world_to_pixels(np.asarray(corners), camera, width, height)
    return (
        float(np.min(projected[:, 0])),
        float(np.min(projected[:, 1])),
        float(np.max(projected[:, 0])),
        float(np.max(projected[:, 1])),
    )


def fit_text_to_rect(text: str, rect: list[float], bold: bool = False) -> tuple[str, ImageFont.ImageFont]:
    width = max(1.0, rect[2] - rect[0] - 4.0)
    height = max(1.0, rect[3] - rect[1] - 2.0)
    for size in range(12, 5, -1):
        font = load_font(size, bold)
        candidate = text
        while candidate:
            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= width and bbox[3] - bbox[1] <= height:
                return candidate, font
            if len(candidate) <= 4:
                break
            candidate = candidate[:-4] + "..."
    return "", load_font(6, bold)

def render_clean_obj(scene: Scene, view: str, width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    obj_path = obj_path_for(scene, "main")
    vertices, faces, colors = parse_clean_obj_mesh(obj_path)
    if not len(vertices) or not faces:
        return img
    camera = CAMERAS[view]
    projected = project_world_to_pixels(vertices, camera, width, height)
    camera_points = world_to_camera(vertices, camera)
    depths = -camera_points[:, 2]
    order = sorted(range(len(faces)), key=lambda idx: float(np.mean(depths[faces[idx]])), reverse=True)
    for idx in order:
        face = faces[idx]
        points = [(float(projected[v, 0]), float(projected[v, 1])) for v in face]
        if max(p[0] for p in points) < 0 or min(p[0] for p in points) > width or max(p[1] for p in points) < 0 or min(p[1] for p in points) > height:
            continue
        draw.polygon(points, fill=colors[idx])
    return img

def obj_path_for(scene: Scene, view: str) -> Path:
    return scene.obj_dir / f"{scene.sample}-{view}-O.obj"


def scene_bounds(scene: Scene, labels_list: list[np.ndarray], view: str, obj_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    pts = [to_view_coords(scene, view, scene.anchors)[:, :2], to_view_coords(scene, view, scene.object_centers)[:, :2]]
    pts.extend(to_view_coords(scene, view, labels)[:, :2] for labels in labels_list)
    if obj_points is not None and len(obj_points):
        pts.append(to_view_coords(scene, view, obj_points)[:, :2])
    all_pts = np.vstack(pts)
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = np.maximum((mx - mn) * 0.12, 0.18)
    return mn - pad, mx + pad


def panel_geometry(width: int, height: int) -> tuple[int, int, np.ndarray]:
    header_h = 58
    footer_h = 38
    render_w = min(PREVIEW_WIDTH, width - 32)
    render_h = min(PREVIEW_HEIGHT, height - header_h - footer_h - 10)
    origin = np.asarray([(width - render_w) * 0.5, float(header_h)], dtype=float)
    return render_w, render_h, origin


def draw_obj_layout_panel(scene: Scene, labels: np.ndarray, view: str, width: int, height: int, title: str) -> Image.Image:
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = load_font(16, True)
    small_font = load_font(9)
    chip_font = load_font(10)
    render_w, render_h, render_origin = panel_geometry(width, height)
    method_colors = {"ManualData": "#7b8794", "Baseline": "#d9822b", "BinoForce": "#1f78b4"}
    accent = method_colors.get(title, "#7b8794")

    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=4, outline="#dfe4ea", width=1, fill="#ffffff")
    draw.rectangle([0, 0, width - 1, 58], fill="#fbfcfd")
    draw.rectangle([12, 14, 16, 31], fill=accent)
    draw.text((26, 11), title, font=title_font, fill="#111827")
    draw.text((26, 32), f"{scene.category}/{scene.sample} {view}", font=small_font, fill="#66707c")

    body = render_clean_obj(scene, view, render_w, render_h)
    img.paste(body, (int(render_origin[0]), int(render_origin[1])))

    camera = CAMERAS[view]
    anchor_px = project_world_to_pixels(scene.anchors, camera, render_w, render_h) + render_origin
    label_px = project_world_to_pixels(labels, camera, render_w, render_h) + render_origin

    for text, size, anchor, label, center in zip(scene.texts, scene.sizes, anchor_px, label_px, labels):
        ax, ay = float(anchor[0]), float(anchor[1])
        lx, ly = float(label[0]), float(label[1])
        draw.line([ax, ay, lx, ly], fill="#ef6f61", width=1)
        draw.ellipse([ax - 2, ay - 2, ax + 2, ay + 2], fill="#111827")

        rect_local = projected_label_bounds_pixels(center, np.r_[size, 0.02], camera, render_w, render_h)
        rect = [
            rect_local[0] + float(render_origin[0]),
            rect_local[1] + float(render_origin[1]),
            rect_local[2] + float(render_origin[0]),
            rect_local[3] + float(render_origin[1]),
        ]
        # Keep the data-derived label rectangle size; only clamp the rendered
        # location enough to keep it visible inside the image panel.
        rect_w = rect[2] - rect[0]
        rect_h = rect[3] - rect[1]
        min_x = float(render_origin[0]) + 1.0
        max_x = float(render_origin[0]) + render_w - rect_w - 1.0
        min_y = float(render_origin[1]) + 1.0
        max_y = float(render_origin[1]) + render_h - rect_h - 1.0
        if rect_w < render_w - 2:
            dx = max(min_x, min(max_x, rect[0])) - rect[0]
            rect[0] += dx
            rect[2] += dx
        if rect_h < render_h - 2:
            dy = max(min_y, min(max_y, rect[1])) - rect[1]
            rect[1] += dy
            rect[3] += dy
        draw.rectangle(rect, fill="#f5f3ea", outline="#777d85")
        shown, font = fit_text_to_rect(shorten(text, 28), rect)
        if shown:
            bbox = draw.textbbox((0, 0), shown, font=font)
            tx = rect[0] + max(2.0, (rect[2] - rect[0] - (bbox[2] - bbox[0])) * 0.5)
            ty = rect[1] + max(1.0, (rect[3] - rect[1] - (bbox[3] - bbox[1])) * 0.5 - 1.0)
            draw.text((tx, ty), shown, font=font, fill="#111827")

    draw_metric_chips(draw, 12, height - 31, metric_chip_text(scene, labels, view), chip_font)
    return img


def draw_manual_image_panel(scene: Scene, view: str, width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = load_font(16, True)
    small_font = load_font(9)
    chip_font = load_font(10)
    render_w, render_h, render_origin = panel_geometry(width, height)
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=4, outline="#dfe4ea", width=1, fill="#ffffff")
    draw.rectangle([0, 0, width - 1, 58], fill="#fbfcfd")
    draw.rectangle([12, 14, 16, 31], fill="#7b8794")
    draw.text((26, 11), "ManualData", font=title_font, fill="#111827")
    draw.text((26, 32), f"{scene.category}/{scene.sample} {view} - dataset Mutiviews", font=small_font, fill="#66707c")
    src = scene.mutiviews_dir / f"{scene.sample}-{view}.png"
    if src.exists():
        manual = Image.open(src).convert("RGB")
        if manual.size != (render_w, render_h):
            manual = manual.resize((render_w, render_h), Image.Resampling.LANCZOS)
        img.paste(manual, (int(render_origin[0]), int(render_origin[1])))
    else:
        draw.text((int(render_origin[0]) + 20, int(render_origin[1]) + render_h // 2), "dataset Mutiviews image missing", font=small_font, fill="#64748b")
    draw_metric_chips(draw, 12, height - 31, metric_chip_text(scene, scene.manual_labels, view), chip_font)
    return img

def metric_chip_text(scene: Scene, labels: np.ndarray, view: str) -> list[str]:
    metrics = metric_values(scene, labels, view)
    return [f"PCK@5 {metrics['PCK_005']:.2f}", f"OLR {metrics['OLR']:.3f}", f"LCD {metrics['LCD']:.3f}"]


def draw_metric_chips(draw: ImageDraw.ImageDraw, x: int, y: int, chips: list[str], font: ImageFont.ImageFont) -> None:
    cursor = x
    for chip in chips:
        bbox = draw.textbbox((0, 0), chip, font=font)
        tw = bbox[2] - bbox[0]
        draw.rounded_rectangle([cursor, y, cursor + tw + 18, y + 25], radius=12, fill="#e8eef5", outline="#d8e0ea")
        draw.text((cursor + 9, y + 5), chip, font=font, fill="#64748b")
        cursor += tw + 28


def draw_generated_render_panel(scene: Scene, labels: np.ndarray, view: str, width: int, height: int, title: str) -> Image.Image:
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = load_font(16, True)
    label_font = load_font(10)
    chip_font = load_font(11)
    small_font = load_font(9)
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=4, outline="#d8e0ea", width=1, fill="#ffffff")
    draw.text((14, 10), title, font=title_font, fill="#111827")
    draw.text((14, 30), f"{scene.category}/{scene.sample} {view}", font=small_font, fill="#64748b")

    obj_points = parse_obj_vertices(obj_path_for(scene, view))
    label_xy = to_view_coords(scene, view, labels)[:, :2]
    anchor_xy = to_view_coords(scene, view, scene.anchors)[:, :2]
    pts = [label_xy, anchor_xy]
    if len(obj_points):
        pts.append(obj_points[:, :2])
    all_pts = np.vstack(pts)
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = np.maximum((mx - mn) * 0.18, 0.18)
    mn, mx = mn - pad, mx + pad
    plot = [24, 54, width - 24, height - 44]

    def to_px(p: np.ndarray) -> tuple[float, float]:
        x = plot[0] + (p[0] - mn[0]) / max(mx[0] - mn[0], 1e-6) * (plot[2] - plot[0])
        y = plot[3] - (p[1] - mn[1]) / max(mx[1] - mn[1], 1e-6) * (plot[3] - plot[1])
        return float(x), float(y)

    if len(obj_points):
        obj_xy = obj_points[:, :2]
        step = max(1, len(obj_xy) // 9000)
        for p in obj_xy[::step]:
            draw.point(to_px(p), fill="#94a3b8")

    for text, a, l in zip(scene.texts, anchor_xy, label_xy):
        ax, ay = to_px(a)
        lx, ly = to_px(l)
        draw.line([ax, ay, lx, ly], fill="#f19985", width=1)
        draw.ellipse([ax - 2, ay - 2, ax + 2, ay + 2], fill="#111827")
        shown = shorten(text, 22)
        bbox = draw.textbbox((0, 0), shown, font=label_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect = [lx - tw / 2 - 5, ly - th / 2 - 3, lx + tw / 2 + 5, ly + th / 2 + 3]
        draw.rectangle(rect, fill="#f5f3ea", outline="#9ca3af")
        draw.text((lx - tw / 2, ly - th / 2 - 1), shown, font=label_font, fill="#111827")

    draw_metric_chips(draw, 12, height - 34, metric_chip_text(scene, labels, view), chip_font)
    return img


def draw_image_overlay_panel(scene: Scene, labels: np.ndarray, view: str, width: int, height: int, title: str) -> Image.Image:
    img = draw_manual_image_panel(scene, view, width, height)
    draw = ImageDraw.Draw(img)
    title_font = load_font(18, True)
    label_font = load_font(10)
    small_font = load_font(10)
    draw.rectangle([8, 8, width - 8, 50], fill="#f8fafc")
    draw.text((16, 13), title, font=title_font, fill="#111827")
    draw.text((17, 36), f"{scene.category} {scene.sample} / {view}", font=small_font, fill="#64748b")

    src = scene.mutiviews_dir / f"{scene.sample}-{view}.png"
    if not src.exists():
        return img
    bg = Image.open(src).convert("RGB")
    bg.thumbnail((width - 34, height - 64), Image.Resampling.LANCZOS)
    image_origin = np.asarray([(width - bg.width) / 2, 58 + (height - 64 - bg.height) / 2], dtype=float)
    image_size = np.asarray([bg.width, bg.height], dtype=float)
    label_xy = to_view_coords(scene, view, labels)[:, :2]
    anchor_xy = to_view_coords(scene, view, scene.anchors)[:, :2]
    manual_xy = to_view_coords(scene, view, scene.manual_labels)[:, :2]
    pts = np.vstack([label_xy, anchor_xy, manual_xy])
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    pad = np.maximum((mx - mn) * 0.18, 0.15)
    mn, mx = mn - pad, mx + pad

    def to_px(p: np.ndarray) -> tuple[float, float]:
        x = image_origin[0] + (p[0] - mn[0]) / max(mx[0] - mn[0], 1e-6) * image_size[0]
        y = image_origin[1] + (1.0 - (p[1] - mn[1]) / max(mx[1] - mn[1], 1e-6)) * image_size[1]
        return float(x), float(y)

    for text, a, l in zip(scene.texts, anchor_xy, label_xy):
        ax, ay = to_px(a)
        lx, ly = to_px(l)
        draw.line([ax, ay, lx, ly], fill="#2563eb", width=1)
        draw.ellipse([ax - 2, ay - 2, ax + 2, ay + 2], fill="#2563eb")
        shown = shorten(text, 17)
        bbox = draw.textbbox((0, 0), shown, font=label_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect = [lx - tw / 2 - 5, ly - th / 2 - 3, lx + tw / 2 + 5, ly + th / 2 + 3]
        draw.rounded_rectangle(rect, radius=3, fill="#ffffff", outline="#0f766e", width=1)
        draw.text((lx - tw / 2, ly - th / 2 - 1), shown, font=label_font, fill="#0f172a")
    return img


def write_obj_overlay(out_dir: Path, scene: Scene, labels: np.ndarray) -> Path:
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / "obj_overlay_best.png"
    panel = draw_obj_layout_panel(scene, labels, "main", 980, 680, "BinoForce on provided 3D OBJ")
    panel.save(path)
    return path


def draw_manual_comparison_panel(scene: Scene, view: str, width: int, height: int) -> Image.Image:
    panel = draw_manual_image_panel(scene, view, width, height)
    draw = ImageDraw.Draw(panel)
    chip_font = load_font(11)
    draw_metric_chips(draw, 12, height - 34, metric_chip_text(scene, scene.manual_labels, view), chip_font)
    return panel


def write_layout_comparison(out_dir: Path, scene: Scene, layout_cache: dict[tuple[str, str, str, str], np.ndarray]) -> Path:
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 820, 610
    methods = [
        ("ManualData", scene.manual_labels),
        ("Baseline", layout_cache[(scene.category, scene.sample, "main", "Baseline")]),
        ("BinoForce", layout_cache[(scene.category, scene.sample, "main", "BinoForce")]),
    ]
    canvas = Image.new("RGB", (len(methods) * panel_w + 30, panel_h + 132), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(21, True)
    sub_font = load_font(13)
    draw.text((18, 20), f"{scene.category} / {scene.sample}", font=title_font, fill="#111827")
    draw.text((18, 50), "main view", font=sub_font, fill="#64748b")
    colors = {"ManualData": "#94a3b8", "Baseline": "#d97706", "BinoForce": "#16a34a"}
    for i, (name, labels) in enumerate(methods):
        x = 15 + i * panel_w
        y = 108
        draw.rectangle([x, y - 28, x + 5, y - 8], fill=colors[name])
        draw.text((x + 18, y - 30), name, font=load_font(16, True), fill="#111827")
        panel = draw_manual_image_panel(scene, "main", panel_w - 28, panel_h - 18) if name == "ManualData" else draw_obj_layout_panel(scene, labels, "main", panel_w - 28, panel_h - 18, name)
        canvas.paste(panel, (x, y))
    path = vis_dir / "layout_comparison_main.png"
    canvas.save(path)
    return path


def write_camera_comparison(out_dir: Path, scene: Scene, layout_cache: dict[tuple[str, str, str, str], np.ndarray]) -> Path:
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 820, 610
    cols = 3
    rows = len(VIEW_ORDER)
    header_h = 96
    canvas = Image.new("RGB", (cols * panel_w + 44, rows * panel_h + header_h + 20), "#edf2f7")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, True)
    sub_font = load_font(13)
    draw.text((22, 18), f"Camera-view comparison: {scene.category} {scene.sample}", font=title_font, fill="#0f172a")
    draw.text((24, 51), f"Fixed photo views: {', '.join(VIEW_ORDER)}. Dynamic labels are projected onto the clean colored OBJ body.", font=sub_font, fill="#475569")
    for r, view in enumerate(VIEW_ORDER):
        y = header_h + r * panel_h
        panels = [
            draw_manual_image_panel(scene, view, panel_w - 16, panel_h - 14),
            draw_obj_layout_panel(scene, layout_cache[(scene.category, scene.sample, view, "Baseline")], view, panel_w - 16, panel_h - 14, "Baseline"),
            draw_obj_layout_panel(scene, layout_cache[(scene.category, scene.sample, view, "BinoForce")], view, panel_w - 16, panel_h - 14, "BinoForce"),
        ]
        for c, panel in enumerate(panels):
            canvas.paste(panel, (22 + c * panel_w, y))
    path = vis_dir / "camera_view_comparison.png"
    canvas.save(path)
    return path


def write_report(
    out_dir: Path,
    rows: list[dict[str, str]],
    method_md: Path,
    category_md: Path,
    layout_comparison: Path,
    comparison: Path,
    samples: int,
    warmup_frames: int,
    eval_frames: int,
    hedgehog_rows: int,
) -> Path:
    def md_to_html_table(md_path: Path) -> str:
        lines = [line for line in md_path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
        head = [cell.strip() for cell in lines[0].strip("|").split("|")]
        body = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]
        th = "".join(f"<th>{html.escape(cell)}</th>" for cell in head)
        trs = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in body)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"

    method_table = md_to_html_table(method_md)
    category_table = md_to_html_table(category_md)
    dbv_path = out_dir / "dbv_comparison.md"
    dbv_table = md_to_html_table(dbv_path) if dbv_path.exists() else ""
    rel_layout = layout_comparison.relative_to(out_dir).as_posix()
    rel_camera = comparison.relative_to(out_dir).as_posix()
    metric_notes = [
        ("PCK@5 / PCK@10", "higher is better", "Share of labels whose projected centers stay within 5% or 10% of the manual layout image diagonal."),
        ("OLR", "lower is better", "Average label overlap ratio. It penalizes label-label and label-object overlap."),
        ("LCD", "lower is better", "Average leader-line crossing degree. Zero means no crossing in the projected view."),
        ("DBV", "lower is better", "Double Vision Degree. It measures left/right-eye overlap imbalance; lower means less binocular ghosting risk."),
        ("avg_leader_length", "balanced is better", "Normalized projected leader length. Too short hides labels inside the object; too long hurts readability."),
        ("overlap_pairs", "lower is better", "Number of label pairs with non-zero overlap in the rendered view."),
        ("occluded_points", "lower is better", "Number of anchor points covered by other label rectangles."),
        ("intersections", "lower is better", "Raw count of crossing leader-line pairs."),
        ("quality_score", "higher is better", "Composite score used for quick comparison: PCK reward minus OLR/LCD penalties."),
    ]
    metric_cards = "".join(
        "<article class='metric'><strong>" + html.escape(name) + "</strong><span>" + html.escape(direction) + "</span><p>" + html.escape(desc) + "</p></article>"
        for name, direction, desc in metric_notes
    )
    html_text = f"""<!doctype html>
<meta charset="utf-8">
<title>BinoForce 2025 dynamic reproduction</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;color:#20242a;background:#f4f6f8}}
.top{{position:sticky;top:0;z-index:2;background:#ffffffea;backdrop-filter:blur(10px);border-bottom:1px solid #dfe4ea;padding:18px 28px}}
h1{{font-size:23px;margin:0 0 6px;letter-spacing:0}}
.top p{{margin:0;color:#66707c;font-size:13px;line-height:1.6;max-width:1180px}}
main{{padding:22px 28px 40px;display:grid;gap:22px}}
section{{background:#fff;border:1px solid #dfe4ea;border-radius:8px;overflow:hidden;box-shadow:0 8px 26px rgba(31,42,55,.06)}}
section header{{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:14px 16px;border-bottom:1px solid #e7ebf0;background:#fbfcfd}}
section header strong{{font-size:15px}}
section header span{{display:block;margin-top:3px;color:#687482;font-size:12px}}
.content{{padding:16px}}
.grid{{display:grid;grid-template-columns:1fr;gap:18px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px}}
.metric{{border:1px solid #dfe4ea;border-radius:8px;padding:12px;background:#fbfcfd}}
.metric strong{{display:block;font-size:13px;margin-bottom:4px}}
.metric span{{display:inline-block;font-size:12px;color:#1f5f8b;background:#e8f2fb;border:1px solid #cfe0ef;border-radius:999px;padding:2px 7px;margin-bottom:8px}}
.metric p{{margin:0;color:#52606d;font-size:12px;line-height:1.55}}
img{{width:100%;height:auto;display:block;background:white;border:1px solid #e2e7ed}} figure{{margin:0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid #edf0f3;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{background:#fbfcfd;color:#52606d;font-weight:700}}
.note{{display:flex;gap:8px;flex-wrap:wrap;color:#52606d;font-size:13px;line-height:1.6}}
.note span{{background:#eef2f6;border:1px solid #dce3ea;border-radius:999px;padding:4px 8px}}
@media(max-width:1100px){{.metrics{{grid-template-columns:repeat(2,minmax(180px,1fr))}}.grid{{grid-template-columns:1fr}}}}
@media(max-width:640px){{main{{padding:14px}}.top{{padding:16px}}.metrics{{grid-template-columns:1fr}}}}
</style>
<div class="top">
  <h1>BinoForce 2025 Dynamic Reproduction</h1>
  <p>Labels are updated in 3D under a deterministic dynamic camera trajectory, then photographed from the five fixed camera views. Manual, Hedgehog, and plane rows are imported from the existing Hedgehog results for metric-consistent comparison.</p>
</div>
<main>
  <section>
    <header><div><strong>Workflow</strong><span>3D dynamic layout first, five-view projection second</span></div></header>
    <div class="content note">
      <span>samples {samples}</span>
      <span>warmup frames {warmup_frames}</span>
      <span>evaluation frames {eval_frames}</span>
      <span>imported Hedgehog/plane rows {hedgehog_rows}</span>
      <span>views {', '.join(VIEW_ORDER)}</span>
      <span>clean colored OBJ body rendering</span>
    </div>
  </section>
  <section>
    <header><div><strong>Method Summary</strong><span>Hedgehog-compatible metric schema</span></div></header>
    <div class="content">{method_table}</div>
  </section>
  <section>
    <header><div><strong>DBV Comparison</strong><span>MonocularForce vs BinoForce; lower DBV is better</span></div></header>
    <div class="content">{dbv_table}</div>
  </section>
  <section>
    <header><div><strong>How To Read Metrics</strong><span>direction of improvement</span></div></header>
    <div class="content metrics">{metric_cards}</div>
  </section>
  <section>
    <header><div><strong>Rendered Previews</strong><span>colored clean OBJ body plus projected labels</span></div></header>
    <div class="content grid">
      <figure><img src="{rel_layout}" alt="main-view layout comparison"></figure>
      <figure><img src="{rel_camera}" alt="five-view camera comparison"></figure>
    </div>
  </section>
  <section>
    <header><div><strong>Category Summary</strong><span>per category and method</span></div></header>
    <div class="content">{category_table}</div>
  </section>
</main>
"""
    path = out_dir / "report.html"
    path.write_text(html_text, encoding="utf-8")
    return path

def write_notes(root: Path, out_dir: Path, samples: int, rows: int, warmup_frames: int, eval_frames: int, hedgehog_rows: int) -> Path:
    lines = [
        "# BinoForce 2025 Reproduction Notes",
        "",
        "This workflow places labels in 3D first and only then projects them into the requested camera views for evaluation and preview.",
        "",
        "## Required Local Data Handling",
        "",
        "- 3D model geometry is loaded from `../data/Layout/*/*/layout1/Obj-O`; no new model/proxy assets are generated.",
        "- View order is `main`, `up`, `down`, `left`, `right`, matching `regenerate_layout_assets(1).py`.",
        "- Camera parameters follow the existing Hedgehog reproduction so all methods are projected and evaluated in the same view state.",
        "- Hedgehog 1D/3D and plane rows are imported from `../Hedgehog/results/hedgehog_results.csv`; this script does not regenerate them.",
        "- Final metrics use the Hedgehog schema plus DBV: PCK_005, PCK_010, OLR, LCD, DBV, avg_leader_length, overlap_pairs, occluded_points, intersections, quality_score.",
        "",
        "## Methods Computed Here",
        "",
        "- Baseline: labels are placed at a fixed radial distance from the scene center through each anchor.",
        "- MonocularForce: same dynamic force update using the center view only for overlap optimization.",
        "- BinoForce: same dynamic force update using left/right-eye max overlap for binocular optimization.",
        "",
        "## Scale",
        "",
        f"- Samples: {samples}",
        f"- Computed Baseline/MonocularForce/BinoForce rows: {rows - hedgehog_rows}",
        f"- Imported Hedgehog/manual/plane rows: {hedgehog_rows}",
        f"- Total result rows: {rows}",
        f"- BinoForce dynamic warmup frames: {warmup_frames}",
        f"- BinoForce evaluation frames averaged per time step: {eval_frames}",
    ]
    path = out_dir / "reproduction_report.md"
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    (root / "paper_notes.md").write_text(text, encoding="utf-8")
    return path

def validation_subset(scenes: list[Scene]) -> list[Scene]:
    selected: list[Scene] = []
    seen: set[str] = set()
    for scene in sorted(scenes, key=lambda item: (item.category, item.sample)):
        if scene.category in seen:
            continue
        selected.append(scene)
        seen.add(scene.category)
    return selected


def mean_bino_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    subset = [row for row in rows if row["method"] == "BinoForce"]
    return {field: float(np.mean([float(row[field]) for row in subset])) if subset else 0.0 for field in METRIC_FIELDS}


def relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-6)


def run_validation_sweep(
    scenes: list[Scene],
    candidates: tuple[int, ...],
    eval_frames: int,
    out_dir: Path,
    fps: int = DEFAULT_FPS,
) -> tuple[int, Path]:
    val_scenes = validation_subset(scenes)
    records: list[dict[str, object]] = []
    metrics_by_candidate: dict[int, dict[str, float]] = {}
    candidates = tuple(sorted(set(int(value) for value in candidates if int(value) > 0))) or DEFAULT_SWEEP_CANDIDATES

    for candidate in candidates:
        candidate_rows: list[dict[str, str]] = []
        for scene in val_scenes:
            scene_rows, _ = run_scene_experiment(scene, candidate, eval_frames, fps, include_monocular=False)
            candidate_rows.extend(scene_rows)
        metrics = mean_bino_metrics(candidate_rows)
        metrics_by_candidate[candidate] = metrics
        records.append(
            {
                "warmup_frames": candidate,
                "validation_samples": [f"{scene.category}/{scene.sample}" for scene in val_scenes],
                "metrics": metrics,
            }
        )

    selected = DEFAULT_WARMUP_FRAMES if DEFAULT_WARMUP_FRAMES in candidates else candidates[min(1, len(candidates) - 1)]
    stable_fields = ("OLR", "LCD", "avg_leader_length")
    for current, nxt in zip(candidates, candidates[1:]):
        current_metrics = metrics_by_candidate[current]
        next_metrics = metrics_by_candidate[nxt]
        max_delta = max(relative_change(current_metrics[field], next_metrics[field]) for field in stable_fields)
        for record in records:
            if record["warmup_frames"] == current:
                record["max_relative_change_to_next"] = max_delta
                record["next_warmup_frames"] = nxt
                break
        if max_delta <= STABILITY_TOLERANCE:
            selected = nxt if current <= 300 <= nxt else current
            break
    else:
        selected = candidates[-1]

    payload = {
        "selection_rule": "Pick the first stable neighboring pair by OLR/LCD/avg_leader_length, preferring 500 when 300 and 500 are already stable; this is not an all-test best-metric search.",
        "stability_tolerance": STABILITY_TOLERANCE,
        "eval_frames": eval_frames,
        "fps": fps,
        "selected_warmup_frames": selected,
        "records": records,
    }
    path = out_dir / "validation_sweep.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected, path

def load_existing_hedgehog_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    keep_methods = {"manual", "hedgehog_1d", "hedgehog_3d", "plane"}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") not in keep_methods:
                continue
            rows.append({field: row.get(field, "") for field in RESULT_FIELDNAMES})
    return rows

def clean_output(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root.parent / "data")
    parser.add_argument("--out", type=Path, default=root / "results")
    parser.add_argument("--warmup-frames", type=int, default=0, help="Dynamic warmup frames. 0 means choose by validation sweep.")
    parser.add_argument("--eval-frames", type=int, default=DEFAULT_EVAL_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--sweep-candidates", default=",".join(str(v) for v in DEFAULT_SWEEP_CANDIDATES))
    parser.add_argument("--no-sweep", action="store_true", help="Skip validation sweep and use --warmup-frames or the default 500 frames.")
    parser.add_argument("--hedgehog-results", type=Path, default=root.parent / "Hedgehog" / "results" / "hedgehog_results.csv")
    args = parser.parse_args()
    clean_output(args.out)

    scenes = [load_scene(path) for path in iter_annotations(args.data)]
    scenes = [scene for scene in scenes if len(scene.anchors) > 0]
    candidates = tuple(int(part.strip()) for part in str(args.sweep_candidates).split(",") if part.strip())
    sweep_path: Path | None = None
    if args.no_sweep:
        warmup_frames = args.warmup_frames or DEFAULT_WARMUP_FRAMES
    else:
        warmup_frames, sweep_path = run_validation_sweep(scenes, candidates, args.eval_frames, args.out, args.fps)
        if args.warmup_frames > 0:
            warmup_frames = args.warmup_frames

    computed_rows: list[dict[str, str]] = []
    layout_cache: dict[tuple[str, str, str, str], np.ndarray] = {}

    max_workers = min(max(os.cpu_count() or 1, 1), 6, len(scenes) or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for scene_rows, scene_layouts in executor.map(
            run_scene_experiment,
            scenes,
            [warmup_frames] * len(scenes),
            [args.eval_frames] * len(scenes),
            [args.fps] * len(scenes),
            [True] * len(scenes),
        ):
            computed_rows.extend(scene_rows)
            for category, sample, view, method, labels in scene_layouts:
                layout_cache[(category, sample, view, method)] = labels

    hedgehog_rows = load_existing_hedgehog_rows(args.hedgehog_results)
    rows = hedgehog_rows + computed_rows
    write_csv(args.out / "binoforce2025_results.csv", rows, RESULT_FIELDNAMES)
    summary = aggregate(rows, ("method",))
    by_category_method = aggregate(rows, ("category", "method"))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "category_summary.json").write_text(json.dumps(by_category_method, ensure_ascii=False, indent=2), encoding="utf-8")
    method_md, category_md = write_metric_tables(args.out, rows)
    dbv_csv, dbv_md = write_dbv_comparison(args.out, rows)
    layout_path = write_layout_jsonl(args.out, layout_cache, scenes)

    scene_map = {(s.category, s.sample): s for s in scenes}
    best_key = min(
        [(cat, sample, view, method) for (cat, sample, view, method) in layout_cache if view == "main" and method == "BinoForce"],
        key=lambda key: metric_values(scene_map[(key[0], key[1])], layout_cache[key], "main")["OLR"],
    )
    best_scene = scene_map[(best_key[0], best_key[1])]
    layout_comparison = write_layout_comparison(args.out, best_scene, layout_cache)
    comparison = write_camera_comparison(args.out, best_scene, layout_cache)
    report = write_report(args.out, rows, method_md, category_md, layout_comparison, comparison, len(scenes), warmup_frames, args.eval_frames, len(hedgehog_rows))
    notes = write_notes(root, args.out, len(scenes), len(rows), warmup_frames, args.eval_frames, len(hedgehog_rows))

    print(json.dumps({
        "samples": len(scenes),
        "computed_rows": len(computed_rows),
        "imported_hedgehog_rows": len(hedgehog_rows),
        "rows": len(rows),
        "selected_warmup_frames": warmup_frames,
        "eval_frames": args.eval_frames,
        "fps": args.fps,
        "validation_sweep": str(sweep_path) if sweep_path else None,
        "layout_path": str(layout_path),
        "method_summary": str(method_md),
        "category_summary": str(category_md),
        "dbv_comparison": str(dbv_md),
        "layout_comparison": str(layout_comparison),
        "comparison": str(comparison),
        "report": str(report),
        "notes": str(notes),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



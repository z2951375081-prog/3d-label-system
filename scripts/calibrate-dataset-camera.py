"""Estimate dataset multiview cameras from annotation leader lines in PNG files.

The dataset does not store camera matrices.  This calibration uses only train
samples: 3D annotation segments (anchor -> final label center) are projected
and aligned to thin red leader-line pixels.  Validation/test images are used
only to report held-out alignment error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 750, 500
VIEWS = ("main", "right", "left", "up", "down")
DISTANCE = 10.0


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-12)


def frame_from_direction(outward: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outward = normalize(outward)
    forward = -outward
    right = normalize(np.cross(forward, normalize(up_hint)))
    up = normalize(np.cross(right, forward))
    return outward, right, up


def protocol_frames() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    main, right, up = frame_from_direction(np.asarray([1.0, 1.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))

    def rotate(axis: np.ndarray, degrees: float = 45.0) -> np.ndarray:
        radians = math.radians(degrees)
        return normalize(main * math.cos(radians) + normalize(axis) * math.sin(radians))

    directions = {"main": main, "right": rotate(right), "left": rotate(-right), "up": rotate(up), "down": rotate(-up)}
    return {name: (main, right, up) if name == "main" else frame_from_direction(direction, up) for name, direction in directions.items()}


def protocol_directions() -> dict[str, np.ndarray]:
    return {name: frame[0] for name, frame in protocol_frames().items()}


def spherical(direction: np.ndarray) -> tuple[float, float]:
    direction = normalize(direction)
    return math.atan2(float(direction[2]), float(direction[0])), math.asin(float(direction[1]))


def camera_basis(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth, elevation, roll = parameters[:3]
    outward = np.asarray([
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.sin(azimuth),
    ])
    forward = -outward
    up_hint = np.asarray([0.0, 1.0, 0.0])
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    rolled_right = right * math.cos(roll) + up * math.sin(roll)
    rolled_up = -right * math.sin(roll) + up * math.cos(roll)
    return outward, normalize(rolled_right), normalize(rolled_up)


def protocol_parameters(view: str) -> np.ndarray:
    outward, desired_right, _ = protocol_frames()[view]
    azimuth, elevation = spherical(outward)
    base = np.asarray([azimuth, elevation, 0.0, 1.0, 1.0, 0.0, 0.0])
    _, base_right, base_up = camera_basis(base)
    roll = math.atan2(float(desired_right @ base_up), float(desired_right @ base_right))
    return np.asarray([azimuth, elevation, roll, 50.0 * (WIDTH - 1) / 36.0, 50.0 * (HEIGHT - 1) / 24.0, (WIDTH - 1) / 2, (HEIGHT - 1) / 2])


def project(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    outward, right, up = camera_basis(parameters)
    fx, fy, cx, cy = parameters[3:]
    position = outward * DISTANCE
    delta = points - position
    depth = np.maximum(1e-5, -(delta @ outward))
    return np.column_stack((cx + fx * (delta @ right) / depth, cy - fy * (delta @ up) / depth))


def sample_segments(annotation: dict, samples_per_segment: int = 36) -> np.ndarray:
    points: list[np.ndarray] = []
    steps = np.linspace(0.0, 1.0, samples_per_segment)
    for group in annotation.get("groups", []):
        anchor = np.asarray(group.get("anchor", {}).get("point"), dtype=float)
        center = np.asarray(group.get("label", {}).get("center"), dtype=float)
        if anchor.shape != (3,) or center.shape != (3,):
            continue
        points.append(anchor[None, :] * (1.0 - steps[:, None]) + center[None, :] * steps[:, None])
    return np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=float)


def thin_red_pixels(image_file: Path, maximum: int = 900) -> np.ndarray:
    rgb = np.asarray(Image.open(image_file).convert("RGB"))
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    mask = (red > 165) & (red - green > 55) & (red - blue > 55) & (green < 175)
    local = ndimage.convolve(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), mode="constant")
    thin = mask & (local <= 13)
    y, x = np.nonzero(thin)
    pixels = np.column_stack((x, y)).astype(float)
    if len(pixels) > maximum:
        indices = np.linspace(0, len(pixels) - 1, maximum, dtype=int)
        pixels = pixels[indices]
    return pixels


def load_cases(manifest: dict, split: str, view: str, limit: int) -> list[dict]:
    cases = []
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        annotation_file = ROOT / sample["target"]["annotation_json"]
        annotation = json.loads(annotation_file.read_text(encoding="utf-8"))
        image_file = annotation_file.parents[1] / "Mutiviews" / f"{sample['sample_id']}-{view}.png"
        if not image_file.exists():
            continue
        segments = sample_segments(annotation)
        observed = thin_red_pixels(image_file)
        if len(segments) and len(observed) >= 20:
            cases.append({"sample": f"{sample['category']}/{sample['sample_id']}", "segments": segments, "observed": observed})
        if limit and len(cases) >= limit:
            break
    return cases


def case_error(case: dict, parameters: np.ndarray) -> float:
    projected = project(case["segments"], parameters)
    inside = projected[(projected[:, 0] >= -40) & (projected[:, 0] <= WIDTH + 40) & (projected[:, 1] >= -40) & (projected[:, 1] <= HEIGHT + 40)]
    if len(inside) < 10:
        return 100.0
    distances = cKDTree(inside).query(case["observed"], k=1)[0]
    distances = np.sort(np.minimum(distances, 50.0))
    kept = distances[: max(10, int(len(distances) * 0.82))]
    return float(np.sqrt(np.mean(kept * kept)))


def mean_error(cases: list[dict], parameters: np.ndarray) -> float:
    return float(np.mean([case_error(case, parameters) for case in cases])) if cases else float("nan")


def calibrate(view: str, train_cases: list[dict], maxiter: int) -> tuple[np.ndarray, float, float]:
    initial = protocol_parameters(view)
    azimuth, elevation = initial[:2]
    initial_error = mean_error(train_cases, initial)
    bounds = [
        (azimuth - math.radians(35), azimuth + math.radians(35)),
        (max(-1.45, elevation - math.radians(30)), min(1.45, elevation + math.radians(30))),
        (-math.radians(35), math.radians(35)),
        (450.0, 1800.0),
        (350.0, 1600.0),
        (300.0, 450.0),
        (180.0, 320.0),
    ]
    objective = lambda values: mean_error(train_cases, np.asarray(values))
    global_result = differential_evolution(objective, bounds, seed=1709, maxiter=maxiter, popsize=6, polish=False, updating="immediate")
    local_global = minimize(objective, global_result.x, method="Powell", bounds=bounds, options={"maxiter": 240, "xtol": 1e-4, "ftol": 1e-4})
    local_initial = minimize(objective, initial, method="Powell", bounds=bounds, options={"maxiter": 240, "xtol": 1e-4, "ftol": 1e-4})
    candidates = [(initial_error, initial), (float(global_result.fun), global_result.x), (float(local_global.fun), local_global.x), (float(local_initial.fun), local_initial.x)]
    _, best_values = min(candidates, key=lambda item: item[0])
    best = np.asarray(best_values)
    return best, initial_error, mean_error(train_cases, best)


def serialize_camera(view: str, parameters: np.ndarray) -> dict:
    outward, right, up = camera_basis(parameters)
    fx, fy, cx, cy = parameters[3:]
    return {
        "view": view,
        "outward_direction": [round(float(value), 9) for value in outward],
        "right_direction": [round(float(value), 9) for value in right],
        "up_direction": [round(float(value), 9) for value in up],
        "roll_degrees": round(math.degrees(float(parameters[2])), 6),
        "camera_distance_fixed": DISTANCE,
        "fx_pixels": round(float(fx), 6),
        "fy_pixels": round(float(fy), 6),
        "equivalent_focal_length_x_mm_for_36mm_sensor": round(float(fx) * 36.0 / (WIDTH - 1), 6),
        "equivalent_focal_length_y_mm_for_24mm_sensor": round(float(fy) * 24.0 / (HEIGHT - 1), 6),
        "principal_point_pixels": [round(float(cx), 6), round(float(cy), 6)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--output", default=str(ROOT / "experiments" / "dataset_camera_calibration.json"))
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments" / "dataset_manifest.json").read_text(encoding="utf-8"))
    results = {}
    for view in [item.strip() for item in args.views.split(",") if item.strip()]:
        train = load_cases(manifest, "train", view, args.train_limit)
        validation = load_cases(manifest, "val", view, 0)
        test = load_cases(manifest, "test", view, 0)
        parameters, initial_train, fitted_train = calibrate(view, train, args.maxiter)
        protocol_direction = protocol_directions()[view]
        protocol = protocol_parameters(view)
        fitted_direction = camera_basis(parameters)[0]
        angle = math.degrees(math.acos(float(np.clip(protocol_direction @ fitted_direction, -1.0, 1.0))))
        results[view] = {
            "camera": serialize_camera(view, parameters),
            "direction_change_from_reproduction_degrees": round(angle, 6),
            "cases": {"train": len(train), "val": len(validation), "test": len(test)},
            "red_line_alignment_rmse_pixels": {
                "reproduction_protocol": {"train": round(initial_train, 6), "val": round(mean_error(validation, protocol), 6), "test": round(mean_error(test, protocol), 6)},
                "calibrated": {"train": round(fitted_train, 6), "val": round(mean_error(validation, parameters), 6), "test": round(mean_error(test, parameters), 6)},
            },
        }
        print(view, json.dumps(results[view], ensure_ascii=False))
    report = {
        "version": "dataset_camera_red_leader_line_calibration_v1",
        "status": "image_derived_estimate_not_original_camera_metadata",
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "method": "Fit shared per-view perspective camera to thin red leader-line pixels using train annotation anchor-to-final-center 3D segments; report held-out val/test alignment.",
        "limitations": [
            "Camera distance is fixed to 10 because focal length and distance are weakly identifiable from these normalized scenes.",
            "Leader lines are partly hidden by labels and objects; a trimmed one-way Chamfer RMSE is used.",
            "This is an image-derived calibration estimate, not recovered camera metadata.",
        ],
        "views": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

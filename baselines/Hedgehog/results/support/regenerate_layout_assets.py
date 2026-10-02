from __future__ import annotations

import argparse
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from check_layout_dataset import (
    DATASET_ROOT,
    LAYOUT_LEVELS,
    discover_dataset_layouts,
    discover_external_layout1,
    expected_group_materials,
    label_leader_missing_groups,
    load_json,
    parse_obj,
    path_identity,
    sample_key,
)


DEFAULT_APP_ROOT = Path(__file__).resolve().parent / "vendored_manual_adjust"
VIEW_ORDER = ("main", "up", "down", "left", "right")


def import_app(app_root: Path) -> Any:
    app_root = app_root.expanduser().resolve()
    sys.path.insert(0, str(app_root))
    module_path = app_root / "web_projection_editor.py"
    spec = importlib.util.spec_from_file_location("manual_adjust_app", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import app from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge_layout_maps(*maps):
    merged = {}
    for mapping in maps:
        for key, instances in mapping.items():
            merged.setdefault(key, []).extend(instances)
    for instances in merged.values():
        instances.sort(key=lambda item: (item.level, item.source, str(item.annotation_path).lower()))
    return merged


def selected_instances(args: argparse.Namespace):
    dataset_root = args.dataset_root.expanduser().resolve()
    maps = [discover_dataset_layouts(dataset_root)]
    for root in args.layout1_root:
        maps.append(discover_external_layout1(root.expanduser().resolve()))
    all_layouts = merge_layout_maps(*maps)
    categories = set(args.category or [])
    samples = set(args.sample or [])
    levels = set(args.levels)
    result = []
    for key in sorted(all_layouts):
        category, sample_id = key
        if categories and category not in categories:
            continue
        if samples and sample_id not in samples and sample_key(category, sample_id) not in samples:
            continue
        for inst in all_layouts[key]:
            if inst.level in levels:
                result.append(inst)
    if args.limit > 0:
        result = result[: args.limit]
    return result


def validate_layout_identity(annotation_path: Path, payload: dict[str, Any]) -> tuple[str, str, str]:
    identity = path_identity(annotation_path)
    if identity is None:
        raise ValueError(f"Cannot infer Layout identity from {annotation_path}")
    path_category, path_sample, path_level = identity
    json_category = str(payload.get("category") or payload.get("model_cat") or "")
    json_sample = str(payload.get("sample_id") or annotation_path.stem)
    if path_category != json_category or path_sample != json_sample:
        raise ValueError(
            f"Layout identity mismatch: path={path_category}/{path_sample}/{path_level}, "
            f"json={json_category}/{json_sample}"
        )
    return path_category, path_sample, path_level


def validate_generated_obj_o_groups(paths: Path | dict[str, Path | str], annotation: dict[str, Any]) -> None:
    expected_materials = expected_group_materials(annotation)
    path_items = paths.items() if isinstance(paths, dict) else [("main", paths)]
    failures: list[str] = []
    for view_name, path in path_items:
        parsed = parse_obj(Path(path))
        missing_label_groups = label_leader_missing_groups(parsed)
        if missing_label_groups:
            failures.append(f"{view_name}: label/leader objects missing matching groups {missing_label_groups[:8]}")
        for group_name, expected_material in expected_materials.items():
            actual = parsed["group_materials"].get(group_name, set())
            if group_name not in parsed["groups"]:
                failures.append(f"{view_name}: missing target group {group_name}")
                continue
            extra = sorted(actual - {expected_material})
            if extra:
                failures.append(f"{view_name}: {group_name} has unexpected materials {extra}")
            if expected_material not in actual:
                failures.append(f"{view_name}: {group_name} missing material {expected_material}")
    if failures:
        raise ValueError("Generated Obj-O group/material validation failed: " + "; ".join(failures[:8]))


def regenerate_instance(app: Any, dataset_root: Path, inst, write: bool, overwrite_json: bool) -> tuple[str, str]:
    annotation = load_json(inst.annotation_path)
    category, sample_id, level = validate_layout_identity(inst.annotation_path, annotation)
    obj_p_path = dataset_root / "Obj-P" / category / sample_id / f"{sample_id}-P.obj"
    if not obj_p_path.is_file():
        raise FileNotFoundError(f"Obj-P missing: {obj_p_path}")
    text_objs_dir = app.find_text_objs_dir(inst.annotation_path, annotation, dataset_root, None, obj_p_path)
    if text_objs_dir is None or not text_objs_dir.is_dir():
        checked_dirs = app.text_objs_dir_candidates(inst.annotation_path, annotation, dataset_root, None, obj_p_path)
        checked_text = "; ".join(str(path) for path in checked_dirs[:8])
        raise FileNotFoundError(f"Text_objs/TEXT_OBJS missing. Checked: {checked_text}")

    annotation = deepcopy(annotation)
    app.normalize_group_ids(annotation)
    app.apply_internal_camera(annotation)
    settings = app.settings_from_annotation(annotation)
    obj_o_dir = inst.layout_dir / "Obj-O"
    mutiviews_dir = inst.layout_dir / "Mutiviews"
    if not write:
        return "dry-run", f"{level} {category}/{sample_id}: would regenerate Obj-O and Mutiviews"

    obj_o_paths = app.export_obj_o_for_orientation(obj_p_path, annotation, text_objs_dir, obj_o_dir, sample_id, settings)
    validate_generated_obj_o_groups(obj_o_paths, annotation)
    app.render_all_obj_o_views(
        obj_o_paths,
        annotation=annotation,
        projection_root=mutiviews_dir,
        category="",
        sample_name=sample_id,
        settings=settings,
    )
    if overwrite_json:
        stored = app.annotation_for_storage(annotation, layout_level=level)
        # Keep the source version unless the caller intentionally marks reviewed elsewhere.
        stored["version"] = load_json(inst.annotation_path).get("version", stored.get("version"))
        inst.annotation_path.write_text(app.json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8")
    return "written", f"{level} {category}/{sample_id}: regenerated {obj_o_dir} and {mutiviews_dir}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Obj-O and projection images for checked 3DLPD layouts.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--layout1-root", type=Path, action="append", default=[], help="Pending/export root containing Layout/*/*/layout1. Can be repeated.")
    parser.add_argument("--app-root", type=Path, default=DEFAULT_APP_ROOT, help="Vendored manual_adjust app root containing web_projection_editor.py; defaults to check_code/vendored_manual_adjust.")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--sample", action="append", default=[], help="Sample id or Category/Sample key.")
    parser.add_argument("--levels", nargs="+", default=list(LAYOUT_LEVELS), choices=list(LAYOUT_LEVELS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="Actually overwrite Obj-O and Mutiviews. Without this, the script is dry-run.")
    parser.add_argument("--overwrite-json", action="store_true", help="Rewrite annotation JSON with normalized internal fields stripped by app storage logic.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    instances = selected_instances(args)
    if not instances:
        print("No layouts selected.")
        return 0
    app = import_app(args.app_root)
    errors = 0
    print(
        "Leader-line rule: generated Obj-O leader tubes are clipped at the label box boundary "
        "by manual_adjust_app.clipped_label_boundary_point, so they do not enter the label interior."
    )
    for inst in instances:
        try:
            status, message = regenerate_instance(app, dataset_root, inst, args.write, args.overwrite_json)
            print(f"[{status}] {message}")
        except Exception as exc:
            errors += 1
            print(f"[error] {inst.level} {inst.category}/{inst.sample_id}: {exc}")
    if errors:
        print(f"Failed layouts: {errors}")
        return 2
    if not args.write:
        print("Dry-run only. Re-run with --write to regenerate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def stable_key(seed: int, category: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category}/{sample_id}".encode("utf-8")).hexdigest()


def largest_remainder(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {key: target * value / total for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = target - sum(result.values())
    order = sorted(counts, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


def discover_samples(names: list[str]):
    samples = set()
    for name in names:
        match = re.match(r"3DLPD-main/Layout/([^/]+)/([^/]+)/layout1/", name)
        if match:
            samples.add((match.group(1), match.group(2)))
    return sorted(samples, key=lambda item: (natural_key(item[0]), natural_key(item[1])))


def make_joined_archive(parts_dir: Path) -> Path:
    parts = sorted(parts_dir.glob("3DLPD-main.zip.???"), key=lambda p: p.name)
    expected = [f"3DLPD-main.zip.{index:03d}" for index in range(1, len(parts) + 1)]
    actual = [part.name for part in parts]
    if actual != expected:
        raise RuntimeError(f"archive parts are incomplete or out of order: expected {expected}, found {actual}")
    temp = tempfile.NamedTemporaryFile(prefix="3dlpd-", suffix=".zip", delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, temp, length=8 * 1024 * 1024)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    joined = args.archive_dir / "3DLPD-main-joined.zip"
    temporary_joined = None
    if not joined.exists():
        temporary_joined = make_joined_archive(args.archive_dir)
        joined = temporary_joined

    try:
        with zipfile.ZipFile(joined) as archive:
            names = archive.namelist()
            samples = discover_samples(names)
            if len(samples) < args.train + args.test:
                raise RuntimeError(f"archive contains {len(samples)} samples, but {args.train + args.test} are required")
            names_set = set(names)

            available_by_category = defaultdict(list)
            for category, sample_id in samples:
                available_by_category[category].append(sample_id)
            category_counts = {category: len(ids) for category, ids in available_by_category.items()}
            train_quota = largest_remainder(category_counts, args.train)
            test_quota = largest_remainder(category_counts, args.test)

            old_manifest = args.manifest.parent / "dataset_manifest.json"
            old_splits = {}
            if old_manifest.exists():
                old = json.loads(old_manifest.read_text(encoding="utf-8"))
                old_splits = {(item["category"], str(item["sample_id"])): item["split"] for item in old.get("samples", [])}

            selected = []
            for category in sorted(available_by_category, key=natural_key):
                ids = available_by_category[category]
                fixed_train = [sid for sid in ids if old_splits.get((category, sid)) in {"train", "val"}]
                fixed_test = [sid for sid in ids if old_splits.get((category, sid)) == "test"]
                if len(fixed_train) > train_quota[category] or len(fixed_test) > test_quota[category]:
                    raise RuntimeError(f"existing samples exceed quota in {category}")
                ordered = sorted(ids, key=lambda sid: stable_key(args.seed, category, sid))
                chosen_train = list(dict.fromkeys(fixed_train))
                chosen_test = list(dict.fromkeys(fixed_test))
                used = set(chosen_train) | set(chosen_test)
                for sid in ordered:
                    if sid in used:
                        continue
                    if len(chosen_train) < train_quota[category]:
                        chosen_train.append(sid)
                        used.add(sid)
                    elif len(chosen_test) < test_quota[category]:
                        chosen_test.append(sid)
                        used.add(sid)
                    if len(chosen_train) == train_quota[category] and len(chosen_test) == test_quota[category]:
                        break
                if len(chosen_train) != train_quota[category] or len(chosen_test) != test_quota[category]:
                    raise RuntimeError(f"could not fill quotas in {category}")
                selected.extend((category, sid, "train") for sid in chosen_train)
                selected.extend((category, sid, "test") for sid in chosen_test)

            if len(selected) != args.train + args.test:
                raise RuntimeError(f"selected {len(selected)} samples instead of {args.train + args.test}")
            if len({(category, sid) for category, sid, _ in selected}) != len(selected):
                raise RuntimeError("selection contains duplicate samples")

            if args.output_root.exists():
                raise RuntimeError(f"output already exists; refusing to overwrite: {args.output_root}")
            args.output_root.mkdir(parents=True)
            manifest_samples = []
            csv_rows = []
            for category, sample_id, split in sorted(selected, key=lambda row: (row[2], natural_key(row[0]), natural_key(row[1]))):
                prefix = f"3DLPD-main/Layout/{category}/{sample_id}/layout1/"
                member_names = [name for name in names if name.startswith(prefix)]
                if not member_names:
                    raise RuntimeError(f"missing archive members for {category}/{sample_id}")
                for name in member_names:
                    relative = Path(name[len("3DLPD-main/"):])
                    destination = args.output_root / relative
                    if name.endswith("/"):
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name) as source, destination.open("wb") as target:
                            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                obj = args.output_root / "Layout" / category / sample_id / "layout1" / "Obj-O" / f"{sample_id}-main-O.obj"
                annotation = args.output_root / "Layout" / category / sample_id / "layout1" / "Annotation" / f"{sample_id}.json"
                views_dir = args.output_root / "Layout" / category / sample_id / "layout1" / "Mutiviews"
                views = sorted((path for path in views_dir.glob("*.png") if not path.name.endswith("-combined.png")), key=lambda path: natural_key(path.name))
                if not obj.exists() or not annotation.exists() or len(views) != 5:
                    raise RuntimeError(f"incomplete extracted sample: {category}/{sample_id}")
                rel = lambda path: path.relative_to(args.manifest.parent.parent).as_posix()
                manifest_samples.append({
                    "category": category,
                    "sample_id": sample_id,
                    "split": split,
                    "input": {"source_obj": rel(obj), "clean_obj": None, "cleaning": {"remove_groups": ["label_*", "leader_*"], "remove_materials": ["label_*", "leader_*"], "normalize_materials": ["anchor_region_*"], "emitted_material": "object_default", "mode": "rebuild_faces_and_reindex"}},
                    "target": {"annotation_json": rel(annotation), "labeled_obj": None, "generated_at_runtime": True},
                    "views": {"expected": 5, "files": [rel(path) for path in views]},
                })
                csv_rows.append({"category": category, "sample_id": sample_id, "split": split, "archive_prefix": prefix})

            counts = {"train": args.train, "val": 0, "test": args.test, "by_category": {}}
            for row in manifest_samples:
                category = row["category"]
                counts["by_category"].setdefault(category, {"train": 0, "val": 0, "test": 0})
                counts["by_category"][category][row["split"]] += 1
            manifest = {
                "version": "1.1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "policy": "3dlpd_archive_stratified_with_existing_55_preserved",
                "seed": args.seed,
                "source_archive": "3DLPD-main.zip.001 through 3DLPD-main.zip.006",
                "note": "Selected layout1 from the 3DLPD archive. The previous 55 samples are preserved: former train/val samples remain train and former test samples remain test.",
                "counts": counts,
                "samples": manifest_samples,
            }
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            args.selection_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.selection_csv.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["category", "sample_id", "split", "archive_prefix"])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(json.dumps({"samples": len(manifest_samples), "counts": counts, "output_root": str(args.output_root), "manifest": str(args.manifest), "selection_csv": str(args.selection_csv)}, ensure_ascii=False, indent=2))
    finally:
        if temporary_joined:
            temporary_joined.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

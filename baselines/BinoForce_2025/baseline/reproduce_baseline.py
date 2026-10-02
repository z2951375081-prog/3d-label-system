from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproduce_binoforce2025 import (  # noqa: E402
    VIEW_ORDER,
    RESULT_FIELDNAMES,
    aggregate,
    baseline_layout,
    iter_annotations,
    load_scene,
    metric_values,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "baseline" / "results")
    parser.add_argument("--from-results", type=Path, default=ROOT / "results" / "binoforce2025_results.csv")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if args.from_results.exists():
        with args.from_results.open("r", encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["method"] == "Baseline"]
    else:
        scenes = [load_scene(path) for path in iter_annotations(args.data)]
        for scene in scenes:
            labels = baseline_layout(scene)
            for view in VIEW_ORDER:
                metrics = metric_values(scene, labels, view)
                row = {
                    "category": scene.category,
                    "sample": scene.sample,
                    "view": view,
                    "method": "Baseline",
                    "num_labels": str(len(scene.anchors)),
                }
                row.update({key: f"{value:.8f}" for key, value in metrics.items()})
                rows.append(row)

    fieldnames = RESULT_FIELDNAMES
    with (args.out / "baseline_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows, ("method",))
    category_summary = aggregate(rows, ("category", "method"))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "category_summary.json").write_text(json.dumps(category_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

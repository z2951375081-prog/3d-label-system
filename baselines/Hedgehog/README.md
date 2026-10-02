# Hedgehog Labeling Reproduction

This folder contains a local reproduction of:

Tatzgern et al., **Hedgehog Labeling: View Management Techniques for External Labels in 3D Space**, IEEE VR 2014.

## What is reproduced

- 3D pole constraint: each label uses the radial pole from the object's bounding-sphere center to the anchor point.
- One-degree-of-freedom hedgehog update: labels move only by changing pole length.
- Three-degree-of-freedom hedgehog update: labels move along the pole plus bounded X/Y motion in the annotation's local image plane.
- Plane constraint: labels are assigned to equidistant view-parallel planes and optimized by a deterministic spring/repulsion embedding in those planes.
- Evaluation against the manual label placement stored in `../data/Layout`.

The paper reports mainly qualitative results, so this reproduction uses proxy metrics computed in the same camera state used for preview rendering: PCK against the manual labels in `../data`, OLR, LCD, normalized leader-line length, and a higher-is-better quality score.

## Run

```powershell
& 'C:\Users\chenyv\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\src\reproduce_hedgehog.py --data ..\data --out .\results
```

Generated files:

- `results/hedgehog_results.csv`: per-sample, per-view metrics.
- `results/summary.json`: aggregate metrics by method.
- `results/comparison.md` and `results/reproduction_protocol.md`: method basis, parameter notes, and reproduction summary.
- `results/layouts/<Category>/<Sample>/<View>/<Method>.json`: reproduced layout results with updated label centers and leader-line ends.
- `results/previews/*.png` and `results/layout_preview.html`: four-method comparison previews for representative sample/view pairs. The manual column uses the dataset-provided `Mutiviews` images; generated-method columns use `pyrender` to render the same OBJ body and then overlay labels.
- `results/support/regenerate_layout_assets.py`: a local copy of the provided asset-regeneration script linked from the preview page.

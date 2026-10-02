# BinoForce 2025 Reproduction Notes

This workflow places labels in 3D first and only then projects them into the requested camera views for evaluation and preview.

## Required Local Data Handling

- Clean 3D model geometry is loaded from `../data/Layout/*/*/layout1/Obj-O`; label and leader groups are skipped for generated previews.
- The BinoForce update is dynamic: labels start at anchors, inherit positions frame by frame, and are updated under a deterministic camera trajectory with circular motion plus 5-second stop/linear-motion segments.
- Fixed photo/evaluation views are `main`, `up`, `down`, `left`, `right`, matching `regenerate_layout_assets(1).py` view order.
- Camera parameters follow the existing Hedgehog reproduction so all methods are projected and evaluated in the same view state.
- Hedgehog 1D/3D and plane rows are imported from `../Hedgehog/results/hedgehog_results.csv`; this script does not regenerate them.
- Final metrics use the Hedgehog schema plus DBV: PCK_005, PCK_010, OLR, LCD, DBV, avg_leader_length, overlap_pairs, occluded_points, intersections, quality_score.
- Layout forces use local object proxies around anchors; evaluation and preview use the clean colored OBJ body projection.

## Methods Computed Here

- Baseline: labels are placed at a fixed radial distance from the scene center through each anchor.
- MonocularForce: same dynamic force update using the center view only for overlap optimization.
- BinoForce: same dynamic force update using left/right-eye max overlap for binocular optimization.

## Scale

- Samples: 55
- Computed Baseline/MonocularForce/BinoForce rows: 825
- Imported Hedgehog/manual/plane rows: 1100
- Total result rows: 1925
- BinoForce dynamic warmup frames: 800
- BinoForce evaluation frames averaged per time step: 300
- Dynamic camera FPS: 30
- DBV comparison output: `results/dbv_comparison.md` and `results/dbv_comparison.csv`.
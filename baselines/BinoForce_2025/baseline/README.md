# BinoForce Baseline Reproduction

This folder computes only the Baseline method used for the BinoForce comparison.

Baseline definition from the paper: labels are positioned at a fixed distance along the vector from the scene center to each object, and label orientation faces the user each frame. In this local reproduction, the fixed distance is calibrated from the provided manual annotation leader lengths because the paper does not publish its Unity scene distance value.

The output fields match the Hedgehog reproduction metrics so Baseline can be compared directly with manual, Hedgehog 1D/3D, plane, and BinoForce rows.

## Run

```powershell
& 'C:\Users\chenyv\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\baseline\reproduce_baseline.py --data ..\data --out .\baseline\results
```
"""Command-line entry point and compatibility facade for the BinoForce reproduction.

The implementation is split into small public modules (data_loader, binoforce,
evaluate, visualize, reporting). This facade intentionally re-exports the legacy
symbols so existing scripts, including baseline/reproduce_baseline.py, keep the
same imports and behavior.
"""

from __future__ import annotations

from _pipeline import *  # noqa: F401,F403
from _pipeline import main as _main


if __name__ == "__main__":
    _main()

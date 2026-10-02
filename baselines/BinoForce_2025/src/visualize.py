"""Visualization and report-image generation helpers."""

from __future__ import annotations

from _pipeline import (
    draw_image_overlay_panel,
    draw_manual_comparison_panel,
    draw_manual_image_panel,
    draw_metric_chips,
    draw_obj_layout_panel,
    fit_text_to_rect,
    load_font,
    metric_chip_text,
    panel_geometry,
    parse_clean_obj_mesh,
    render_clean_obj,
    write_camera_comparison,
    write_layout_comparison,
    write_obj_overlay,
)

__all__ = [
    "load_font",
    "fit_text_to_rect",
    "panel_geometry",
    "parse_clean_obj_mesh",
    "render_clean_obj",
    "metric_chip_text",
    "draw_metric_chips",
    "draw_obj_layout_panel",
    "draw_manual_image_panel",
    "draw_manual_comparison_panel",
    "draw_image_overlay_panel",
    "write_obj_overlay",
    "write_layout_comparison",
    "write_camera_comparison",
]

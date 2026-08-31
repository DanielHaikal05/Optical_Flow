"""DSO-style deterministic pixel selection for normal-flow experiments."""

from .selector import (
    DSOSelectorConfig,
    build_dso_confidence,
    draw_selected_points,
    save_confidence_heatmap,
    select_dso_points,
)

__all__ = [
    "DSOSelectorConfig",
    "build_dso_confidence",
    "draw_selected_points",
    "save_confidence_heatmap",
    "select_dso_points",
]

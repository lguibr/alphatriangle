# File: src/visualization/__init__.py
# File: src/visualization/__init__.py
"""
Visualization module for rendering the game state using Pygame.
"""

from .core.visualizer import Visualizer
from .core.game_renderer import GameRenderer
from .core.dashboard_renderer import DashboardRenderer
from .core.layout import (
    calculate_interactive_layout,
    calculate_training_layout,
)
from .core.fonts import load_fonts
from .core import colors  # Keep colors accessible
from .core.coord_mapper import (
    get_grid_coords_from_screen,
    get_preview_index_from_screen,
)

from .drawing.grid import (
    draw_grid_background,
    draw_grid_triangles,
    draw_grid_indices,
)
from .drawing.shapes import draw_shape
from .drawing.previews import (
    render_previews,
    draw_placement_preview,
    draw_floating_preview,
)
from .drawing.hud import render_hud
from .drawing.highlight import draw_debug_highlight

from .ui.progress_bar import ProgressBar

# REMOVED: Import and potentially export utils
# from . import utils as vis_utils

from src.config import VisConfig


__all__ = [
    # Core Renderers & Layout
    "Visualizer",
    "GameRenderer",
    "DashboardRenderer",
    "calculate_interactive_layout",
    "calculate_training_layout",
    "load_fonts",
    "colors",  # Export colors module
    "get_grid_coords_from_screen",
    "get_preview_index_from_screen",
    # Drawing Functions
    "draw_grid_background",
    "draw_grid_triangles",
    "draw_grid_indices",
    "draw_shape",
    "render_previews",
    "draw_placement_preview",
    "draw_floating_preview",
    "render_hud",
    "draw_debug_highlight",
    # UI Components
    "ProgressBar",
    # REMOVED: Utils export
    # "vis_utils",
    # Config
    "VisConfig",
]

"""
Visualization module for rendering the game state using Pygame.
"""

from ..config import VisConfig
from .core import colors
from .core.coord_mapper import (
    get_grid_coords_from_screen,
    get_preview_index_from_screen,
)
from .core.dashboard_renderer import DashboardRenderer
from .core.fonts import load_fonts
from .core.game_renderer import GameRenderer
from .core.layout import (
    calculate_interactive_layout,
    calculate_training_layout,
)
from .core.visualizer import Visualizer
from .drawing.grid import (
    draw_grid_background,
    draw_grid_indices,
    draw_grid_triangles,
)
from .drawing.highlight import draw_debug_highlight
from .drawing.hud import render_hud
from .drawing.previews import (
    draw_floating_preview,
    draw_placement_preview,
    render_previews,
)
from .drawing.shapes import draw_shape
from .ui.progress_bar import ProgressBar

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
    # Config
    "VisConfig",
]

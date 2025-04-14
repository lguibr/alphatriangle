import logging
from typing import TYPE_CHECKING

import pygame

from ...structs import Shape
from ..drawing import grid as grid_drawing
from ..drawing import highlight as highlight_drawing
from ..drawing import hud as hud_drawing
from ..drawing import previews as preview_drawing
from ..drawing.previews import (
    draw_floating_preview,
    draw_placement_preview,
)
from . import colors, layout

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig
    from ...environment.core.game_state import GameState

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Orchestrates rendering of a single game state for interactive modes.
    Receives interaction state (hover, selection) via render parameters.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.preview_rects: dict[int, pygame.Rect] = {}  # Cache preview rects
        self._layout_calculated_for_size: tuple[int, int] = (0, 0)
        self.ensure_layout()  # Initial layout calculation

    def ensure_layout(self) -> dict[str, pygame.Rect]:
        """Returns cached layout or calculates it if needed."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Use the interactive layout calculation
            self.layout_rects = layout.calculate_interactive_layout(
                current_w, current_h, self.vis_config
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated interactive layout for size {current_size}: {self.layout_rects}"
            )
            # Clear preview rect cache when layout changes
            self.preview_rects = {}

        return self.layout_rects if self.layout_rects is not None else {}

    def render(
        self,
        game_state: "GameState",
        mode: str,
        # Interaction state passed in:
        selected_shape_idx: int = -1,
        hover_shape: Shape | None = None,
        hover_grid_coord: tuple[int, int] | None = None,
        hover_is_valid: bool = False,
        hover_screen_pos: tuple[int, int] | None = None,
        debug_highlight_coord: tuple[int, int] | None = None,
    ):
        """
        Renders the entire game visualization for interactive modes.
        Uses interaction state passed as parameters for visual feedback.
        """
        self.screen.fill(colors.GRID_BG_DEFAULT)  # Clear screen
        layout_rects = self.ensure_layout()
        grid_rect = layout_rects.get("grid")
        preview_rect = layout_rects.get("preview")

        # Render Grid Area
        if grid_rect and grid_rect.width > 0 and grid_rect.height > 0:
            try:
                grid_surf = self.screen.subsurface(grid_rect)
                self._render_grid_area(
                    grid_surf,
                    game_state,
                    mode,
                    grid_rect,
                    hover_shape,
                    hover_grid_coord,
                    hover_is_valid,
                    hover_screen_pos,
                    debug_highlight_coord,
                )
            except ValueError as e:
                logger.error(f"Error creating grid subsurface ({grid_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, grid_rect, 1)

        # Render Preview Area
        if preview_rect and preview_rect.width > 0 and preview_rect.height > 0:
            try:
                preview_surf = self.screen.subsurface(preview_rect)
                # Pass selected_shape_idx for highlighting
                self._render_preview_area(
                    preview_surf, game_state, mode, preview_rect, selected_shape_idx
                )
            except ValueError as e:
                logger.error(f"Error creating preview subsurface ({preview_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, preview_rect, 1)

        # Render HUD
        # --- CORRECTED CALL ---
        hud_drawing.render_hud(
            surface=self.screen,
            mode=mode,
            fonts=self.fonts,
            display_stats=None,  # Pass None for display_stats in interactive modes
        )
        # --- END CORRECTION ---

    def _render_grid_area(
        self,
        grid_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        grid_rect: pygame.Rect,  # Pass grid_rect for hover calculations
        hover_shape: Shape | None,
        hover_grid_coord: tuple[int, int] | None,
        hover_is_valid: bool,
        hover_screen_pos: tuple[int, int] | None,
        debug_highlight_coord: tuple[int, int] | None,
    ):
        """Renders the main game grid and overlays onto the provided grid_surf."""
        # Background
        bg_color = (
            colors.GRID_BG_GAME_OVER if game_state.is_over() else colors.GRID_BG_DEFAULT
        )
        grid_drawing.draw_grid_background(grid_surf, bg_color)

        # Grid Triangles
        grid_drawing.draw_grid_triangles(
            grid_surf, game_state.grid_data, self.env_config
        )

        # Debug Indices
        if mode == "debug":
            grid_drawing.draw_grid_indices(
                grid_surf, game_state.grid_data, self.env_config, self.fonts
            )

        # Play Mode Hover Previews
        if mode == "play" and hover_shape:
            if hover_grid_coord:  # Snapped preview
                draw_placement_preview(
                    grid_surf,
                    hover_shape,
                    hover_grid_coord[0],
                    hover_grid_coord[1],
                    is_valid=hover_is_valid,  # Use validity passed in
                    config=self.env_config,
                )
            elif hover_screen_pos:  # Floating preview (relative to grid_surf)
                # Adjust screen pos to be relative to grid_surf
                local_hover_pos = (
                    hover_screen_pos[0] - grid_rect.left,
                    hover_screen_pos[1] - grid_rect.top,
                )
                if grid_surf.get_rect().collidepoint(local_hover_pos):
                    draw_floating_preview(
                        grid_surf,
                        hover_shape,
                        local_hover_pos,
                        self.env_config,
                    )

        # Debug Mode Highlight
        if mode == "debug" and debug_highlight_coord:
            r, c = debug_highlight_coord
            highlight_drawing.draw_debug_highlight(grid_surf, r, c, self.env_config)

        # --- ADDED: Display Score in Grid Area for Interactive Modes ---
        score_font = self.fonts.get("score")
        if score_font:
            score_text = f"Score: {game_state.game_score:.0f}"
            score_surf = score_font.render(score_text, True, colors.YELLOW)
            # Position score at top-left of grid area
            score_rect = score_surf.get_rect(topleft=(5, 5))
            grid_surf.blit(score_surf, score_rect)
        # --- END ADDED ---

    def _render_preview_area(
        self,
        preview_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        preview_rect: pygame.Rect,
        selected_shape_idx: int,  # Pass selected index
    ):
        """Renders the shape preview slots onto preview_surf and caches rects."""
        # Pass selected_shape_idx to render_previews for highlighting
        current_preview_rects = preview_drawing.render_previews(
            preview_surf,
            game_state,
            preview_rect.topleft,  # Pass absolute top-left
            mode,
            self.env_config,
            self.vis_config,
            selected_shape_idx=selected_shape_idx,  # Pass selection state
        )
        # Update cache only if it changed (or first time)
        if not self.preview_rects or self.preview_rects != current_preview_rects:
            self.preview_rects = current_preview_rects

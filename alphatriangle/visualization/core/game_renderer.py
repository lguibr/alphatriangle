import logging
from typing import TYPE_CHECKING, Any

import pygame

from ...environment import GameState
from ..drawing import grid as grid_drawing
from ..drawing import previews as preview_drawing
from . import colors

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig

logger = logging.getLogger(__name__)


class GameRenderer:
    """
    Renders a single GameState (grid and previews) within a specified area.
    Used by DashboardRenderer for displaying worker states.
    Also displays latest per-step stats for the worker.
    """

    def __init__(
        self,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
    ):
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.preview_width_ratio = 0.2
        self.min_preview_width = 30
        self.max_preview_width = 60
        self.padding = 5

    def render_worker_state(
        self,
        target_surface: pygame.Surface,
        area_rect: pygame.Rect,
        worker_id: int,
        game_state: GameState | None,
        # Add worker_step_stats parameter
        worker_step_stats: dict[str, Any] | None = None,
    ):
        """
        Renders the game state of a single worker into the specified area_rect
        on the target_surface. Includes per-step stats display.
        """
        if not game_state:
            # Optionally draw a placeholder if state is None
            pygame.draw.rect(target_surface, colors.DARK_GRAY, area_rect)
            pygame.draw.rect(target_surface, colors.GRAY, area_rect, 1)
            id_font = self.fonts.get("help")
            if id_font:
                id_surf = id_font.render(
                    f"W{worker_id} (No State)", True, colors.LIGHT_GRAY
                )
                id_rect = id_surf.get_rect(center=area_rect.center)
                target_surface.blit(id_surf, id_rect)
            return

        # Calculate layout within the worker's area_rect
        preview_w = max(
            self.min_preview_width,
            min(area_rect.width * self.preview_width_ratio, self.max_preview_width),
        )
        grid_w = max(0, area_rect.width - preview_w - self.padding)
        grid_h = area_rect.height
        preview_h = area_rect.height

        grid_rect_local = pygame.Rect(0, 0, grid_w, grid_h)
        preview_rect_local = pygame.Rect(grid_w + self.padding, 0, preview_w, preview_h)

        # Create subsurfaces relative to the target_surface
        try:
            worker_surface = target_surface.subsurface(area_rect)
            worker_surface.fill(colors.DARK_GRAY)  # Background for the worker area
            pygame.draw.rect(
                target_surface, colors.GRAY, area_rect, 1
            )  # Border around worker area

            # Render Grid
            if grid_rect_local.width > 0 and grid_rect_local.height > 0:
                grid_surf = worker_surface.subsurface(grid_rect_local)
                bg_color = (
                    colors.GRID_BG_GAME_OVER
                    if game_state.is_over()
                    else colors.GRID_BG_DEFAULT
                )
                grid_drawing.draw_grid_background(grid_surf, bg_color)
                grid_drawing.draw_grid_triangles(
                    grid_surf, game_state.grid_data, self.env_config
                )

                # --- Render Worker Info Text ---
                id_font = self.fonts.get("help")
                if id_font:
                    line_y = 3
                    line_height = id_font.get_height() + 1

                    # Worker ID and Game Step
                    id_text = f"W{worker_id} (Step {game_state.current_step})"
                    id_surf = id_font.render(id_text, True, colors.LIGHT_GRAY)
                    grid_surf.blit(id_surf, (3, line_y))
                    line_y += line_height

                    # Current Score
                    score_text = f"Score: {game_state.game_score:.0f}"
                    score_surf = id_font.render(score_text, True, colors.YELLOW)
                    grid_surf.blit(score_surf, (3, line_y))
                    line_y += line_height

                    # MCTS Stats (if available)
                    if worker_step_stats:
                        visits = worker_step_stats.get("mcts_visits", "?")
                        depth = worker_step_stats.get("mcts_depth", "?")
                        mcts_text = f"MCTS: V={visits} D={depth}"
                        mcts_surf = id_font.render(mcts_text, True, colors.CYAN)
                        grid_surf.blit(mcts_surf, (3, line_y))
                        line_y += line_height

            # Render Previews
            if preview_rect_local.width > 0 and preview_rect_local.height > 0:
                preview_surf = worker_surface.subsurface(preview_rect_local)
                # Pass 0,0 as topleft because preview_surf is already positioned
                _ = preview_drawing.render_previews(
                    preview_surf,
                    game_state,
                    (0, 0),  # Relative to preview_surf
                    "training_visual",  # Mode context
                    self.env_config,
                    self.vis_config,
                    selected_shape_idx=-1,  # No selection in training view
                )

        except ValueError as e:
            # Handle cases where subsurface creation fails (e.g., zero dimensions)
            if "subsurface rectangle is invalid" not in str(e):
                logger.error(
                    f"Error creating subsurface for W{worker_id} in area {area_rect}: {e}"
                )
            # Draw error indicator if subsurface fails
            pygame.draw.rect(target_surface, colors.RED, area_rect, 2)
            id_font = self.fonts.get("help")
            if id_font:
                id_surf = id_font.render(f"W{worker_id} (Render Err)", True, colors.RED)
                id_rect = id_surf.get_rect(center=area_rect.center)
                target_surface.blit(id_surf, id_rect)

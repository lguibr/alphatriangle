# File: alphatriangle/visualization/core/dashboard_renderer.py
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import pygame
import ray  # Import ray

from ...environment import GameState
from ...stats.plotter import Plotter
from ..drawing import hud as hud_drawing
from ..ui import ProgressBar
from . import colors, layout
from .game_renderer import GameRenderer

if TYPE_CHECKING:
    from ...config import EnvConfig, ModelConfig, VisConfig
    from ...stats import StatsCollectorData


logger = logging.getLogger(__name__)


class DashboardRenderer:
    """
    Renders the training dashboard with minimal spacing.
    Displays worker states, plots, and progress bars with specific info lines.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
        stats_collector_actor: ray.actor.ActorHandle | None = None,
        model_config: Optional["ModelConfig"] = None,
        total_params: int | None = None,
        trainable_params: int | None = None,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.stats_collector_actor = stats_collector_actor
        self.model_config = model_config
        self.total_params = total_params
        self.trainable_params = trainable_params
        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.worker_sub_rects: dict[int, pygame.Rect] = {}
        self.last_worker_grid_size = (0, 0)
        self.last_num_workers = 0

        self.single_game_renderer = GameRenderer(vis_config, env_config, fonts)
        self.plotter = Plotter(plot_update_interval=0.2)

        self.progress_bar_height_per_bar = 25
        self.num_progress_bars = 2
        self.progress_bar_spacing = 2
        self.progress_bars_total_height = (
            (
                (self.progress_bar_height_per_bar * self.num_progress_bars)
                + (self.progress_bar_spacing * (self.num_progress_bars - 1))
            )
            if self.num_progress_bars > 0
            else 0
        )

        self._layout_calculated_for_size: tuple[int, int] = (0, 0)
        # Don't call ensure_layout here, wait for first render

    def ensure_layout(self):
        """Calculates or retrieves the main layout areas."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Pass the calculated total height needed for progress bars
            self.layout_rects = layout.calculate_training_layout(
                current_w,
                current_h,
                self.vis_config,
                progress_bars_total_height=self.progress_bars_total_height,
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated dashboard layout for size {current_size}: {self.layout_rects}"
            )
            self.last_worker_grid_size = (0, 0)
            self.worker_sub_rects = {}
        return self.layout_rects if self.layout_rects is not None else {}

    def _calculate_worker_sub_layout(
        self, worker_grid_area: pygame.Rect, worker_ids: list[int]
    ):
        """Calculates the grid layout within the worker_grid_area with NO padding."""
        area_w, area_h = worker_grid_area.size
        num_workers = len(worker_ids)

        if (
            area_w,
            area_h,
        ) == self.last_worker_grid_size and num_workers == self.last_num_workers:
            return

        logger.debug(
            f"Recalculating worker sub-layout for {num_workers} workers in area {area_w}x{area_h}"
        )
        self.last_worker_grid_size = (area_w, area_h)
        self.last_num_workers = num_workers
        self.worker_sub_rects = {}

        if area_h <= 10 or area_w <= 10 or num_workers <= 0:
            if num_workers > 0:
                logger.warning(
                    f"Worker grid area too small ({area_w}x{area_h}). Cannot calculate sub-layout."
                )
            return

        aspect_ratio = area_w / max(1, area_h)
        cols = math.ceil(math.sqrt(num_workers * aspect_ratio))
        rows = math.ceil(num_workers / cols)

        cols = max(1, cols)
        rows = max(1, rows)

        cell_w = max(1, area_w / cols)
        cell_h = max(1, area_h / rows)

        min_cell_w, min_cell_h = 60, 40
        if cell_w < min_cell_w or cell_h < min_cell_h:
            logger.warning(
                f"Worker grid cells potentially too small ({cell_w:.1f}x{cell_h:.1f})."
            )

        logger.info(
            f"Calculated worker sub-layout (no pad): {rows}x{cols} for {num_workers} workers. Cell: {cell_w:.1f}x{cell_h:.1f}"
        )

        sorted_worker_ids = sorted(worker_ids)
        for i, worker_id in enumerate(sorted_worker_ids):
            row = i // cols
            col = i % cols
            worker_area_x = int(worker_grid_area.left + col * cell_w)
            worker_area_y = int(worker_grid_area.top + row * cell_h)
            worker_w = int((col + 1) * cell_w) - int(col * cell_w)
            worker_h = int((row + 1) * cell_h) - int(row * cell_h)

            worker_rect = pygame.Rect(worker_area_x, worker_area_y, worker_w, worker_h)
            self.worker_sub_rects[worker_id] = worker_rect.clip(worker_grid_area)

    def render(
        self,
        worker_states: dict[int, GameState],
        global_stats: dict[str, Any] | None = None,
    ):
        """Renders the entire training dashboard."""
        self.screen.fill(colors.DARK_GRAY)
        layout_rects = self.ensure_layout()
        if not layout_rects:
            return

        worker_grid_area = layout_rects.get("worker_grid")
        plots_rect = layout_rects.get("plots")
        progress_bar_area_rect = layout_rects.get("progress_bar_area")

        worker_step_stats = (
            global_stats.get("worker_step_stats", {}) if global_stats else {}
        )

        # --- Render Worker Grid Area ---
        if (
            worker_grid_area
            and worker_grid_area.width > 0
            and worker_grid_area.height > 0
        ):
            worker_ids = list(worker_states.keys())
            if not worker_ids and global_stats and "num_workers" in global_stats:
                worker_ids = list(range(global_stats["num_workers"]))

            self._calculate_worker_sub_layout(worker_grid_area, worker_ids)

            for worker_id in self.worker_sub_rects:
                worker_area_rect = self.worker_sub_rects[worker_id]
                game_state = worker_states.get(worker_id)
                step_stats = worker_step_stats.get(worker_id)
                self.single_game_renderer.render_worker_state(
                    self.screen,
                    worker_area_rect,
                    worker_id,
                    game_state,
                    worker_step_stats=step_stats,
                )
                pygame.draw.rect(self.screen, colors.GRAY, worker_area_rect, 1)
        else:
            logger.warning("Worker grid area not available or too small.")

        # --- Render Plot Area ---
        if global_stats:
            plot_surface = None
            if plots_rect and plots_rect.width > 0 and plots_rect.height > 0:
                stats_data_for_plot: StatsCollectorData | None = global_stats.get(
                    "stats_data"
                )
                if stats_data_for_plot is not None:
                    has_any_metric_data = any(
                        isinstance(dq, deque) and dq
                        for dq in stats_data_for_plot.values()
                    )
                    if has_any_metric_data:
                        plot_surface = self.plotter.get_plot_surface(
                            stats_data_for_plot,
                            int(plots_rect.width),
                            int(plots_rect.height),
                        )
                    else:
                        logger.debug(
                            "Plot data received but all metric deques are empty."
                        )
                else:
                    logger.debug(
                        "No 'stats_data' key found in global_stats for plotting."
                    )

                if plot_surface:
                    self.screen.blit(plot_surface, plots_rect.topleft)
                else:
                    pygame.draw.rect(self.screen, colors.DARK_GRAY, plots_rect)
                    plot_font = self.fonts.get("help")
                    if plot_font:
                        wait_text = (
                            "Plot Area (Waiting for data...)"
                            if stats_data_for_plot is None
                            else "Plot Area (No data yet)"
                        )
                        wait_surf = plot_font.render(wait_text, True, colors.LIGHT_GRAY)
                        wait_rect = wait_surf.get_rect(center=plots_rect.center)
                        self.screen.blit(wait_surf, wait_rect)
                    pygame.draw.rect(self.screen, colors.GRAY, plots_rect, 1)

            # --- Render Progress Bars (in their dedicated area) ---
            if progress_bar_area_rect:
                current_y = (
                    progress_bar_area_rect.top
                )  # Start at the top of the PB area
                progress_bar_font = self.fonts.get("help")

                if progress_bar_font:
                    bar_width = progress_bar_area_rect.width  # Use the area width
                    bar_x = progress_bar_area_rect.left
                    bar_height = self.progress_bar_height_per_bar

                    # --- Generate Info Strings for Each Bar ---
                    train_bar_info_str = ""
                    buffer_bar_info_str = ""

                    # Info for Training Bar (Model/Params)
                    train_info_parts = []
                    if self.model_config:
                        model_str = f"CNN:{len(self.model_config.CONV_FILTERS)}L"
                        if self.model_config.NUM_RESIDUAL_BLOCKS > 0:
                            model_str += (
                                f"/Res:{self.model_config.NUM_RESIDUAL_BLOCKS}L"
                            )
                        if (
                            self.model_config.USE_TRANSFORMER
                            and self.model_config.TRANSFORMER_LAYERS > 0
                        ):
                            model_str += f"/TF:{self.model_config.TRANSFORMER_LAYERS}L"
                        train_info_parts.append(model_str)
                    if self.total_params is not None:
                        train_info_parts.append(
                            f"Params:{self.total_params / 1e6:.1f}M"
                        )
                    train_bar_info_str = " | ".join(train_info_parts)

                    # Info for Buffer Bar (Weight Updates, Episodes, Sims, Workers)
                    buffer_info_parts = []
                    # Use .get with default '?' for robustness
                    updates = global_stats.get("worker_weight_updates", "?")
                    episodes = global_stats.get("total_episodes", "?")
                    sims = global_stats.get("total_simulations_run", "?")  # Correct key
                    num_workers = global_stats.get("num_workers", "?")
                    pending_tasks = global_stats.get("pending_tasks", "?")

                    buffer_info_parts.append(f"Weight Updates:{updates}")
                    buffer_info_parts.append(f"Episodes:{episodes}")
                    if isinstance(sims, int | float):
                        sims_str = (
                            f"{sims / 1e6:.1f}M"
                            if sims >= 1e6
                            else (
                                f"{sims / 1e3:.0f}k" if sims >= 1000 else str(int(sims))
                            )
                        )
                        buffer_info_parts.append(f"Simulations:{sims_str}")
                    else:
                        buffer_info_parts.append(f"Simulations:{sims}")

                    # --- CHANGED: Robust worker status formatting ---
                    if isinstance(pending_tasks, int) and isinstance(num_workers, int):
                        buffer_info_parts.append(
                            f"Workers:{pending_tasks}/{num_workers}"
                        )
                    else:
                        buffer_info_parts.append(
                            f"Workers:{pending_tasks or '?'}/{num_workers or '?'}"
                        )
                    # --- END CHANGED ---

                    buffer_bar_info_str = " | ".join(buffer_info_parts)
                    # --- End Generate Info Strings ---

                    # Training Progress Bar (with model/param info)
                    train_progress = global_stats.get("train_progress")
                    if isinstance(train_progress, ProgressBar):
                        train_progress.render(
                            self.screen,
                            (bar_x, current_y),
                            int(bar_width),
                            bar_height,
                            progress_bar_font,
                            border_radius=3,
                            info_line=train_bar_info_str,  # Pass specific info
                        )
                        current_y += bar_height + self.progress_bar_spacing
                    else:
                        logger.debug(
                            "Train progress bar data not available or invalid type."
                        )

                    # Buffer Progress Bar (with global stats info)
                    buffer_progress = global_stats.get("buffer_progress")
                    if isinstance(buffer_progress, ProgressBar):
                        buffer_progress.render(
                            self.screen,
                            (bar_x, current_y),
                            int(bar_width),
                            bar_height,
                            progress_bar_font,
                            border_radius=3,
                            info_line=buffer_bar_info_str,  # Pass specific info
                        )
                    else:
                        logger.debug(
                            "Buffer progress bar data not available or invalid type."
                        )

        elif not global_stats:
            logger.debug("No global_stats provided to DashboardRenderer.")

        # --- Render HUD (Help Text Only) ---
        hud_drawing.render_hud(
            self.screen,
            mode="training_visual",
            fonts=self.fonts,
            display_stats=None,
        )

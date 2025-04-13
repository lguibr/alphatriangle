# File: src/visualization/core/dashboard_renderer.py
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import pygame

from src.environment import GameState
from src.stats import StatsCollectorData  # Keep this import

# Import Plotter directly from its module
from src.stats.plotter import Plotter

from ..drawing import hud as hud_drawing
from ..ui import ProgressBar
from . import colors, layout
from .game_renderer import GameRenderer

if TYPE_CHECKING:
    from src.stats import StatsCollectorActor

    from ...config import EnvConfig, ModelConfig, VisConfig  # Add ModelConfig

logger = logging.getLogger(__name__)


class DashboardRenderer:
    """
    Renders the training dashboard, including multiple worker game states
    in a top grid area, and statistics plots/progress bars/model info
    in a bottom area.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
        stats_collector_actor: Optional["StatsCollectorActor"] = None,  # Keep handle
        model_config: Optional["ModelConfig"] = None,  # Add model_config
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.stats_collector_actor = stats_collector_actor  # Store handle
        self.model_config = model_config  # Store model config
        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.worker_sub_rects: dict[int, pygame.Rect] = {}
        self.last_worker_grid_size = (0, 0)
        self.last_num_workers = 0

        self.single_game_renderer = GameRenderer(vis_config, env_config, fonts)
        self.plotter = Plotter(plot_update_interval=0.2)

        # Adjust progress bar layout slightly to make space for model info
        self.progress_bar_height_per_bar = 40  # Slightly smaller
        self.num_progress_bars = 2
        self.progress_bar_spacing = 5
        self.progress_bars_total_height = (
            self.progress_bar_height_per_bar + self.progress_bar_spacing
        ) * self.num_progress_bars

        self.model_info_height = 60  # Reserve space for model info text
        self.model_info_padding = 5

        self._layout_calculated_for_size: tuple[int, int] = (0, 0)
        self.ensure_layout()

    def ensure_layout(self):
        """Calculates or retrieves the main layout areas."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Calculate required bottom margin including model info
            required_bottom_margin_for_stats = (
                self.progress_bars_total_height
                + self.model_info_height
                + self.model_info_padding  # Padding between model info and progress bars
                + self.vis_config.PADDING  # Padding below model info
            )
            self.layout_rects = layout.calculate_training_layout(
                current_w,
                current_h,
                self.vis_config,
                bottom_margin=required_bottom_margin_for_stats,
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated dashboard layout for size {current_size}: {self.layout_rects}"
            )
            self.last_worker_grid_size = (0, 0)
            self.worker_sub_rects = {}
        return self.layout_rects

    def _calculate_worker_sub_layout(
        self, worker_grid_area: pygame.Rect, worker_ids: list[int]
    ):
        """Calculates the grid layout within the worker_grid_area."""
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
        pad = 5

        if area_h <= 10 or area_w <= 10 or num_workers <= 0:
            if num_workers > 0:
                logger.warning(
                    f"Worker grid area too small ({area_w}x{area_h}). Cannot calculate sub-layout."
                )
            return

        # Try to make layout more square-ish
        aspect_ratio = area_w / max(1, area_h)
        cols = math.ceil(math.sqrt(num_workers * aspect_ratio))
        rows = math.ceil(num_workers / cols)

        # Ensure cols is at least 1
        cols = max(1, cols)
        rows = max(1, rows)

        cell_w = max(1, (area_w - (cols - 1) * pad) / cols)
        cell_h = max(1, (area_h - (rows - 1) * pad) / rows)

        min_cell_w, min_cell_h = 80, 60
        if cell_w < min_cell_w or cell_h < min_cell_h:
            logger.warning(
                f"Worker grid cells potentially too small ({cell_w:.1f}x{cell_h:.1f})."
            )

        logger.info(
            f"Calculated worker sub-layout: {rows}x{cols} for {num_workers} workers. Cell: {cell_w:.1f}x{cell_h:.1f}"
        )

        sorted_worker_ids = sorted(worker_ids)
        for i, worker_id in enumerate(sorted_worker_ids):
            row = i // cols
            col = i % cols
            worker_area_x = worker_grid_area.left + col * (cell_w + pad)
            worker_area_y = worker_grid_area.top + row * (cell_h + pad)
            worker_rect = pygame.Rect(worker_area_x, worker_area_y, cell_w, cell_h)
            self.worker_sub_rects[worker_id] = worker_rect.clip(worker_grid_area)

    def _render_model_info(self, area_rect: pygame.Rect):
        """Renders model configuration details in the specified area."""
        if not self.model_config or not self.fonts.get("help"):
            return

        font = self.fonts["help"]
        text_color = colors.LIGHT_GRAY
        bg_color = colors.DARK_GRAY  # Match stats area background
        border_color = colors.GRAY

        pygame.draw.rect(self.screen, bg_color, area_rect)
        pygame.draw.rect(self.screen, border_color, area_rect, 1)

        y_offset = area_rect.top + 5
        x_offset = area_rect.left + 10
        line_height = font.get_height() + 2

        def render_line(text: str, y: int):
            # Truncate long lines if needed
            max_width = area_rect.width - 2 * x_offset
            original_text = text
            while font.size(text)[0] > max_width and len(text) > 10:
                text = text[:-4] + "..."  # Truncate
            if text != original_text:
                logger.debug(f"Truncated model info line: {original_text} -> {text}")

            surf = font.render(text, True, text_color)
            self.screen.blit(surf, (x_offset, y))
            return y + line_height

        current_y = render_line("Model Config:", y_offset)
        current_y = render_line(
            f"  CNN Filters: {self.model_config.CONV_FILTERS}", current_y
        )
        current_y = render_line(
            f"  Res Blocks: {self.model_config.NUM_RESIDUAL_BLOCKS} x {self.model_config.RESIDUAL_BLOCK_FILTERS}",
            current_y,
        )
        if self.model_config.USE_TRANSFORMER:
            current_y = render_line(
                f"  Transformer: {self.model_config.TRANSFORMER_LAYERS}L x {self.model_config.TRANSFORMER_HEADS}H (Dim:{self.model_config.TRANSFORMER_DIM})",
                current_y,
            )
        else:
            current_y = render_line("  Transformer: Disabled", current_y)
        # Only render next line if space permits
        if current_y + line_height <= area_rect.bottom - 5:
            current_y = render_line(
                f"  Shared FC: {self.model_config.FC_DIMS_SHARED}", current_y
            )
        # Only render next line if space permits
        if current_y + line_height <= area_rect.bottom - 5:
            current_y = render_line(
                f"  Other Feats Dim: {self.model_config.OTHER_NN_INPUT_FEATURES_DIM}",
                current_y,
            )

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
        stats_area = layout_rects.get("stats_area")
        plots_rect = layout_rects.get("plots")

        # Extract per-worker step stats from global_stats if available
        worker_step_stats = (
            global_stats.get("worker_step_stats", {}) if global_stats else {}
        )

        # --- Render Worker Grid Area ---
        if (
            worker_grid_area
            and worker_grid_area.width > 0
            and worker_grid_area.height > 0
        ):
            pygame.draw.rect(self.screen, colors.DARK_GRAY, worker_grid_area)
            # Use worker_states keys to determine layout, even if some states are None
            worker_ids = list(worker_states.keys())
            # If worker_states is empty but we know workers exist from stats, use those IDs
            if not worker_ids and global_stats and "num_workers" in global_stats:
                worker_ids = list(range(global_stats["num_workers"]))

            self._calculate_worker_sub_layout(worker_grid_area, worker_ids)

            # Render each worker area
            for worker_id in self.worker_sub_rects.keys():
                worker_area_rect = self.worker_sub_rects[worker_id]
                game_state = worker_states.get(worker_id)
                step_stats = worker_step_stats.get(
                    worker_id
                )  # Get stats for this worker
                self.single_game_renderer.render_worker_state(
                    self.screen,
                    worker_area_rect,
                    worker_id,
                    game_state,
                    worker_step_stats=step_stats,  # Pass the specific worker's step stats
                )
        else:
            logger.warning("Worker grid area not available or too small.")

        # --- Render Stats Area (Plots, Progress Bars, Model Info) ---
        if stats_area and global_stats:
            pygame.draw.rect(self.screen, colors.DARK_GRAY, stats_area)

            # Render Plots
            plot_surface = None
            if plots_rect and plots_rect.width > 0 and plots_rect.height > 0:
                stats_data_for_plot: StatsCollectorData | None = global_stats.get(
                    "stats_data"
                )

                if stats_data_for_plot is not None:
                    # Check if *any* metric has data
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
                    # Draw placeholder if no plot surface generated
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

            # --- Render Progress Bars and Model Info below plots ---
            current_y = (
                plots_rect.bottom + self.progress_bar_spacing
                if plots_rect
                else stats_area.top + self.vis_config.PADDING
            )

            # Render Progress Bars
            progress_bar_font = self.fonts.get("help")
            if progress_bar_font:
                bar_width = stats_area.width
                bar_x = stats_area.left
                bar_height = self.progress_bar_height_per_bar

                # Training Progress Bar
                if current_y + bar_height <= stats_area.bottom:
                    train_progress = global_stats.get("train_progress")
                    if isinstance(train_progress, ProgressBar):
                        train_progress.render(
                            self.screen,
                            (bar_x, current_y),
                            int(bar_width),
                            bar_height,
                            progress_bar_font,
                            bar_color=colors.GREEN,
                        )
                        current_y += bar_height + self.progress_bar_spacing
                    else:
                        logger.debug(
                            "Train progress bar data not available or invalid type."
                        )

                # Buffer Progress Bar
                if current_y + bar_height <= stats_area.bottom:
                    buffer_progress = global_stats.get("buffer_progress")
                    if isinstance(buffer_progress, ProgressBar):
                        buffer_progress.render(
                            self.screen,
                            (bar_x, current_y),
                            int(bar_width),
                            bar_height,
                            progress_bar_font,
                            bar_color=colors.ORANGE,
                        )
                        current_y += bar_height + self.progress_bar_spacing
                    else:
                        logger.debug(
                            "Buffer progress bar data not available or invalid type."
                        )
                elif current_y <= stats_area.bottom:
                    logger.warning(
                        "Not enough vertical space for the second progress bar."
                    )

            # Render Model Info below progress bars
            model_info_y = current_y  # Start right after progress bars
            model_info_rect = pygame.Rect(
                stats_area.left,
                model_info_y,
                stats_area.width,
                # Use remaining height in stats_area
                max(0, stats_area.bottom - model_info_y - self.vis_config.PADDING),
            )
            # Ensure it fits within the stats area and has minimum height
            if (
                model_info_rect.height >= self.model_info_height
            ):  # Check if enough space
                self._render_model_info(model_info_rect)
            else:
                logger.warning(
                    f"Not enough space to render model info (Available: {model_info_rect.height}, Need: {self.model_info_height})."
                )

        elif not global_stats:
            logger.debug("No global_stats provided to DashboardRenderer.")

        # --- Render HUD ---
        # Pass mode and stats, but no GameState
        hud_drawing.render_hud(
            self.screen,
            mode="training_visual",
            fonts=self.fonts,
            display_stats=global_stats,
        )

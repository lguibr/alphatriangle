# File: src/stats/plotter.py
import logging
import time
from collections import deque
from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib

# Move numpy import into TYPE_CHECKING block
if TYPE_CHECKING:
    import numpy as np

import pygame

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import normalize_color_for_matplotlib from the new location
from src.utils.helpers import normalize_color_for_matplotlib
from src.visualization.core import colors as vis_colors

from .collector import StatsCollectorData
from .plot_utils import render_single_plot  # Keep this import

logger = logging.getLogger(__name__)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self, plot_update_interval: float = 0.5):
        self.plot_surface_cache: pygame.Surface | None = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = plot_update_interval
        self.rolling_window_sizes: list[int] = [10, 50, 100, 500, 1000, 5000]
        self.colors = self._init_colors()

        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None  # type: ignore # numpy is type-checked only
        self.last_target_size: tuple[int, int] = (0, 0)
        self.last_data_hash: int | None = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> dict[str, tuple[float, float, float]]:
        """Initializes plot colors using vis_colors."""
        return {
            "SelfPlay/Episode_Score": normalize_color_for_matplotlib(vis_colors.YELLOW),
            "Loss/Total": normalize_color_for_matplotlib(vis_colors.RED),
            "Loss/Value": normalize_color_for_matplotlib(vis_colors.BLUE),
            "Loss/Policy": normalize_color_for_matplotlib(vis_colors.GREEN),
            "LearningRate": normalize_color_for_matplotlib(vis_colors.CYAN),
            "SelfPlay/Episode_Length": normalize_color_for_matplotlib(
                vis_colors.ORANGE
            ),
            "Buffer/Size": normalize_color_for_matplotlib(vis_colors.PURPLE),
            "MCTS/Avg_Root_Visits": normalize_color_for_matplotlib(
                vis_colors.LIGHT_GRAY
            ),
            "MCTS/Avg_Tree_Depth": normalize_color_for_matplotlib(vis_colors.LIGHTG),
            "placeholder": normalize_color_for_matplotlib(
                vis_colors.GRAY
            ),  # Keep placeholder color
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes."""
        logger.info(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Error closing previous figure: {e}")

        dpi = 96
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            nrows, ncols = 2, 4
            self.fig, self.axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width_in, fig_height_in),
                dpi=dpi,
                sharex=False,
            )
            # Set figure background color
            self.fig.patch.set_facecolor((0.1, 0.1, 0.1))  # Dark background for figure
            self.fig.subplots_adjust(
                hspace=0.4, wspace=0.35, left=0.08, right=0.98, bottom=0.15, top=0.92
            )
            self.last_target_size = (target_width, target_height)
            logger.info(
                f"[Plotter] Matplotlib figure initialized ({nrows}x{ncols} grid)."
            )
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

    def _get_data_hash(self, plot_data: StatsCollectorData) -> int:
        """Generates a simple hash based on data lengths and last elements."""
        hash_val = 0
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            if not dq:
                continue
            hash_val ^= hash(key) ^ len(dq)
            try:
                last_step, last_val = dq[-1]
                hash_val ^= hash(last_step) ^ hash(f"{last_val:.6f}")
            except IndexError:
                pass
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            plot_defs = [
                ("SelfPlay/Episode_Score", "Ep Score", False),
                ("Loss/Total", "Total Loss", True),
                ("MCTS/Avg_Root_Visits", "Root Visits", False),
                ("LearningRate", "Learn Rate", True),
                ("SelfPlay/Episode_Length", "Ep Length", False),
                ("Loss/Value", "Value Loss", True),
                ("Loss/Policy", "Policy Loss", True),
                ("MCTS/Avg_Tree_Depth", "Tree Depth", False),
            ]

            data_values: dict[str, list[float]] = {}
            data_steps: dict[str, list[int]] = {}
            has_any_data = False
            for key, _, _ in plot_defs:
                dq = plot_data.get(key, deque())
                if dq:
                    steps, values = zip(*dq, strict=False)
                    data_values[key] = list(values)
                    data_steps[key] = list(steps)
                    if values:
                        has_any_data = True
                else:
                    data_values[key], data_steps[key] = [], []

            if not has_any_data:
                logger.debug("[Plotter] No data available for any defined plot.")
                for ax in axes_flat:
                    ax.clear()
                    ax.set_title(
                        "(No Data)", loc="left", fontsize=8, color="white"
                    )  # Ensure text is visible
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_facecolor((0.15, 0.15, 0.15))  # Set axes background
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["bottom"].set_color("gray")
                    ax.spines["left"].set_color("gray")
                return True

            placeholder_color_mpl = self.colors.get(
                "placeholder", (0.5, 0.5, 0.5)
            )  # Get placeholder color

            for i, (data_key, label, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                ax.clear()

                current_values = data_values.get(data_key, [])
                current_steps = data_steps.get(data_key, [])
                color_mpl = self.colors.get(data_key, (0.5, 0.5, 0.5))

                render_single_plot(
                    ax,
                    current_steps,
                    current_values,
                    label,
                    color_mpl,
                    placeholder_color=placeholder_color_mpl,  # Pass placeholder color
                    rolling_window_sizes=self.rolling_window_sizes,
                    show_placeholder=(not current_values),
                    placeholder_text=label,
                    y_log_scale=log_scale,
                )
                nrows, ncols = self.axes.shape
                if i < (nrows - 1) * ncols:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)

            plot_update_duration = time.monotonic() - plot_update_start
            logger.debug(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass
            return False

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            self.fig.canvas.draw()
            buf = BytesIO()
            self.fig.savefig(
                buf,
                format="png",
                transparent=False,  # Keep false if figure background is set
                facecolor=self.fig.get_facecolor(),  # Use figure's facecolor
            )
            buf.seek(0)
            # Use convert() if no alpha needed, convert_alpha() otherwise
            plot_img_surface = pygame.image.load(buf, "png").convert()
            buf.close()

            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                plot_img_surface = pygame.transform.smoothscale(
                    plot_img_surface, (target_width, target_height)
                )
            render_duration = time.monotonic() - render_start
            logger.debug(
                f"[Plotter] Figure rendered to surface in {render_duration:.4f}s"
            )
            return plot_img_surface

        except Exception as e:
            logger.error(f"Error rendering Matplotlib figure: {e}", exc_info=True)
            return None

    def get_plot_surface(
        self, plot_data: StatsCollectorData, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(
            isinstance(dq, deque) and dq for dq in plot_data.values()
        )  # Check if any deque has data
        target_size = (target_width, target_height)

        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed
        can_create_plot = target_width > 50 and target_height > 50

        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info("[Plotter] Target size too small, clearing cache/figure.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes, self.last_target_size = None, None, (0, 0)
            return None

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.debug("[Plotter] No plot data available, clearing cache/figure.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes, self.last_target_size = None, None, (0, 0)
            return None

        # cache_status = "HIT" # Removed unused variable
        try:
            if needs_reinit:
                # cache_status = "MISS (Re-init)" # Removed unused variable
                self._init_figure(target_width, target_height)
                if self.fig and self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    self.plot_surface_cache = None
            elif needs_update:
                # cache_status = f"MISS (Update - Data: {data_changed}, Time: {time_elapsed})" # Removed unused variable
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot update failed, returning stale cache."
                    )
                    # cache_status = "ERROR (Update Failed)" # Removed unused variable
            elif self.plot_surface_cache is None:
                # cache_status = "MISS (Cache None)" # Removed unused variable
                if self.fig is None:
                    self._init_figure(target_width, target_height)
                if self.fig and self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash

        except Exception as e:
            logger.error(f"[Plotter] Error in get_plot_surface: {e}", exc_info=True)
            self.plot_surface_cache = None
            if self.fig:
                plt.close(self.fig)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed."""
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")

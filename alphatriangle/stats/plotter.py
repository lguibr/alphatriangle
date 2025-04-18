# File: alphatriangle/stats/plotter.py
import contextlib
import logging
import time
from collections import deque
from io import BytesIO
from typing import TYPE_CHECKING, Any

import matplotlib

if TYPE_CHECKING:
    import numpy as np

    # --- MOVED: Import vis_colors only for type checking ---

import pygame

# Use Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- MOVED: Import normalize_color_for_matplotlib here ---
from ..utils.helpers import normalize_color_for_matplotlib  # noqa: E402

# --- CHANGED: Import StepInfo ---
from ..utils.types import StatsCollectorData  # noqa: E402

# --- END CHANGED ---
from .plot_definitions import (  # noqa: E402
    WEIGHT_UPDATE_METRIC_KEY,  # Import key
    PlotDefinitions,
)
from .plot_rendering import render_subplot  # Import subplot rendering logic

logger = logging.getLogger(__name__)


class Plotter:
    """
    Handles creation and caching of the multi-plot Matplotlib surface.
    Uses PlotDefinitions for layout and plot_rendering for drawing subplots.
    """

    def __init__(self, plot_update_interval: float = 0.75):  # Increased interval
        self.plot_surface_cache: pygame.Surface | None = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = plot_update_interval
        self.colors = self._init_colors()
        self.plot_definitions = PlotDefinitions(self.colors)  # Instantiate definitions

        self.rolling_window_sizes: list[int] = [
            10,
            50,
            100,
            500,
            1000,
            5000,
        ]

        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None  # type: ignore # numpy is type-checked \only
        self.last_target_size: tuple[int, int] = (0, 0)
        self.last_data_hash: int | None = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> dict[str, tuple[float, float, float]]:
        """Initializes plot colors using hardcoded values to avoid vis import."""
        # This breaks the circular import. Ensure these match vis_colors.py
        colors_rgb = {
            "YELLOW": (230, 230, 40),
            "WHITE": (255, 255, 255),
            "LIGHT_GRAY": (180, 180, 180),
            "LIGHTG": (144, 238, 144),
            "RED": (220, 40, 40),
            "BLUE": (60, 60, 220),
            "GREEN": (40, 200, 40),
            "CYAN": (40, 200, 200),
            "PURPLE": (140, 40, 140),
            "BLACK": (0, 0, 0),
            "GRAY": (100, 100, 100),
            "ORANGE": (240, 150, 20),
            "HOTPINK": (255, 105, 180),
        }

        return {
            "RL/Current_Score": normalize_color_for_matplotlib(colors_rgb["YELLOW"]),
            "RL/Step_Reward": normalize_color_for_matplotlib(colors_rgb["WHITE"]),
            "MCTS/Step_Visits": normalize_color_for_matplotlib(
                colors_rgb["LIGHT_GRAY"]
            ),
            "MCTS/Step_Depth": normalize_color_for_matplotlib(colors_rgb["LIGHTG"]),
            "Loss/Total": normalize_color_for_matplotlib(colors_rgb["RED"]),
            "Loss/Value": normalize_color_for_matplotlib(colors_rgb["BLUE"]),
            "Loss/Policy": normalize_color_for_matplotlib(colors_rgb["GREEN"]),
            "LearningRate": normalize_color_for_matplotlib(colors_rgb["CYAN"]),
            "Buffer/Size": normalize_color_for_matplotlib(colors_rgb["PURPLE"]),
            WEIGHT_UPDATE_METRIC_KEY: normalize_color_for_matplotlib(
                colors_rgb["BLACK"]
            ),
            "placeholder": normalize_color_for_matplotlib(colors_rgb["GRAY"]),
            "Rate/Steps_Per_Sec": normalize_color_for_matplotlib(colors_rgb["ORANGE"]),
            "Rate/Episodes_Per_Sec": normalize_color_for_matplotlib(
                colors_rgb["HOTPINK"]
            ),
            "Rate/Simulations_Per_Sec": normalize_color_for_matplotlib(
                colors_rgb["LIGHTG"]
            ),
            "PER/Beta": normalize_color_for_matplotlib(colors_rgb["ORANGE"]),
            "Loss/Entropy": normalize_color_for_matplotlib(colors_rgb["PURPLE"]),
            "Loss/Mean_TD_Error": normalize_color_for_matplotlib(colors_rgb["RED"]),
            "Progress/Train_Step_Percent": normalize_color_for_matplotlib(
                colors_rgb["GREEN"]
            ),
            "Progress/Total_Simulations": normalize_color_for_matplotlib(
                colors_rgb["CYAN"]
            ),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes based on plot definitions."""
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
            nrows = self.plot_definitions.nrows
            ncols = self.plot_definitions.ncols
            self.fig, self.axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width_in, fig_height_in),
                dpi=dpi,
                sharex=False,
            )
            if self.axes is None:
                raise RuntimeError("Failed to create Matplotlib subplots.")

            self.fig.patch.set_facecolor((0.1, 0.1, 0.1))
            self.fig.subplots_adjust(
                hspace=0.40,
                wspace=0.08,
                left=0.03,
                right=0.99,
                bottom=0.05,
                top=0.98,
            )
            self.last_target_size = (target_width, target_height)
            logger.info(
                f"[Plotter] Matplotlib figure initialized ({nrows}x{ncols} grid)."
            )
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

    def _get_data_hash(self, plot_data: StatsCollectorData) -> int:
        """Generates a hash based on data lengths and recent values."""
        hash_val = 0
        sample_size = 5
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            hash_val ^= hash(key) ^ len(dq)
            if not dq:
                continue
            try:
                num_to_sample = min(len(dq), sample_size)
                for i in range(-1, -num_to_sample - 1, -1):
                    # Hash StepInfo dict and value
                    step_info, val = dq[i]
                    # Simple hash for dict: hash tuple of sorted items
                    step_info_hash = hash(tuple(sorted(step_info.items())))
                    hash_val ^= step_info_hash ^ hash(f"{val:.6f}")
            except IndexError:
                pass
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData) -> bool:
        """Updates the data on the existing Matplotlib axes using render_subplot."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            plot_defs = self.plot_definitions.get_definitions()
            num_plots = len(plot_defs)

            # Extract weight update steps (global_step values)
            weight_update_steps: list[int] = []
            if WEIGHT_UPDATE_METRIC_KEY in plot_data:
                dq = plot_data[WEIGHT_UPDATE_METRIC_KEY]
                if dq:
                    # Extract global_step from StepInfo
                    weight_update_steps = [
                        step_info["global_step"]
                        for step_info, _ in dq
                        if "global_step" in step_info
                    ]

            for i, plot_def in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                # Pass weight_update_steps
                render_subplot(
                    ax=ax,
                    plot_data=plot_data,
                    plot_def=plot_def,
                    colors=self.colors,
                    rolling_window_sizes=self.rolling_window_sizes,
                    weight_update_steps=weight_update_steps,  # Pass the list
                )

            for i in range(num_plots, len(axes_flat)):
                ax = axes_flat[i]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("gray")
                ax.spines["left"].set_color("gray")

            self._apply_final_axis_formatting(axes_flat)

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

    def _apply_final_axis_formatting(self, axes_flat: Any):
        """Hides x-axis labels for plots not in the bottom row."""
        if not hasattr(axes_flat, "__iter__"):
            logger.error("axes_flat is not iterable in _apply_final_axis_formatting")
            return

        nrows = self.plot_definitions.nrows
        ncols = self.plot_definitions.ncols
        num_plots = len(self.plot_definitions.get_definitions())

        for i, ax in enumerate(axes_flat):
            if i >= num_plots:
                continue

            if i < (nrows - 1) * ncols:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0)

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            self.fig.canvas.draw_idle()
            buf = BytesIO()
            self.fig.savefig(
                buf, format="png", transparent=False, facecolor=self.fig.get_facecolor()
            )
            buf.seek(0)
            plot_img_surface = pygame.image.load(buf, "png").convert_alpha()
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
            isinstance(dq, deque) and dq
            for key, dq in plot_data.items()
            if not key.startswith("Internal/")
        )
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
            logger.debug("[Plotter] No plot data available, returning cache (if any).")
            return self.plot_surface_cache

        try:
            if needs_reinit:
                self._init_figure(target_width, target_height)
                needs_update = True

            if needs_update and self.fig:
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot update failed, returning stale cache if available."
                    )
            elif (
                self.plot_surface_cache is None
                and self.fig
                and self._update_plot_data(plot_data)
            ):
                self.plot_surface_cache = self._render_figure_to_surface(
                    target_width, target_height
                )
                self.last_plot_update_time = current_time
                self.last_data_hash = current_data_hash

        except Exception as e:
            logger.error(f"[Plotter] Error in get_plot_surface: {e}", exc_info=True)
            self.plot_surface_cache = None
            if self.fig:
                with contextlib.suppress(Exception):
                    plt.close(self.fig)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed."""
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                print(f"[Plotter] Error closing figure in destructor: {e}")

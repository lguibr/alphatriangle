# File: alphatriangle/stats/plotter.py
import contextlib  # Added import
import logging
import time
from collections import deque
from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib

if TYPE_CHECKING:
    import numpy as np

import pygame

# Use Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from ..utils.helpers import normalize_color_for_matplotlib  # noqa: E402
from ..visualization.core import colors as vis_colors  # noqa: E402
from .collector import StatsCollectorData  # noqa: E402
from .plot_utils import calculate_rolling_average, format_value  # noqa: E402

logger = logging.getLogger(__name__)

WEIGHT_UPDATE_METRIC_KEY = "Internal/Weight_Update_Step"


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self, plot_update_interval: float = 0.5):
        self.plot_surface_cache: pygame.Surface | None = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = plot_update_interval
        # Use smaller windows too
        self.rolling_window_sizes: list[int] = [
            5,
            10,
            20,
            50,
            100,
            500,
            1000,
            5000,
        ]
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
        # Define colors based on the *conceptual* metric keys
        return {
            "RL/Current_Score": normalize_color_for_matplotlib(vis_colors.YELLOW),
            "RL/Step_Reward": normalize_color_for_matplotlib(vis_colors.WHITE),
            "MCTS/Step_Visits": normalize_color_for_matplotlib(vis_colors.LIGHT_GRAY),
            "MCTS/Step_Depth": normalize_color_for_matplotlib(vis_colors.LIGHTG),
            "Loss/Total": normalize_color_for_matplotlib(vis_colors.RED),
            "Loss/Value": normalize_color_for_matplotlib(vis_colors.BLUE),
            "Loss/Policy": normalize_color_for_matplotlib(vis_colors.GREEN),
            "LearningRate": normalize_color_for_matplotlib(vis_colors.CYAN),
            "Buffer/Size": normalize_color_for_matplotlib(vis_colors.PURPLE),
            WEIGHT_UPDATE_METRIC_KEY: normalize_color_for_matplotlib(vis_colors.BLACK),
            "placeholder": normalize_color_for_matplotlib(vis_colors.GRAY),
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
            # Use 3x3 layout
            nrows, ncols = 3, 3
            self.fig, self.axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width_in, fig_height_in),
                dpi=dpi,
                sharex=False,  # Keep sharex=False for independent y-axes
            )
            if self.axes is None:  # Check if axes creation failed
                raise RuntimeError("Failed to create Matplotlib subplots.")

            self.fig.patch.set_facecolor((0.1, 0.1, 0.1))
            # Adjust spacing for 3x3
            self.fig.subplots_adjust(
                hspace=0.5, wspace=0.35, left=0.08, right=0.98, bottom=0.1, top=0.92
            )
            self.last_target_size = (target_width, target_height)
            logger.info(
                f"[Plotter] Matplotlib figure initialized ({nrows}x{ncols} grid)."
            )
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

    def _get_data_hash(self, plot_data: StatsCollectorData) -> int:
        """
        Generates a more sensitive hash based on data lengths and a sample of recent values.
        """
        hash_val = 0
        sample_size = 5  # Check last 5 points for changes
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            hash_val ^= hash(key) ^ len(dq)  # Include length
            if not dq:
                continue
            try:
                # Hash a sample of recent points instead of just the last one
                num_to_sample = min(len(dq), sample_size)
                for i in range(-1, -num_to_sample - 1, -1):
                    step, val = dq[i]
                    hash_val ^= hash(step) ^ hash(f"{val:.6f}")
            except IndexError:
                pass  # Should not happen if dq is not empty
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData) -> bool:
        """
        Updates the data on the existing Matplotlib axes.
        Plots raw data as scatter and rolling average against appropriate x-axis
        (global step or data index).
        Applies x-axis limits independently per plot.
        """
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            # Define conceptual plots and their x-axis type ('global_step' or 'index')
            plot_defs = [
                ("RL/Current_Score", "Score", False, "index"),
                ("Loss/Total", "Total Loss", True, "global_step"),
                ("LearningRate", "Learn Rate", True, "global_step"),
                ("RL/Step_Reward", "Step Reward", False, "index"),
                ("Loss/Value", "Value Loss", True, "global_step"),
                ("Loss/Policy", "Policy Loss", True, "global_step"),
                ("MCTS/Step_Visits", "MCTS Visits", False, "index"),
                ("MCTS/Step_Depth", "MCTS Depth", False, "index"),
                ("Buffer/Size", "Buffer Size", False, "global_step"),
            ]

            # Extract weight update steps (always global)
            weight_update_steps: list[int] = []
            if WEIGHT_UPDATE_METRIC_KEY in plot_data:
                dq = plot_data[WEIGHT_UPDATE_METRIC_KEY]
                if dq:
                    weight_update_steps = [step for step, _ in dq]

            # Store max x-value per axis (can be step or index)
            max_x_per_axis: dict[int, int] = {}
            has_any_data_at_all = False

            # Loop 1: Plot data and find max x-value PER AXIS
            for i, (conceptual_key, label, log_scale, x_axis_type) in enumerate(
                plot_defs
            ):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                ax.clear()
                ax.set_facecolor((0.15, 0.15, 0.15))

                max_x_for_plot = -1  # Use -1 to indicate no data yet

                combined_steps_values: list[tuple[int, float]] = []
                dq = plot_data.get(conceptual_key, deque())
                if dq:
                    combined_steps_values = list(dq)  # Use data in received order
                    if combined_steps_values:
                        if x_axis_type == "global_step":
                            # Max x is the maximum step value
                            max_x_for_plot = max(s for s, v in combined_steps_values)
                        else:  # x_axis_type == "index"
                            # Max x is the maximum index
                            max_x_for_plot = len(combined_steps_values) - 1
                        has_any_data_at_all = True

                max_x_per_axis[i] = max_x_for_plot

                color_mpl = self.colors.get(conceptual_key, (0.5, 0.5, 0.5))
                placeholder_color_mpl = self.colors.get("placeholder", (0.5, 0.5, 0.5))

                if not combined_steps_values:
                    ax.text(
                        0.5,
                        0.5,
                        f"{label}\n(No Data)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color=placeholder_color_mpl,
                        fontsize=9,
                    )
                    ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    # Select X-Data based on type
                    if x_axis_type == "global_step":
                        x_data = [s for s, v in combined_steps_values]
                        x_label_text = "Train Step"
                    else:  # x_axis_type == "index"
                        x_data = list(range(len(combined_steps_values)))
                        x_label_text = "Data Index"
                    combined_values = [v for s, v in combined_steps_values]

                    # Plot raw data as scatter against appropriate x-axis
                    ax.scatter(
                        x_data,
                        combined_values,
                        color=color_mpl,
                        alpha=0.1,
                        s=2,
                        label="_nolegend_",
                        zorder=2,  # Draw scatter points below avg line
                    )

                    num_points = len(combined_values)
                    best_window = 0
                    for window in sorted(self.rolling_window_sizes, reverse=True):
                        if num_points >= window:
                            best_window = window
                            break

                    # PLOT ROLLING AVERAGE (Single Prominent Line) against appropriate x-axis
                    if best_window > 0:
                        rolling_avg = calculate_rolling_average(
                            combined_values, best_window
                        )
                        if len(rolling_avg) == len(
                            x_data
                        ):  # Check against x_data length
                            ax.plot(
                                x_data,
                                rolling_avg,
                                color=color_mpl,
                                alpha=0.9,
                                linewidth=1.5,
                                label=f"Avg {best_window}",
                                zorder=3,  # Ensure average line is on top
                            )
                            ax.legend(
                                fontsize=6,
                                loc="upper right",
                                frameon=False,
                                labelcolor="lightgray",
                            )
                        else:
                            logger.warning(
                                f"Length mismatch for rolling avg ({len(rolling_avg)}) vs x_data ({len(x_data)}) for {label}. Skipping avg plot."
                            )

                    # Axis Formatting
                    ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
                    ax.tick_params(
                        axis="both", which="major", labelsize=7, colors="lightgray"
                    )
                    ax.grid(True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4))

                    if log_scale:
                        ax.set_yscale("log")
                        min_val = min(
                            (v for v in combined_values if v > 0), default=1e-6
                        )
                        max_val = max(combined_values, default=1.0)
                        ylim_bottom = max(1e-9, min_val * 0.1)
                        ylim_top = max_val * 10
                        if ylim_bottom < ylim_top:
                            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
                        else:
                            ax.set_ylim(bottom=1e-9, top=1.0)
                    else:
                        ax.set_yscale("linear")

                    # Format x-axis using index or step value
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                    ax.xaxis.set_major_formatter(
                        FuncFormatter(
                            lambda x, _: (
                                f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"
                            )
                        )
                    )
                    ax.set_xlabel(
                        x_label_text, fontsize=8, color="gray"
                    )  # Use determined label

                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda y, _: format_value(y))
                    )

                    # Get last value from original list (last point added)
                    current_val_str = format_value(combined_values[-1])
                    ax.text(
                        1.0,
                        1.01,
                        f"Cur: {current_val_str}",
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="white",
                    )

                # Common Axis Styling
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("gray")
                ax.spines["left"].set_color("gray")
                nrows, ncols = self.axes.shape
                if i < (nrows - 1) * ncols:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)

            # Loop 2: Apply final axis limits (using per-axis max x-value)
            if has_any_data_at_all:
                logger.debug(
                    f"[Plotter] Applying individual xlims. Max x-values per axis: {max_x_per_axis}"
                )
                for i, ax in enumerate(axes_flat):
                    # Use per-axis max x-value
                    plot_max_x = max_x_per_axis.get(i, -1)
                    if plot_max_x >= 0:  # Only set limit if data exists
                        # Add a small buffer to the max x for xlim
                        effective_max_xlim = max(1, plot_max_x * 1.05)
                        ax.set_xlim(left=0, right=effective_max_xlim)

                        # Draw weight update lines only on global step axes
                        _, _, _, x_axis_type = plot_defs[i]
                        if weight_update_steps and x_axis_type == "global_step":
                            relevant_updates = [
                                step
                                for step in weight_update_steps
                                if step <= effective_max_xlim
                            ]
                            for update_step in relevant_updates:
                                ax.axvline(
                                    x=update_step,  # Plot line at the global step value
                                    color="white",
                                    linestyle="--",
                                    linewidth=0.5,
                                    alpha=0.6,
                                    zorder=1,
                                )
                    else:  # No data for this axis
                        ax.set_xlim(left=0, right=1)  # Default xlim if no data

                    # Ensure x-label visibility is correct based on final layout
                    nrows, ncols = self.axes.shape
                    is_bottom_row = i >= (nrows - 1) * ncols
                    has_data_this_axis = max_x_per_axis.get(i, -1) >= 0
                    if not is_bottom_row and ax.get_xlabel():
                        ax.set_xlabel("")
                    elif is_bottom_row and has_data_this_axis and not ax.get_xlabel():
                        # Re-set label based on plot type
                        _, _, _, x_axis_type = plot_defs[i]
                        x_label_text = (
                            "Train Step"
                            if x_axis_type == "global_step"
                            else "Data Index"
                        )
                        ax.set_xlabel(x_label_text, fontsize=8, color="gray")

            # Clear any unused axes
            for i in range(len(plot_defs), len(axes_flat)):
                ax = axes_flat[i]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("gray")
                ax.spines["left"].set_color("gray")

            plot_update_duration = time.monotonic() - plot_update_start
            logger.debug(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True  # Indicate update happened

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass
            return False  # Indicate update failed

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
                transparent=False,
                facecolor=self.fig.get_facecolor(),
            )
            buf.seek(0)
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
            logger.debug("[Plotter] No plot data available, returning None.")
            # Keep stale cache if no new data
            return self.plot_surface_cache

        try:
            if needs_reinit:
                self._init_figure(target_width, target_height)
                if self.fig and self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    self.plot_surface_cache = None  # Clear cache if update failed
            elif needs_update:
                if self.fig and self._update_plot_data(plot_data):
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
                self.plot_surface_cache is None and self.fig
            ):  # Render if cache is empty but update not triggered (e.g., first time after init)
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash

        except Exception as e:
            logger.error(f"[Plotter] Error in get_plot_surface: {e}", exc_info=True)
            self.plot_surface_cache = None
            if self.fig:
                # --- CHANGED: Use contextlib.suppress ---
                with contextlib.suppress(Exception):  # Ignore errors during cleanup
                    plt.close(self.fig)
                # --- END CHANGED ---
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed."""
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                # Use print directly as logging might not work during deletion
                print(f"[Plotter] Error closing figure in destructor: {e}")

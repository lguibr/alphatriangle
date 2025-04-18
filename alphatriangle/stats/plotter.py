# File: alphatriangle/stats/plotter.py
# Changes:
# - Further reduced wspace and left margin in subplots_adjust.

import contextlib
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
            "Rate/Steps_Per_Sec": normalize_color_for_matplotlib(vis_colors.ORANGE),
            "Rate/Episodes_Per_Sec": normalize_color_for_matplotlib(vis_colors.HOTPINK),
            "Rate/Simulations_Per_Sec": normalize_color_for_matplotlib(
                vis_colors.LIGHTG
            ),
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
            nrows, ncols = 4, 3
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
            # --- CHANGED: Further reduced spacing and margins ---
            self.fig.subplots_adjust(
                hspace=0.40,  # Keep reduced vertical space
                wspace=0.08,  # Drastically reduced horizontal space
                left=0.03,  # Minimal left margin
                right=0.99,  # Maximize right edge
                bottom=0.05,  # Minimal bottom margin
                top=0.98,  # Maximize top edge
            )
            # --- END CHANGED ---
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
                    step, val = dq[i]
                    hash_val ^= hash(step) ^ hash(f"{val:.6f}")
            except IndexError:
                pass
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData) -> bool:
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            plot_defs = [
                ("RL/Current_Score", "Score", False, "index"),
                ("Rate/Episodes_Per_Sec", "Episodes/sec", False, "buffer_size"),
                ("Loss/Total", "Total Loss", True, "global_step"),
                ("RL/Step_Reward", "Step Reward", False, "index"),
                ("Rate/Simulations_Per_Sec", "Sims/sec", False, "buffer_size"),
                ("Loss/Policy", "Policy Loss", True, "global_step"),
                ("MCTS/Step_Visits", "MCTS Visits", False, "index"),
                ("Buffer/Size", "Buffer Size", False, "buffer_size"),
                ("Loss/Value", "Value Loss", True, "global_step"),
                ("MCTS/Step_Depth", "MCTS Depth", False, "index"),
                ("Rate/Steps_Per_Sec", "Steps/sec", False, "global_step"),
                ("LearningRate", "Learn Rate", True, "global_step"),
            ]

            weight_update_steps: list[int] = []
            if WEIGHT_UPDATE_METRIC_KEY in plot_data:
                dq = plot_data[WEIGHT_UPDATE_METRIC_KEY]
                if dq:
                    weight_update_steps = [step for step, _ in dq]

            max_x_per_axis: dict[int, int] = {}
            has_any_data_at_all = False
            debug_keys = {
                "Rate/Episodes_Per_Sec",
                "Rate/Simulations_Per_Sec",
                "Buffer/Size",
            }

            # Loop 1: Plot data and find max x-value PER AXIS
            for i, (conceptual_key, label, log_scale, x_axis_type) in enumerate(
                plot_defs
            ):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                ax.clear()
                ax.set_facecolor((0.15, 0.15, 0.15))

                max_x_for_plot = -1
                x_data: list[int] = []
                combined_values: list[float] = []
                x_label_text = "Index"  # Default label

                combined_steps_values: list[tuple[int, float]] = []
                dq = plot_data.get(conceptual_key, deque())

                if conceptual_key in debug_keys:
                    logger.debug(
                        f"[Plotter Debug] Key: {conceptual_key}, Deque Length: {len(dq)}"
                    )
                    if dq:
                        logger.debug(
                            f"[Plotter Debug]   Last 5 points: {list(dq)[-5:]}"
                        )

                if dq:
                    combined_steps_values = list(dq)
                    if combined_steps_values:
                        if x_axis_type == "global_step":
                            x_data = [s for s, v in combined_steps_values]
                            x_label_text = "Train Step"
                        elif x_axis_type == "buffer_size":
                            x_data = [s for s, v in combined_steps_values]
                            x_label_text = "Buffer Size"
                        else:  # Default to index
                            x_data = list(range(len(combined_steps_values)))
                            if (
                                "Score" in conceptual_key
                                or "Reward" in conceptual_key
                                or "MCTS" in conceptual_key
                            ):
                                x_label_text = "Game Step Index"
                            else:
                                x_label_text = "Data Point Index"

                        combined_values = [v for s, v in combined_steps_values]
                        max_x_for_plot = max(x_data) if x_data else -1
                        has_any_data_at_all = True

                max_x_per_axis[i] = max_x_for_plot

                color_mpl = self.colors.get(conceptual_key, (0.5, 0.5, 0.5))
                placeholder_color_mpl = self.colors.get("placeholder", (0.5, 0.5, 0.5))

                if not x_data or not combined_values:
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
                    if conceptual_key in debug_keys:
                        logger.debug(
                            f"[Plotter Debug]   Plotting {len(x_data)} points for {conceptual_key}."
                        )
                        logger.debug(
                            f"[Plotter Debug]     x_data (first 5): {x_data[:5]}"
                        )
                        logger.debug(
                            f"[Plotter Debug]     y_data (first 5): {combined_values[:5]}"
                        )
                        logger.debug(
                            f"[Plotter Debug]     x_data (last 5): {x_data[-5:]}"
                        )
                        logger.debug(
                            f"[Plotter Debug]     y_data (last 5): {combined_values[-5:]}"
                        )

                    ax.scatter(
                        x_data,
                        combined_values,
                        color=color_mpl,
                        alpha=0.1,
                        s=2,
                        label="_nolegend_",
                        zorder=2,
                    )

                    num_points = len(combined_values)
                    best_window = 0
                    for window in sorted(self.rolling_window_sizes, reverse=True):
                        if num_points >= window:
                            best_window = window
                            break

                    if best_window > 0:
                        rolling_avg = calculate_rolling_average(
                            combined_values, best_window
                        )
                        if len(rolling_avg) == len(x_data):
                            ax.plot(
                                x_data,
                                rolling_avg,
                                color=color_mpl,
                                alpha=0.9,
                                linewidth=1.5,
                                label=f"Avg {best_window}",
                                zorder=3,
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
                        min_val = min(combined_values) if combined_values else 0.0
                        max_val = max(combined_values) if combined_values else 0.0
                        val_range = max_val - min_val

                        if abs(val_range) < 1e-6:
                            if abs(min_val) < 1e-6 and abs(max_val) < 1e-6:
                                ylim_bottom = 0.0
                                ylim_top = 1.0
                            else:
                                center_val = (min_val + max_val) / 2.0
                                buffer = max(abs(center_val * 0.1), 0.5)
                                ylim_bottom = center_val - buffer
                                ylim_top = center_val + buffer
                        else:
                            buffer = val_range * 0.1
                            ylim_bottom = min_val - buffer
                            ylim_top = max_val + buffer

                        if all(v >= 0 for v in combined_values) and ylim_bottom < 0:
                            ylim_bottom = 0.0

                        if ylim_bottom >= ylim_top:
                            ylim_bottom = min_val - 0.5
                            ylim_top = max_val + 0.5
                            if ylim_bottom >= ylim_top:
                                ylim_bottom = 0.0
                                ylim_top = max(1.0, max_val)

                        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                    ax.xaxis.set_major_formatter(
                        FuncFormatter(
                            lambda x, _: (
                                f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"
                            )
                        )
                    )
                    ax.set_xlabel(x_label_text, fontsize=8, color="gray")
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda y, _: format_value(y))
                    )

                    current_val_str = format_value(combined_values[-1])
                    min_val_overall = min(combined_values)
                    max_val_overall = max(combined_values)
                    min_str = format_value(min_val_overall)
                    max_str = format_value(max_val_overall)
                    info_text = f"Min:{min_str} | Max:{max_str} | Cur:{current_val_str}"
                    ax.text(
                        1.0,
                        1.01,
                        info_text,
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=6,
                        color="white",
                    )

                # Common Axis Styling
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("gray")
                ax.spines["left"].set_color("gray")
                nrows, ncols = self.axes.shape
                if i < (nrows - 1) * ncols:
                    ax.set_xticklabels([])  # Hide tick labels only
                ax.tick_params(axis="x", rotation=0)

            # Loop 2: Apply final axis limits
            if has_any_data_at_all:
                logger.debug(
                    f"[Plotter] Applying individual xlims. Max x-values per axis: {max_x_per_axis}"
                )
                for i, ax in enumerate(axes_flat):
                    plot_max_x = max_x_per_axis.get(i, -1)
                    if plot_max_x >= 0:
                        effective_max_xlim = max(1, plot_max_x * 1.05)
                        ax.set_xlim(left=0, right=effective_max_xlim)

                        _, _, _, x_axis_type = plot_defs[i]
                        if weight_update_steps and x_axis_type == "global_step":
                            relevant_updates = [
                                step
                                for step in weight_update_steps
                                if step <= effective_max_xlim
                            ]
                            for update_step in relevant_updates:
                                ax.axvline(
                                    x=update_step,
                                    color="white",
                                    linestyle="--",
                                    linewidth=0.5,
                                    alpha=0.6,
                                    zorder=1,
                                )
                    else:
                        ax.set_xlim(left=0, right=1)

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
                buf, format="png", transparent=False, facecolor=self.fig.get_facecolor()
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
                    self.plot_surface_cache = None
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
            elif self.plot_surface_cache is None and self.fig:
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

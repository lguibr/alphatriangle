# File: alphatriangle/stats/plot_rendering.py
import logging
from collections import deque
from typing import TYPE_CHECKING  # Added cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..utils.types import StepInfo
from .plot_definitions import PlotDefinition
from .plot_utils import calculate_rolling_average, format_value

if TYPE_CHECKING:
    from .collector import StatsCollectorData

logger = logging.getLogger(__name__)


def _find_closest_index_for_global_step(
    target_global_step: int,
    step_info_list: list[StepInfo],
) -> int | None:
    """
    Finds the index in the step_info_list where the stored 'global_step'
    is closest to the target_global_step.
    Returns None if no suitable point is found or list is empty.
    """
    if not step_info_list:
        return None

    best_match_idx = None
    min_step_diff = float("inf")

    for i, step_info in enumerate(step_info_list):
        global_step_in_info = step_info.get("global_step")

        if global_step_in_info is not None:
            step_diff = abs(global_step_in_info - target_global_step)
            if step_diff < min_step_diff:
                min_step_diff = step_diff
                best_match_idx = i
            # Optimization: If we found an exact match, we can stop early
            # Also, if the steps start increasing again, we passed the best point
            if step_diff == 0 or (
                best_match_idx is not None and global_step_in_info > target_global_step
            ):
                break

    # Optional: Add a tolerance? If min_step_diff is too large, maybe don't return a match?
    # For now, return the index of the closest found value.
    return best_match_idx


def render_subplot(
    ax: plt.Axes,
    plot_data: "StatsCollectorData",
    plot_def: PlotDefinition,
    colors: dict[str, tuple[float, float, float]],
    rolling_window_sizes: list[int],
    weight_update_steps: list[int] | None = None,  # Global steps where updates happened
) -> bool:
    """
    Renders data for a single metric onto the given Matplotlib Axes object.
    Scatter point size/alpha decrease linearly as more data/longer averages are shown.
    Draws semi-transparent black, solid vertical lines for weight updates on all plots.
    Returns True if data was plotted, False otherwise.
    """
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    metric_key = plot_def.metric_key
    label = plot_def.label
    log_scale = plot_def.y_log_scale
    x_axis_type = plot_def.x_axis_type  # e.g., "global_step", "buffer_size", "index"

    x_data: list[int] = []
    y_data: list[float] = []
    x_label_text = "Index"  # Default label
    step_info_list: list[StepInfo] = []  # Store step info for mapping

    dq = plot_data.get(metric_key, deque())
    if dq:
        # Extract x-axis value and store StepInfo
        temp_x = []
        temp_y = []
        for i, (step_info, value) in enumerate(dq):
            x_val: int | None = None
            if x_axis_type == "global_step":
                x_val = step_info.get("global_step")
                x_label_text = "Train Step"
            elif x_axis_type == "buffer_size":
                x_val = step_info.get("buffer_size")
                x_label_text = "Buffer Size"
            else:  # index
                x_val = i  # Use the simple index 'i' as the x-value
                if (
                    "Score" in metric_key
                    or "Reward" in metric_key
                    or "MCTS" in metric_key
                ):
                    x_label_text = "Game Step Index"  # Label remains descriptive
                else:
                    x_label_text = "Data Point Index"

            if x_val is not None:
                temp_x.append(x_val)
                temp_y.append(value)
                step_info_list.append(
                    step_info
                )  # Keep StepInfo aligned with data points
            else:
                logger.warning(
                    f"Missing x-axis key '{x_axis_type}' in step_info for metric '{metric_key}'. Skipping point."
                )

        x_data = temp_x
        y_data = temp_y

    color_mpl = colors.get(metric_key, (0.5, 0.5, 0.5))
    placeholder_color_mpl = colors.get("placeholder", (0.5, 0.5, 0.5))

    data_plotted = False
    if not x_data or not y_data:
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
        data_plotted = True

        # Determine best rolling average window
        num_points = len(y_data)
        best_window = 0
        for window in sorted(rolling_window_sizes, reverse=True):
            if num_points >= window:
                best_window = window
                break

        # Determine scatter size/alpha based on best_window
        # Initialize as float
        scatter_size: float = 0.0
        scatter_alpha: float = 0.0
        max_scatter_size = 10.0  # Use float
        min_scatter_size = 1.0  # Use float
        max_scatter_alpha = 0.3
        min_scatter_alpha = 0.03
        max_window_for_interp = float(max(rolling_window_sizes))

        if best_window == 0:
            scatter_size = max_scatter_size
            scatter_alpha = max_scatter_alpha
        elif best_window >= max_window_for_interp:
            scatter_size = min_scatter_size
            scatter_alpha = min_scatter_alpha
        else:
            interp_fraction = best_window / max_window_for_interp
            # Cast result of np.interp to float
            scatter_size = float(
                np.interp(interp_fraction, [0, 1], [max_scatter_size, min_scatter_size])
            )
            scatter_alpha = float(
                np.interp(
                    interp_fraction, [0, 1], [max_scatter_alpha, min_scatter_alpha]
                )
            )

        # Plot raw data with dynamic size/alpha
        ax.scatter(
            x_data,
            y_data,
            color=color_mpl,
            alpha=scatter_alpha,
            s=scatter_size,  # Pass float size
            label="_nolegend_",
            zorder=2,
        )

        # Plot best rolling average
        if best_window > 0:
            rolling_avg = calculate_rolling_average(y_data, best_window)
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
                    fontsize=6, loc="upper right", frameon=False, labelcolor="lightgray"
                )
            else:
                logger.warning(
                    f"Length mismatch for rolling avg ({len(rolling_avg)}) vs x_data ({len(x_data)}) for {label}. Skipping avg plot."
                )

        # Draw vertical lines by mapping global_step to current x-axis value
        if weight_update_steps and step_info_list:
            plotted_line_x_values: set[float] = set()  # Store plotted x-values as float
            for update_global_step in weight_update_steps:
                x_index_for_line = _find_closest_index_for_global_step(
                    update_global_step, step_info_list
                )

                if x_index_for_line is not None and x_index_for_line < len(x_data):
                    actual_x_value: int | float
                    if x_axis_type == "index":
                        actual_x_value = x_index_for_line  # int
                    else:
                        # Explicitly cast list access to int to satisfy MyPy
                        actual_x_value = int(x_data[x_index_for_line])  # int

                    # Cast to float for axvline and check if already plotted
                    actual_x_float = float(actual_x_value)
                    if actual_x_float not in plotted_line_x_values:
                        ax.axvline(
                            x=actual_x_float,  # Pass float
                            color="black",
                            linestyle="-",
                            linewidth=0.7,
                            alpha=0.5,
                            zorder=10,
                        )
                        plotted_line_x_values.add(actual_x_float)
                else:
                    logger.debug(
                        f"Could not map global_step {update_global_step} to an index for plot '{label}'"
                    )

        # Formatting
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.tick_params(axis="both", which="major", labelsize=7, colors="lightgray")
        ax.grid(
            True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4), zorder=1
        )  # Ensure grid is behind lines

        # Set y-axis scale
        if log_scale:
            ax.set_yscale("log")
            min_val = min((v for v in y_data if v > 0), default=1e-6)
            max_val = max(y_data, default=1.0)
            ylim_bottom = max(1e-9, min_val * 0.1)
            ylim_top = max_val * 10
            if ylim_bottom < ylim_top:
                ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
            else:
                ax.set_ylim(bottom=1e-9, top=1.0)
        else:
            ax.set_yscale("linear")
            min_val = min(y_data) if y_data else 0.0
            max_val = max(y_data) if y_data else 0.0
            val_range = max_val - min_val
            if abs(val_range) < 1e-6:
                center_val = (min_val + max_val) / 2.0
                buffer = max(abs(center_val * 0.1), 0.5)
                ylim_bottom, ylim_top = center_val - buffer, center_val + buffer
            else:
                buffer = val_range * 0.1
                ylim_bottom, ylim_top = min_val - buffer, max_val + buffer
            if all(v >= 0 for v in y_data) and ylim_bottom < 0:
                ylim_bottom = 0.0
            if ylim_bottom >= ylim_top:
                ylim_bottom, ylim_top = min_val - 0.5, max_val + 0.5
                if ylim_bottom >= ylim_top:
                    ylim_bottom, ylim_top = 0.0, max(1.0, max_val)
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

        # Format x-axis
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"
            )
        )
        ax.set_xlabel(x_label_text, fontsize=8, color="gray")

        # Format y-axis
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_value(y)))

        # Add info text (min/max/current)
        current_val_str = format_value(y_data[-1])
        min_val_overall = min(y_data)
        max_val_overall = max(y_data)
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

    # Common Axis Styling (applied regardless of data presence)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")

    return data_plotted

# File: alphatriangle/stats/plot_rendering.py
import logging
from collections import deque
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .plot_definitions import PlotDefinition
from .plot_utils import calculate_rolling_average, format_value

if TYPE_CHECKING:
    from .collector import StatsCollectorData

logger = logging.getLogger(__name__)


def render_subplot(
    ax: plt.Axes,
    plot_data: "StatsCollectorData",
    plot_def: PlotDefinition,
    colors: dict[str, tuple[float, float, float]],
    rolling_window_sizes: list[int],
    weight_update_steps: list[int] | None = None,  # Added argument
) -> bool:
    """
    Renders data for a single metric onto the given Matplotlib Axes object.
    Draws black, solid vertical lines for weight updates on all plots if available,
    drawn on top of other data.
    Returns True if data was plotted, False otherwise.
    """
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    metric_key = plot_def.metric_key
    label = plot_def.label
    log_scale = plot_def.y_log_scale
    x_axis_type = plot_def.x_axis_type

    x_data: list[int] = []
    y_data: list[float] = []
    x_label_text = "Index"  # Default label

    dq = plot_data.get(metric_key, deque())
    if dq:
        combined_steps_values = list(dq)
        if combined_steps_values:
            if x_axis_type == "global_step":
                x_data = [s for s, v in combined_steps_values]
                x_label_text = "Train Step"
            elif x_axis_type == "buffer_size":
                # Use the step value (which represents buffer size for these metrics)
                x_data = [s for s, v in combined_steps_values]
                x_label_text = "Buffer Size"
            else:  # Default to index
                x_data = list(range(len(combined_steps_values)))
                if (
                    "Score" in metric_key
                    or "Reward" in metric_key
                    or "MCTS" in metric_key
                ):
                    x_label_text = "Game Step Index"
                else:
                    x_label_text = "Data Point Index"
            y_data = [v for s, v in combined_steps_values]

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

        # Plot raw data (thin, semi-transparent) - Lower zorder
        ax.scatter(
            x_data,
            y_data,
            color=color_mpl,
            alpha=0.1,
            s=2,
            label="_nolegend_",
            zorder=2,  # Draw above background/grid
        )

        # Plot best rolling average - Mid zorder
        num_points = len(y_data)
        best_window = 0
        for window in sorted(rolling_window_sizes, reverse=True):
            if num_points >= window:
                best_window = window
                break

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
                    zorder=3,  # Draw above scatter
                )
                ax.legend(
                    fontsize=6, loc="upper right", frameon=False, labelcolor="lightgray"
                )
            else:
                logger.warning(
                    f"Length mismatch for rolling avg ({len(rolling_avg)}) vs x_data ({len(x_data)}) for {label}. Skipping avg plot."
                )

        # --- CHANGED: Draw vertical lines if weight_update_steps exist, ON ALL PLOTS, ON TOP ---
        if weight_update_steps:
            # NOTE: These lines represent the global_step value when a weight update occurred.
            # On plots where the x-axis is NOT global_step (e.g., index, buffer_size),
            # the line's position relative to the x-axis values requires careful interpretation.
            # It indicates *when* the update happened in training time, not necessarily
            # at the corresponding x-value shown on that specific plot's axis.
            plot_max_x = max(x_data) if x_data else 0
            relevant_updates = weight_update_steps
            # Only filter if x-axis is global_step to avoid drawing lines way off-screen
            if x_axis_type == "global_step":
                relevant_updates = [
                    step for step in weight_update_steps if step <= plot_max_x * 1.05
                ]

            for update_step in relevant_updates:
                # Draw the line at the global_step value on the current axis
                ax.axvline(
                    x=update_step,
                    color="black",  # Black color
                    linestyle="-",  # Solid line
                    linewidth=0.7,  # Slightly thicker? Or keep 0.5? Let's try 0.7
                    alpha=0.8,  # Slightly transparent if needed, or 1.0 for solid
                    zorder=10,  # High zorder to draw on top
                )
        # --- END CHANGED ---

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

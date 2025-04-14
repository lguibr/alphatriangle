# File: src/stats/plot_utils.py
# File: src/stats/plot_utils.py
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Import normalize_color_for_matplotlib from the new location

logger = logging.getLogger(__name__)


def calculate_rolling_average(data: list[float], window: int) -> list[float]:
    """Calculates the rolling average with handling for edges."""
    if not data or window <= 0:
        return []
    if window > len(data):
        # If window is larger than data, return average of all data for all points
        avg = np.mean(data)
        # Cast to float explicitly
        return [float(avg)] * len(data)
    # Use convolution for efficient rolling average
    weights = np.ones(window) / window
    rolling_avg = np.convolve(data, weights, mode="valid")
    # Pad the beginning to match the original length
    padding = [float(np.mean(data[:i])) for i in range(1, window)]
    # Cast result to list of floats
    return padding + [float(x) for x in rolling_avg]


def calculate_trend_line(
    steps: list[int], values: list[float]
) -> tuple[list[int], list[float]]:
    """Calculates a simple linear trend line."""
    if len(steps) < 2:
        return [], []
    try:
        coeffs = np.polyfit(steps, values, 1)
        poly = np.poly1d(coeffs)
        trend_values = poly(steps)
        return steps, list(trend_values)
    except Exception as e:
        logger.warning(f"Could not calculate trend line: {e}")
        return [], []


def format_value(value: float) -> str:
    """Formats a float value for display on the plot."""
    if abs(value) < 1e-6:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}k"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def render_single_plot(
    ax: plt.Axes,
    steps: list[int],
    values: list[float],
    label: str,
    color: tuple[float, float, float],
    placeholder_color: tuple[float, float, float],
    rolling_window_sizes: list[int],
    show_placeholder: bool = False,
    placeholder_text: str | None = None,
    y_log_scale: bool = False,
):
    """Renders a single metric plot onto a Matplotlib Axes object."""
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    if show_placeholder or not steps or not values:
        text_to_display = placeholder_text if placeholder_text else "(No Data)"
        ax.text(
            0.5,
            0.5,
            text_to_display,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=placeholder_color,
            fontsize=9,
        )
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        return

    # Plot raw data (thin, semi-transparent)
    ax.plot(steps, values, color=color, alpha=0.3, linewidth=0.7, label="_nolegend_")

    # Plot rolling averages
    num_points = len(steps)
    plotted_rolling = False
    for i, window in enumerate(reversed(rolling_window_sizes)):
        if num_points >= window:
            rolling_avg = calculate_rolling_average(values, window)
            alpha = (
                0.6 + 0.4 * (i / (len(rolling_window_sizes) - 1))
                if len(rolling_window_sizes) > 1
                else 1.0
            )
            linewidth = (
                1.0 + 0.5 * (i / (len(rolling_window_sizes) - 1))
                if len(rolling_window_sizes) > 1
                else 1.5
            )
            ax.plot(
                steps,
                rolling_avg,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=f"Avg {window}" if not plotted_rolling else "_nolegend_",
            )
            plotted_rolling = True
            break  # Only plot the largest applicable rolling average

    # Plot trend line
    # trend_steps, trend_values = calculate_trend_line(steps, values)
    # if trend_steps:
    #     ax.plot(trend_steps, trend_values, color=color, linestyle='--', linewidth=1.0, alpha=0.8, label='Trend')

    # Formatting
    ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
    ax.tick_params(axis="both", which="major", labelsize=7, colors="lightgray")
    ax.grid(True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")

    # Set y-axis scale
    if y_log_scale:
        ax.set_yscale("log")
        # Ensure positive values for log scale, adjust limits if needed
        min_val = (
            min(v for v in values if v > 0) if any(v > 0 for v in values) else 1e-6
        )
        max_val = max(values) if values else 1.0
        ax.set_ylim(bottom=max(1e-9, min_val * 0.5), top=max_val * 1.5)
    else:
        ax.set_yscale("linear")

    # Format x-axis (steps)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(
        # Remove unused 'p' argument
        FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}")
    )
    ax.set_xlabel("Step", fontsize=8, color="gray")

    # Format y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # Remove unused 'p' argument
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_value(y)))

    # Add current value text
    current_val_str = format_value(values[-1])
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

    # Optional legend for rolling average
    # if plotted_rolling:
    #     ax.legend(fontsize=6, loc='upper right', frameon=False, labelcolor='lightgray')

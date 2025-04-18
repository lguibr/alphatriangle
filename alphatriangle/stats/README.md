# File: alphatriangle/stats/README.md
# Statistics Module (`alphatriangle.stats`)

## Purpose and Architecture

This module provides utilities for collecting, storing, and visualizing time-series statistics generated during the reinforcement learning training process using Matplotlib rendered onto Pygame surfaces.

-   **[`collector.py`](collector.py):** Defines the `StatsCollectorActor` class, a **Ray actor**. This actor uses dictionaries of `deque`s to store metric values (like losses, rewards, learning rate) associated with **step context information** ([`StepInfo`](../utils/types.py) dictionary containing `global_step`, `buffer_size`, etc.). It provides **remote methods** (`log`, `log_batch`) for asynchronous logging from multiple sources and methods (`get_data`, `get_metric_data`) for fetching the stored data. It supports limiting the history size and includes `get_state` and `set_state` methods for checkpointing.
-   **[`plot_definitions.py`](plot_definitions.py):** Defines the structure and properties of each plot in the dashboard (`PlotDefinition`, `PlotDefinitions`), including which step information (`x_axis_type`) should be used for the x-axis. **Also defines the `WEIGHT_UPDATE_METRIC_KEY` constant.**
-   **[`plot_rendering.py`](plot_rendering.py):** Contains the `render_subplot` function, responsible for drawing a single metric onto a Matplotlib `Axes` object based on a `PlotDefinition`. **It now accepts a list of `global_step` values where weight updates occurred and draws semi-transparent black, solid vertical lines on all plots by mapping the `global_step` to the corresponding value on the plot's specific x-axis. The raw data scatter points are now rendered with dynamically adjusted size and opacity, starting larger and fading as more data accumulates.**
-   **[`plot_utils.py`](plot_utils.py):** Contains helper functions for Matplotlib plotting, including calculating rolling averages and formatting values.
-   **[`plotter.py`](plotter.py):** Defines the `Plotter` class which manages the overall Matplotlib figure and axes.
    -   It orchestrates the plotting of multiple metrics onto a grid within the figure using `render_subplot`.
    -   It handles rendering the Matplotlib figure to an in-memory buffer and then converting it to a `pygame.Surface`.
    -   It implements caching logic.
    -   **It now extracts the weight update steps (`global_step` values) from the collected data and passes them to `render_subplot`.**

## Exposed Interfaces

-   **Classes:**
    -   `StatsCollectorActor`: Ray actor for collecting stats.
        -   `log.remote(metric_name: str, value: float, step_info: StepInfo)`
        -   `log_batch.remote(metrics: Dict[str, Tuple[float, StepInfo]])`
        -   `get_data.remote() -> StatsCollectorData`
        -   `get_metric_data.remote(metric_name: str) -> Optional[Deque[Tuple[StepInfo, float]]]`
        -   (Other methods: `clear`, `get_state`, `set_state`)
    -   `Plotter`:
        -   `get_plot_surface(plot_data: StatsCollectorData, target_width: int, target_height: int) -> Optional[pygame.Surface]`
    -   `PlotDefinitions`: Holds the layout and properties of all plots.
    -   `PlotDefinition`: NamedTuple defining a single plot.
-   **Types:**
    -   `StatsCollectorData`: Type alias `Dict[str, Deque[Tuple[StepInfo, float]]]` representing the stored data.
    -   `StepInfo`: TypedDict holding step context.
    -   `PlotType`: Alias for `PlotDefinition`.
-   **Functions:**
    -   `render_subplot`: Renders a single subplot, including mapped weight update lines and dynamic scatter points.
-   **Modules:**
    -   `plot_utils`: Contains helper functions.
-   **Constants:**
    -   `WEIGHT_UPDATE_METRIC_KEY`: Key used for logging/retrieving weight update events.

## Dependencies

-   **[`alphatriangle.visualization`](../visualization/README.md)**: `colors` (used indirectly via `Plotter`).
-   **[`alphatriangle.utils`](../utils/README.md)**: `helpers`, `types` (including `StepInfo`).
-   **`pygame`**: Used by `plotter.py` to create the final surface.
-   **`matplotlib`**: Used by `plotter.py`, `plot_rendering.py`, and `plot_utils.py` for generating plots.
-   **`numpy`**: Used by `plot_utils.py` and `plot_rendering.py` for calculations.
-   **`ray`**: Used by `StatsCollectorActor`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `math`, `time`, `io`, `contextlib`.

## Integration

-   The `TrainingLoop` ([`alphatriangle.training.loop`](../training/loop.py)) instantiates `StatsCollectorActor` and calls its remote `log` or `log_batch` methods, **passing `StepInfo` dictionaries**. It logs the `WEIGHT_UPDATE_METRIC_KEY` when worker weights are updated.
-   The `SelfPlayWorker` ([`alphatriangle.rl.self_play.worker`](../rl/self_play/worker.py)) calls `log_batch` **passing `StepInfo` dictionaries containing `game_step_index` and `global_step` (of its current weights).**
-   The `DashboardRenderer` ([`alphatriangle.visualization.core.dashboard_renderer`](../visualization/core/dashboard_renderer.py)) holds a handle to the `StatsCollectorActor` and calls `get_data.remote()` periodically to fetch data for plotting.
-   The `DashboardRenderer` instantiates `Plotter` and calls `get_plot_surface` using the fetched stats data and the target plot area dimensions. It then blits the returned surface.
-   The `DataManager` ([`alphatriangle.data.data_manager`](../data/data_manager.py)) interacts with the `StatsCollectorActor` via `get_state.remote()` and `set_state.remote()` during checkpoint saving and loading.

---

**Note:** Please keep this README updated when changing the data collection methods (especially the `StepInfo` structure), the plotting functions, or the way statistics are managed and displayed. Accurate documentation is crucial for maintainability.
# File: src/stats/README.md
# Statistics Module (`src.stats`)

## Purpose and Architecture

This module provides utilities for collecting, storing, and visualizing time-series statistics generated during the reinforcement learning training process using Matplotlib rendered onto Pygame surfaces.

-   **`collector.py`:** Defines the `StatsCollectorActor` class, a **Ray actor**. This actor uses dictionaries of `deque`s to store metric values (like losses, rewards, learning rate) associated with training steps or episodes. It provides **remote methods** (`log`, `log_batch`) for asynchronous logging from multiple sources (e.g., orchestrator, workers) and methods (`get_data`, `get_metric_data`) for fetching the stored data (e.g., by the visualizer). It supports limiting the history size for performance and memory management. It also includes `get_state` and `set_state` methods for checkpointing.
-   **`plot_utils.py`:** Contains helper functions for Matplotlib plotting, including calculating rolling averages, trend lines, formatting values, and the core `render_single_plot` function which draws a single metric onto a Matplotlib `Axes` object.
-   **`plotter.py`:** Defines the `Plotter` class which manages the overall Matplotlib figure and axes.
    -   It orchestrates the plotting of multiple metrics onto a grid within the figure using `render_single_plot`.
    -   It handles rendering the Matplotlib figure to an in-memory buffer and then converting it to a `pygame.Surface`.
    -   It implements caching logic to avoid regenerating the plot surface on every frame if the data or target size hasn't changed significantly, improving performance.

## Exposed Interfaces

-   **Classes:**
    -   `StatsCollectorActor`: Ray actor for collecting stats.
        -   `__init__(max_history: Optional[int] = 1000)`
        -   `log.remote(metric_name: str, value: float, step: int)`
        -   `log_batch.remote(metrics: Dict[str, Tuple[float, int]])`
        -   `get_data.remote() -> StatsCollectorData`
        -   `get_metric_data.remote(metric_name: str) -> Optional[Deque[Tuple[int, float]]]`
        -   `clear.remote()`
        -   `get_state.remote() -> Dict[str, Any]`
        -   `set_state.remote(state: Dict[str, Any])`
    -   `Plotter`:
        -   `__init__(plot_update_interval: float = 2.0)`
        -   `get_plot_surface(plot_data: StatsCollectorData, target_width: int, target_height: int) -> Optional[pygame.Surface]`
-   **Types:**
    -   `StatsCollectorData`: Type alias `Dict[str, Deque[Tuple[int, float]]]` representing the stored data.
-   **Modules:**
    -   `plot_utils`: Contains helper functions like `render_single_plot`.

## Dependencies

-   **`src.visualization.core.colors`**: Used by `plot_utils.py`.
-   **`pygame`**: Used by `plotter.py` to create the final surface.
-   **`matplotlib`**: Used by `plotter.py` and `plot_utils.py` for generating plots.
-   **`numpy`**: Used by `plot_utils.py` for calculations.
-   **`ray`**: Used by `StatsCollectorActor`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `math`, `time`, `io`.

## Integration

-   The `TrainingOrchestrator` (`src.rl.core.orchestrator`) instantiates `StatsCollectorActor` and calls its remote `log` or `log_batch` methods.
-   The `GameRenderer` (`src.visualization.core.game_renderer`) holds a handle to the `StatsCollectorActor` and calls `get_data.remote()` periodically to fetch data for plotting.
-   The `GameRenderer` instantiates `Plotter` and calls `get_plot_surface` using the fetched stats data and the target plot area dimensions. It then blits the returned surface.
-   The `DataManager` interacts with the `StatsCollectorActor` via `get_state.remote()` and `set_state.remote()` during checkpoint saving and loading.

---

**Note:** Please keep this README updated when changing the data collection methods, the plotting functions, or the way statistics are managed and displayed, especially regarding the actor-based approach.

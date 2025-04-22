

# Statistics Module (`alphatriangle.stats`)

## Purpose and Architecture

This module provides utilities for collecting, processing, and logging time-series statistics generated during the reinforcement learning training process. It aims for asynchronous collection and configurable, centralized processing and logging.

-   **Configuration ([`alphatriangle.config.stats_config`](../config/stats_config.py)):** A dedicated Pydantic model (`StatsConfig`) defines *which* metrics to track, their *source*, *how* they should be aggregated (mean, sum, rate, latest, etc.), *how often* they should be logged (based on steps or time), and *where* they should be logged (MLflow, TensorBoard, console). **It includes metrics for losses, learning rate, episode outcomes (score, length, triangles cleared), MCTS stats, buffer size, system info (workers, tasks), progress (simulations, episodes, weight updates), and rates.**
-   **Types ([`stats_types.py`](stats_types.py)):** Defines Pydantic models for data structures:
    -   `RawMetricEvent`: Represents a single raw data point sent to the collector, tagged with its name, value, `global_step`, and optional context.
    -   `LogContext`: Contains timing and state information passed to the processor during logging cycles.
-   **Collector ([`collector.py`](collector.py)):** Defines the `StatsCollectorActor` class, a **Ray actor**.
    -   Its primary role is to asynchronously receive `RawMetricEvent` objects via `log_event` or `log_batch_events`.
    -   It buffers these raw events internally, grouped by `global_step`.
    -   It periodically triggers (or is triggered by the `TrainingLoop`) the processing and logging logic.
    -   It instantiates and uses a `StatsProcessor` to handle the aggregation and logging.
    -   It still handles tracking the latest `GameState` from workers for debugging/inspection purposes.
    -   Includes `get_state` and `set_state` for checkpointing minimal necessary state (like the last processed step/time).
-   **Processor ([`processor.py`](processor.py)):** Contains the `StatsProcessor` class.
    -   Takes buffered raw data and the `StatsConfig`.
    -   Aggregates raw values for each metric based on its configured `aggregation` method (mean, sum, rate, etc.). **It can extract values from the `RawMetricEvent` context dictionary based on `MetricConfig.context_key`.**
    -   Calculates rates based on event counts and time deltas.
    -   Determines if a metric should be logged based on `log_frequency_steps` or `log_frequency_seconds`.
    -   Handles the actual logging calls to MLflow and TensorBoard, ensuring the correct `global_step` is used as the x-axis.
    -   **Important:** The `StatsProcessor` is responsible for *logging* processed metrics to external tracking systems (MLflow, TensorBoard). It does **not** save general training artifacts like model checkpoints or replay buffers; that responsibility lies with the [`DataManager`](../data/README.md).

## Exposed Interfaces

-   **Classes:**
    -   `StatsCollectorActor`: Ray actor for collecting stats.
        -   `log_event.remote(event: RawMetricEvent)`
        -   `log_batch_events.remote(events: List[RawMetricEvent])`
        -   `process_and_log.remote(current_global_step: int)`: Triggers processing and logging.
        -   `update_worker_game_state.remote(...)`
        -   `get_latest_worker_states.remote() -> Dict[int, GameState]`
        -   (Other methods: `get_state`, `set_state`, `close_tb_writer`)
    -   `StatsProcessor`: Handles aggregation and logging logic (used internally by actor).
-   **Types (from `stats_types.py`):**
    -   `StatsConfig`, `MetricConfig`, `RawMetricEvent`, `LogContext`.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `StatsConfig`.
-   **[`alphatriangle.utils`](../utils/README.md)**: General utilities.
-   **`trianglengin`**: `GameState`.
-   **`ray`**: Used by `StatsCollectorActor`.
-   **`mlflow`**: Used by `StatsProcessor` for logging.
-   **`torch.utils.tensorboard`**: Used by `StatsProcessor` for logging.
-   **`numpy`**: Used for aggregation in `StatsProcessor`.
-   **Standard Libraries:** `typing`, `logging`, `collections`, `time`, `threading`.

## Integration

-   The `TrainingLoop` ([`alphatriangle.training.loop`](../training/loop.py)) instantiates `StatsCollectorActor` (passing `StatsConfig`) and calls its remote `log_event`/`log_batch_events` methods with `RawMetricEvent` objects (e.g., for losses, buffer size, **total weight updates**). It periodically calls `process_and_log.remote()`.
-   The `SelfPlayWorker` ([`alphatriangle.rl.self_play.worker`](../rl/self_play/worker.py)) sends `RawMetricEvent` objects for things like step rewards, MCTS simulations, and episode completion (including score, length, **total triangles cleared** in context).
-   The `Trainer` ([`alphatriangle.rl.core.trainer`](../rl/core/trainer.py)) sends `RawMetricEvent` objects for loss components (e.g., `Loss/Policy`, `Loss/Value`, `Loss/Mean_Abs_TD_Error`).
-   The `DataManager` ([`alphatriangle.data.data_manager`](../data/data_manager.py)) interacts with the `StatsCollectorActor` via `get_state.remote()` and `set_state.remote()` during checkpoint saving and loading.

---

**Note:** Please keep this README updated when changing the statistics configuration, event structures, or the processing/logging logic. Accurate documentation is crucial for maintainability.

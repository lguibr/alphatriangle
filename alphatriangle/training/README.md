
# Training Module (`alphatriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the **headless** reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **`setup.py`:** Contains `setup_training_components` which initializes Ray, detects resources, **adjusts the requested worker count based on available CPU cores (reserving some for stability),** loads configurations (including `trianglengin.EnvConfig`, `AlphaTriangleMCTSConfig`, and **`StatsConfig`**), creates the `trimcts.SearchConfiguration`, initializes the `StatsCollectorActor` (passing `StatsConfig`), and bundles the core components (`TrainingComponents`).
-   **`components.py`:** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs including `trimcts.SearchConfiguration` and **`StatsConfig`**) required by the `TrainingLoop`.
-   **`loop.py`:** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, passing the current `global_step` to the workers.
    -   **Sending raw metric events (`RawMetricEvent`) to the `StatsCollectorActor` for various occurrences (e.g., training step losses, episode completion, buffer size changes, weight updates).**
    -   **Periodically triggering the `StatsCollectorActor` to process and log the collected raw metrics based on the `StatsConfig`.**
    -   Logging simple progress strings to the console.
    -   Handling stop requests.
-   **`worker_manager.py`:** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. It passes the `trimcts.SearchConfiguration` and the `run_base_dir` to workers during initialization.
-   **`loop_helpers.py`:** Contains simplified helper functions used by `TrainingLoop`, primarily for formatting console progress/ETA strings and validating experiences. **Direct logging responsibilities have been removed.**
-   **`runner.py`:** Contains the top-level logic for running the headless training pipeline. It handles argument parsing (via CLI), setup logging, calls `setup_training_components`, loads initial state, runs the `TrainingLoop`, and manages overall cleanup (MLflow, TensorBoard, Ray shutdown, **closing the stats actor's TB writer**).
-   **`runners.py`:** Re-exports the main entry point function (`run_training`) from `runner.py`.
-   **`logging_utils.py`:** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`runner`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify. Statistics collection is now handled asynchronously by the `StatsCollectorActor`, which processes and logs data according to the flexible `StatsConfig`.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingLoop`: Contains the core async loop logic.
    -   `TrainingComponents`: Dataclass holding initialized components.
    -   `WorkerManager`: Manages worker actors.
    -   `LoopHelpers`: Provides simplified helper functions for the loop.
-   **Functions (from `runners.py`):**
    -   `run_training(...) -> int`
-   **Functions (from `setup.py`):**
    -   `setup_training_components(...) -> Tuple[Optional[TrainingComponents], bool]`
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `AlphaTriangleMCTSConfig`, `ModelConfig`, `PersistenceConfig`, `TrainConfig`, **`StatsConfig`**.
-   **`trianglengin`**: `GameState`, `EnvConfig`.
-   **`trimcts`**: `SearchConfiguration`.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.rl`](../rl/README.md)**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **[`alphatriangle.data`](../data/README.md)**: `DataManager`, `LoadedTrainingState`.
-   **[`alphatriangle.stats`](../stats/README.md)**: `StatsCollectorActor`, **`RawMetricEvent`**.
-   **[`alphatriangle.utils`](../utils/README.md)**: Helper functions and types.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **`torch.utils.tensorboard`**: For TensorBoard logging.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`, `pathlib`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the new statistics collection and logging mechanism. Accurate documentation is crucial for maintainability.

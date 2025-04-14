# File: src/training/README.md
# Training Module (`src.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **`pipeline.py`:** Defines the `TrainingPipeline` class. This is the main entry point for starting a training run. It handles:
    -   Initialization of Ray and MLflow.
    -   Loading configurations.
    -   Instantiating core components (`TrainingComponents`).
    -   Loading initial state (checkpoints, buffers) via `DataManager`.
    -   Creating and running the `TrainingLoop`.
    -   Saving the final state.
    -   Overall error handling and cleanup (Ray shutdown, MLflow run termination).
-   **`loop.py`:** Defines the `TrainingLoop` class (refactored from the old `TrainingOrchestrator`). This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **`components.py`:** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **`logging_utils.py`:** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`pipeline`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingPipeline`: Main class to run the training process.
        -   `__init__(components: TrainingComponents, visual_mode: bool, visual_state_queue: Optional[queue.Queue])`
        -   `run()`
        -   `request_stop()`
        -   `cleanup()`
    -   `TrainingLoop`: Contains the core async loop logic (primarily used internally by `TrainingPipeline`).
        -   `__init__(components: TrainingComponents, visual_state_queue: Optional[queue.Queue])`
        -   `run()`
        -   `request_stop()`
        -   `cleanup_actors()`
        -   `set_initial_state(...)`
        -   `initialize_workers()`
    -   `TrainingComponents`: Dataclass holding initialized components.
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`
    -   `log_metrics_to_mlflow(...)`

## Dependencies

-   **`src.config`**: All configuration classes.
-   **`src.nn`**: `NeuralNetwork`.
-   **`src.rl`**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **`src.data`**: `DataManager`, `LoadedTrainingState`.
-   **`src.stats`**: `StatsCollectorActor`.
-   **`src.environment`**: `GameState`.
-   **`src.utils`**: Helper functions and types.
-   **`src.visualization`**: `ProgressBar`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the `TrainingPipeline` or `TrainingLoop`, or the way components interact.
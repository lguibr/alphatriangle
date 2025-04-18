# File: alphatriangle/training/README.md
# Training Module (`alphatriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **`setup.py`:** Contains `setup_training_components` which initializes Ray, detects resources, adjusts worker counts, loads configurations, and creates the core components bundle (`TrainingComponents`).
-   **`components.py`:** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **`loop.py`:** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, **passing the current `global_step` to the workers**, and logging a special event (`Internal/Weight_Update_Step`) with the `global_step` to the `StatsCollectorActor` when updates occur.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **`worker_manager.py`:** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. **It now passes the `global_step` to workers when updating weights.**
-   **`loop_helpers.py`:** Contains helper functions used by `TrainingLoop` for tasks like logging rates, updating the visual queue, validating experiences, and logging results. **It constructs the `StepInfo` dictionary containing relevant step counters (`global_step`, `buffer_size`) for logging.** It also includes logic to log the weight update event.
-   **`runners.py`:** Re-exports the main entry point functions (`run_training_headless_mode`, `run_training_visual_mode`) from their respective modules.
-   **`headless_runner.py` / `visual_runner.py`:** Contain the top-level logic for running training in either headless or visual mode. They handle argument parsing (via CLI), setup logging, call `setup_training_components`, load initial state, run the `TrainingLoop` (potentially in a separate thread for visual mode), handle visualization setup (visual mode), and manage overall cleanup (MLflow, Ray shutdown).
-   **`logging_utils.py`:** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`headless_runner`/`visual_runner`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingLoop`: Contains the core async loop logic.
    -   `TrainingComponents`: Dataclass holding initialized components.
    -   `WorkerManager`: Manages worker actors.
    -   `LoopHelpers`: Provides helper functions for the loop.
-   **Functions (from `runners.py`):**
    -   `run_training_headless_mode(...) -> int`
    -   `run_training_visual_mode(...) -> int`
-   **Functions (from `setup.py`):**
    -   `setup_training_components(...) -> Tuple[Optional[TrainingComponents], bool]`
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **`alphatriangle.config`**: All configuration classes.
-   **`alphatriangle.nn`**: `NeuralNetwork`.
-   **`alphatriangle.rl`**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **`alphatriangle.data`**: `DataManager`, `LoadedTrainingState`.
-   **`alphatriangle.stats`**: `StatsCollectorActor`, `PlotDefinitions`.
-   **`alphatriangle.environment`**: `GameState`.
-   **`alphatriangle.utils`**: Helper functions and types (including `StepInfo`).
-   **`alphatriangle.visualization`**: `ProgressBar`, `DashboardRenderer`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the logging of step context information (`StepInfo`) and worker weight updates.
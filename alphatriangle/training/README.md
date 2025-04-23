# Training Module (`alphatriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the **headless** reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class. **It now leverages the `trieye` library for asynchronous statistics collection and data persistence.**

-   **`setup.py`:** Contains `setup_training_components` which initializes Ray, detects resources, **adjusts the requested worker count based on available CPU cores (reserving some for stability),** loads configurations (including `trianglengin.EnvConfig`, `AlphaTriangleMCTSConfig`), creates the `trimcts.SearchConfiguration`, **initializes the `TrieyeActor` (passing `TrieyeConfig`)**, and bundles the core components (`TrainingComponents`).
-   **`components.py`:** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, **`TrieyeActor` handle**, Configs including `trimcts.SearchConfiguration`) required by the `TrainingLoop`.
-   **`loop.py`:** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, passing the current `global_step` to the workers.
    -   **Sending raw metric events (`RawMetricEvent` from `trieye`) to the `TrieyeActor` for various occurrences (e.g., training step losses, buffer size changes, total weight updates).**
    -   **Processing results from workers and sending `episode_end` events with context (score, length, triangles cleared) to the `TrieyeActor`.**
    -   **Triggering the `TrieyeActor` to save checkpoints and buffers.**
    -   Logging simple progress strings to the console.
    -   Handling stop requests.
-   **`worker_manager.py`:** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. It passes the `trimcts.SearchConfiguration`, the **`TrieyeActor` name**, and the **run base directory path** (within `.trieye_data/<app_name>/runs/`) to workers during initialization.
-   **`loop_helpers.py`:** Contains simplified helper functions used by `TrainingLoop`, primarily for formatting console progress/ETA strings and validating experiences.
-   **`runner.py`:** Contains the top-level logic for running the headless training pipeline. It handles argument parsing (via CLI), sets up console logging, calls `setup_training_components` (which starts the `TrieyeActor`), runs the `TrainingLoop`, and manages overall cleanup (**including `TrieyeActor` shutdown**).
-   **`runners.py`:** Re-exports the main entry point function (`run_training`) from `runner.py`.
-   **`logging_utils.py`:** Contains helper functions for logging configurations/metrics to MLflow (now primarily used for logging AlphaTriangle-specific configs, as `trieye` handles its own config logging).

This structure separates the high-level setup/teardown (`runner`) from the core iterative logic (`loop`), making the system more modular. Statistics collection and persistence are now handled asynchronously by the `TrieyeActor`, configured via `TrieyeConfig`.

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
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `AlphaTriangleMCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **`trianglengin`**: `GameState`, `EnvConfig`.
-   **`trimcts`**: `SearchConfiguration`.
-   **`trieye`**: `TrieyeConfig`, `TrieyeActor`, `RawMetricEvent`, `LoadedTrainingState`.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.rl`](../rl/README.md)**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **[`alphatriangle.utils`](../utils/README.md)**: Helper functions and types.
-   **`ray`**: For parallelism and actors.
-   **`mlflow`**: Used by `trieye`.
-   **`torch`**: For neural network operations.
-   **`torch.utils.tensorboard`**: Used by `trieye`.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`, `pathlib`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the interaction with the `trieye` library. Accurate documentation is crucial for maintainability.
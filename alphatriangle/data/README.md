# File: alphatriangle/data/README.md

# Data Management Module (`alphatriangle.data`)

## Purpose and Architecture

This module is responsible for handling the persistence of training artifacts using structured data schemas defined with Pydantic. It manages:

-   Neural network checkpoints (model weights, optimizer state).
-   Experience replay buffers.
-   Statistics collector state.
-   Run configuration files.

All data is stored within the `.alphatriangle_data` directory at the project root. The core component is the [`DataManager`](data_manager.py) class, which centralizes file path management and saving/loading logic based on the [`PersistenceConfig`](../config/persistence_config.py) and [`TrainConfig`](../config/train_config.py). It uses `cloudpickle` for robust serialization of complex Python objects, including Pydantic models containing tensors and deques.

-   **Schemas ([`schemas.py`](schemas.py)):** Defines Pydantic models (`CheckpointData`, `BufferData`, `LoadedTrainingState`) to structure the data being saved and loaded, ensuring clarity and enabling validation.
-   **Path Management ([`path_manager.py`](path_manager.py)):** The `PathManager` class handles constructing file paths within `.alphatriangle_data`, creating directories, and finding previous runs.
-   **Serialization ([`serializer.py`](serializer.py)):** The `Serializer` class handles the actual reading/writing of files using `cloudpickle` and JSON, including validation during loading.
-   **Centralization:** `DataManager` provides a single point of control for saving/loading operations.
-   **Configuration-Driven:** Uses `PersistenceConfig` and `TrainConfig` to determine save locations, filenames, and loading behavior (e.g., auto-resume).
-   **Run Management:** Organizes saved artifacts into subdirectories within `.alphatriangle_data/runs/<run_name>/`.
-   **State Loading:** `DataManager.load_initial_state` determines the correct files, deserializes them, validates the structure, and returns a `LoadedTrainingState` object.
-   **State Saving:** `DataManager.save_training_state` assembles data into Pydantic models, serializes them, and saves to files within the run directory.
-   **MLflow Integration:** Logs saved artifacts (checkpoints, buffers, configs) to the corresponding MLflow run (located in `.alphatriangle_data/mlruns/`) after successful local saving. Checkpoints and buffers are logged with relative paths (e.g., `checkpoints/checkpoint_step_1000.pkl`). **The run configuration JSON is logged to the root artifact directory using its filename (e.g., `configs.json`).**
-   **Buffer Content:** The saved buffer file (`buffer.pkl`) contains `Experience` tuples `(StateType, PolicyTargetMapping, n_step_return)`. The `StateType` contains processed numerical features (`grid`, `other_features`). It does **not** contain raw `GameState` objects or action sequences for full interactive replay.

## Exposed Interfaces

-   **Classes:**
    -   `DataManager`: Orchestrates saving and loading.
        -   `__init__(persist_config: PersistenceConfig, train_config: TrainConfig)`
        -   `load_initial_state() -> LoadedTrainingState`: Loads state, returns Pydantic model.
        -   `save_training_state(...)`: Saves state using Pydantic models and cloudpickle, logs to MLflow.
        -   `save_run_config(configs: Dict[str, Any])`: Saves config JSON locally and logs to MLflow.
        -   `get_checkpoint_path(...) -> Path`
        -   `get_buffer_path(...) -> Path`
    -   `PathManager`: Manages file paths within `.alphatriangle_data`.
    -   `Serializer`: Handles serialization/deserialization.
    -   `CheckpointData` (from `schemas.py`): Pydantic model for checkpoint structure.
    -   `BufferData` (from `schemas.py`): Pydantic model for buffer structure.
    -   `LoadedTrainingState` (from `schemas.py`): Pydantic model wrapping loaded data.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `PersistenceConfig`, `TrainConfig`.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.rl`](../rl/README.md)**: `ExperienceBuffer`.
-   **[`alphatriangle.stats`](../stats/README.md)**: `StatsCollectorActor`.
-   **[`alphatriangle.utils`](../utils/README.md)**: `Experience`, `StateType`.
-   **`torch.optim`**: `Optimizer`.
-   **Standard Libraries:** `os`, `shutil`, `logging`, `glob`, `re`, `json`, `collections.deque`, `pathlib`, `datetime`.
-   **Third-Party:** `pydantic`, `cloudpickle`, `torch`, `ray`, `mlflow`, `numpy`.

---

**Note:** Please keep this README updated when changing the Pydantic schemas, the types of artifacts managed, the saving/loading mechanisms, or the responsibilities of the `DataManager`, `PathManager`, or `Serializer`.
# File: src/data/README.md
# Data Management Module (`src.data`)

## Purpose and Architecture

This module is responsible for handling the persistence of training artifacts using structured data schemas defined with Pydantic. It manages:

-   Neural network checkpoints (model weights, optimizer state).
-   Experience replay buffers.
-   Statistics collector state.
-   Run configuration files.

The core component is the `DataManager` class, which centralizes file path management and saving/loading logic based on the `PersistenceConfig` and `TrainConfig`. It uses `cloudpickle` for robust serialization of complex Python objects, including Pydantic models containing tensors and deques.

-   **Schemas (`schemas.py`):** Defines Pydantic models (`CheckpointData`, `BufferData`, `LoadedTrainingState`) to structure the data being saved and loaded, ensuring clarity and enabling validation.
-   **Centralization:** Provides a single point of control for saving/loading operations.
-   **Configuration-Driven:** Uses `PersistenceConfig` and `TrainConfig` to determine save locations, filenames, and loading behavior (e.g., auto-resume).
-   **Serialization:** Uses `cloudpickle` to serialize/deserialize the Pydantic model instances, which effectively handles nested complex objects like tensors and deques within the models.
-   **Run Management:** Organizes saved artifacts into subdirectories based on the `RUN_NAME`.
-   **State Loading:** Provides `load_initial_state` to determine the correct files, deserialize them using `cloudpickle`, validate the structure with Pydantic models, and return a `LoadedTrainingState` object.
-   **State Saving:** Provides `save_training_state` to assemble data into Pydantic models (`CheckpointData`, `BufferData`), serialize them using `cloudpickle`, and save to files.
-   **MLflow Integration:** Logs saved artifacts (checkpoints, buffers, configs) to MLflow after successful local saving.

## Exposed Interfaces

-   **Classes:**
    -   `DataManager`:
        -   `__init__(persist_config: PersistenceConfig, train_config: TrainConfig)`
        -   `load_initial_state() -> LoadedTrainingState`: Loads state, returns Pydantic model.
        -   `save_training_state(...)`: Saves state using Pydantic models and cloudpickle.
        -   `save_run_config(configs: Dict[str, Any])`: Saves config JSON.
        -   `get_checkpoint_path(...) -> str`
        -   `get_buffer_path(...) -> str`
        -   `find_latest_run_dir(...) -> Optional[str]`
    -   `CheckpointData` (from `schemas.py`): Pydantic model for checkpoint structure.
    -   `BufferData` (from `schemas.py`): Pydantic model for buffer structure.
    -   `LoadedTrainingState` (from `schemas.py`): Pydantic model wrapping loaded data.

## Dependencies

-   **`src.config`**: `PersistenceConfig`, `TrainConfig`.
-   **`src.nn`**: `NeuralNetwork`.
-   **`src.rl.core.buffer`**: `ExperienceBuffer`.
-   **`src.stats`**: `StatsCollectorActor`.
-   **`src.utils.types`**: `Experience`.
-   **`torch.optim`**: `Optimizer`.
-   **Standard Libraries:** `os`, `shutil`, `logging`, `glob`, `re`, `json`, `collections.deque`.
-   **Third-Party:** `pydantic`, `cloudpickle`, `torch`, `ray`, `mlflow`.

---

**Note:** Please keep this README updated when changing the Pydantic schemas, the types of artifacts managed, the saving/loading mechanisms, or the responsibilities of the `DataManager`.
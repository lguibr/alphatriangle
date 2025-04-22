
# Configuration Module (`alphatriangle.config`)

## Purpose and Architecture

This module centralizes all configuration parameters for the AlphaTriangle project *except* for the core environment settings. It uses separate **Pydantic models** for different aspects of the application (model, training, persistence, MCTS, **statistics**) to promote modularity, clarity, and automatic validation.

**Core environment configuration (`EnvConfig`) is now defined and imported directly from the `trianglengin` library.**

-   **Modularity:** Separating configurations makes it easier to manage parameters for different components.
-   **Type Safety & Validation:** Using Pydantic models (`BaseModel`) provides strong type hinting, automatic parsing, and validation of configuration values based on defined types and constraints (e.g., `Field(gt=0)`).
-   **Validation Script:** The [`validation.py`](validation.py) script instantiates all configuration models (including importing and validating `trianglengin.EnvConfig`), triggering Pydantic's validation, and prints a summary.
-   **Dynamic Defaults:** Some configurations, like `RUN_NAME` in `TrainConfig`, use `default_factory` for dynamic defaults (e.g., timestamp).
-   **Computed Fields:** Properties like `MLFLOW_TRACKING_URI` in `PersistenceConfig` are defined using `@computed_field` for clarity.
-   **Tuned Defaults:** The default values in `TrainConfig` and `ModelConfig` are tuned for substantial learning runs. `AlphaTriangleMCTSConfig` defaults to 128 simulations. `StatsConfig` defines a default set of metrics to track.
-   **Data Paths:** `PersistenceConfig` defines the structure within the `.alphatriangle_data` directory where all local artifacts (runs, checkpoints, logs, TensorBoard data) and MLflow data (`mlruns`) are stored.

## Exposed Interfaces

-   **Pydantic Models:**
    -   `EnvConfig` (Imported from `trianglengin`): Environment parameters (grid size, shapes, rewards).
    -   [`ModelConfig`](model_config.py): Neural network architecture parameters.
    -   [`TrainConfig`](train_config.py): Training loop hyperparameters (batch size, learning rate, workers, PER settings, etc.).
    -   [`PersistenceConfig`](persistence_config.py): Data saving/loading parameters (directories within `.alphatriangle_data`, filenames).
    -   [`AlphaTriangleMCTSConfig`](mcts_config.py): MCTS parameters (simulations, exploration constants, temperature).
    -   [`StatsConfig`](stats_config.py): Statistics collection and logging parameters (metrics, aggregation, frequency).
-   **Constants:**
    -   [`APP_NAME`](app_config.py): The name of the application.
-   **Functions:**
    -   `print_config_info_and_validate(mcts_config_instance: AlphaTriangleMCTSConfig | None)`: Validates and prints a summary of all configurations.

## Dependencies

This module primarily defines configurations and relies heavily on **Pydantic**.

-   **`pydantic`**: The core library used for defining models and validation.
-   **`trianglengin`**: Imports `EnvConfig`.
-   **`trimcts`**: Used by `AlphaTriangleMCTSConfig` (implicitly, as it mirrors the structure).
-   **Standard Libraries:** `typing`, `time`, `os`, `logging`, `pathlib`.

---

**Note:** Please keep this README updated when adding, removing, or significantly modifying configuration parameters or the structure of the Pydantic models. Accurate documentation is crucial for maintainability.
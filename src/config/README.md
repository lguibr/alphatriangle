# File: src/config/README.md
# Configuration Module (`src.config`)

## Purpose and Architecture

This module centralizes all configuration parameters for the AlphaTriangle project. It uses separate **Pydantic models** for different aspects of the application (environment, model, training, visualization, persistence) to promote modularity, clarity, and automatic validation.

-   **Modularity:** Separating configurations makes it easier to manage parameters for different components.
-   **Type Safety & Validation:** Using Pydantic models (`BaseModel`) provides strong type hinting, automatic parsing, and validation of configuration values based on defined types and constraints (e.g., `Field(gt=0)`).
-   **Validation Script:** The `validation.py` script instantiates all configuration models, triggering Pydantic's validation, and prints a summary.
-   **Dynamic Defaults:** Some configurations, like `RUN_NAME` in `TrainConfig`, use `default_factory` for dynamic defaults (e.g., timestamp).
-   **Computed Fields:** Properties like `ACTION_DIM` in `EnvConfig` or `MLFLOW_TRACKING_URI` in `PersistenceConfig` are defined using `@computed_field` for clarity.

## Exposed Interfaces

-   **Pydantic Models:**
    -   `EnvConfig`: Environment parameters (grid size, shapes).
    -   `ModelConfig`: Neural network architecture parameters.
    -   `TrainConfig`: Training loop hyperparameters (batch size, learning rate, workers, **PER settings**, etc.).
    -   `VisConfig`: Visualization parameters (screen size, FPS, layout).
    -   `PersistenceConfig`: Data saving/loading parameters (directories, filenames).
    -   `MCTSConfig`: MCTS parameters (simulations, exploration constants, temperature). - *Note: Defined in `src.mcts.core.config` but often used alongside other configs.*
-   **Constants:**
    -   `APP_NAME`: The name of the application.
-   **Functions:**
    -   `print_config_info_and_validate(mcts_config_instance: MCTSConfig)`: Validates and prints a summary of all configurations by instantiating the Pydantic models.

## Dependencies

This module primarily defines configurations and relies heavily on **Pydantic**.

-   **`pydantic`**: The core library used for defining models and validation.
-   **`src.mcts.core.config`**: The `validation.py` script imports `MCTSConfig` from this location.
-   **Standard Libraries:** `typing`, `time`, `os`.

---

**Note:** Please keep this README updated when adding, removing, or significantly modifying configuration parameters or the structure of the Pydantic models. Accurate documentation is crucial for maintainability.

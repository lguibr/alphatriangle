# File: src/features/README.md
# Feature Extraction Module (`src.features`)

## Purpose and Architecture

This module is solely responsible for converting raw `GameState` objects from the `src.environment` module into numerical representations (features) suitable for input into the neural network (`src.nn`). It acts as a bridge between the game's internal state and the requirements of the machine learning model.

-   **Decoupling:** This module completely decouples feature engineering from the core game environment logic. The `environment` module focuses only on game rules and state transitions, while this module handles the transformation for the NN.
-   **Feature Engineering:**
    -   `extractor.py`: Contains the `GameStateFeatures` class and the main `extract_state_features` function. This orchestrates the extraction process, calling helper functions to generate different feature types. It uses `Triangle` and `Shape` from `src.structs`.
    -   `grid_features.py`: Contains low-level, potentially performance-optimized (e.g., using Numba) functions for calculating specific scalar metrics derived from the grid state (like column heights, holes, bumpiness).
-   **Output Format:** The `extract_state_features` function returns a `StateType` (a `TypedDict` defined in `src.utils.types` containing `grid` and `other_features` numpy arrays), which is the standard input format expected by the `NeuralNetwork` interface.
-   **Configuration Dependency:** The extractor requires `ModelConfig` to ensure the dimensions of the extracted features match the expectations of the neural network architecture.

## Exposed Interfaces

-   **Functions:**
    -   `extract_state_features(game_state: GameState, model_config: ModelConfig) -> StateType`: The main function to perform feature extraction.
    -   Low-level grid feature functions from `grid_features` (e.g., `get_column_heights`, `count_holes`, `get_bumpiness`).
-   **Classes:**
    -   `GameStateFeatures`: Class containing the feature extraction logic (primarily used internally by `extract_state_features`).

## Dependencies

-   **`src.environment`**:
    -   `GameState`: The input object for feature extraction.
    -   `GridData`: Accessed via `GameState` to get grid information.
-   **`src.config`**:
    -   `EnvConfig`: Accessed via `GameState` for environment dimensions.
    -   `ModelConfig`: Required by `extract_state_features` to ensure output dimensions match the NN input layer.
-   **`src.structs`**:
    -   Uses `Triangle`, `Shape`.
-   **`src.utils.types`**:
    -   `StateType`: The return type dictionary format.
-   **`numpy`**:
    -   Used extensively for creating and manipulating the numerical feature arrays.
-   **`numba`**:
    -   Used in `grid_features` for performance optimization.
-   **Standard Libraries:** `typing`.

---

**Note:** Please keep this README updated when changing the feature extraction logic, the set of extracted features, or the output format (`StateType`). Accurate documentation is crucial for maintainability.
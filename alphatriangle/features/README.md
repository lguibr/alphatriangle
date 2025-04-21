
# Feature Extraction Module (`alphatriangle.features`)

## Purpose and Architecture

This module is solely responsible for converting raw [`GameState`](../../trianglengin/game_interface.py) objects from the `trianglengin` library into numerical representations (features) suitable for input into the neural network ([`alphatriangle.nn`](../nn/README.md)). It acts as a bridge between the game's internal state and the requirements of the machine learning model.

-   **Decoupling:** This module completely decouples feature engineering from the core game engine logic. The `trianglengin` library focuses only on game rules and state transitions (now largely in C++), while this module handles the transformation for the NN.
-   **Feature Engineering:**
    -   [`extractor.py`](extractor.py): Contains the `GameStateFeatures` class and the main `extract_state_features` function. This orchestrates the extraction process, calling helper functions to generate different feature types. It uses the `Shape` class from `trianglengin.game_interface`. **It now also extracts the geometry (triangle list and color ID) of shapes available in the slots.**
    -   [`grid_features.py`](grid_features.py): Contains low-level, potentially performance-optimized (e.g., using Numba) functions for calculating specific scalar metrics derived from the grid state (like column heights, holes, bumpiness). This module operates directly on NumPy arrays obtained from the `GameState` wrapper.
-   **Output Format:** The `extract_state_features` function returns a `StateType` (a `TypedDict` defined in [`alphatriangle.utils.types`](../utils/types.py) containing `grid`, `other_features`, and `available_shapes_geometry` numpy arrays/lists), which is the standard input format expected by the `NeuralNetwork` interface and stored in the `ExperienceBuffer`.
-   **Configuration Dependency:** The extractor requires [`ModelConfig`](../config/model_config.py) to ensure the dimensions of the extracted numerical features (`other_features`) match the expectations of the neural network architecture.

## Exposed Interfaces

-   **Functions:**
    -   `extract_state_features(game_state: GameState, model_config: ModelConfig) -> StateType`: The main function to perform feature extraction.
    -   Low-level grid feature functions from `grid_features` (e.g., `get_column_heights`, `count_holes`, `get_bumpiness`).
-   **Classes:**
    -   `GameStateFeatures`: Class containing the feature extraction logic (primarily used internally by `extract_state_features`).

## Dependencies

-   **`trianglengin`**:
    -   `GameState`: The input object for feature extraction.
    -   `EnvConfig`: Accessed via `GameState` for environment dimensions.
    -   `Shape`: Used for processing shape features.
-   **[`alphatriangle.config`](../config/README.md)**:
    -   `ModelConfig`: Required by `extract_state_features` to ensure output dimensions match the NN input layer.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   `StateType`: The return type dictionary format.
    -   `SerializableShapeInfo`, `ShapeGeometry`: Types for shape geometry.
-   **`numpy`**:
    -   Used extensively for creating and manipulating the numerical feature arrays.
-   **`numba`**:
    -   Used in `grid_features` for performance optimization.
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when changing the feature extraction logic, the set of extracted features, or the output format (`StateType`). Accurate documentation is crucial for maintainability.

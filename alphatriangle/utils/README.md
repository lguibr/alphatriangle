# File: src/utils/README.md
# Utilities Module (`src.utils`)

## Purpose and Architecture

This module provides common utility functions and type definitions used across various parts of the AlphaTriangle project. Its goal is to avoid code duplication and provide central definitions for shared concepts.

-   **Helper Functions (`helpers.py`):** Contains miscellaneous helper functions:
    -   `get_device`: Determines the appropriate PyTorch device (CPU, CUDA, MPS) based on availability and preference.
    -   `set_random_seeds`: Initializes random number generators for Python, NumPy, and PyTorch for reproducibility.
    -   `format_eta`: Converts a time duration (in seconds) into a human-readable string (HH:MM:SS).
-   **Type Definitions (`types.py`):** Defines common type aliases and `TypedDict`s used throughout the codebase, particularly for data structures passed between modules (like RL components, NN, and environment). This improves code readability and enables better static analysis. Examples include:
    -   `StateType`: A `TypedDict` defining the structure of the state representation passed to the NN and stored in the buffer (e.g., `{'grid': np.ndarray, 'other_features': np.ndarray}`).
    -   `ActionType`: An alias for `int`, representing encoded actions.
    -   `PolicyTargetMapping`: A mapping from `ActionType` to `float`, representing the policy target from MCTS.
    -   `Experience`: A tuple representing `(GameState, PolicyTargetMapping, float)` stored in the replay buffer.
    -   `ExperienceBatch`: A list of `Experience` tuples.
    -   `PolicyValueOutput`: A tuple representing `(PolicyTargetMapping, float)` returned by the NN's `evaluate` method.
    -   **`PERBatchSample`**: A `TypedDict` defining the output of the PER buffer's sample method, including the batch, indices, and importance sampling weights.
    -   `StatsCollectorData`: Type alias for the data structure holding collected statistics.
-   **Geometry Utilities (`geometry.py`):** Contains geometric helper functions.
    -   `is_point_in_polygon`: Checks if a 2D point lies inside a given polygon.

## Exposed Interfaces

-   **Functions:**
    -   `get_device(device_preference: str = "auto") -> torch.device`
    -   `set_random_seeds(seed: int = 42)`
    -   `format_eta(seconds: Optional[float]) -> str`
    -   `is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool`
-   **Types:**
    -   `StateType` (TypedDict)
    -   `ActionType` (TypeAlias for `int`)
    -   `PolicyTargetMapping` (TypeAlias for `Mapping[ActionType, float]`)
    -   `Experience` (TypeAlias for `Tuple[GameState, PolicyTargetMapping, float]`)
    -   `ExperienceBatch` (TypeAlias for `List[Experience]`)
    -   `PolicyValueOutput` (TypeAlias for `Tuple[Mapping[ActionType, float], float]`)
    -   `PERBatchSample` (TypedDict)
    -   `StatsCollectorData` (TypeAlias for `Dict[str, Deque[Tuple[int, float]]]`)

## Dependencies

-   **`torch`**:
    -   Used by `get_device` and `set_random_seeds`.
-   **`numpy`**:
    -   Used by `set_random_seeds` and potentially in type definitions (`np.ndarray`).
-   **`src.environment`**:
    -   `GameState` (used in `Experience` type hint via TYPE_CHECKING).
-   **Standard Libraries:** `typing`, `random`, `os`, `math`, `logging`, `collections.deque`.

---

**Note:** Please keep this README updated when adding or modifying utility functions or type definitions, especially those used as interfaces between different modules. Accurate documentation is crucial for maintainability.
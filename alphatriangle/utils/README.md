# Utilities Module (alphatriangle.utils)

## Purpose and Architecture

This module provides common utility functions and type definitions used across various parts of the AlphaTriangle project. Its goal is to avoid code duplication and provide central definitions for shared concepts.

-   **Helper Functions ([helpers.py](helpers.py)):** Contains miscellaneous helper functions:
    -   get_device: Determines the appropriate PyTorch device (CPU, CUDA, MPS) based on availability and preference.
    -   set_random_seeds: Initializes random number generators for Python, NumPy, and PyTorch for reproducibility.
    -   format_eta: Converts a time duration (in seconds) into a human-readable string (HH:MM:SS).
    -   normalize_color_for_matplotlib: Converts RGB (0-255) to Matplotlib format (0.0-1.0).
-   **Type Definitions ([types.py](types.py)):** Defines common type aliases and TypedDicts used throughout the codebase, particularly for data structures passed between modules (like RL components, NN, and environment). This improves code readability and enables better static analysis. Examples include:
    -   StateType: A TypedDict defining the structure of the state representation passed to the NN and stored in the buffer. Contains grid (occupancy features) and other_features (numerical features).
    -   ActionType: An alias for int, representing encoded actions.
    -   PolicyTargetMapping: A mapping from ActionType to float, representing the policy target from MCTS.
    -   Experience: A tuple representing (StateType, PolicyTargetMapping, float) stored in the replay buffer (the float is the n-step return).
    -   ExperienceBatch: A list of Experience tuples.
    -   PolicyValueOutput: A tuple representing (PolicyTargetMapping, float) returned by the NN's evaluate method (the float is the expected value).
    -   PERBatchSample: A TypedDict defining the output of the PER buffer's sample method, including the batch, indices, and importance sampling weights.
-   **Geometry Utilities ([geometry.py](geometry.py)):** Contains geometric helper functions.
    -   is_point_in_polygon: Checks if a 2D point lies inside a given polygon.
-   **Data Structures ([sumtree.py](sumtree.py)):**
    -   SumTree: A simple SumTree implementation used for Prioritized Experience Replay.

## Exposed Interfaces

-   **Functions:**
    -   get_device(device_preference: str = "auto") -> torch.device
    -   set_random_seeds(seed: int = 42)
    -   format_eta(seconds: Optional[float]) -> str
    -   normalize_color_for_matplotlib(color_tuple_0_255: tuple[int, int, int]) -> tuple[float, float, float]
    -   is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool
-   **Classes:**
    -   SumTree: For PER.
-   **Types:**
    -   StateType (TypedDict)
    -   ActionType (TypeAlias for int)
    -   PolicyTargetMapping (TypeAlias for Mapping[ActionType, float])
    -   Experience (TypeAlias for Tuple[StateType, PolicyTargetMapping, float])
    -   ExperienceBatch (TypeAlias for List[Experience])
    -   PolicyValueOutput (TypeAlias for Tuple[Mapping[ActionType, float], float])
    -   PERBatchSample (TypedDict)

## Dependencies

-   **torch**:
    -   Used by get_device and set_random_seeds.
-   **numpy**:
    -   Used by set_random_seeds and potentially in type definitions (np.ndarray).
-   **Standard Libraries:** typing, random, os, math, logging, collections.deque.
-   **typing_extensions**: For TypedDict.

---

**Note:** Please keep this README updated when adding or modifying utility functions or type definitions, especially those used as interfaces between different modules. Accurate documentation is crucial for maintainability.

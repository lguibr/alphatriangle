# File: src/environment/grid/README.md
# Environment Grid Submodule (`src.environment.grid`)

## Purpose and Architecture

This submodule manages the game's grid structure and related logic. It defines the triangular cells, their properties, relationships, and operations like placement validation and line clearing.

-   **Cell Representation:** The `Triangle` class (defined in `src.structs`) represents a single cell, storing its position, orientation (`is_up`), state (`is_occupied`, `is_death`), color, and references to its immediate neighbors.
-   **Grid Data Structure:** The `GridData` class holds the 2D array of `Triangle` objects. It also maintains optimized `numpy` arrays (`_occupied_np`, `_death_np`) for faster state access and manages information about potential lines for efficient clearing checks.
-   **Grid Logic:** The `logic.py` module (exposed as `GridLogic`) contains functions operating on `GridData` and `Triangle` objects. This includes:
    -   Initializing the grid based on `EnvConfig` (defining death zones).
    -   Linking triangle neighbors.
    -   Precomputing potential lines (`_precompute_lines`) and indexing them (`initialize_lines_and_index`) for efficient checking.
    -   Checking if a shape can be placed (`can_place`), **including matching triangle orientations**.
    -   Checking for and clearing completed lines (`check_and_clear_lines`). **This function does NOT implement gravity.**
-   **Grid Features:** Note: The `grid_features.py` module, which provided functions to calculate scalar metrics (heights, holes, bumpiness), has been **moved** to the top-level `src/features` module (`src/features/grid_features.py`) as part of decoupling feature extraction from the core environment.

## Exposed Interfaces

-   **Classes:**
    -   `GridData`: Holds the grid state.
        -   `__init__(config: EnvConfig)`
        -   `valid(r: int, c: int) -> bool`
        -   `get_occupied_state() -> np.ndarray`
        -   `get_death_state() -> np.ndarray`
        -   `deepcopy() -> GridData`
-   **Modules/Namespaces:**
    -   `logic` (often imported as `GridLogic`):
        -   `link_neighbors(grid_data: GridData)`
        -   `initialize_lines_and_index(grid_data: GridData)`
        -   `can_place(grid_data: GridData, shape: Shape, r: int, c: int) -> bool`
        -   `check_and_clear_lines(grid_data: GridData, newly_occupied: Set[Triangle]) -> Tuple[int, Set[Triangle], Set[frozenset[Triangle]]]` **(Returns: lines_cleared_count, unique_triangles_cleared_set, set_of_cleared_lines)**

## Dependencies

-   **`src.config`**:
    -   `EnvConfig`: Used by `GridData` initialization and logic functions.
-   **`src.structs`**:
    -   Uses `Triangle`, `Shape`.
-   **`numpy`**:
    -   Used extensively in `GridData`.
-   **Standard Libraries:** `typing`, `logging`, `numpy`.

---

**Note:** Please keep this README updated when changing the grid structure, cell properties, placement rules, or line clearing logic. Accurate documentation is crucial for maintainability.
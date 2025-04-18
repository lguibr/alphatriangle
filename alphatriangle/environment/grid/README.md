# File: alphatriangle/environment/grid/README.md
# Environment Grid Submodule (`alphatriangle.environment.grid`)

## Purpose and Architecture

This submodule manages the game's grid structure and related logic. It defines the triangular cells, their properties, relationships, and operations like placement validation and line clearing.

-   **Cell Representation:** The `Triangle` class (defined in [`alphatriangle.structs`](../../structs/README.md)) represents a single cell, storing its position and orientation (`is_up`). The actual state (occupied, death, color) is managed within `GridData`.
-   **Grid Data Structure:** The [`GridData`](grid_data.py) class holds the grid state using efficient `numpy` arrays (`_occupied_np`, `_death_np`, `_color_id_np`). It also manages precomputed information about potential lines (sets of coordinates) for efficient clearing checks.
-   **Grid Logic:** The [`logic.py`](logic.py) module (exposed as `GridLogic`) contains functions operating on `GridData`. This includes:
    -   Initializing the grid based on `EnvConfig` (defining death zones).
    -   Precomputing potential lines (`_precompute_lines`) and indexing them (`initialize_lines_and_index`) for efficient checking.
    -   Checking if a shape can be placed (`can_place`), **including matching triangle orientations**.
    -   Checking for and clearing completed lines (`check_and_clear_lines`). **This function does NOT implement gravity.**
-   **Grid Features:** Note: The `grid_features.py` module, which previously provided functions to calculate scalar metrics (heights, holes, bumpiness), has been **moved** to the top-level [`alphatriangle.features`](../../features/README.md) module (`alphatriangle/features/grid_features.py`) as part of decoupling feature extraction from the core environment.

## Exposed Interfaces

-   **Classes:**
    -   `GridData`: Holds the grid state using NumPy arrays.
        -   `__init__(config: EnvConfig)`
        -   `valid(r: int, c: int) -> bool`
        -   `is_death(r: int, c: int) -> bool`
        -   `is_occupied(r: int, c: int) -> bool`
        -   `get_color_id(r: int, c: int) -> int`
        -   `get_occupied_state() -> np.ndarray`
        -   `get_death_state() -> np.ndarray`
        -   `get_color_id_state() -> np.ndarray`
        -   `deepcopy() -> GridData`
-   **Modules/Namespaces:**
    -   `logic` (often imported as `GridLogic`):
        -   `initialize_lines_and_index(grid_data: GridData)`
        -   `can_place(grid_data: GridData, shape: Shape, r: int, c: int) -> bool`
        -   `check_and_clear_lines(grid_data: GridData, newly_occupied_coords: Set[Tuple[int, int]]) -> Tuple[int, Set[Tuple[int, int]], Set[frozenset[Tuple[int, int]]]]` **(Returns: lines_cleared_count, unique_coords_cleared_set, set_of_cleared_lines_coord_sets)**

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used by `GridData` initialization and logic functions.
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `NO_COLOR_ID`.
-   **`numpy`**:
    -   Used extensively in `GridData`.
-   **Standard Libraries:** `typing`, `logging`, `numpy`, `copy`.

---

**Note:** Please keep this README updated when changing the grid structure, cell properties, placement rules, or line clearing logic. Accurate documentation is crucial for maintainability.
# File: src/environment/shapes/README.md
# Environment Shapes Submodule (`src.environment.shapes`)

## Purpose and Architecture

This submodule defines the logic for managing placeable shapes within the game environment.

-   **Shape Representation:** The `Shape` class (defined in `src.structs`) stores the geometry of a shape as a list of relative triangle coordinates (`(dr, dc, is_up)`) and its color.
-   **Shape Templates:** The `templates.py` file contains the `PREDEFINED_SHAPE_TEMPLATES` list, which defines the geometry of all possible shapes used in the game. **This list should not be modified.**
-   **Shape Logic:** The `logic.py` module (exposed as `ShapeLogic`) contains functions related to shapes:
    -   `generate_random_shape`: Creates a new `Shape` instance by randomly selecting a template from `PREDEFINED_SHAPE_TEMPLATES` and assigning a random color (using `SHAPE_COLORS` from `src.structs`).
    -   `refill_shape_slots`: **Refills ALL empty shape slots** in the `GameState` with new random shapes. This function is now typically called only when all shape slots have become empty after pieces are placed.

## Exposed Interfaces

-   **Modules/Namespaces:**
    -   `logic` (often imported as `ShapeLogic`):
        -   `generate_random_shape(rng: random.Random) -> Shape`
        -   `refill_shape_slots(game_state: GameState, rng: random.Random)` **(Refills all empty slots)**
-   **Constants:**
    -   `PREDEFINED_SHAPE_TEMPLATES` (from `templates.py`): The list of shape geometries.

## Dependencies

-   **`src.environment.core`**:
    -   `GameState`: Used by `ShapeLogic.refill_shape_slots` to access and modify the list of available shapes.
-   **`src.config`**:
    -   `EnvConfig`: Accessed via `GameState` (e.g., for `NUM_SHAPE_SLOTS`).
-   **`src.structs`**:
    -   Uses `Shape`, `Triangle`, `SHAPE_COLORS`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the shape generation algorithm or the logic for managing shape slots in the game state (especially the batch refill mechanism). Accurate documentation is crucial for maintainability. **Do not modify `templates.py`.**
# File: alphatriangle/environment/shapes/README.md
# Environment Shapes Submodule (`alphatriangle.environment.shapes`)

## Purpose and Architecture

This submodule defines the logic for managing placeable shapes within the game environment.

-   **Shape Representation:** The `Shape` class (defined in [`alphatriangle.structs`](../../structs/README.md)) stores the geometry of a shape as a list of relative triangle coordinates (`(dr, dc, is_up)`) and its color.
-   **Shape Templates:** The [`templates.py`](templates.py) file contains the `PREDEFINED_SHAPE_TEMPLATES` list, which defines the geometry of all possible shapes used in the game. **This list should not be modified.**
-   **Shape Logic:** The [`logic.py`](logic.py) module (exposed as `ShapeLogic`) contains functions related to shapes:
    -   `generate_random_shape`: Creates a new `Shape` instance by randomly selecting a template from `PREDEFINED_SHAPE_TEMPLATES` and assigning a random color (using `SHAPE_COLORS` from [`alphatriangle.structs`](../../structs/README.md)).
    -   `refill_shape_slots`: **Refills ALL empty shape slots** in the `GameState`, but **only if ALL slots are currently empty**. This implements batch refilling.

## Exposed Interfaces

-   **Modules/Namespaces:**
    -   `logic` (often imported as `ShapeLogic`):
        -   `generate_random_shape(rng: random.Random) -> Shape`
        -   `refill_shape_slots(game_state: GameState, rng: random.Random)` **(Refills all slots only if all are empty)**
-   **Constants:**
    -   `PREDEFINED_SHAPE_TEMPLATES` (from `templates.py`): The list of shape geometries.

## Dependencies

-   **[`alphatriangle.environment.core`](../core/README.md)**:
    -   `GameState`: Used by `ShapeLogic.refill_shape_slots` to access and modify the list of available shapes.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Accessed via `GameState` (e.g., for `NUM_SHAPE_SLOTS`).
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   Uses `Shape`, `Triangle`, `SHAPE_COLORS`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the shape generation algorithm or the logic for managing shape slots in the game state (especially the batch refill mechanism). Accurate documentation is crucial for maintainability. **Do not modify `templates.py`.**
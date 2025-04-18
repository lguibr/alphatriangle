# File: alphatriangle/environment/logic/README.md
# Environment Logic Submodule (`alphatriangle.environment.logic`)

## Purpose and Architecture

This submodule contains higher-level game logic that operates on the `GameState` and its components (`GridData`, `Shape`). It bridges the gap between basic actions/rules and the overall game flow.

-   **`actions.py`:**
    -   `get_valid_actions`: Determines all possible valid moves (shape placements) from the current `GameState` by iterating through available shapes and grid positions, checking placement validity using [`GridLogic.can_place`](../grid/logic.py). Returns a list of encoded `ActionType` integers.
-   **`step.py`:**
    -   `execute_placement`: Performs the core logic when a shape is placed. It updates the `GridData` (occupancy and color), checks for and clears completed lines using [`GridLogic.check_and_clear_lines`](../grid/logic.py), calculates the reward for the step using `calculate_reward`, updates the game score and step counters, and **triggers a batch refill of shape slots using [`ShapeLogic.refill_shape_slots`](../shapes/logic.py) only if all slots become empty after the placement.**
    -   `calculate_reward`: Calculates the reward based on the number of triangles placed, triangles cleared, and whether the game ended.

## Exposed Interfaces

-   **Functions:**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float`
    -   `calculate_reward(placed_count: int, unique_coords_cleared: Set[Tuple[int, int]], is_game_over: bool, config: EnvConfig) -> float`

## Dependencies

-   **[`alphatriangle.environment.core`](../core/README.md)**:
    -   `GameState`: The primary object operated upon.
    -   `ActionCodec`: Used by `get_valid_actions`.
-   **[`alphatriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used for placement checks and line clearing.
-   **[`alphatriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used for shape refilling.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used for reward calculation and action encoding.
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   `Shape`, `Triangle`, `COLOR_TO_ID_MAP`, `NO_COLOR_ID`.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the logic for determining valid actions, executing placements (including reward calculation and shape refilling), or modifying dependencies.
# File: alphatriangle/environment/core/README.md
# Environment Core Submodule (`alphatriangle.environment.core`)

## Purpose and Architecture

This submodule contains the most fundamental components of the game environment: the [`GameState`](game_state.py) class and the [`action_codec`](action_codec.py).

-   **`GameState`:** This class acts as the central hub for the environment's state. It holds references to the [`GridData`](../grid/grid_data.py), the current shapes, score, game status, and other relevant information. It provides the primary interface (`reset`, `step`, `valid_actions`, `is_over`, `get_outcome`, `copy`) for agents (like MCTS or self-play workers) to interact with the game. It delegates specific logic (like placement validation, line clearing, shape generation) to other submodules ([`grid`](../grid/README.md), [`shapes`](../shapes/README.md), [`logic`](../logic/README.md)).
-   **`action_codec`:** Provides simple, stateless functions (`encode_action`, `decode_action`) to translate between the agent's integer action representation and the game's internal representation (shape index, row, column). This decouples the agent's action space from the internal game logic.

## Exposed Interfaces

-   **Classes:**
    -   `GameState`: The main state class (see [`alphatriangle/environment/README.md`](../README.md) for methods).
-   **Functions:**
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used by `GameState` and `action_codec`.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`: Used for method signatures and return types.
-   **[`alphatriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used internally by `GameState`.
-   **[`alphatriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used internally by `GameState`.
-   **[`alphatriangle.environment.logic`](../logic/README.md)**:
    -   `get_valid_actions`, `execute_placement`: Used internally by `GameState`.
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`.

---

**Note:** Please keep this README updated when modifying the core `GameState` interface or the action encoding/decoding scheme. Accurate documentation is crucial for maintainability.
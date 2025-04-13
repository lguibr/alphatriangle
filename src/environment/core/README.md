# File: src/environment/core/README.md
# Environment Core Submodule (`src.environment.core`)

## Purpose and Architecture

This submodule contains the most fundamental components of the game environment: the `GameState` class and the `action_codec`.

-   **`GameState`:** This class acts as the central hub for the environment's state. It holds references to the `GridData`, the current shapes, score, game status, and other relevant information. It provides the primary interface (`reset`, `step`, `get_state`, `valid_actions`, `is_over`, `get_outcome`, `copy`) for agents (like MCTS or self-play workers) to interact with the game. It delegates specific logic (like placement validation, line clearing, shape generation) to other submodules (`grid`, `shapes`, `logic`, `features`).
-   **`action_codec`:** Provides simple, stateless functions (`encode_action`, `decode_action`) to translate between the agent's integer action representation and the game's internal representation (shape index, row, column). This decouples the agent's action space from the internal game logic.

## Exposed Interfaces

-   **Classes:**
    -   `GameState`: The main state class (see `src/environment/README.md` for methods).
-   **Functions:**
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`

## Dependencies

-   **`src.config`**:
    -   `EnvConfig`: Used by `GameState` and `action_codec`.
-   **`src.utils.types`**:
    -   `StateType`, `ActionType`: Used for method signatures and return types.
-   **`src.environment.grid`**:
    -   `GridData`, `GridLogic`: Used internally by `GameState`.
-   **`src.environment.shapes`**:
    -   `Shape`, `ShapeLogic`: Used internally by `GameState`.
-   **`src.environment.features`**:
    -   `extract_state_features`: Used internally by `GameState.get_state()`.
-   **`src.environment.logic`**:
    -   `get_valid_actions`, `execute_placement`: Used internally by `GameState`.
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`.

---

**Note:** Please keep this README updated when modifying the core `GameState` interface or the action encoding/decoding scheme. Accurate documentation is crucial for maintainability.
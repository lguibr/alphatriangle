# File: alphatriangle/environment/README.md
# Environment Module (`alphatriangle.environment`)

## Purpose and Architecture

This module defines the game world for AlphaTriangle. It encapsulates the rules, state representation, actions, and core game logic. **Crucially, this module is now independent of any feature extraction logic specific to the neural network.** Its sole focus is the simulation of the game itself.

-   **State Representation:** [`GameState`](core/game_state.py) holds the current board ([`GridData`](grid/grid_data.py)), available shapes (`List[Shape]`), score, and game status. It represents the canonical state of the game. It uses core structures like `Shape` and `Triangle` defined in [`alphatriangle.structs`](../structs/README.md).
-   **Core Logic:** Submodules ([`grid`](grid/README.md), [`shapes`](shapes/README.md), [`logic`](logic/README.md)) handle specific aspects like checking valid placements, clearing lines, managing shape generation, and calculating rewards. These logic modules operate on `GridData`, `Shape`, and `Triangle`. **Shape refilling now happens in batches: all slots are refilled only when all slots become empty.**
-   **Action Handling:** [`action_codec`](core/action_codec.py) provides functions to convert between a structured action (shape index, row, column) and a single integer representation used by the RL agent and MCTS.
-   **Modularity:** Separating grid logic, shape logic, and core state makes the code easier to understand and modify.

**Note:** Feature extraction (converting `GameState` to NN input tensors) is handled by the separate [`alphatriangle.features`](../features/README.md) module. Core data structures (`Triangle`, `Shape`) are defined in [`alphatriangle.structs`](../structs/README.md).

## Exposed Interfaces

-   **Core ([`core/README.md`](core/README.md)):**
    -   `GameState`: The main class representing the environment state.
        -   `reset()`
        -   `step(action_index: ActionType) -> Tuple[float, bool]`
        -   `valid_actions() -> List[ActionType]`
        -   `is_over() -> bool`
        -   `get_outcome() -> float`
        -   `copy() -> GameState`
        -   Public attributes like `grid_data`, `shapes`, `game_score`, `current_step`, etc.
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`
-   **Grid ([`grid/README.md`](grid/README.md)):**
    -   `GridData`: Class holding grid triangle data and line information.
    -   `GridLogic`: Namespace containing functions like `link_neighbors`, `initialize_lines_and_index`, `can_place`, `check_and_clear_lines`.
-   **Shapes ([`shapes/README.md`](shapes/README.md)):**
    -   `ShapeLogic`: Namespace containing functions like `refill_shape_slots`, `generate_random_shape`. **Includes `PREDEFINED_SHAPE_TEMPLATES` constant.**
-   **Logic ([`logic/README.md`](logic/README.md)):**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float` **(Triggers batch refill)**
    -   `calculate_reward(...) -> float` (Used internally by `execute_placement`)
-   **Config:**
    -   `EnvConfig`: Configuration class (re-exported for convenience).

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**:
    -   Uses `EnvConfig` extensively to define grid dimensions, shape slots, etc.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `SHAPE_COLORS`, `NO_COLOR_ID`, `DEBUG_COLOR_ID`, `COLOR_TO_ID_MAP`.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   Uses `ActionType`.
-   **`numpy`**:
    -   Used for grid representation (`GridData`).
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`, `copy`.

---

**Note:** Please keep this README updated when changing game rules, state representation, action space, or the module's internal structure (especially the shape refill logic). Accurate documentation is crucial for maintainability.
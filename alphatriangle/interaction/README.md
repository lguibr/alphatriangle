# File: alphatriangle/interaction/README.md
# Interaction Module (`alphatriangle.interaction`)

## Purpose and Architecture

This module handles user input (keyboard and mouse) for interactive modes of the application, such as "play" and "debug". It bridges the gap between raw Pygame events and actions within the game simulation ([`GameState`](../environment/core/game_state.py)).

-   **Event Processing:** [`event_processor.py`](event_processor.py) handles common Pygame events like quitting (QUIT, ESC) and window resizing. It acts as a generator, yielding other events for mode-specific processing.
-   **Input Handler:** The [`InputHandler`](input_handler.py) class is the main entry point.
    -   It receives Pygame events (via the `event_processor`).
    -   It **manages interaction-specific state** internally (e.g., `selected_shape_idx`, `hover_grid_coord`, `debug_highlight_coord`).
    -   It determines the current interaction mode ("play" or "debug") and delegates event handling and hover updates to specific handler functions ([`play_mode_handler`](play_mode_handler.py), [`debug_mode_handler`](debug_mode_handler.py)).
    -   It provides the necessary interaction state to the [`Visualizer`](../visualization/core/visualizer.py) for rendering feedback (hover previews, selection highlights).
-   **Mode-Specific Handlers:** `play_mode_handler.py` and `debug_mode_handler.py` contain the logic specific to each mode, operating on the `InputHandler`'s state and the `GameState`.
    -   `play`: Handles selecting shapes, checking placement validity, and triggering `GameState.step` on valid clicks. Updates hover state in the `InputHandler`.
    -   `debug`: Handles toggling the state of individual triangles directly on the `GameState.grid_data`. Updates hover state in the `InputHandler`.
-   **Decoupling:** It separates input handling logic from the core game simulation ([`environment`](../environment/README.md)) and rendering ([`visualization`](../visualization/README.md)), although it needs references to both to function. The `Visualizer` is now only responsible for drawing based on the state provided by the `GameState` and the `InputHandler`.

## Exposed Interfaces

-   **Classes:**
    -   `InputHandler`:
        -   `__init__(game_state: GameState, visualizer: Visualizer, mode: str, env_config: EnvConfig)`
        -   `handle_input() -> bool`: Processes events for one frame, returns `False` if quitting.
        -   `get_render_interaction_state() -> dict`: Returns interaction state needed by `Visualizer.render`.
-   **Functions:**
    -   `process_pygame_events(visualizer: Visualizer) -> Generator[pygame.event.Event, Any, bool]`: Processes common events, yields others.
    -   `handle_play_click(event: pygame.event.Event, handler: InputHandler)`: Handles clicks in play mode.
    -   `update_play_hover(handler: InputHandler)`: Updates hover state in play mode.
    -   `handle_debug_click(event: pygame.event.Event, handler: InputHandler)`: Handles clicks in debug mode.
    -   `update_debug_hover(handler: InputHandler)`: Updates hover state in debug mode.

## Dependencies

-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: Modifies the game state based on user actions (placing shapes, toggling debug cells).
    -   `EnvConfig`: Used for coordinate mapping and action encoding.
    -   `GridLogic`, `ActionCodec`: Used by mode-specific handlers.
-   **[`alphatriangle.visualization`](../visualization/README.md)**:
    -   `Visualizer`: Used to get layout information (`grid_rect`, `preview_rects`) and for coordinate mapping (`get_grid_coords_from_screen`, `get_preview_index_from_screen`). Also updated directly during resize events.
    -   `VisConfig`: Accessed via `Visualizer`.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   `Shape`, `Triangle`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **`pygame`**:
    -   Relies heavily on Pygame for event handling (`pygame.event`, `pygame.mouse`) and constants (`MOUSEBUTTONDOWN`, `KEYDOWN`, etc.).
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when adding new interaction modes, changing input handling logic, or modifying the interfaces between interaction, environment, and visualization. Accurate documentation is crucial for maintainability.
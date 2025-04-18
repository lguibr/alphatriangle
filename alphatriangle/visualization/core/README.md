# File: alphatriangle/visualization/core/README.md
# Visualization Core Submodule (`alphatriangle.visualization.core`)

## Purpose and Architecture

This submodule contains the central classes and foundational elements for the visualization system. It orchestrates rendering, manages layout and coordinate systems, and defines core visual properties like colors and fonts.

-   **Render Orchestration:**
    -   `Visualizer`: The main class for rendering in **interactive modes** ("play", "debug"). It maintains the Pygame screen, calculates layout using `layout.py`, manages cached preview area rectangles, and calls appropriate drawing functions from `alphatriangle.visualization.drawing`. **It receives interaction state (hover position, selected index) via its `render` method to display visual feedback.**
    -   `GameRenderer`: **Simplified renderer** responsible for drawing a **single** worker's `GameState` (grid and previews) within a specified sub-rectangle. Used by the `DashboardRenderer`.
    -   `DashboardRenderer` **(New)**: Renderer specifically for the **training visualization mode**. It uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects (using `GameRenderer` instances) in the top grid and displays statistics plots (using `alphatriangle.stats.Plotter`) and progress bars in the bottom area. **The training progress bar shows model/parameter info, while the buffer progress bar shows global training stats (worker weight updates, episodes, sims, worker status). Plots now include black, solid vertical lines (drawn on top) indicating the `global_step` when worker weights were updated.** It takes a dictionary mapping worker IDs to `GameState` objects and a dictionary of global statistics.
-   **Layout Management:**
    -   `layout.py`: Contains functions (`calculate_interactive_layout`, `calculate_training_layout`) to determine the size and position of the main UI areas based on the screen dimensions, mode, and `VisConfig`.
-   **Coordinate System:**
    -   `coord_mapper.py`: Provides essential mapping functions:
        -   `_calculate_render_params`: Internal helper to get scaling and offset for grid rendering.
        -   `get_grid_coords_from_screen`: Converts mouse/screen coordinates into logical grid (row, column) coordinates.
        -   `get_preview_index_from_screen`: Converts mouse/screen coordinates into the index of the shape preview slot being pointed at.
-   **Visual Properties:**
    -   `colors.py`: Defines a centralized palette of named color constants (RGB tuples).
    -   `fonts.py`: Contains the `load_fonts` function to load and manage Pygame font objects.

## Exposed Interfaces

-   **Classes:**
    -   `Visualizer`: Renderer for interactive modes.
        -   `__init__(...)`
        -   `render(game_state: GameState, mode: str, **interaction_state)`: Renders based on game state and interaction hints.
        -   `ensure_layout() -> Dict[str, pygame.Rect]`
        -   `screen`: Public attribute (Pygame Surface).
        -   `preview_rects`: Public attribute (cached preview area rects).
    -   `GameRenderer`: Renderer for a single worker's game state.
        -   `__init__(...)`
        -   `render_worker_state(target_surface: pygame.Surface, area_rect: pygame.Rect, worker_id: int, game_state: Optional[GameState])`
    -   `DashboardRenderer`: Renderer for combined multi-game/stats training visualization.
        -   `__init__(...)`
        -   `render(worker_states: Dict[int, GameState], global_stats: Optional[Dict[str, Any]])`
        -   `screen`: Public attribute (Pygame Surface).
-   **Functions:**
    -   `calculate_interactive_layout(...) -> Dict[str, pygame.Rect]`
    -   `calculate_training_layout(...) -> Dict[str, pygame.Rect]`
    -   `load_fonts() -> Dict[str, Optional[pygame.font.Font]]`
    -   `get_grid_coords_from_screen(...) -> Optional[Tuple[int, int]]`
    -   `get_preview_index_from_screen(...) -> Optional[int]`
-   **Modules:**
    -   `colors`: Provides color constants (e.g., `colors.RED`).

## Dependencies

-   **`alphatriangle.config`**: `VisConfig`, `EnvConfig`.
-   **`alphatriangle.environment`**: `GameState`, `GridData`.
-   **`alphatriangle.stats`**: `Plotter` (imported under `TYPE_CHECKING` in `dashboard_renderer`), `StatsCollectorActor`.
-   **`alphatriangle.utils`**: `types`, `helpers`.
-   **`alphatriangle.visualization.drawing`**: Drawing functions are called by renderers.
-   **`alphatriangle.visualization.ui`**: `ProgressBar` (used by `DashboardRenderer`).
-   **`pygame`**: Used for surfaces, rectangles, fonts, display management.
-   **Standard Libraries:** `typing`, `logging`, `math`.

---

**Note:** Please keep this README updated when changing the core rendering logic, layout calculations, coordinate mapping, or the interfaces of the renderers. Accurate documentation is crucial for maintainability.
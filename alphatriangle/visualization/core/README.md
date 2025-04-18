# File: alphatriangle/visualization/core/README.md
# Visualization Core Submodule (`alphatriangle.visualization.core`)

## Purpose and Architecture

This submodule contains the central classes and foundational elements for the visualization system. It orchestrates rendering, manages layout and coordinate systems, and defines core visual properties like colors and fonts.

-   **Render Orchestration:**
    -   [`Visualizer`](visualizer.py): The main class for rendering in **interactive modes** ("play", "debug"). It maintains the Pygame screen, calculates layout using `layout.py`, manages cached preview area rectangles, and calls appropriate drawing functions from [`alphatriangle.visualization.drawing`](../drawing/README.md). **It receives interaction state (hover position, selected index) via its `render` method to display visual feedback.**
    -   [`GameRenderer`](game_renderer.py): **Simplified renderer** responsible for drawing a **single** worker's `GameState` (grid and previews) within a specified sub-rectangle. Used by the `DashboardRenderer`.
    -   [`DashboardRenderer`](dashboard_renderer.py): Renderer specifically for the **training visualization mode**. It uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects (using `GameRenderer` instances) in the top grid and displays statistics plots (using [`alphatriangle.stats.Plotter`](../../stats/plotter.py)) and progress bars in the bottom area. **The training progress bar shows model/parameter info, while the buffer progress bar shows global training stats (worker weight updates, episodes, sims, worker status). Plots now include black, solid vertical lines (drawn on top) indicating the `global_step` when worker weights were updated, mapped to the appropriate position on each plot's x-axis.** It takes a dictionary mapping worker IDs to `GameState` objects and a dictionary of global statistics.
-   **Layout Management:**
    -   [`layout.py`](layout.py): Contains functions (`calculate_interactive_layout`, `calculate_training_layout`) to determine the size and position of the main UI areas based on the screen dimensions, mode, and `VisConfig`.
-   **Coordinate System:**
    -   [`coord_mapper.py`](coord_mapper.py): Provides essential mapping functions:
        -   `_calculate_render_params`: Internal helper to get scaling and offset for grid rendering.
        -   `get_grid_coords_from_screen`: Converts mouse/screen coordinates into logical grid (row, column) coordinates.
        -   `get_preview_index_from_screen`: Converts mouse/screen coordinates into the index of the shape preview slot being pointed at.
-   **Visual Properties:**
    -   [`colors.py`](colors.py): Defines a centralized palette of named color constants (RGB tuples).
    -   [`fonts.py`](fonts.py): Contains the `load_fonts` function to load and manage Pygame font objects.

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
        -   `render_worker_state(target_surface: pygame.Surface, area_rect: pygame.Rect, worker_id: int, game_state: Optional[GameState], worker_step_stats: Optional[Dict[str, Any]])`
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

-   **[`alphatriangle.config`](../../config/README.md)**: `VisConfig`, `EnvConfig`, `ModelConfig`.
-   **[`alphatriangle.environment`](../../environment/README.md)**: `GameState`, `GridData`.
-   **[`alphatriangle.stats`](../../stats/README.md)**: `Plotter`, `StatsCollectorActor`.
-   **[`alphatriangle.utils`](../../utils/README.md)**: `types`, `helpers`.
-   **[`alphatriangle.visualization.drawing`](../drawing/README.md)**: Drawing functions are called by renderers.
-   **[`alphatriangle.visualization.ui`](../ui/README.md)**: `ProgressBar` (used by `DashboardRenderer`).
-   **`pygame`**: Used for surfaces, rectangles, fonts, display management.
-   **`ray`**: Used by `DashboardRenderer` (for actor handle type hint).
-   **Standard Libraries:** `typing`, `logging`, `math`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the core rendering logic, layout calculations, coordinate mapping, or the interfaces of the renderers. Accurate documentation is crucial for maintainability.
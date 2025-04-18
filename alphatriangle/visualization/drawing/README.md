# File: alphatriangle/visualization/drawing/README.md
# Visualization Drawing Submodule (`alphatriangle.visualization.drawing`)

## Purpose and Architecture

This submodule contains specialized functions responsible for drawing specific visual elements of the game onto Pygame surfaces. These functions are typically called by the core renderers (`Visualizer`, `GameRenderer`) in [`alphatriangle.visualization.core`](../core/README.md). Separating drawing logic makes the core renderers cleaner and promotes reusability of drawing code.

-   **[`grid.py`](grid.py):** Functions for drawing the grid background (`draw_grid_background`), the individual triangles within it colored based on occupancy/emptiness (`draw_grid_triangles`), and optional indices (`draw_grid_indices`). Uses `Triangle` from [`alphatriangle.structs`](../../structs/README.md) and `GridData` from [`alphatriangle.environment`](../../environment/README.md).
-   **[`shapes.py`](shapes.py):** Contains `draw_shape`, a function to render a given `Shape` object at a specific location on a surface (used primarily for previews). Uses `Shape` and `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
-   **[`previews.py`](previews.py):** Handles rendering related to shape previews:
    -   `render_previews`: Draws the dedicated preview area, including borders and the shapes within their slots, handling selection highlights. Uses `Shape` from [`alphatriangle.structs`](../../structs/README.md).
    -   `draw_placement_preview`: Draws a semi-transparent version of a shape snapped to the grid, indicating a potential placement location (used in play mode hover). Uses `Shape` and `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
    -   `draw_floating_preview`: Draws a semi-transparent shape directly under the mouse cursor when hovering over the grid but not snapped (used in play mode hover). Uses `Shape` and `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
-   **[`hud.py`](hud.py):** `render_hud` draws Heads-Up Display elements like the game score, help text, and optional training statistics onto the main screen surface.
-   **[`highlight.py`](highlight.py):** `draw_debug_highlight` draws a distinct border around a specific triangle, used for visual feedback in debug mode. Uses `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
-   **[`utils.py`](utils.py):** Contains general drawing utility functions (currently empty).

## Exposed Interfaces

-   **Grid Drawing:**
    -   `draw_grid_background(surface: pygame.Surface, bg_color: tuple)`
    -   `draw_grid_triangles(surface: pygame.Surface, grid_data: GridData, config: EnvConfig)`
    -   `draw_grid_indices(surface: pygame.Surface, grid_data: GridData, config: EnvConfig, fonts: Dict[str, Optional[pygame.font.Font]])`
-   **Shape Drawing:**
    -   `draw_shape(surface: pygame.Surface, shape: Shape, topleft: Tuple[int, int], cell_size: float, is_selected: bool = False, origin_offset: Tuple[int, int] = (0, 0))`
-   **Preview Drawing:**
    -   `render_previews(surface: pygame.Surface, game_state: GameState, area_topleft: Tuple[int, int], mode: str, env_config: EnvConfig, vis_config: VisConfig, selected_shape_idx: int = -1) -> Dict[int, pygame.Rect]`
    -   `draw_placement_preview(surface: pygame.Surface, shape: Shape, r: int, c: int, is_valid: bool, config: EnvConfig)`
    -   `draw_floating_preview(surface: pygame.Surface, shape: Shape, screen_pos: Tuple[int, int], config: EnvConfig)`
-   **HUD Drawing:**
    -   `render_hud(surface: pygame.Surface, mode: str, fonts: Dict[str, Optional[pygame.font.Font]], display_stats: Optional[Dict[str, Any]] = None)`
-   **Highlight Drawing:**
    -   `draw_debug_highlight(surface: pygame.Surface, r: int, c: int, config: EnvConfig)`
-   **Utility Functions:**
    -   (Currently empty or contains other drawing-specific utils)

## Dependencies

-   **[`alphatriangle.visualization.core`](../core/README.md)**:
    -   `colors`: Used extensively for drawing colors.
    -   `coord_mapper`: Used internally (e.g., by `draw_placement_preview`) or relies on its calculations passed in.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`, `VisConfig`: Provide dimensions, padding, etc., needed for drawing calculations.
-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`, `GridData`: Provide the data to be drawn.
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `COLOR_ID_MAP`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **[`alphatriangle.visualization.ui`](../ui/README.md)**:
    -   `ProgressBar` (used by `hud.py`).
-   **`pygame`**:
    -   The core library used for all drawing operations (`pygame.draw.polygon`, `surface.fill`, `surface.blit`, etc.) and font rendering.
-   **Standard Libraries:** `typing`, `logging`, `math`.

---

**Note:** Please keep this README updated when adding new drawing functions, modifying existing ones, or changing their dependencies on configuration or environment data structures. Accurate documentation is crucial for maintainability.
# File: alphatriangle/visualization/README.md
# Visualization Module (`alphatriangle.visualization`)

## Purpose and Architecture

This module is responsible for rendering the game state visually using the Pygame library. It provides components for drawing the grid, shapes, previews, HUD elements, and statistics plots. **In training visualization mode, it now renders the states of multiple self-play workers in a grid layout alongside plots and progress bars (with specific information displayed on each bar).**

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Visualizer`: Orchestrates the rendering process for interactive modes ("play", "debug"). It manages the layout, calls drawing functions, and handles hover/selection states specific to visualization.
    -   `GameRenderer`: **Adapted renderer** for displaying **multiple** game states and statistics during training visualization (`run_training_visual.py`). It uses `layout.py` to divide the screen. It renders worker game states in one area and statistics plots/progress bars in another. It re-instantiates [`alphatriangle.stats.Plotter`](../stats/plotter.py).
    -   `DashboardRenderer`: Renderer specifically for the **training visualization mode**. It uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects (using `GameRenderer` instances) in the top grid and displays statistics plots (using `alphatriangle.stats.Plotter`) and progress bars in the bottom area. **The training progress bar shows model/parameter info, while the buffer progress bar shows global training stats (updates, episodes, sims, workers).** It takes a dictionary mapping worker IDs to `GameState` objects and a dictionary of global statistics.
    -   `layout`: Calculates the screen positions and sizes for different UI areas (worker grid, stats area, plots).
    -   `fonts`: Loads necessary font files.
    -   `colors`: Defines a centralized palette of RGB color tuples.
    -   `coord_mapper`: Provides functions to map screen coordinates to grid coordinates (`get_grid_coords_from_screen`) and preview indices (`get_preview_index_from_screen`).
-   **Drawing Components ([`drawing/README.md`](drawing/README.md)):**
    -   Contains specific functions for drawing different elements onto Pygame surfaces:
        -   `grid`: Draws the grid background and occupied/empty triangles.
        -   `shapes`: Draws individual shapes (used by previews).
        -   `previews`: Renders the shape preview area.
        -   `hud`: Renders text information like global training stats and help text at the bottom.
        -   `highlight`: Draws debug highlights.
-   **UI Components ([`ui/README.md`](ui/README.md)):**
    -   Contains reusable UI elements like `ProgressBar`.

## Exposed Interfaces

-   **Core Classes & Functions:**
    -   `Visualizer`: Main renderer for interactive modes.
    -   `GameRenderer`: Renderer for a single worker's game state.
    -   `DashboardRenderer`: Renderer for combined multi-game/stats training visualization.
    -   `calculate_interactive_layout`, `calculate_training_layout`: Calculates UI layout rectangles.
    -   `load_fonts`: Loads Pygame fonts.
    -   `colors`: Module containing color constants (e.g., `colors.WHITE`).
    -   `get_grid_coords_from_screen`: Maps screen to grid coordinates.
    -   `get_preview_index_from_screen`: Maps screen to preview index.
-   **Drawing Functions (primarily used internally by Visualizer/GameRenderer but exposed):**
    -   `draw_grid_background`, `draw_grid_triangles`, `draw_grid_indices`
    -   `draw_shape`
    -   `render_previews`, `draw_placement_preview`, `draw_floating_preview`
    -   `render_hud`
    -   `draw_debug_highlight`
-   **UI Components:**
    -   `ProgressBar`: Class for rendering progress bars.
-   **Config:**
    -   `VisConfig`: Configuration class (re-exported from [`alphatriangle.config`](../config/README.md)).

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**:
    -   `VisConfig`, `EnvConfig`, `ModelConfig`: Used extensively for layout, sizing, and coordinate mapping.
-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: The primary object whose state is visualized.
    -   `GridData`: Accessed via `GameState` or passed directly to drawing functions.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `COLOR_ID_MAP`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **[`alphatriangle.stats`](../stats/README.md)**:
    -   Uses `Plotter` within `DashboardRenderer`.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   Uses `geometry.is_point_in_polygon`, `helpers.format_eta`, `types.StatsCollectorData`.
-   **`pygame`**:
    -   The core library used for all drawing, surface manipulation, event handling (via `interaction`), and font rendering.
-   **`matplotlib`**:
    -   Used by `alphatriangle.stats.Plotter`.
-   **Standard Libraries:** `typing`, `logging`, `math`, `time`.

---

**Note:** Please keep this README updated when changing rendering logic, adding new visual elements, modifying layout calculations, or altering the interfaces exposed to other modules (like `interaction` or the main application scripts). Accurate documentation is crucial for maintainability.
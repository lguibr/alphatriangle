# File: src/visualization/README.md
# Visualization Module (`src.visualization`)

## Purpose and Architecture

This module is responsible for rendering the game state visually using the Pygame library. It provides components for drawing the grid, shapes, previews, HUD elements, and statistics plots. **In training visualization mode, it now renders the states of multiple self-play workers in a grid layout alongside plots and progress bars.**

-   **Core Components (`src.visualization.core`):**
    -   `Visualizer`: Orchestrates the rendering process for interactive modes ("play", "debug"). It manages the layout, calls drawing functions, and handles hover/selection states specific to visualization.
    -   `GameRenderer`: **Adapted renderer** for displaying **multiple** game states and statistics during training visualization (`run_training_visual.py`). It uses `layout.py` to divide the screen. It renders worker game states in one area and statistics plots/progress bars in another. It re-instantiates `src.stats.Plotter`.
    -   `layout`: Calculates the screen positions and sizes for different UI areas (worker grid, stats area, plots).
    -   `fonts`: Loads necessary font files.
    -   `colors`: Defines a centralized palette of RGB color tuples.
    -   `coord_mapper`: Provides functions to map screen coordinates to grid coordinates (`get_grid_coords_from_screen`) and preview indices (`get_preview_index_from_screen`).
-   **Drawing Components (`src.visualization.drawing`):**
    -   Contains specific functions for drawing different elements onto Pygame surfaces:
        -   `grid`: Draws the grid background and occupied/empty triangles.
        -   `shapes`: Draws individual shapes (used by previews).
        -   `previews`: Renders the shape preview area.
        -   `hud`: Renders text information like global training stats and help text at the bottom.
        -   `highlight`: Draws debug highlights.
-   **UI Components (`src.visualization.ui`):**
    -   Contains reusable UI elements like `ProgressBar`.

## Exposed Interfaces

-   **Core Classes & Functions:**
    -   `Visualizer`: Main renderer for interactive modes.
    -   `GameRenderer`: Renderer for combined multi-game/stats training visualization.
    -   `calculate_layout`: Calculates UI layout rectangles.
    -   `load_fonts`: Loads Pygame fonts.
    -   `colors`: Module containing color constants (e.g., `colors.WHITE`).
    -   `get_grid_coords_from_screen`: Maps screen to grid coordinates.
    -   `get_preview_index_from_screen`: Maps screen to preview index.
-   **Drawing Functions (primarily used internally by Visualizer/GameRenderer but exposed):**
    -   `draw_grid_background`, `draw_grid_triangles`
    -   `draw_shape`
    -   `render_previews`
    -   `render_hud`
    -   `draw_debug_highlight`
-   **UI Components:**
    -   `ProgressBar`: Class for rendering progress bars.
-   **Config:**
    -   `VisConfig`: Configuration class (re-exported from `src.config`).

## Dependencies

-   **`src.config`**:
    -   `VisConfig`, `EnvConfig`: Used extensively for layout, sizing, and coordinate mapping.
-   **`src.environment`**:
    -   `GameState`: The primary object whose state is visualized.
    -   `GridData`: Accessed via `GameState` or passed directly to drawing functions.
-   **`src.structs`**:
    -   Uses `Triangle`, `Shape`.
-   **`src.stats`**:
    -   Uses `Plotter` within `GameRenderer`.
-   **`src.utils`**:
    -   Uses `geometry.is_point_in_polygon`, `helpers.format_eta`, `types.StatsCollectorData`.
-   **`pygame`**:
    -   The core library used for all drawing, surface manipulation, event handling (via `interaction`), and font rendering.
-   **`matplotlib`**:
    -   Used by `src.stats.Plotter`.
-   **Standard Libraries:** `typing`, `logging`, `math`, `time`.

---

**Note:** Please keep this README updated when changing rendering logic, adding new visual elements, modifying layout calculations, or altering the interfaces exposed to other modules (like `interaction` or the main application scripts). Accurate documentation is crucial for maintainability.
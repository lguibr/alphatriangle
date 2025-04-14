# File: src/structs/README.md
# Core Structures Module (`src.structs`)

## Purpose and Architecture

This module defines fundamental data structures and constants that are shared across multiple major components of the application (like `environment`, `visualization`, `features`). Its primary purpose is to break potential circular dependencies that arise when these components need to know about the same basic building blocks.

-   **`triangle.py`:** Defines the `Triangle` class, representing a single cell on the game grid.
-   **`shape.py`:** Defines the `Shape` class, representing a placeable piece composed of triangles.
-   **`constants.py`:** Defines shared constants, such as the list of possible `SHAPE_COLORS`.

By placing these core definitions in a low-level module with minimal dependencies, other modules can import them without creating import cycles.

## Exposed Interfaces

-   **Classes:**
    -   `Triangle`: Represents a grid cell.
    -   `Shape`: Represents a placeable piece.
-   **Constants:**
    -   `SHAPE_COLORS`: A list of RGB tuples for shape generation.

## Dependencies

This module has minimal dependencies, primarily relying on standard Python libraries (`typing`). It should **not** import from higher-level modules like `environment`, `visualization`, `nn`, `rl`, etc.

---

**Note:** This module should only contain widely shared, fundamental data structures and constants. More complex logic or structures specific to a particular domain (like game rules or rendering details) should remain in their respective modules.
# File: alphatriangle/visualization/core/colors.py
"""Centralized color definitions (RGB tuples 0-255)."""

WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHT_GRAY: tuple[int, int, int] = (180, 180, 180)
GRAY: tuple[int, int, int] = (100, 100, 100)
DARK_GRAY: tuple[int, int, int] = (40, 40, 40)
RED: tuple[int, int, int] = (220, 40, 40)
DARK_RED: tuple[int, int, int] = (100, 10, 10)
BLUE: tuple[int, int, int] = (60, 60, 220)
YELLOW: tuple[int, int, int] = (230, 230, 40)
GREEN: tuple[int, int, int] = (40, 200, 40)
DARK_GREEN: tuple[int, int, int] = (10, 80, 10)
ORANGE: tuple[int, int, int] = (240, 150, 20)
PURPLE: tuple[int, int, int] = (140, 40, 140)
CYAN: tuple[int, int, int] = (40, 200, 200)
LIGHTG: tuple[int, int, int] = (144, 238, 144)
HOTPINK: tuple[int, int, int] = (255, 105, 180)  # Added for plots

GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]

# Game Specific Visuals
GRID_BG_DEFAULT: tuple[int, int, int] = (20, 20, 30)
GRID_BG_GAME_OVER: tuple[int, int, int] = DARK_RED
GRID_LINE_COLOR: tuple[int, int, int] = GRAY
TRIANGLE_EMPTY_COLOR: tuple[int, int, int] = (60, 60, 70)
PREVIEW_BG: tuple[int, int, int] = (30, 30, 40)
PREVIEW_BORDER: tuple[int, int, int] = GRAY
PREVIEW_SELECTED_BORDER: tuple[int, int, int] = BLUE
PLACEMENT_VALID_COLOR: tuple[int, int, int, int] = (*GREEN, 150)  # RGBA
PLACEMENT_INVALID_COLOR: tuple[int, int, int, int] = (*RED, 100)  # RGBA
DEBUG_TOGGLE_COLOR: tuple[int, int, int] = YELLOW

# --- ADDED: Colors for Progress Bar Cycling ---
PROGRESS_BAR_CYCLE_COLORS: list[tuple[int, int, int]] = [
    GREEN,
    BLUE,
    YELLOW,
    ORANGE,
    PURPLE,
    CYAN,
    HOTPINK,
    RED,  # Add red towards the end
]
# --- END ADDED ---

import numpy as np
from numba import njit, prange


@njit(cache=True)
def get_column_heights(
    occupied: np.ndarray, death: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    """Calculates the height of each column (highest occupied non-death cell)."""
    heights = np.zeros(cols, dtype=np.int32)
    for c in prange(cols):
        max_r = -1
        for r in range(rows):
            if occupied[r, c] and not death[r, c]:
                max_r = r
        heights[c] = max_r + 1
    return heights


@njit(cache=True)
def count_holes(
    occupied: np.ndarray, death: np.ndarray, heights: np.ndarray, _rows: int, cols: int
) -> int:
    """Counts the number of empty, non-death cells below the column height."""
    holes = 0
    for c in prange(cols):
        col_height = heights[c]
        for r in range(col_height):
            if not occupied[r, c] and not death[r, c]:
                holes += 1
    return holes


@njit(cache=True)
def get_bumpiness(heights: np.ndarray) -> float:
    """Calculates the total absolute difference between adjacent column heights."""
    bumpiness = 0.0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

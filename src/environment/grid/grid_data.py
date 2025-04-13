# File: src/environment/grid/grid_data.py
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import logging

from ...config import EnvConfig

# Import logic functions needed within this file
from . import logic as GridLogic

# Import Triangle from structs
from src.structs import Triangle

logger = logging.getLogger(__name__)


# --- Line Precomputation (Moved here to break circular import) ---
def _precompute_lines(config: EnvConfig) -> List[List[Tuple[int, int]]]:
    """
    Generates all potential horizontal and diagonal lines based on grid geometry.
    Returns a list of lines, where each line is a list of (row, col) tuples.
    """
    lines = []
    rows, cols = config.ROWS, config.COLS
    min_len = config.MIN_LINE_LENGTH

    # Create a temporary grid to easily access triangles by (r, c)
    temp_grid: List[List[Optional[Triangle]]] = [
        [None for _ in range(cols)] for _ in range(rows)
    ]
    playable_mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        playable_width = config.COLS_PER_ROW[r]
        padding = cols - playable_width
        pad_left = padding // 2
        playable_start_col = pad_left
        playable_end_col = pad_left + playable_width
        for c in range(cols):
            is_up = (r + c) % 2 != 0
            is_death = not (playable_start_col <= c < playable_end_col)
            if not is_death:
                temp_grid[r][c] = Triangle(r, c, is_up, is_death)
                playable_mask[r, c] = True

    # Link neighbors for the temporary grid to trace lines
    # Need GridData class definition here - use a temporary structure
    class TempGridHolder:
        def __init__(self, r, c, t):
            self.rows = r
            self.cols = c
            self.triangles = t

        # Add the missing 'valid' method
        def valid(self, r_coord: int, c_coord: int) -> bool:
            """Checks if coordinates are within grid bounds."""
            return 0 <= r_coord < self.rows and 0 <= c_coord < self.cols

    temp_grid_data = TempGridHolder(rows, cols, temp_grid)
    GridLogic.link_neighbors(temp_grid_data)  # Link neighbors on temp grid

    visited_in_line: Set[Tuple[int, int, str]] = set()  # (r, c, direction)

    for r_start in range(rows):
        for c_start in range(cols):
            start_tri = temp_grid[r_start][c_start]
            if not start_tri:
                continue  # Skip death cells

            # Trace Horizontal
            if (r_start, c_start, "h") not in visited_in_line:
                current_line_h = []
                # Trace left to find the true start
                curr = start_tri
                while curr and curr.neighbor_left:
                    curr = curr.neighbor_left
                # Trace right from the true start
                while curr and (curr.row, curr.col, "h") not in visited_in_line:
                    current_line_h.append((curr.row, curr.col))
                    visited_in_line.add((curr.row, curr.col, "h"))
                    curr = curr.neighbor_right
                if len(current_line_h) >= min_len:
                    lines.append(current_line_h)

            # Trace Diagonal TL-BR (Down-Right)
            if (r_start, c_start, "d1") not in visited_in_line:
                current_line_d1 = []
                # Trace backwards first (Up-Left)
                curr = start_tri
                while curr:
                    is_up = (curr.row + curr.col) % 2 != 0
                    prev_tri = curr.neighbor_left if is_up else curr.neighbor_vert
                    if prev_tri:
                        curr = prev_tri
                    else:
                        break  # Reached start of backward trace

                # Trace forwards from the actual start
                while curr and (curr.row, curr.col, "d1") not in visited_in_line:
                    current_line_d1.append((curr.row, curr.col))
                    visited_in_line.add((curr.row, curr.col, "d1"))
                    is_up = (curr.row + curr.col) % 2 != 0
                    curr = curr.neighbor_vert if is_up else curr.neighbor_right
                if len(current_line_d1) >= min_len:
                    lines.append(current_line_d1)

            # Trace Diagonal BL-TR (Up-Right)
            if (r_start, c_start, "d2") not in visited_in_line:
                current_line_d2 = []
                # Trace backwards first (Down-Left)
                curr = start_tri
                while curr:
                    is_up = (curr.row + curr.col) % 2 != 0
                    prev_tri = curr.neighbor_vert if is_up else curr.neighbor_left
                    if prev_tri:
                        curr = prev_tri
                    else:
                        break

                # Trace forwards from the actual start
                while curr and (curr.row, curr.col, "d2") not in visited_in_line:
                    current_line_d2.append((curr.row, curr.col))
                    visited_in_line.add((curr.row, curr.col, "d2"))
                    is_up = (curr.row + curr.col) % 2 != 0
                    curr = curr.neighbor_right if is_up else curr.neighbor_vert
                if len(current_line_d2) >= min_len:
                    lines.append(current_line_d2)

    # Remove duplicate lines (can happen with trace logic)
    unique_lines_tuples = {tuple(sorted(line)) for line in lines}
    unique_lines = [list(line_tuple) for line_tuple in unique_lines_tuples]

    # Filter out lines shorter than min_len again after removing duplicates
    final_lines = [line for line in unique_lines if len(line) >= min_len]

    return final_lines


class GridData:
    """Holds the grid state (triangles, occupancy, death zones)."""

    def __init__(self, config: EnvConfig):
        self.rows = config.ROWS
        self.cols = config.COLS
        self.config = config
        self.triangles: List[List[Triangle]] = self._create(config)
        GridLogic.link_neighbors(self)  # Use logic from logic.py

        self._occupied_np = np.array(
            [[t.is_occupied for t in r] for r in self.triangles], dtype=bool
        )
        self._death_np = np.array(
            [[t.is_death for t in r] for r in self.triangles], dtype=bool
        )

        self.potential_lines: Set[frozenset[Triangle]] = set()
        self._triangle_to_lines_map: Dict[Triangle, Set[frozenset[Triangle]]] = {}
        self._initialize_lines_and_index()  # Call internal method
        # Changed log level from INFO to DEBUG
        logger.debug(
            f"GridData initialized ({self.rows}x{self.cols}). Found {len(self.potential_lines)} potential lines."
        )

    def _create(self, config: EnvConfig) -> List[List[Triangle]]:
        """
        Initializes the grid, marking death cells based on COLS_PER_ROW.
        COLS_PER_ROW defines the number of *playable* cells, centered horizontally.
        Inverts the is_up calculation for <> edge pattern.
        """
        cols_per_row = config.COLS_PER_ROW
        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"COLS_PER_ROW length mismatch: {len(cols_per_row)} vs {self.rows}"
            )

        grid = []
        for r in range(self.rows):
            row_tris = []
            playable_width = cols_per_row[r]
            padding = self.cols - playable_width
            pad_left = padding // 2
            playable_start_col = pad_left
            playable_end_col = pad_left + playable_width

            for c in range(self.cols):
                is_playable = playable_start_col <= c < playable_end_col
                is_death = not is_playable

                is_up = (r + c) % 2 != 0

                row_tris.append(Triangle(r, c, is_up, is_death=is_death))
            grid.append(row_tris)
        return grid

    def _initialize_lines_and_index(self):
        """
        Precomputes potential lines and creates a map from triangle coords to lines.
        (Moved from logic.py)
        """
        self.potential_lines = set()
        self._triangle_to_lines_map = {}

        # Generate lines based on geometry using the function now local to this module
        potential_lines_coords = _precompute_lines(self.config)

        for line_coords in potential_lines_coords:
            line_triangles: Set[Triangle] = set()
            valid_line = True
            for r, c in line_coords:
                if self.valid(r, c):
                    # Use the actual grid_data triangles now
                    tri = self.triangles[r][c]
                    if not tri.is_death:
                        line_triangles.add(tri)
                    else:
                        valid_line = False
                        break
                else:
                    valid_line = False
                    break

            if valid_line and len(line_triangles) >= self.config.MIN_LINE_LENGTH:
                frozen_line = frozenset(line_triangles)
                self.potential_lines.add(frozen_line)
                # Add to the reverse map
                for tri in line_triangles:
                    if tri not in self._triangle_to_lines_map:
                        self._triangle_to_lines_map[tri] = set()
                    self._triangle_to_lines_map[tri].add(frozen_line)

        logger.debug(
            f"Initialized {len(self.potential_lines)} potential lines and mapping for {len(self._triangle_to_lines_map)} triangles."
        )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_occupied_state(self) -> np.ndarray:
        """Returns a copy of the occupancy numpy array."""
        return self._occupied_np.copy()

    def get_death_state(self) -> np.ndarray:
        """Returns a copy of the death zone numpy array."""
        return self._death_np.copy()

    def deepcopy(self) -> "GridData":
        """
        Creates a deep copy of the grid data.
        Copies precomputed line data instead of re-initializing.
        """
        new_grid = GridData.__new__(GridData)
        new_grid.rows = self.rows
        new_grid.cols = self.cols
        new_grid.config = self.config

        # 1. Copy triangles
        new_grid.triangles = [[tri.copy() for tri in row] for row in self.triangles]

        # 2. Link neighbors for the new triangles
        GridLogic.link_neighbors(new_grid)

        # 3. Copy numpy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()

        # 4. Copy precomputed line data, mapping old triangles to new ones
        new_grid.potential_lines = set()
        new_grid._triangle_to_lines_map = {}
        # Create a mapping from old triangle hash to new triangle object
        old_to_new_tri_map: Dict[int, Triangle] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                old_tri = self.triangles[r][c]
                new_tri = new_grid.triangles[r][c]
                old_to_new_tri_map[hash(old_tri)] = new_tri

        # Rebuild potential_lines and _triangle_to_lines_map using new triangles
        for old_frozen_line in self.potential_lines:
            new_line_triangles: Set[Triangle] = set()
            valid_new_line = True
            for old_tri in old_frozen_line:
                new_tri = old_to_new_tri_map.get(hash(old_tri))
                if new_tri:
                    new_line_triangles.add(new_tri)
                else:
                    # This shouldn't happen if the map is built correctly
                    logger.error(
                        f"Deepcopy error: Could not find new triangle corresponding to old {old_tri}"
                    )
                    valid_new_line = False
                    break
            if valid_new_line:
                new_frozen_line = frozenset(new_line_triangles)
                new_grid.potential_lines.add(new_frozen_line)
                # Add to the reverse map for the new grid
                for new_tri in new_line_triangles:
                    if new_tri not in new_grid._triangle_to_lines_map:
                        new_grid._triangle_to_lines_map[new_tri] = set()
                    new_grid._triangle_to_lines_map[new_tri].add(new_frozen_line)

        # logger.debug(f"GridData deepcopy complete. Copied {len(new_grid.potential_lines)} lines.") # Optional: reduce log noise
        return new_grid

    def __str__(self) -> str:
        return f"GridData({self.rows}x{self.cols})"

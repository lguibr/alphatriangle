# File: alphatriangle/environment/grid/grid_data.py
import copy
import logging

import numpy as np

from ...config import EnvConfig
from ...structs import NO_COLOR_ID

logger = logging.getLogger(__name__)


def _precompute_lines(config: EnvConfig) -> list[list[tuple[int, int]]]:
    """
    Generates all potential horizontal and diagonal lines based on grid geometry.
    Returns a list of lines, where each line is a list of (row, col) tuples.
    This function no longer needs actual Triangle objects.
    """
    lines = []
    rows, cols = config.ROWS, config.COLS
    min_len = config.MIN_LINE_LENGTH

    # --- Determine playable cells based on config ---
    playable_mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        playable_width = config.COLS_PER_ROW[r]
        padding = cols - playable_width
        pad_left = padding // 2
        playable_start_col = pad_left
        playable_end_col = pad_left + playable_width
        for c in range(cols):
            if playable_start_col <= c < playable_end_col:
                playable_mask[r, c] = True
    # --- End Playable Mask ---

    # Helper to check validity and playability
    def is_valid_playable(r, c):
        return 0 <= r < rows and 0 <= c < cols and playable_mask[r, c]

    # --- Trace Lines using Coordinates ---
    visited_in_line: set[tuple[int, int, str]] = set()  # (r, c, direction)

    for r_start in range(rows):
        for c_start in range(cols):
            if not is_valid_playable(r_start, c_start):
                continue

            # --- Trace Horizontal ---
            if (r_start, c_start, "h") not in visited_in_line:
                current_line_h = []
                # Trace left
                cr, cc = r_start, c_start
                while is_valid_playable(cr, cc - 1):
                    cc -= 1
                # Trace right from the start
                while is_valid_playable(cr, cc):
                    if (cr, cc, "h") not in visited_in_line:
                        current_line_h.append((cr, cc))
                        visited_in_line.add((cr, cc, "h"))
                    else:
                        # If we hit a visited cell, the rest of the line was already processed
                        break
                    cc += 1
                if len(current_line_h) >= min_len:
                    lines.append(current_line_h)

            # --- Trace Diagonal TL-BR (Down-Right) ---
            if (r_start, c_start, "d1") not in visited_in_line:
                current_line_d1 = []
                # Trace backwards (Up-Left)
                cr, cc = r_start, c_start
                while True:
                    is_up = (cr + cc) % 2 != 0
                    prev_r, prev_c = (cr, cc - 1) if is_up else (cr - 1, cc)
                    if is_valid_playable(prev_r, prev_c):
                        cr, cc = prev_r, prev_c
                    else:
                        break
                # Trace forwards
                while is_valid_playable(cr, cc):
                    if (cr, cc, "d1") not in visited_in_line:
                        current_line_d1.append((cr, cc))
                        visited_in_line.add((cr, cc, "d1"))
                    else:
                        break
                    is_up = (cr + cc) % 2 != 0
                    next_r, next_c = (cr + 1, cc) if is_up else (cr, cc + 1)
                    cr, cc = next_r, next_c
                if len(current_line_d1) >= min_len:
                    lines.append(current_line_d1)

            # --- Trace Diagonal BL-TR (Up-Right) ---
            if (r_start, c_start, "d2") not in visited_in_line:
                current_line_d2 = []
                # Trace backwards (Down-Left)
                cr, cc = r_start, c_start
                while True:
                    is_up = (cr + cc) % 2 != 0
                    prev_r, prev_c = (cr + 1, cc) if is_up else (cr, cc - 1)
                    if is_valid_playable(prev_r, prev_c):
                        cr, cc = prev_r, prev_c
                    else:
                        break
                # Trace forwards
                while is_valid_playable(cr, cc):
                    if (cr, cc, "d2") not in visited_in_line:
                        current_line_d2.append((cr, cc))
                        visited_in_line.add((cr, cc, "d2"))
                    else:
                        break
                    is_up = (cr + cc) % 2 != 0
                    next_r, next_c = (cr, cc + 1) if is_up else (cr - 1, cc)
                    cr, cc = next_r, next_c
                if len(current_line_d2) >= min_len:
                    lines.append(current_line_d2)
    # --- End Line Tracing ---

    # Remove duplicates (lines traced from different start points)
    unique_lines_tuples = {tuple(sorted(line)) for line in lines}
    unique_lines = [list(line_tuple) for line_tuple in unique_lines_tuples]

    # Final filter by length (should be redundant but safe)
    final_lines = [line for line in unique_lines if len(line) >= min_len]

    return final_lines


class GridData:
    """
    Holds the grid state using NumPy arrays for occupancy, death zones, and color IDs.
    Manages precomputed line information based on coordinates.
    """

    def __init__(self, config: EnvConfig):
        self.rows = config.ROWS
        self.cols = config.COLS
        self.config = config

        # --- NumPy Array State ---
        self._occupied_np: np.ndarray = np.zeros((self.rows, self.cols), dtype=bool)
        self._death_np: np.ndarray = np.zeros((self.rows, self.cols), dtype=bool)
        # Stores color ID, NO_COLOR_ID (-1) means empty/no color
        self._color_id_np: np.ndarray = np.full(
            (self.rows, self.cols), NO_COLOR_ID, dtype=np.int8
        )
        # --- End NumPy Array State ---

        self._initialize_death_zone(config)
        self._occupied_np[self._death_np] = True  # Death cells are considered occupied

        # --- Line Information (Coordinate Based) ---
        # Stores frozensets of (r, c) tuples
        self.potential_lines: set[frozenset[tuple[int, int]]] = set()
        # Maps (r, c) tuple to a set of line frozensets it belongs to
        self._coord_to_lines_map: dict[
            tuple[int, int], set[frozenset[tuple[int, int]]]
        ] = {}
        # --- End Line Information ---

        self._initialize_lines_and_index()
        logger.debug(
            f"GridData initialized ({self.rows}x{self.cols}) using NumPy arrays. Found {len(self.potential_lines)} potential lines."
        )

    def _initialize_death_zone(self, config: EnvConfig):
        """Initializes the death zone numpy array."""
        cols_per_row = config.COLS_PER_ROW
        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"COLS_PER_ROW length mismatch: {len(cols_per_row)} vs {self.rows}"
            )

        for r in range(self.rows):
            playable_width = cols_per_row[r]
            padding = self.cols - playable_width
            pad_left = padding // 2
            playable_start_col = pad_left
            playable_end_col = pad_left + playable_width
            for c in range(self.cols):
                if not (playable_start_col <= c < playable_end_col):
                    self._death_np[r, c] = True

    def _initialize_lines_and_index(self) -> None:
        """
        Precomputes potential lines (as coordinate sets) and creates a map
        from coordinates to the lines they belong to.
        """
        self.potential_lines = set()
        self._coord_to_lines_map = {}

        potential_lines_coords = _precompute_lines(self.config)

        for line_coords in potential_lines_coords:
            # Filter out lines containing death cells
            valid_line = True
            line_coord_set: set[tuple[int, int]] = set()
            for r, c in line_coords:
                # Use self.valid() and self._death_np directly
                if self.valid(r, c) and not self._death_np[r, c]:
                    line_coord_set.add((r, c))
                else:
                    valid_line = False
                    break  # Skip this line if any part is invalid/death

            if valid_line and len(line_coord_set) >= self.config.MIN_LINE_LENGTH:
                frozen_line = frozenset(line_coord_set)
                self.potential_lines.add(frozen_line)
                # Add to the reverse map
                for coord in line_coord_set:
                    if coord not in self._coord_to_lines_map:
                        self._coord_to_lines_map[coord] = set()
                    self._coord_to_lines_map[coord].add(frozen_line)

        logger.debug(
            f"Initialized {len(self.potential_lines)} potential lines and mapping for {len(self._coord_to_lines_map)} coordinates."
        )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_death(self, r: int, c: int) -> bool:
        """Checks if a cell is a death cell."""
        if not self.valid(r, c):
            return True  # Out of bounds is considered death
        # Cast NumPy bool_ to Python bool for type consistency
        return bool(self._death_np[r, c])

    def is_occupied(self, r: int, c: int) -> bool:
        """Checks if a cell is occupied (includes death cells)."""
        if not self.valid(r, c):
            return True  # Out of bounds is considered occupied
        # Cast NumPy bool_ to Python bool for type consistency
        return bool(self._occupied_np[r, c])

    def get_color_id(self, r: int, c: int) -> int:
        """Gets the color ID of a cell."""
        if not self.valid(r, c):
            return NO_COLOR_ID
        # Cast NumPy int8 to Python int for type consistency
        return int(self._color_id_np[r, c])

    def get_occupied_state(self) -> np.ndarray:
        """Returns a copy of the occupancy numpy array."""
        return self._occupied_np.copy()

    def get_death_state(self) -> np.ndarray:
        """Returns a copy of the death zone numpy array."""
        return self._death_np.copy()

    def get_color_id_state(self) -> np.ndarray:
        """Returns a copy of the color ID numpy array."""
        return self._color_id_np.copy()

    def deepcopy(self) -> "GridData":
        """
        Creates a deep copy of the grid data using NumPy array copying
        and standard dictionary/set copying for line data.
        """
        new_grid = GridData.__new__(
            GridData
        )  # Create new instance without calling __init__
        new_grid.rows = self.rows
        new_grid.cols = self.cols
        new_grid.config = self.config  # Config is likely immutable, shallow copy ok

        # 1. Copy NumPy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()
        new_grid._color_id_np = self._color_id_np.copy()

        # 2. Copy Line Data (Set of frozensets and Dict[Tuple, Set[frozenset]])
        # potential_lines contains immutable frozensets, shallow copy is fine
        new_grid.potential_lines = self.potential_lines.copy()
        # _coord_to_lines_map values are sets, need deepcopy
        new_grid._coord_to_lines_map = copy.deepcopy(self._coord_to_lines_map)

        # No Triangle objects or neighbors to handle anymore

        return new_grid

    def __str__(self) -> str:
        # Basic representation, could be enhanced to show grid visually
        occupied_count = np.sum(self._occupied_np & ~self._death_np)
        return f"GridData({self.rows}x{self.cols}, Occupied: {occupied_count})"

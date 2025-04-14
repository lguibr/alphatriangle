from __future__ import annotations


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up
        self.is_death = is_death
        self.is_occupied = is_death
        self.color: tuple[int, int, int] | None = None

        self.neighbor_left: Triangle | None = None
        self.neighbor_right: Triangle | None = None
        self.neighbor_vert: Triangle | None = None

    def get_points(
        self, ox: float, oy: float, cw: float, ch: float
    ) -> list[tuple[float, float]]:
        """Calculates vertex points for drawing, relative to origin (ox, oy)."""
        x = ox + self.col * (cw * 0.75)
        y = oy + self.row * ch
        if self.is_up:
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def copy(self) -> Triangle:
        """Creates a copy of the Triangle object's state (neighbors are not copied)."""
        new_tri = Triangle(self.row, self.col, self.is_up, self.is_death)
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color
        return new_tri

    def __repr__(self) -> str:
        state = "D" if self.is_death else ("O" if self.is_occupied else ".")
        orient = "^" if self.is_up else "v"
        return f"T({self.row},{self.col} {orient}{state})"

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.row == other.row and self.col == other.col

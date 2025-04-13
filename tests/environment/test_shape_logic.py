# File: tests/environment/test_shape_logic.py
import pytest
import random
from typing import List, Tuple, Set

from src.environment.core.game_state import GameState
from src.environment.grid.grid_data import GridData
from src.config import EnvConfig
from src.structs import Shape, Triangle, SHAPE_COLORS  # Import SHAPE_COLORS

# Correct the import name here
from src.environment.shapes.logic import (
    generate_random_shape,
    refill_shape_slots,
    is_shape_connected,
    get_neighbors,
    PREDEFINED_SHAPE_TEMPLATES,  # Use the correct name
)


def test_refill_shape_slots_empty(game_state: GameState):
    """Test refilling when all slots are empty."""
    gs = game_state
    # Ensure all slots are initially empty for this test scenario
    gs.shapes = [None] * gs.env_config.NUM_SHAPE_SLOTS
    assert all(s is None for s in gs.shapes)

    rng = random.Random(456)
    refill_shape_slots(gs, rng)

    assert all(isinstance(s, Shape) for s in gs.shapes)
    assert len(gs.shapes) == gs.env_config.NUM_SHAPE_SLOTS


def test_is_shape_connected():
    """Test the connectivity check."""
    # Connected L-shape (vertex connection)
    connected = [(0, 0, False), (0, 1, True), (1, 1, False)]
    assert is_shape_connected(connected), "L-shape should be connected"

    # Disconnected shape
    disconnected = [(0, 0, False), (2, 2, True)]
    assert not is_shape_connected(disconnected), "Disconnected shape failed"

    # Single triangle
    single = [(0, 0, False)]
    assert is_shape_connected(single), "Single triangle should be connected"

    # Empty list
    empty: List[Tuple[int, int, bool]] = []
    assert is_shape_connected(empty), "Empty list should be connected"

    # More complex connected shape
    complex_connected = [
        (0, 0, False),
        (0, 1, True),
        (1, 0, True),
        (1, 1, False),
    ]  # 2x2 block
    assert is_shape_connected(complex_connected), "2x2 block should be connected"

    # Complex disconnected
    complex_disconnected = [(0, 0, False), (0, 1, True), (2, 2, False), (2, 3, True)]
    assert not is_shape_connected(complex_disconnected), "Complex disconnected failed"


def test_get_neighbors():
    """Test neighbor calculation (including vertex neighbors)."""
    neighbors_down = get_neighbors(0, 0, False)
    expected_down = {(0, -1), (0, 1), (-1, 0)}
    assert set(neighbors_down) == expected_down

    neighbors_up = get_neighbors(0, 1, True)
    expected_up = {(0, 0), (0, 2), (1, 1)}
    assert set(neighbors_up) == set(expected_up)


def test_predefined_shapes_are_connected():
    """Verify that all predefined shapes are connected."""
    for i, template in enumerate(PREDEFINED_SHAPE_TEMPLATES):
        assert is_shape_connected(
            template
        ), f"Predefined shape {i} is not connected: {template}"

# File: tests/environment/test_shape_logic.py
import random

import pytest

from alphatriangle.environment import GameState
from alphatriangle.environment.shapes import logic as ShapeLogic
from alphatriangle.structs import Shape

# Fixtures are now implicitly injected from tests/environment/conftest.py


@pytest.fixture
def fixed_rng() -> random.Random:
    """Provides a Random instance with a fixed seed."""
    return random.Random(12345)


def test_generate_random_shape(fixed_rng: random.Random):
    """Test generating a single random shape."""
    shape = ShapeLogic.generate_random_shape(fixed_rng)
    assert isinstance(shape, Shape)
    assert shape.triangles is not None
    assert shape.color is not None
    assert len(shape.triangles) > 0
    # Check connectivity (optional but good)
    assert ShapeLogic.is_shape_connected(shape)


def test_generate_multiple_shapes(fixed_rng: random.Random):
    """Test generating multiple shapes to ensure variety (or lack thereof with fixed seed)."""
    shape1 = ShapeLogic.generate_random_shape(fixed_rng)
    # Re-seed or use different rng instance if true randomness is needed per call
    # For this test, using the same fixed_rng will likely produce the same shape again
    shape2 = ShapeLogic.generate_random_shape(fixed_rng)
    # --- REMOVED INCORRECT ASSERTION ---
    # assert shape1 == shape2  # Expect same shape due to fixed seed - THIS IS INCORRECT
    # --- END REMOVED ---
    # Check that subsequent calls produce different results with the same RNG instance
    assert shape1 != shape2, (
        "Two consecutive calls with the same RNG produced the exact same shape (template and color), which is highly unlikely."
    )

    # Use a different seed for variation
    rng2 = random.Random(54321)
    shape3 = ShapeLogic.generate_random_shape(rng2)
    # Check that different RNGs produce different results (highly likely)
    assert shape1 != shape3 or shape1.color != shape3.color


def test_refill_shape_slots_empty(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when all slots are initially empty."""
    game_state.shapes = [None] * game_state.env_config.NUM_SHAPE_SLOTS
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)
    assert len(game_state.shapes) == game_state.env_config.NUM_SHAPE_SLOTS


def test_refill_shape_slots_partial(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when some slots are empty - SHOULD NOT REFILL."""
    num_slots = game_state.env_config.NUM_SHAPE_SLOTS
    if num_slots < 2:
        pytest.skip("Test requires at least 2 shape slots")

    # Start with full slots
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)

    # Empty one slot
    game_state.shapes[0] = None
    # Store original state (important: copy shapes if they are mutable)
    original_shapes = [s.copy() if s else None for s in game_state.shapes]

    # Attempt refill - it should do nothing
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)

    # Check that shapes remain unchanged
    assert game_state.shapes == original_shapes, "Refill happened unexpectedly"


def test_refill_shape_slots_full(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when all slots are already full - SHOULD NOT REFILL."""
    # Start with full slots
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)
    original_shapes = [s.copy() if s else None for s in game_state.shapes]

    # Attempt refill - should do nothing
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)

    # Check shapes are unchanged
    assert game_state.shapes == original_shapes, "Refill happened when slots were full"


def test_refill_shape_slots_batch_trigger(game_state: GameState) -> None:
    """Test that refill only happens when ALL slots are empty."""
    num_slots = game_state.env_config.NUM_SHAPE_SLOTS
    if num_slots < 2:
        pytest.skip("Test requires at least 2 shape slots")

    # Fill all slots initially
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    initial_shapes = [s.copy() if s else None for s in game_state.shapes]
    assert all(s is not None for s in initial_shapes)

    # Empty one slot - refill should NOT happen
    game_state.shapes[0] = None
    shapes_after_one_empty = [s.copy() if s else None for s in game_state.shapes]
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    assert game_state.shapes == shapes_after_one_empty, (
        "Refill happened when only one slot was empty"
    )

    # Empty all slots - refill SHOULD happen
    game_state.shapes = [None] * num_slots
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    assert all(s is not None for s in game_state.shapes), (
        "Refill did not happen when all slots were empty"
    )
    # Check that the shapes are different from the initial ones (probabilistically)
    assert game_state.shapes != initial_shapes, (
        "Shapes after refill are identical to initial shapes (unlikely)"
    )

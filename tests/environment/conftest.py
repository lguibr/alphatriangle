import pytest

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig
    from alphatriangle.environment import GameState
    from alphatriangle.structs import Shape
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig
    from alphatriangle.environment import GameState
    from alphatriangle.structs import Shape


# Use session-scoped config from top-level conftest
@pytest.fixture(scope="session")
def default_env_config() -> EnvConfig:
    """Provides the default EnvConfig used in the specification (session-scoped)."""
    # Pydantic models with defaults can be instantiated without args
    return EnvConfig()


@pytest.fixture
def game_state(default_env_config: EnvConfig) -> GameState:
    """Provides a fresh GameState instance for testing."""
    # Use a fixed seed for reproducibility within tests if needed
    return GameState(config=default_env_config, initial_seed=123)


@pytest.fixture
def game_state_with_fixed_shapes(default_env_config: EnvConfig) -> GameState:
    """
    Provides a game state with predictable initial shapes.
    Uses a modified EnvConfig with NUM_SHAPE_SLOTS=3 for this specific fixture.
    """
    # Create a specific config for this fixture
    config_3_slots = default_env_config.model_copy(update={"NUM_SHAPE_SLOTS": 3})
    gs = GameState(config=config_3_slots, initial_seed=456)

    # Override the random shapes with fixed ones for testing placement/refill
    fixed_shapes = [
        Shape([(0, 0, False)], (255, 0, 0)),  # Single down (matches grid at 0,0)
        Shape([(0, 0, True)], (0, 255, 0)),  # Single up (matches grid at 0,1)
        Shape(
            [(0, 0, False), (0, 1, True)], (0, 0, 255)
        ),  # Domino (matches grid at 0,0 and 0,1)
    ]
    # This fixture now guarantees NUM_SHAPE_SLOTS is 3
    assert len(fixed_shapes) == gs.env_config.NUM_SHAPE_SLOTS

    for i in range(len(fixed_shapes)):
        gs.shapes[i] = fixed_shapes[i]
    return gs


@pytest.fixture
def simple_shape() -> Shape:
    """Provides a simple 3-triangle shape (Down, Up, Down)."""
    # Example: L-shape (Down at 0,0; Up at 1,0; Down at 1,1 relative)
    # Grid at (r,c) is Down if r+c is even, Up if odd.
    # (0,0) is Down. (1,0) is Up. (1,1) is Down. This shape matches grid orientation.
    triangles = [(0, 0, False), (1, 0, True), (1, 1, False)]
    color = (255, 0, 0)
    return Shape(triangles, color)

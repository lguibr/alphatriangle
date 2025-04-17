# File: tests/nn/test_model.py
import pytest
import torch

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, ModelConfig
    from alphatriangle.nn import AlphaTriangleNet
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, ModelConfig
    from alphatriangle.nn import AlphaTriangleNet


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def model(model_config: ModelConfig, env_config: EnvConfig) -> AlphaTriangleNet:
    """Provides an instance of the AlphaTriangleNet model."""
    return AlphaTriangleNet(model_config, env_config)


def test_model_initialization(
    model: AlphaTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test if the model initializes without errors."""
    assert model is not None
    # Cast ACTION_DIM to int for comparison
    assert model.action_dim == int(env_config.ACTION_DIM)
    # Add more checks based on config if needed (e.g., transformer presence)
    assert model.model_config.USE_TRANSFORMER == model_config.USE_TRANSFORMER
    if model_config.USE_TRANSFORMER:
        assert model.transformer_body is not None
    else:
        assert model.transformer_body is None


def test_model_forward_pass(
    model: AlphaTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test the forward pass with dummy input tensors."""
    batch_size = 4
    device = torch.device("cpu")  # Test on CPU
    model.to(device)
    model.eval()  # Set to eval mode
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)

    # Create dummy input tensors
    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        # --- CHANGED: Expect value_logits ---
        policy_logits, value_logits = model(dummy_grid, dummy_other)
        # --- END CHANGED ---

    # Check output shapes
    assert policy_logits.shape == (
        batch_size,
        action_dim_int,
    ), f"Policy logits shape mismatch: {policy_logits.shape}"
    # --- CHANGED: Check value logits shape ---
    assert value_logits.shape == (
        batch_size,
        model_config.NUM_VALUE_ATOMS,
    ), f"Value logits shape mismatch: {value_logits.shape}"
    # --- END CHANGED ---

    # Check output types
    assert policy_logits.dtype == torch.float32
    # --- CHANGED: Check value logits type ---
    assert value_logits.dtype == torch.float32
    # --- END CHANGED ---

    # --- REMOVED: Value range check (output is logits) ---
    # assert torch.all(value >= -1.0) and torch.all(value <= 1.0), (
    #     f"Value out of range [-1, 1]: {value}"
    # )
    # --- END REMOVED ---


@pytest.mark.parametrize(
    "use_transformer", [False, True], ids=["CNN_Only", "CNN_Transformer"]
)
def test_model_forward_transformer_toggle(use_transformer: bool, env_config: EnvConfig):
    """Test forward pass with transformer enabled/disabled."""
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)
    # Create a specific model config for this test, providing all required fields
    # --- CHANGED: Use default distributional params from ModelConfig ---
    model_config_test = ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4, 8],  # Simple CNN
        CONV_KERNEL_SIZES=[3, 3],
        CONV_STRIDES=[1, 1],
        CONV_PADDING=[1, 1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=8,
        USE_TRANSFORMER=use_transformer,
        TRANSFORMER_DIM=16,
        TRANSFORMER_HEADS=2,
        TRANSFORMER_LAYERS=1,
        TRANSFORMER_FC_DIM=32,
        FC_DIMS_SHARED=[16],
        POLICY_HEAD_DIMS=[action_dim_int],  # Use casted int
        # VALUE_HEAD_DIMS=[1], # Use default from ModelConfig
        OTHER_NN_INPUT_FEATURES_DIM=10,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=True,
        # NUM_VALUE_ATOMS=51, # Use default
        # VALUE_MIN=-10.0, # Use default
        # VALUE_MAX=10.0, # Use default
    )
    # --- END CHANGED ---
    model = AlphaTriangleNet(model_config_test, env_config)
    batch_size = 2
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config_test.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config_test.OTHER_NN_INPUT_FEATURES_DIM)
    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        # --- CHANGED: Expect value_logits ---
        policy_logits, value_logits = model(dummy_grid, dummy_other)
        # --- END CHANGED ---

    assert policy_logits.shape == (batch_size, action_dim_int)
    # --- CHANGED: Check value logits shape ---
    assert value_logits.shape == (batch_size, model_config_test.NUM_VALUE_ATOMS)
    # --- END CHANGED ---

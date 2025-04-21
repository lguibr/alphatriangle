# File: tests/nn/test_model.py
import pytest
import torch

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig  # UPDATED IMPORT

# Keep alphatriangle imports
from alphatriangle.config import ModelConfig
from alphatriangle.nn import AlphaTriangleNet


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:  # Uses trianglengin.EnvConfig
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def model(model_config: ModelConfig, env_config: EnvConfig) -> AlphaTriangleNet:
    """Provides an instance of the AlphaTriangleNet model."""
    # Pass trianglengin.EnvConfig
    return AlphaTriangleNet(model_config, env_config)


def test_model_initialization(
    model: AlphaTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test if the model initializes without errors."""
    assert model is not None
    # Calculate action_dim manually for comparison
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    assert model.action_dim == action_dim_int
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
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    # Calculate action_dim manually
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)

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
        policy_logits, value_logits = model(dummy_grid, dummy_other)

    assert policy_logits.shape == (
        batch_size,
        action_dim_int,
    ), f"Policy logits shape mismatch: {policy_logits.shape}"
    assert value_logits.shape == (
        batch_size,
        model_config.NUM_VALUE_ATOMS,
    ), f"Value logits shape mismatch: {value_logits.shape}"

    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32


@pytest.mark.parametrize(
    "use_transformer", [False, True], ids=["CNN_Only", "CNN_Transformer"]
)
def test_model_forward_transformer_toggle(use_transformer: bool, env_config: EnvConfig):
    """Test forward pass with transformer enabled/disabled."""
    # Calculate action_dim manually
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    model_config_test = ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4, 8],
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
        POLICY_HEAD_DIMS=[action_dim_int],  # Use calculated int
        OTHER_NN_INPUT_FEATURES_DIM=10,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=True,
    )
    # Pass trianglengin.EnvConfig
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
        policy_logits, value_logits = model(dummy_grid, dummy_other)

    assert policy_logits.shape == (batch_size, action_dim_int)
    assert value_logits.shape == (batch_size, model_config_test.NUM_VALUE_ATOMS)

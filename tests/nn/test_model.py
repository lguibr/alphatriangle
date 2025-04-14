# File: tests/nn/test_model.py
import pytest
import torch

from src.config import EnvConfig, ModelConfig
from src.nn import AlphaTriangleNet


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
    assert model.action_dim == env_config.ACTION_DIM
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
        policy_logits, value = model(dummy_grid, dummy_other)

    # Check output shapes
    assert policy_logits.shape == (
        batch_size,
        env_config.ACTION_DIM,
    ), f"Policy logits shape mismatch: {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Value shape mismatch: {value.shape}"

    # Check output types
    assert policy_logits.dtype == torch.float32
    assert value.dtype == torch.float32

    # Check value range (should be within [-1, 1] due to Tanh)
    assert torch.all(value >= -1.0) and torch.all(
        value <= 1.0
    ), f"Value out of range [-1, 1]: {value}"


@pytest.mark.parametrize(
    "use_transformer", [False, True], ids=["CNN_Only", "CNN_Transformer"]
)
def test_model_forward_transformer_toggle(use_transformer: bool, env_config: EnvConfig):
    """Test forward pass with transformer enabled/disabled."""
    # Create a specific model config for this test, providing all required fields
    model_config_test = ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4, 8],  # Simple CNN
        CONV_KERNEL_SIZES=[3, 3],
        CONV_STRIDES=[1, 1],
        CONV_PADDING=[1, 1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=8,  # Provide default even if blocks=0
        USE_TRANSFORMER=use_transformer,
        TRANSFORMER_DIM=16,  # Make different from CNN output
        TRANSFORMER_HEADS=2,
        TRANSFORMER_LAYERS=1,
        TRANSFORMER_FC_DIM=32,
        FC_DIMS_SHARED=[16],
        # ACTION_DIM is already int
        POLICY_HEAD_DIMS=[env_config.ACTION_DIM],
        VALUE_HEAD_DIMS=[1],
        OTHER_NN_INPUT_FEATURES_DIM=10,
        ACTIVATION_FUNCTION="ReLU",  # Provide default
        USE_BATCH_NORM=True,  # Provide default
    )
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
        policy_logits, value = model(dummy_grid, dummy_other)

    assert policy_logits.shape == (batch_size, env_config.ACTION_DIM)
    assert value.shape == (batch_size, 1)

# File: tests/nn/test_network.py
import numpy as np
import pytest
import torch

# Import GameState and EnvConfig from trianglengin's top level
from trianglengin import EnvConfig, GameState

# Keep alphatriangle imports
from alphatriangle.config import ModelConfig, TrainConfig
from alphatriangle.features import extract_state_features
from alphatriangle.nn import NeuralNetwork
from alphatriangle.nn.network import NetworkEvaluationError
from alphatriangle.utils.types import StateType


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:  # Uses trianglengin.EnvConfig
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    # Explicitly disable compilation for tests unless specifically testing compilation
    cfg = mock_train_config.model_copy(deep=True)
    cfg.COMPILE_MODEL = False
    return cfg


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
    train_config: TrainConfig,  # Use the modified train_config fixture
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing."""
    device = torch.device("cpu")
    # Pass trianglengin.EnvConfig and the modified train_config
    nn_interface_instance = NeuralNetwork(
        model_config, env_config, train_config, device
    )
    nn_interface_instance.model.eval()
    return nn_interface_instance


@pytest.fixture
def game_state(env_config: EnvConfig) -> GameState:  # Uses trianglengin.GameState
    """Provides a fresh GameState instance."""
    # Pass trianglengin.EnvConfig
    return GameState(config=env_config, initial_seed=123)


def test_network_initialization(
    nn_interface: NeuralNetwork,
    model_config: ModelConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
):
    """Test if the NeuralNetwork wrapper initializes correctly."""
    assert nn_interface.model is not None
    assert nn_interface.device == torch.device("cpu")
    assert nn_interface.model_config == model_config
    assert nn_interface.env_config == env_config  # Check env_config storage
    # Calculate action_dim manually for comparison
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    assert nn_interface.action_dim == action_dim_int


def test_state_to_tensors(
    nn_interface: NeuralNetwork,
    game_state: GameState,  # Uses trianglengin.GameState
    model_config: ModelConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
):
    """Test the conversion of a GameState to tensors."""
    grid_tensor, other_tensor = nn_interface._state_to_tensors(game_state)

    assert isinstance(grid_tensor, torch.Tensor)
    assert isinstance(other_tensor, torch.Tensor)

    assert grid_tensor.shape == (
        1,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    assert other_tensor.shape == (1, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    assert grid_tensor.device == nn_interface.device
    assert other_tensor.device == nn_interface.device


def test_batch_states_to_tensors(
    nn_interface: NeuralNetwork,
    game_state: GameState,  # Uses trianglengin.GameState
    model_config: ModelConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
):
    """Test the conversion of a batch of GameStates to tensors."""
    batch_size = 3
    states = [game_state.copy() for _ in range(batch_size)]
    grid_tensor, other_tensor = nn_interface._batch_states_to_tensors(states)

    assert isinstance(grid_tensor, torch.Tensor)
    assert isinstance(other_tensor, torch.Tensor)

    assert grid_tensor.shape == (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    assert other_tensor.shape == (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    assert grid_tensor.device == nn_interface.device
    assert other_tensor.device == nn_interface.device


def test_evaluate_state(
    nn_interface: NeuralNetwork,
    game_state: GameState,  # Uses trianglengin.GameState
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
):
    """Test evaluating a single GameState."""
    policy_dict, value = nn_interface.evaluate_state(game_state)

    assert isinstance(policy_dict, dict)
    assert isinstance(value, float)
    # Calculate action_dim manually for comparison
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    assert len(policy_dict) == action_dim_int
    assert all(isinstance(k, int) for k in policy_dict)
    assert all(isinstance(v, float) for v in policy_dict.values())
    assert abs(sum(policy_dict.values()) - 1.0) < 1e-5


def test_evaluate_batch(
    nn_interface: NeuralNetwork,
    game_state: GameState,  # Uses trianglengin.GameState
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
):
    """Test evaluating a batch of GameStates."""
    batch_size = 3
    states = [game_state.copy() for _ in range(batch_size)]
    results = nn_interface.evaluate_batch(states)

    assert isinstance(results, list)
    assert len(results) == batch_size

    # Calculate action_dim manually for comparison
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    for policy_dict, value in results:
        assert isinstance(policy_dict, dict)
        assert isinstance(value, float)
        assert len(policy_dict) == action_dim_int
        assert all(isinstance(k, int) for k in policy_dict)
        assert all(isinstance(v, float) for v in policy_dict.values())
        assert abs(sum(policy_dict.values()) - 1.0) < 1e-5


def test_get_set_weights(nn_interface: NeuralNetwork):
    """Test getting and setting model weights."""
    initial_weights = nn_interface.get_weights()
    assert isinstance(initial_weights, dict)
    assert all(isinstance(v, torch.Tensor) for v in initial_weights.values())
    assert all(v.device == torch.device("cpu") for v in initial_weights.values())

    # Modify weights slightly by creating NEW tensors
    modified_weights = {}
    params_modified = False
    float_keys_modified = []
    for k, v in initial_weights.items():
        if v.dtype.is_floating_point:
            # Create a completely new tensor of ONES to ensure difference
            # (assuming not all weights are initialized to 1)
            modified_weights[k] = torch.ones_like(v)
            params_modified = True
            float_keys_modified.append(k)
            # REMOVED internal assertion that failed when initial weight was already 1
            # assert not torch.equal(
            #     initial_weights[k], modified_weights[k]
            # ), f"Weight {k} did not change after creating ones_like!"
        else:
            modified_weights[k] = v  # Keep non-float tensors (buffers) unchanged

    assert params_modified, "No floating point parameters found to modify in state_dict"

    nn_interface.set_weights(modified_weights)
    new_weights = nn_interface.get_weights()

    # Check if weights were actually updated
    weights_changed = False
    for k in float_keys_modified:
        # Check if the tensor retrieved AFTER setting is different from the INITIAL tensor
        if not torch.equal(initial_weights[k], new_weights[k]):
            weights_changed = True
            break

    assert weights_changed, "Floating point weights did not change after set_weights"

    # Check if the loaded weights match the modified ones we intended to set
    assert all(
        torch.allclose(modified_weights[k], new_weights[k], atol=1e-6)
        for k in modified_weights
    )


def test_evaluate_state_with_nan_features(
    nn_interface: NeuralNetwork,
    game_state: GameState,  # Uses trianglengin.GameState
    mocker,
):
    """Test that evaluation raises error if features contain NaN."""

    def mock_extract_nan(*args, **kwargs) -> StateType:
        state_dict = extract_state_features(*args, **kwargs)
        state_dict["other_features"][0] = np.nan  # Inject NaN
        return state_dict

    mocker.patch("alphatriangle.nn.network.extract_state_features", mock_extract_nan)

    with pytest.raises(NetworkEvaluationError, match="Non-finite values found"):
        nn_interface.evaluate_state(game_state)


def test_evaluate_batch_with_nan_features(
    nn_interface: NeuralNetwork,
    game_state: GameState,  # Uses trianglengin.GameState
    mocker,
):
    """Test that batch evaluation raises error if features contain NaN."""
    batch_size = 2
    states = [game_state.copy() for _ in range(batch_size)]

    def mock_extract_nan_batch(*args, **kwargs) -> StateType:
        state_dict = extract_state_features(*args, **kwargs)
        # Inject NaN into the first element of the batch only
        if args[0] is states[0]:
            state_dict["other_features"][0] = np.nan
        return state_dict

    mocker.patch(
        "alphatriangle.nn.network.extract_state_features", mock_extract_nan_batch
    )

    with pytest.raises(NetworkEvaluationError, match="Non-finite values found"):
        nn_interface.evaluate_batch(states)

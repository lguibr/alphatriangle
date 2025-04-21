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
    # Ensure model is in eval mode for consistency, although set_weights should handle it
    nn_interface_instance.model.eval()
    nn_interface_instance._orig_model.eval()
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
    # Ensure model is on CPU for this test
    nn_interface._orig_model.cpu()
    nn_interface.device = torch.device("cpu")

    initial_weights_state_dict = nn_interface.get_weights()
    assert isinstance(initial_weights_state_dict, dict)
    assert all(isinstance(v, torch.Tensor) for v in initial_weights_state_dict.values())
    assert all(
        v.device == torch.device("cpu") for v in initial_weights_state_dict.values()
    )

    # Create a deep copy of initial weights for comparison later
    initial_weights_copy = {
        k: v.clone().detach() for k, v in initial_weights_state_dict.items()
    }

    # Modify weights: Create a new state dict with zeros for float params
    modified_weights_state_dict = {}
    params_modified = False
    float_keys_modified = []

    for k, v in initial_weights_state_dict.items():
        if v.dtype.is_floating_point:
            modified_weights_state_dict[k] = torch.zeros_like(v)
            params_modified = True
            float_keys_modified.append(k)
            # Check if modification actually happened (unless initial was already zero)
            if not torch.all(initial_weights_state_dict[k] == 0):
                assert not torch.equal(
                    initial_weights_state_dict[k], modified_weights_state_dict[k]
                ), f"Weight {k} did not change after creating zeros_like!"
        else:
            # Keep non-float tensors (buffers) unchanged, ensure they are detached copies
            modified_weights_state_dict[k] = v.clone().detach()

    assert params_modified, "No floating point parameters found to modify in state_dict"

    # --- Set the modified weights ---
    nn_interface.set_weights(modified_weights_state_dict)

    # --- Direct Check: Compare model parameters immediately after setting ---
    direct_check_passed = True
    mismatched_params = []
    with torch.no_grad():
        # Use named_parameters which only includes trainable parameters
        for name, param in nn_interface._orig_model.named_parameters():
            if name in modified_weights_state_dict:
                expected_tensor = modified_weights_state_dict[name]
                actual_tensor = param.data
                if not torch.allclose(actual_tensor, expected_tensor, atol=1e-6):
                    direct_check_passed = False
                    mismatched_params.append(name)
                    print(f"Direct Param Check Mismatch for key '{name}':")
                    print(f"  Expected (modified): {expected_tensor.flatten()[:5]}...")
                    print(f"  Got (actual param):  {actual_tensor.flatten()[:5]}...")
            else:
                print(
                    f"Warning: Parameter '{name}' not found in modified_weights_state_dict during direct check."
                )

    assert direct_check_passed, (
        f"Direct parameter check failed after set_weights for keys: {mismatched_params}"
    )

    # --- Get weights again using the interface method ---
    new_weights_state_dict = nn_interface.get_weights()

    # --- Comparison using state dicts ---
    weights_changed_from_initial = False
    mismatched_keys_final = []

    for k in float_keys_modified:
        initial_tensor = initial_weights_copy[k]  # Compare against the initial copy
        new_tensor = new_weights_state_dict[k]
        modified_tensor = modified_weights_state_dict[k]

        # Check 1: Did the tensor change from the initial state?
        if not torch.equal(initial_tensor, new_tensor):
            weights_changed_from_initial = True

        # Check 2: Does the new tensor match the one we tried to set?
        if not torch.allclose(new_tensor, modified_tensor, atol=1e-6):
            mismatched_keys_final.append(k)
            print(f"Final State Dict Mismatch for key '{k}':")
            print(f"  Expected (modified): {modified_tensor.flatten()[:5]}...")
            print(f"  Got (new state dict):{new_tensor.flatten()[:5]}...")
            print(f"  Initial:             {initial_tensor.flatten()[:5]}...")

    # Assert that at least one weight changed from the initial state
    assert weights_changed_from_initial, (
        "Floating point weights did not change from initial state after set_weights/get_weights cycle"
    )

    # Assert that all modified weights match the expected values in the final state dict
    assert not mismatched_keys_final, (
        f"Weights in final state_dict did not match expected values after set_weights for keys: {mismatched_keys_final}"
    )

    # Final check: ensure all keys match (including buffers)
    assert all(
        (
            torch.allclose(
                modified_weights_state_dict[k], new_weights_state_dict[k], atol=1e-6
            )
            if modified_weights_state_dict[k].dtype.is_floating_point
            else torch.equal(modified_weights_state_dict[k], new_weights_state_dict[k])
        )
        for k in modified_weights_state_dict
    ), (
        "Mismatch between modified_weights and new_weights across all keys in final state dict."
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

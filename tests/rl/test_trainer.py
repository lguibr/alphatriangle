# File: tests/rl/test_trainer.py
import numpy as np
import pytest
import torch

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig  # UPDATED IMPORT

# Keep alphatriangle imports
from alphatriangle.config import ModelConfig, TrainConfig
from alphatriangle.nn import NeuralNetwork
from alphatriangle.rl import ExperienceBuffer, Trainer
from alphatriangle.utils.types import Experience, PERBatchSample, StateType


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:  # Uses trianglengin.EnvConfig
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    mock_model_config.OTHER_NN_INPUT_FEATURES_DIM = 10
    return mock_model_config


@pytest.fixture
def train_config_uniform(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = False
    return cfg


@pytest.fixture
def train_config_per(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = True
    cfg.PER_BETA_ANNEAL_STEPS = 100
    return cfg


@pytest.fixture
def nn_interface(
    mock_model_config: ModelConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
    train_config_uniform: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing, configured for uniform buffer."""
    device = torch.device("cpu")
    # Pass trianglengin.EnvConfig
    nn_interface_instance = NeuralNetwork(
        mock_model_config, env_config, train_config_uniform, device
    )
    nn_interface_instance.model.to(device)
    nn_interface_instance.model.eval()
    return nn_interface_instance


@pytest.fixture
def trainer_uniform(
    nn_interface: NeuralNetwork,
    train_config_uniform: TrainConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
) -> Trainer:
    """Provides a Trainer instance configured for uniform sampling."""
    # Pass trianglengin.EnvConfig
    return Trainer(nn_interface, train_config_uniform, env_config)


@pytest.fixture
def trainer_per(
    nn_interface: NeuralNetwork,
    train_config_per: TrainConfig,
    env_config: EnvConfig,  # Uses trianglengin.EnvConfig
) -> Trainer:
    """Provides a Trainer instance configured for PER."""
    # Pass trianglengin.EnvConfig
    return Trainer(nn_interface, train_config_per, env_config)


@pytest.fixture
def buffer_uniform(
    train_config_uniform: TrainConfig, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a filled uniform buffer."""
    buffer = ExperienceBuffer(train_config_uniform)
    for i in range(buffer.min_size_to_train + 5):
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (
            state_copy,
            mock_experience[1],
            mock_experience[2] + i * 0.1,
        )
        buffer.add(exp_copy)
    return buffer


@pytest.fixture
def buffer_per(
    train_config_per: TrainConfig, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a filled PER buffer."""
    buffer = ExperienceBuffer(train_config_per)
    for i in range(buffer.min_size_to_train + 5):
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (
            state_copy,
            mock_experience[1],
            mock_experience[2] + i * 0.1,
        )
        buffer.add(exp_copy)
    return buffer


def test_trainer_initialization(trainer_uniform: Trainer):
    assert trainer_uniform.nn is not None
    assert trainer_uniform.model is not None
    assert trainer_uniform.optimizer is not None
    assert hasattr(trainer_uniform, "scheduler")


def test_prepare_batch(trainer_uniform: Trainer, mock_experience: Experience):
    """Test the internal _prepare_batch method."""
    batch_size = trainer_uniform.train_config.BATCH_SIZE
    batch = [mock_experience] * batch_size
    grid_t, other_t, policy_target_t, n_step_return_t = trainer_uniform._prepare_batch(
        batch
    )

    assert grid_t.shape == (
        batch_size,
        trainer_uniform.model_config.GRID_INPUT_CHANNELS,
        trainer_uniform.env_config.ROWS,
        trainer_uniform.env_config.COLS,
    )
    assert other_t.shape == (
        batch_size,
        trainer_uniform.model_config.OTHER_NN_INPUT_FEATURES_DIM,
    )
    # Calculate action_dim manually for comparison
    action_dim_int = int(
        trainer_uniform.env_config.NUM_SHAPE_SLOTS
        * trainer_uniform.env_config.ROWS
        * trainer_uniform.env_config.COLS
    )
    assert policy_target_t.shape == (batch_size, action_dim_int)
    assert n_step_return_t.shape == (batch_size,)

    assert grid_t.device == trainer_uniform.device
    assert other_t.device == trainer_uniform.device
    assert policy_target_t.device == trainer_uniform.device
    assert n_step_return_t.device == trainer_uniform.device


def test_train_step_uniform(trainer_uniform: Trainer, buffer_uniform: ExperienceBuffer):
    """Test a single training step with uniform sampling."""
    initial_params = [p.clone() for p in trainer_uniform.model.parameters()]
    sample = buffer_uniform.sample(trainer_uniform.train_config.BATCH_SIZE)
    assert sample is not None

    result = trainer_uniform.train_step(sample)

    assert result is not None
    loss_info, td_errors = result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert loss_info["total_loss"] > 0

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer_uniform.train_config.BATCH_SIZE,)

    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer_uniform.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change after training step."


def test_train_step_per(trainer_per: Trainer, buffer_per: ExperienceBuffer):
    """Test a single training step with PER."""
    initial_params = [p.clone() for p in trainer_per.model.parameters()]
    sample = buffer_per.sample(
        trainer_per.train_config.BATCH_SIZE, current_train_step=10
    )
    assert sample is not None

    result = trainer_per.train_step(sample)

    assert result is not None
    loss_info, td_errors = result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert loss_info["total_loss"] > 0

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer_per.train_config.BATCH_SIZE,)
    assert np.all(np.isfinite(td_errors))

    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer_per.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change after training step."


def test_train_step_empty_batch(trainer_uniform: Trainer):
    """Test train_step with an empty batch."""
    empty_sample: PERBatchSample = {
        "batch": [],
        "indices": np.array([]),
        "weights": np.array([]),
    }
    result = trainer_uniform.train_step(empty_sample)
    assert result is None


def test_get_current_lr(trainer_uniform: Trainer):
    """Test retrieving the current learning rate."""
    lr = trainer_uniform.get_current_lr()
    assert isinstance(lr, float)
    assert lr == trainer_uniform.train_config.LEARNING_RATE

    if trainer_uniform.scheduler:
        trainer_uniform.scheduler.step()
        lr_after_step = trainer_uniform.get_current_lr()
        assert isinstance(lr_after_step, float)

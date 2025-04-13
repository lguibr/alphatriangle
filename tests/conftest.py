# File: tests/conftest.py
# Top-level conftest for sharing session-scoped fixtures
import pytest
import torch
import torch.optim as optim  # Added for mock_optimizer
import numpy as np  # Added for mock_state_type
import random  # Added for mock_experience
from collections import deque  # Added for filled_mock_buffer

from src.config import EnvConfig, ModelConfig, TrainConfig, MCTSConfig

# Added imports for moved fixtures
from src.utils.types import StateType, Experience
from src.nn import NeuralNetwork, AlphaTriangleNet
from src.rl import ExperienceBuffer, Trainer


@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    """Provides a default, *valid* EnvConfig for tests (session-scoped)."""
    # Use a smaller, fully playable grid for easier testing of placement logic
    rows = 3
    cols = 3
    cols_per_row = [cols] * rows
    return EnvConfig(ROWS=rows, COLS=cols, COLS_PER_ROW=cols_per_row, NUM_SHAPE_SLOTS=1)


@pytest.fixture(scope="session")
def mock_model_config(mock_env_config: EnvConfig) -> ModelConfig:
    """Provides a default ModelConfig compatible with mock_env_config (session-scoped)."""
    # Simple CNN config for testing
    return ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4],
        CONV_KERNEL_SIZES=[3],
        CONV_STRIDES=[1],
        CONV_PADDING=[1],
        NUM_RESIDUAL_BLOCKS=0,
        USE_TRANSFORMER=False,
        FC_DIMS_SHARED=[8],
        POLICY_HEAD_DIMS=[mock_env_config.ACTION_DIM],  # Match action dim
        VALUE_HEAD_DIMS=[1],
        OTHER_NN_INPUT_FEATURES_DIM=10,  # Simplified feature dim for testing
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    """Provides a default TrainConfig for tests (session-scoped)."""
    return TrainConfig(
        BATCH_SIZE=4,
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        USE_PER=False,  # Default to uniform for simpler tests unless specified
    )


@pytest.fixture(scope="session")
def mock_mcts_config() -> MCTSConfig:
    """Provides a default MCTSConfig for tests (session-scoped)."""
    return MCTSConfig(
        num_simulations=10,
        puct_coefficient=1.5,
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_anneal_steps=5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_search_depth=10,
    )


# --- Fixtures Moved from tests/mcts/conftest.py ---


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    """Creates a mock StateType dictionary with correct shapes."""
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": np.random.rand(*grid_shape).astype(np.float32),
        "other_features": np.random.rand(*other_shape).astype(np.float32),
    }


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_experience(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> Experience:
    """Creates a mock Experience tuple."""
    policy_target = (
        {a: 1.0 / mock_env_config.ACTION_DIM for a in range(mock_env_config.ACTION_DIM)}
        if mock_env_config.ACTION_DIM > 0
        else {0: 1.0}
    )
    value_target = random.uniform(-1, 1)
    return (mock_state_type, policy_target, value_target)


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance with a mock model for testing."""
    device = torch.device("cpu")  # Use CPU for testing
    nn_interface = NeuralNetwork(
        mock_model_config, mock_env_config, mock_train_config, device
    )
    # Optionally replace internal model with a simpler mock if needed,
    # but using the actual AlphaTriangleNet with simple config is often better.
    return nn_interface


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance."""
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    """Provides the optimizer from the mock_trainer."""
    return mock_trainer.optimizer


@pytest.fixture  # Buffer should likely be function-scoped unless state doesn't matter
def mock_experience_buffer(mock_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an ExperienceBuffer instance."""
    return ExperienceBuffer(mock_train_config)


@pytest.fixture  # Buffer should likely be function-scoped unless state doesn't matter
def filled_mock_buffer(
    mock_experience_buffer: ExperienceBuffer, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a buffer filled with some mock experiences."""
    for i in range(mock_experience_buffer.min_size_to_train + 5):
        # Create slightly different experiences
        state_copy = {k: v.copy() for k, v in mock_experience[0].items()}
        # Ensure grid is writeable before modifying
        if not state_copy["grid"].flags.writeable:
            state_copy["grid"] = state_copy["grid"].copy()
        state_copy["grid"] += (
            np.random.randn(*state_copy["grid"].shape).astype(np.float32) * 0.1
        )
        exp_copy = (state_copy, mock_experience[1], random.uniform(-1, 1))
        mock_experience_buffer.add(exp_copy)
    return mock_experience_buffer

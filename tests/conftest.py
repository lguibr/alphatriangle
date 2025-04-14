# File: tests/conftest.py
# Top-level conftest for sharing session-scoped fixtures
import random

import numpy as np
import pytest
import torch
import torch.optim as optim  # Added for mock_optimizer

from src.config import EnvConfig, MCTSConfig, ModelConfig, TrainConfig

# Added imports for moved fixtures
from src.nn import NeuralNetwork
from src.rl import ExperienceBuffer, Trainer
from src.utils.types import Experience, StateType

# Use default NumPy random number generator
rng = np.random.default_rng()


@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    """Provides a default, *valid* EnvConfig for tests (session-scoped)."""
    # Use a smaller, fully playable grid for easier testing of placement logic
    rows = 3
    cols = 3
    cols_per_row = [cols] * rows
    return EnvConfig(
        ROWS=rows,
        COLS=cols,
        COLS_PER_ROW=cols_per_row,
        NUM_SHAPE_SLOTS=1,
        MIN_LINE_LENGTH=3,  # Provide default
    )


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
        RESIDUAL_BLOCK_FILTERS=4,  # Provide default
        USE_TRANSFORMER=False,
        TRANSFORMER_DIM=16,  # Provide default
        TRANSFORMER_HEADS=2,  # Provide default
        TRANSFORMER_LAYERS=0,  # Provide default
        TRANSFORMER_FC_DIM=32,  # Provide default
        FC_DIMS_SHARED=[8],
        # Cast ACTION_DIM to int
        POLICY_HEAD_DIMS=[int(mock_env_config.ACTION_DIM)],
        VALUE_HEAD_DIMS=[1],
        OTHER_NN_INPUT_FEATURES_DIM=10,  # Simplified feature dim for testing
        ACTIVATION_FUNCTION="ReLU",  # Provide default
        USE_BATCH_NORM=True,  # Provide default
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    """Provides a default TrainConfig for tests (session-scoped)."""
    return TrainConfig(
        BATCH_SIZE=4,
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        USE_PER=False,  # Default to uniform for simpler tests unless specified
        # Provide defaults for other required fields
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.0,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=100,
        PER_EPSILON=1e-5,
        MAX_TRAINING_STEPS=200,  # Set a finite value for tests
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
        "grid": rng.random(grid_shape, dtype=np.float32),
        "other_features": rng.random(other_shape, dtype=np.float32),
    }


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_experience(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> Experience:
    """Creates a mock Experience tuple."""
    action_dim = int(mock_env_config.ACTION_DIM)  # Cast to int
    policy_target = (
        dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
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
    for _ in range(mock_experience_buffer.min_size_to_train + 5):
        # Create slightly different experiences
        # Ensure dict values are copied correctly
        state_copy: StateType = {k: v.copy() for k, v in mock_experience[0].items()}
        # Ensure grid is writeable before modifying
        if not state_copy["grid"].flags.writeable:
            state_copy["grid"] = state_copy["grid"].copy()
        state_copy["grid"] += (
            rng.standard_normal(state_copy["grid"].shape, dtype=np.float32) * 0.1
        )
        # Cast state_copy to the correct type for the buffer
        exp_copy: Experience = (state_copy, mock_experience[1], random.uniform(-1, 1))
        mock_experience_buffer.add(exp_copy)
    return mock_experience_buffer

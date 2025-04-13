# File: tests/conftest.py
# Top-level conftest for sharing session-scoped fixtures
import pytest
import torch
from src.config import EnvConfig, ModelConfig, TrainConfig, MCTSConfig


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

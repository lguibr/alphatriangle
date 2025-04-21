from collections import deque
from typing import cast  # Import cast

import numpy as np
import pytest

from alphatriangle.config import TrainConfig
from alphatriangle.rl import ExperienceBuffer
from alphatriangle.utils.sumtree import SumTree
from alphatriangle.utils.types import (
    Experience,
    SerializableShapeInfo,  # Import new type
    StateType,
)

# Use module-level rng from tests/conftest.py
from tests.conftest import rng

# --- Fixtures ---


@pytest.fixture
def uniform_train_config() -> TrainConfig:
    """TrainConfig for uniform buffer."""
    return TrainConfig(
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        BATCH_SIZE=4,
        USE_PER=False,
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
        N_STEP_RETURNS=3,
        GAMMA=0.99,
    )


@pytest.fixture
def per_train_config() -> TrainConfig:
    """TrainConfig for PER buffer."""
    return TrainConfig(
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        BATCH_SIZE=4,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=50,  # Short anneal for testing
        PER_EPSILON=1e-5,
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
        MAX_TRAINING_STEPS=200,  # Set a finite value for tests
        N_STEP_RETURNS=3,
        GAMMA=0.99,
    )


@pytest.fixture
def uniform_buffer(uniform_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an empty uniform ExperienceBuffer."""
    return ExperienceBuffer(uniform_train_config)


@pytest.fixture
def per_buffer(per_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an empty PER ExperienceBuffer."""
    return ExperienceBuffer(per_train_config)


# Use shared mock_experience fixture implicitly from tests/conftest.py


# --- Uniform Buffer Tests ---


def test_uniform_buffer_init(uniform_buffer: ExperienceBuffer):
    assert not uniform_buffer.use_per
    assert isinstance(uniform_buffer.buffer, deque)
    assert uniform_buffer.capacity == 100
    assert len(uniform_buffer) == 0
    assert not uniform_buffer.is_ready()


# Use mock_experience directly injected by pytest
def test_uniform_buffer_add(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    assert len(uniform_buffer) == 0
    uniform_buffer.add(mock_experience)
    assert len(uniform_buffer) == 1
    # Check if the added experience is the same object (or equal)
    # Note: Deep comparison might be needed if copies are made internally
    assert uniform_buffer.buffer[0] == mock_experience


# Use mock_experience directly injected by pytest
def test_uniform_buffer_add_batch(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Create copies for the batch to avoid adding the same object reference multiple times
    batch: list[Experience] = [
        (
            cast(
                "StateType",  # Cast the dictionary to StateType
                {
                    "grid": mock_experience[0]["grid"].copy(),
                    "other_features": mock_experience[0]["other_features"].copy(),
                    "available_shapes_geometry": [
                        (list(geom), cid) if geom_info else None
                        for geom_info in mock_experience[0]["available_shapes_geometry"]
                        if geom_info is not None
                        for geom, cid in [geom_info]
                    ],
                },
            ),
            mock_experience[1],
            mock_experience[2],
        )
        for _ in range(5)
    ]
    uniform_buffer.add_batch(batch)
    assert len(uniform_buffer) == 5


# Use mock_experience directly injected by pytest
def test_uniform_buffer_capacity(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    for i in range(uniform_buffer.capacity + 10):
        # Create slightly different experiences with deep copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid + i))
            else:
                geometry_copy.append(None)

        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        uniform_buffer.add(exp_copy)
    assert len(uniform_buffer) == uniform_buffer.capacity
    # Check if the first added element is gone (check based on value)
    first_added_val = mock_experience[2] + 0
    assert not any(exp[2] == first_added_val for exp in uniform_buffer.buffer)
    # Check if the last added element is present (check based on value)
    last_added_val = mock_experience[2] + uniform_buffer.capacity + 9
    assert any(exp[2] == last_added_val for exp in uniform_buffer.buffer)


# Use mock_experience directly injected by pytest
def test_uniform_buffer_is_ready(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    assert not uniform_buffer.is_ready()
    for i in range(uniform_buffer.min_size_to_train):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy(),
            "other_features": mock_experience[0]["other_features"].copy(),
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        uniform_buffer.add(exp_copy)
    assert uniform_buffer.is_ready()


# Use mock_experience directly injected by pytest
def test_uniform_buffer_sample(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Fill buffer until ready
    for i in range(uniform_buffer.min_size_to_train):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid + i))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        uniform_buffer.add(exp_copy)

    sample = uniform_buffer.sample(uniform_buffer.config.BATCH_SIZE)
    assert sample is not None
    assert isinstance(sample, dict)
    assert "batch" in sample
    assert "indices" in sample
    assert "weights" in sample
    assert len(sample["batch"]) == uniform_buffer.config.BATCH_SIZE
    assert isinstance(sample["batch"][0], tuple)  # Check if it's an Experience tuple
    # Check structure of the first element's StateType
    assert isinstance(sample["batch"][0][0], dict)
    assert "grid" in sample["batch"][0][0]
    assert "other_features" in sample["batch"][0][0]
    assert "available_shapes_geometry" in sample["batch"][0][0]
    assert sample["indices"].shape == (uniform_buffer.config.BATCH_SIZE,)
    assert sample["weights"].shape == (uniform_buffer.config.BATCH_SIZE,)
    assert np.allclose(sample["weights"], 1.0)  # Uniform weights should be 1.0


def test_uniform_buffer_sample_not_ready(uniform_buffer: ExperienceBuffer):
    sample = uniform_buffer.sample(uniform_buffer.config.BATCH_SIZE)
    assert sample is None


def test_uniform_buffer_update_priorities(uniform_buffer: ExperienceBuffer):
    # Should be a no-op
    initial_len = len(uniform_buffer)
    uniform_buffer.update_priorities(np.array([0, 1]), np.array([0.5, 0.1]))
    assert len(uniform_buffer) == initial_len  # No change expected


# --- PER Buffer Tests ---


def test_per_buffer_init(per_buffer: ExperienceBuffer):
    assert per_buffer.use_per
    assert isinstance(per_buffer.tree, SumTree)
    assert per_buffer.capacity == 100
    assert len(per_buffer) == 0
    assert not per_buffer.is_ready()
    assert per_buffer.tree.max_priority == 1.0  # Initial max priority


# Use mock_experience directly injected by pytest
def test_per_buffer_add(per_buffer: ExperienceBuffer, mock_experience: Experience):
    assert len(per_buffer) == 0
    initial_max_p = per_buffer.tree.max_priority
    per_buffer.add(mock_experience)
    assert len(per_buffer) == 1
    # Check if added with initial max priority
    data_idx = (
        per_buffer.tree.data_pointer - 1 + per_buffer.capacity
    ) % per_buffer.capacity
    tree_idx = data_idx + per_buffer.capacity - 1
    assert per_buffer.tree.tree[tree_idx] == initial_max_p
    assert per_buffer.tree.data[data_idx] == mock_experience


# Use mock_experience directly injected by pytest
def test_per_buffer_add_batch(
    per_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Create copies for the batch
    batch: list[Experience] = [
        (
            cast(
                "StateType",  # Cast the dictionary to StateType
                {
                    "grid": mock_experience[0]["grid"].copy(),
                    "other_features": mock_experience[0]["other_features"].copy(),
                    "available_shapes_geometry": [
                        (list(geom), cid) if geom_info else None
                        for geom_info in mock_experience[0]["available_shapes_geometry"]
                        if geom_info is not None
                        for geom, cid in [geom_info]
                    ],
                },
            ),
            mock_experience[1],
            mock_experience[2],
        )
        for _ in range(5)
    ]
    per_buffer.add_batch(batch)
    assert len(per_buffer) == 5


# Use mock_experience directly injected by pytest
def test_per_buffer_capacity(per_buffer: ExperienceBuffer, mock_experience: Experience):
    for i in range(per_buffer.capacity + 10):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid + i))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)  # Adds with current max priority
    assert len(per_buffer) == per_buffer.capacity


# Use mock_experience directly injected by pytest
def test_per_buffer_is_ready(per_buffer: ExperienceBuffer, mock_experience: Experience):
    assert not per_buffer.is_ready()
    for i in range(per_buffer.min_size_to_train):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy(),
            "other_features": mock_experience[0]["other_features"].copy(),
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)
    assert per_buffer.is_ready()


# Use mock_experience directly injected by pytest
def test_per_buffer_sample(per_buffer: ExperienceBuffer, mock_experience: Experience):
    # Fill buffer until ready
    for i in range(per_buffer.min_size_to_train):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid + i))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)

    # Need current_step for beta calculation
    sample = per_buffer.sample(per_buffer.config.BATCH_SIZE, current_train_step=10)
    assert sample is not None
    assert isinstance(sample, dict)
    assert "batch" in sample
    assert "indices" in sample
    assert "weights" in sample
    assert len(sample["batch"]) == per_buffer.config.BATCH_SIZE
    assert isinstance(sample["batch"][0], tuple)
    # Check structure of the first element's StateType
    assert isinstance(sample["batch"][0][0], dict)
    assert "grid" in sample["batch"][0][0]
    assert "other_features" in sample["batch"][0][0]
    assert "available_shapes_geometry" in sample["batch"][0][0]
    assert sample["indices"].shape == (per_buffer.config.BATCH_SIZE,)
    assert sample["weights"].shape == (per_buffer.config.BATCH_SIZE,)
    assert np.all(sample["weights"] >= 0) and np.all(
        sample["weights"] <= 1.0
    )  # Weights are normalized


# Use mock_experience directly injected by pytest
def test_per_buffer_sample_requires_step(
    per_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Fill buffer
    for i in range(per_buffer.min_size_to_train):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy(),
            "other_features": mock_experience[0]["other_features"].copy(),
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)
    with pytest.raises(ValueError, match="current_train_step is required"):
        per_buffer.sample(per_buffer.config.BATCH_SIZE)


# Use mock_experience directly injected by pytest
def test_per_buffer_update_priorities(
    per_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Add some items
    num_items = per_buffer.min_size_to_train
    for i in range(num_items):
        # Create copies
        geometry_copy: list[SerializableShapeInfo | None] = []
        for geom_info in mock_experience[0]["available_shapes_geometry"]:
            if geom_info is not None:
                geom, cid = geom_info
                geom_copy = [(r, c, up) for r, c, up in geom]
                geometry_copy.append((geom_copy, cid + i))
            else:
                geometry_copy.append(None)
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
            "available_shapes_geometry": geometry_copy,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)

    # Sample to get indices
    sample = per_buffer.sample(per_buffer.config.BATCH_SIZE, current_train_step=1)
    assert sample is not None
    indices = sample["indices"]  # These are tree indices

    # Update with some errors
    td_errors = rng.random(per_buffer.config.BATCH_SIZE) * 0.5  # Example errors
    per_buffer.update_priorities(indices, td_errors)

    # --- Verification Adjustment ---
    expected_priorities_map = {}
    calculated_priorities = np.array(
        [per_buffer._get_priority(err) for err in td_errors]
    )
    for tree_idx, expected_p in zip(indices, calculated_priorities, strict=True):
        expected_priorities_map[tree_idx] = expected_p  # Last write wins

    unique_indices = sorted(expected_priorities_map.keys())
    actual_updated_priorities = [per_buffer.tree.tree[idx] for idx in unique_indices]
    expected_final_priorities = [expected_priorities_map[idx] for idx in unique_indices]

    assert np.allclose(
        actual_updated_priorities, expected_final_priorities, rtol=1e-4, atol=1e-4
    ), (
        f"Mismatch between actual tree priorities {actual_updated_priorities} and expected {expected_final_priorities} for unique indices {unique_indices}"
    )


def test_per_buffer_beta_annealing(per_buffer: ExperienceBuffer):
    config = per_buffer.config
    assert per_buffer._calculate_beta(0) == config.PER_BETA_INITIAL
    anneal_steps = per_buffer.per_beta_anneal_steps
    assert anneal_steps is not None and anneal_steps > 0
    mid_step = anneal_steps // 2
    expected_mid_beta = config.PER_BETA_INITIAL + 0.5 * (
        config.PER_BETA_FINAL - config.PER_BETA_INITIAL
    )
    assert per_buffer._calculate_beta(mid_step) == pytest.approx(expected_mid_beta)
    assert per_buffer._calculate_beta(anneal_steps) == config.PER_BETA_FINAL
    assert per_buffer._calculate_beta(anneal_steps * 2) == config.PER_BETA_FINAL

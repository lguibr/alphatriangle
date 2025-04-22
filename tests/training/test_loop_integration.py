# File: tests/training/test_loop_integration.py
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

# Import necessary classes for patching targets
from alphatriangle.config import TrainConfig
from alphatriangle.data import DataManager, PathManager
from alphatriangle.rl import ExperienceBuffer, SelfPlayResult, Trainer
from alphatriangle.training import TrainingComponents, TrainingLoop
from alphatriangle.training.loop_helpers import LoopHelpers
from alphatriangle.training.worker_manager import WorkerManager

if TYPE_CHECKING:
    from alphatriangle.utils.types import Experience, PERBatchSample

# Define logger for the test file
logger = logging.getLogger(__name__)

# --- Fixtures ---


# Mock Worker class to avoid actual Ray actors in unit tests
class MockSelfPlayWorker:
    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        self.weights_set_count = 0
        self.last_weights_received: dict[str, Any] | None = None
        self.step_set_count = 0
        self.current_trainer_step = 0
        self.task_running = False
        # Mock the remote methods directly on the instance
        self.set_weights = MagicMock(side_effect=self._set_weights_impl)
        self.set_current_trainer_step = MagicMock(
            side_effect=self._set_current_trainer_step_impl
        )
        self.run_episode = MagicMock(side_effect=self._run_episode_impl)

        # Mock the .remote attribute to return the mocked methods
        self.set_weights.remote = self.set_weights
        self.set_current_trainer_step.remote = self.set_current_trainer_step
        self.run_episode.remote = self.run_episode

    def _run_episode_impl(self):
        self.task_running = True
        step_at_start = self.current_trainer_step
        time.sleep(0.01)
        dummy_exp: Experience = (
            {"grid": np.array([[[0.0]]]), "other_features": np.array([0.0])},
            {0: 1.0},
            0.5,
        )
        result = SelfPlayResult(
            episode_experiences=[dummy_exp],
            final_score=1.0,
            episode_steps=1,
            trainer_step_at_episode_start=step_at_start,
            total_simulations=10,
            avg_root_visits=10.0,
            avg_tree_depth=1.0,
        )
        self.task_running = False
        return result

    def _set_weights_impl(self, weights: dict):
        self.weights_set_count += 1
        self.last_weights_received = weights
        time.sleep(0.001)

    def _set_current_trainer_step_impl(self, global_step: int):
        self.step_set_count += 1
        self.current_trainer_step = global_step
        time.sleep(0.001)


@pytest.fixture
def mock_training_config(mock_train_config: TrainConfig) -> TrainConfig:
    """Fixture for TrainConfig with settings suitable for integration tests."""
    cfg = mock_train_config.model_copy(deep=True)
    cfg.NUM_SELF_PLAY_WORKERS = 2
    cfg.WORKER_UPDATE_FREQ_STEPS = 5
    cfg.MIN_BUFFER_SIZE_TO_TRAIN = 2
    cfg.BATCH_SIZE = 1
    cfg.MAX_TRAINING_STEPS = 20
    cfg.USE_PER = False
    # Disable compilation for mock tests
    cfg.COMPILE_MODEL = False
    return cfg


@pytest.fixture
def mock_components(
    mocker,
    mock_nn_interface,
    mock_training_config,
    mock_env_config,
    mock_model_config,
    mock_mcts_config,
    mock_persistence_config,
    mock_experience,
) -> TrainingComponents:
    """Fixture to create TrainingComponents with mocks suitable for loop tests."""

    mock_buffer = MagicMock(spec=ExperienceBuffer)
    mock_buffer.config = mock_training_config
    mock_buffer.capacity = mock_training_config.BUFFER_CAPACITY
    mock_buffer.min_size_to_train = mock_training_config.MIN_BUFFER_SIZE_TO_TRAIN
    mock_buffer.use_per = mock_training_config.USE_PER
    mock_buffer.is_ready.return_value = True
    dummy_sample: PERBatchSample = {
        "batch": [mock_experience],
        "indices": np.array([0]),
        "weights": np.array([1.0]),
    }
    mock_buffer.sample.return_value = dummy_sample
    mocker.patch(
        "alphatriangle.rl.core.buffer.ExperienceBuffer", return_value=mock_buffer
    )

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train_config = mock_training_config
    mock_trainer.env_config = mock_env_config
    mock_trainer.model_config = mock_model_config
    mock_trainer.nn = mock_nn_interface
    mock_trainer.device = mock_nn_interface.device
    mock_trainer.train_step.return_value = ({"total_loss": 0.1}, np.array([0.1]))
    mocker.patch("alphatriangle.rl.core.trainer.Trainer", return_value=mock_trainer)

    mock_path_manager = MagicMock(spec=PathManager)
    mock_path_manager.run_base_dir = Path("/tmp/mock_run")
    mock_data_manager = MagicMock(spec=DataManager)
    mock_data_manager.path_manager = mock_path_manager
    # Patch DataManager at its source definition
    mocker.patch("alphatriangle.data.DataManager", return_value=mock_data_manager)

    mock_stats_collector = MagicMock()

    mock_workers = [
        MockSelfPlayWorker(i) for i in range(mock_training_config.NUM_SELF_PLAY_WORKERS)
    ]
    mock_worker_manager_instance = MagicMock(spec=WorkerManager)
    mock_worker_manager_instance.workers = mock_workers
    mock_worker_manager_instance.active_worker_indices = set(
        range(mock_training_config.NUM_SELF_PLAY_WORKERS)
    )

    # Simplified task tracking
    mock_worker_tasks: dict[int, MagicMock] = {}  # worker_idx -> mock_task_ref

    def mock_submit_task(worker_idx):
        if worker_idx in mock_worker_manager_instance.active_worker_indices:
            # Create a mock object to represent the task reference
            mock_task_ref = MagicMock(name=f"task_ref_w{worker_idx}")
            mock_worker_tasks[worker_idx] = mock_task_ref
            logger.debug(f"Mock submit task for worker {worker_idx}")

    def mock_get_completed_tasks(*_args, **_kwargs):
        results = []
        # Simulate one task completing if available
        if mock_worker_tasks:
            # Pop an arbitrary worker index that has a pending task
            worker_idx_to_complete = next(iter(mock_worker_tasks))
            mock_worker_tasks.pop(worker_idx_to_complete)  # Remove task ref

            if (
                worker_idx_to_complete < len(mock_workers)
                and mock_workers[worker_idx_to_complete] is not None
            ):
                # Simulate getting the result (call the worker's mocked method)
                result = mock_workers[worker_idx_to_complete].run_episode()
                results.append((worker_idx_to_complete, result))
                logger.debug(f"Mock completed task for worker {worker_idx_to_complete}")
            else:
                logger.warning(
                    f"Mock worker {worker_idx_to_complete} not found or None during completion."
                )

        return results

    mock_worker_manager_instance.submit_task.side_effect = mock_submit_task
    mock_worker_manager_instance.get_completed_tasks.side_effect = (
        mock_get_completed_tasks
    )
    # Keep track of pending tasks via the dictionary length
    mock_worker_manager_instance.get_num_pending_tasks.side_effect = lambda: len(
        mock_worker_tasks
    )

    def mock_update_worker_networks(global_step: int):
        logger.debug(f"Mock update_worker_networks called for step {global_step}")
        weights = mock_nn_interface.get_weights()
        for worker in mock_workers:
            if worker:  # Check if worker is not None
                worker.set_weights.remote(weights)
                worker.set_current_trainer_step.remote(global_step)
        logger.debug("Mock update_worker_networks finished.")

    mock_worker_manager_instance.update_worker_networks.side_effect = (
        mock_update_worker_networks
    )

    mock_worker_manager_instance.get_num_active_workers.return_value = (
        mock_training_config.NUM_SELF_PLAY_WORKERS
    )

    mocker.patch(
        "alphatriangle.training.loop.WorkerManager",
        return_value=mock_worker_manager_instance,
    )

    mock_loop_helpers = MagicMock(spec=LoopHelpers)
    mock_loop_helpers.log_weight_update_event = MagicMock()
    # Configure validate_experiences to return valid output
    mock_loop_helpers.validate_experiences.side_effect = lambda exps: (exps, 0)
    mocker.patch(
        "alphatriangle.training.loop.LoopHelpers", return_value=mock_loop_helpers
    )

    # Use the mocked DataManager instance here
    components = TrainingComponents(
        nn=mock_nn_interface,
        buffer=mock_buffer,
        trainer=mock_trainer,
        data_manager=mock_data_manager,
        stats_collector_actor=mock_stats_collector,
        train_config=mock_training_config,
        env_config=mock_env_config,
        model_config=mock_model_config,
        mcts_config=mock_mcts_config,
        persist_config=mock_persistence_config,
    )

    return components


@pytest.fixture
def mock_training_loop(mock_components: TrainingComponents) -> TrainingLoop:
    """Fixture for an initialized TrainingLoop with mocked components."""
    loop = TrainingLoop(mock_components)
    loop.set_initial_state(0, 0, 0)
    # Submit initial tasks using the mocked manager
    for i in range(loop.train_config.NUM_SELF_PLAY_WORKERS):
        loop.worker_manager.submit_task(i)
    return loop


# --- Test ---


def test_worker_weight_update_usage(
    mock_training_loop: TrainingLoop, mock_components: TrainingComponents
):
    """
    Verify that worker weights/steps are updated and that subsequent episodes
    use the updated step information.
    """
    loop = mock_training_loop
    components = mock_components
    worker_manager = loop.worker_manager
    mock_workers = worker_manager.workers
    update_freq = components.train_config.WORKER_UPDATE_FREQ_STEPS
    max_steps = components.train_config.MAX_TRAINING_STEPS or 20

    # Access the mock directly from the mocked helper instance
    # Cast to MagicMock to satisfy mypy
    log_event_mock = cast("MagicMock", loop.loop_helpers.log_weight_update_event)

    # Run the loop
    loop.run()  # Let the loop run until max_steps or stop condition

    # --- Assertions after the loop ---
    results_processed = loop.episodes_played  # Get final count from loop state

    total_expected_updates = max_steps // update_freq
    assert log_event_mock.call_count == total_expected_updates, (
        f"Log event calls: expected {total_expected_updates}, got {log_event_mock.call_count}"
    )
    assert results_processed > 0, (
        "No self-play results were processed (loop.episodes_played is 0)"
    )

    for worker_idx, worker in enumerate(mock_workers):
        if worker is not None:
            # Cast mocks for mypy checks
            set_step_mock = cast("MagicMock", worker.set_current_trainer_step)
            set_weights_mock = cast("MagicMock", worker.set_weights)

            # Check call counts
            assert set_step_mock.call_count == total_expected_updates, (
                f"Worker {worker_idx} set_step calls: expected {total_expected_updates}, got {set_step_mock.call_count}"
            )
            assert set_weights_mock.call_count == total_expected_updates, (
                f"Worker {worker_idx} set_weights calls: expected {total_expected_updates}, got {set_weights_mock.call_count}"
            )

            # Check final state and last call argument if updates happened
            if total_expected_updates > 0:
                last_expected_step = total_expected_updates * update_freq
                # Check the *last* step it was called with
                set_step_mock.assert_called_with(last_expected_step)
                # Check the internal state set by the side effect
                assert worker.current_trainer_step == last_expected_step, (
                    f"Worker {worker_idx} final step: expected {last_expected_step}, got {worker.current_trainer_step}"
                )
        else:
            pytest.fail(f"Worker object {worker_idx} became None during test")

    logger.info(
        f"Test finished. Verified {results_processed} results processed. Expected updates: {total_expected_updates}."
    )

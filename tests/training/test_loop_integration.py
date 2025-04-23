# File: tests/training/test_loop_integration.py
import logging
import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Import Trieye config for components
from trieye import Serializer, TrieyeConfig  # Added Serializer
from trieye.schemas import RawMetricEvent  # Import from trieye

# Import necessary classes for patching targets
from alphatriangle.config import TrainConfig
from alphatriangle.rl import ExperienceBuffer, SelfPlayResult, Trainer
from alphatriangle.training import TrainingComponents, TrainingLoop
from alphatriangle.training.loop_helpers import LoopHelpers
from alphatriangle.training.worker_manager import WorkerManager

if TYPE_CHECKING:
    from alphatriangle.utils.types import Experience, PERBatchSample

logger = logging.getLogger(__name__)


# --- Fixtures ---


class MockSelfPlayWorker:
    """Simplified Mock Worker for loop integration tests."""

    def __init__(self, actor_id: int, trieye_actor_mock: MagicMock | None):
        self.actor_id = actor_id
        self.trieye_actor_mock = trieye_actor_mock  # Store mock handle
        self.weights_set_count = 0
        self.last_weights_received: dict[str, Any] | None = None
        self.step_set_count = 0
        self.current_trainer_step = 0
        self.task_running = False
        self.set_weights = MagicMock(side_effect=self._set_weights_impl)
        self.set_current_trainer_step = MagicMock(
            side_effect=self._set_current_trainer_step_impl
        )
        self.run_episode = MagicMock(side_effect=self._run_episode_impl)

        # Simulate remote calls
        self.set_weights.remote = self.set_weights
        self.set_current_trainer_step.remote = self.set_current_trainer_step
        self.run_episode.remote = self.run_episode

    def _run_episode_impl(self) -> SelfPlayResult:
        self.task_running = True
        step_at_start = self.current_trainer_step
        time.sleep(0.01)
        dummy_exp: Experience = (
            {"grid": np.array([[[0.0]]]), "other_features": np.array([0.0])},
            {0: 1.0},
            0.5,
        )
        episode_context = {
            "score": 1.0,
            "length": 1,
            "simulations": 10,
            "triangles_cleared": 0,
            "trainer_step": step_at_start,
        }

        # Simulate worker sending events to Trieye
        if self.trieye_actor_mock:
            self.trieye_actor_mock.log_event.remote(
                RawMetricEvent(
                    name="mcts_step",
                    value=10.0,
                    global_step=step_at_start,
                    context={"game_step": 0},
                )
            )
            self.trieye_actor_mock.log_event.remote(
                RawMetricEvent(
                    name="step_reward",
                    value=0.1,
                    global_step=step_at_start,
                    context={"game_step": 0},
                )
            )
            self.trieye_actor_mock.log_event.remote(
                RawMetricEvent(
                    name="current_score",
                    value=1.0,
                    global_step=step_at_start,
                    context={"game_step": 0},
                )
            )
            self.trieye_actor_mock.log_event.remote(
                RawMetricEvent(
                    name="episode_end",
                    value=1.0,
                    global_step=step_at_start,
                    context=episode_context,
                )
            )

        result = SelfPlayResult(
            episode_experiences=[dummy_exp],
            final_score=1.0,
            episode_steps=1,
            trainer_step_at_episode_start=step_at_start,
            total_simulations=10,
            avg_root_visits=10.0,
            avg_tree_depth=1.0,
            context=episode_context,
        )
        self.task_running = False
        return result

    def _set_weights_impl(self, weights: dict) -> None:
        self.weights_set_count += 1
        self.last_weights_received = weights
        time.sleep(0.001)

    def _set_current_trainer_step_impl(self, global_step: int) -> None:
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
    cfg.COMPILE_MODEL = False
    cfg.PROFILE_WORKERS = False
    cfg.CHECKPOINT_SAVE_FREQ_STEPS = 10  # Save checkpoint during short run
    return cfg


@pytest.fixture
def mock_trieye_config() -> TrieyeConfig:
    """Fixture for a basic TrieyeConfig."""
    # Use default metrics for simplicity in this test
    return TrieyeConfig(app_name="test_loop_app", run_name="test_loop_run")


@pytest.fixture
def mock_components(
    mocker,
    mock_nn_interface,
    mock_training_config,
    mock_trieye_config,  # Use Trieye config
    mock_env_config,
    mock_model_config,
    mock_mcts_config,
    mock_experience,
) -> TrainingComponents:
    """Fixture to create TrainingComponents with mocks suitable for loop tests."""

    mock_buffer = MagicMock(spec=ExperienceBuffer)
    mock_buffer.config = mock_training_config
    mock_buffer.capacity = mock_training_config.BUFFER_CAPACITY
    mock_buffer.min_size_to_train = mock_training_config.MIN_BUFFER_SIZE_TO_TRAIN
    mock_buffer.use_per = mock_training_config.USE_PER
    mock_buffer.is_ready.return_value = True
    mock_buffer.__len__.return_value = mock_training_config.MIN_BUFFER_SIZE_TO_TRAIN + 5
    dummy_sample: PERBatchSample = {
        "batch": [mock_experience],
        "indices": np.array([0]),
        "weights": np.array([1.0]),
    }
    mock_buffer.sample.return_value = dummy_sample
    if mock_training_config.USE_PER:
        mock_buffer._calculate_beta = MagicMock(return_value=0.5)
    mocker.patch(
        "alphatriangle.rl.core.buffer.ExperienceBuffer", return_value=mock_buffer
    )

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train_config = mock_training_config
    mock_trainer.env_config = mock_env_config
    mock_trainer.model_config = mock_model_config
    mock_trainer.nn = mock_nn_interface
    mock_trainer.device = mock_nn_interface.device
    mock_trainer.train_step.return_value = (
        {"total_loss": 0.1, "policy_loss": 0.05, "value_loss": 0.05},
        np.array([0.1]),
    )
    mock_trainer.get_current_lr.return_value = mock_training_config.LEARNING_RATE
    # Mock the optimizer attribute needed for saving state
    mock_trainer.optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_trainer.optimizer.state_dict.return_value = {"opt_state": "dummy"}
    mocker.patch("alphatriangle.rl.core.trainer.Trainer", return_value=mock_trainer)

    # Mock Trieye Actor Handle
    mock_trieye_actor_handle = MagicMock(name="TrieyeActorMockHandle")
    mock_trieye_actor_handle.log_event = MagicMock(name="log_event_remote")
    mock_trieye_actor_handle.log_batch_events = MagicMock(
        name="log_batch_events_remote"
    )
    mock_trieye_actor_handle.save_training_state = MagicMock(
        name="save_training_state_remote"
    )
    # Simulate remote calls
    mock_trieye_actor_handle.log_event.remote = mock_trieye_actor_handle.log_event
    mock_trieye_actor_handle.log_batch_events.remote = (
        mock_trieye_actor_handle.log_batch_events
    )
    mock_trieye_actor_handle.save_training_state.remote = (
        mock_trieye_actor_handle.save_training_state
    )

    # --- ADDED: Create mock serializer ---
    mock_serializer = MagicMock(spec=Serializer)
    mock_serializer.prepare_optimizer_state.return_value = {"opt_state": "dummy"}
    mock_serializer.prepare_buffer_data.return_value = MagicMock(buffer_list=[])
    # Attach serializer to the mock Trieye handle (mimicking real actor structure for loop logic)
    mock_trieye_actor_handle.serializer = mock_serializer

    mock_workers = [
        MockSelfPlayWorker(i, mock_trieye_actor_handle)  # Pass Trieye mock
        for i in range(mock_training_config.NUM_SELF_PLAY_WORKERS)
    ]
    mock_worker_manager_instance = MagicMock(spec=WorkerManager)
    mock_worker_manager_instance.workers = mock_workers
    mock_worker_manager_instance.active_worker_indices = set(
        range(mock_training_config.NUM_SELF_PLAY_WORKERS)
    )

    mock_worker_tasks: dict[int, MagicMock] = {}

    def mock_submit_task(worker_idx):
        if worker_idx in mock_worker_manager_instance.active_worker_indices:
            mock_task_ref = MagicMock(name=f"task_ref_w{worker_idx}")
            mock_worker_tasks[worker_idx] = mock_task_ref
            logger.debug(f"Mock submit task for worker {worker_idx}")

    def mock_get_completed_tasks(*_args, **_kwargs):
        results = []
        if mock_worker_tasks:
            worker_idx_to_complete = next(iter(mock_worker_tasks))
            mock_worker_tasks.pop(worker_idx_to_complete)
            if (
                worker_idx_to_complete < len(mock_workers)
                and mock_workers[worker_idx_to_complete] is not None
            ):
                result = mock_workers[worker_idx_to_complete].run_episode()
                results.append((worker_idx_to_complete, result))
                logger.debug(f"Mock completed task for worker {worker_idx_to_complete}")
            else:
                logger.warning(
                    f"Mock worker {worker_idx_to_complete} not found or None."
                )
        return results

    mock_worker_manager_instance.submit_task.side_effect = mock_submit_task
    mock_worker_manager_instance.get_completed_tasks.side_effect = (
        mock_get_completed_tasks
    )
    mock_worker_manager_instance.get_num_pending_tasks.side_effect = lambda: len(
        mock_worker_tasks
    )

    def mock_update_worker_networks(global_step: int):
        logger.debug(f"Mock update_worker_networks called for step {global_step}")
        weights = mock_nn_interface.get_weights()
        for worker in mock_workers:
            if worker:
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
    mock_loop_helpers.log_progress_eta = MagicMock()
    mock_loop_helpers.validate_experiences.side_effect = lambda exps: (exps, 0)
    mocker.patch(
        "alphatriangle.training.loop.LoopHelpers", return_value=mock_loop_helpers
    )

    # Create TrainingComponents with Trieye actor handle and config
    components = TrainingComponents(
        nn=mock_nn_interface,
        buffer=mock_buffer,
        trainer=mock_trainer,
        trieye_actor=mock_trieye_actor_handle,
        trieye_config=mock_trieye_config,
        serializer=mock_serializer,  # --- ADDED missing argument ---
        train_config=mock_training_config,
        env_config=mock_env_config,
        model_config=mock_model_config,
        mcts_config=mock_mcts_config,
        profile_workers=mock_training_config.PROFILE_WORKERS,
    )

    return components


@pytest.fixture
def mock_training_loop(mock_components: TrainingComponents) -> TrainingLoop:
    """Fixture for an initialized TrainingLoop with mocked components."""
    loop = TrainingLoop(mock_components)
    loop.set_initial_state(0, 0, 0)
    for i in range(loop.train_config.NUM_SELF_PLAY_WORKERS):
        loop.worker_manager.submit_task(i)
    return loop


# --- Test ---


def test_worker_weight_update_and_stats_logging(
    mock_training_loop: TrainingLoop, mock_components: TrainingComponents
):
    """
    Verify that worker weights/steps are updated and that the weight update
    event is sent to the Trieye actor.
    """
    loop = mock_training_loop
    components = mock_components
    worker_manager = loop.worker_manager
    mock_workers = worker_manager.workers
    trieye_actor_mock = components.trieye_actor  # Get mock handle
    assert trieye_actor_mock is not None, "Trieye actor mock handle should not be None"

    update_freq = components.train_config.WORKER_UPDATE_FREQ_STEPS
    max_steps = components.train_config.MAX_TRAINING_STEPS or 20

    loop.run()

    results_processed = loop.episodes_played
    total_expected_updates = max_steps // update_freq
    expected_total_weight_updates = total_expected_updates

    # Check calls to Trieye actor's log_event
    weight_update_event_calls = [
        c
        for c in trieye_actor_mock.log_event.call_args_list
        if isinstance(c.args[0], RawMetricEvent)
        and c.args[0].name == "Progress/Weight_Updates_Total"
    ]
    assert len(weight_update_event_calls) == total_expected_updates, (
        f"Weight update event calls: expected {total_expected_updates}, got {len(weight_update_event_calls)}"
    )

    if weight_update_event_calls:
        last_event_arg = weight_update_event_calls[-1].args[0]
        assert isinstance(last_event_arg, RawMetricEvent)
        assert last_event_arg.value == expected_total_weight_updates, (
            f"Last weight update event value mismatch: expected {expected_total_weight_updates}, got {last_event_arg.value}"
        )
        last_expected_step = total_expected_updates * update_freq
        assert last_event_arg.global_step == last_expected_step, (
            f"Last weight update event step mismatch: expected {last_expected_step}, got {last_event_arg.global_step}"
        )

    assert results_processed > 0, "No self-play results were processed"

    for worker_idx, worker in enumerate(mock_workers):
        if worker is not None:
            set_step_mock = cast("MagicMock", worker.set_current_trainer_step)
            set_weights_mock = cast("MagicMock", worker.set_weights)

            assert set_step_mock.call_count == total_expected_updates, (
                f"Worker {worker_idx} set_step calls"
            )
            assert set_weights_mock.call_count == total_expected_updates, (
                f"Worker {worker_idx} set_weights calls"
            )

            if total_expected_updates > 0:
                last_expected_step = total_expected_updates * update_freq
                set_step_mock.assert_called_with(last_expected_step)
                assert worker.current_trainer_step == last_expected_step, (
                    f"Worker {worker_idx} final step"
                )
        else:
            pytest.fail(f"Worker object {worker_idx} became None during test")

    logger.info(
        f"Test finished. Verified {results_processed} results processed. Expected updates: {total_expected_updates}."
    )


def test_checkpoint_save_trigger(
    mock_training_loop: TrainingLoop, mock_components: TrainingComponents
):
    """Verify that save_training_state is called on the Trieye actor."""
    loop = mock_training_loop
    trieye_actor_mock = mock_components.trieye_actor
    assert trieye_actor_mock is not None

    save_freq = loop.train_config.CHECKPOINT_SAVE_FREQ_STEPS
    loop.train_config.MAX_TRAINING_STEPS = save_freq + 2  # Ensure save is triggered

    loop.run()

    # Check if save_training_state was called at the correct step
    save_calls = trieye_actor_mock.save_training_state.call_args_list
    assert len(save_calls) >= 1, "save_training_state was not called"

    # Check the call at the expected frequency step
    call_found = False
    for call in save_calls:
        if call.kwargs.get("global_step") == save_freq:
            assert call.kwargs.get("is_best") is False
            assert (
                call.kwargs.get("save_buffer") is False
            )  # Default checkpoint save doesn't save buffer
            call_found = True
            break
    assert call_found, f"save_training_state not called at expected step {save_freq}"

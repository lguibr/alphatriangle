# File: tests/stats/test_collector.py
import logging
import time
from typing import Any, cast  # Added cast

import cloudpickle
import pytest
import ray

# Import new types
# Correct the import path for config
from alphatriangle.config import StatsConfig  # Corrected import
from alphatriangle.stats import StatsCollectorActor
from alphatriangle.stats.types import RawMetricEvent


# Mock GameState for testing worker state updates
class MockGameStateForStats:
    def __init__(self, step: int, score: float):
        self.current_step = step
        self._game_score = score
        self.grid_data = True
        self.shapes = True

    def game_score(self) -> float:
        return self._game_score

    def get_grid_data_np(self) -> dict:
        return {}

    def get_shapes(self) -> list:
        return []


# --- Fixtures ---


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    if not ray.is_initialized():
        # Use ERROR level for Ray to minimize noise during tests
        ray.init(logging_level=logging.ERROR, num_cpus=1, log_to_driver=False)
        initialized_here = True
    else:
        initialized_here = False
    yield
    if initialized_here and ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def mock_stats_config() -> StatsConfig:
    """Provides a default StatsConfig for testing."""
    # Use a small positive interval to satisfy validation (gt=0)
    return StatsConfig(processing_interval_seconds=0.01, metrics=[])


# REMOVED mock_processor fixture


@pytest.fixture
def stats_actor(mock_stats_config, tmp_path):
    """Provides a normal StatsCollectorActor instance."""
    tb_dir = tmp_path / "tb_logs"
    # Create the actor normally
    actor = StatsCollectorActor.remote(
        stats_config=mock_stats_config,
        run_name="test_run",
        tb_log_dir=str(tb_dir),
    )
    # Ensure actor is initialized
    ray.get(actor.get_state.remote())  # Use a simple remote call
    yield actor  # Return only the actor handle
    # Clean up
    ray.kill(actor, no_restart=True)


# --- Helper to create RawMetricEvent ---
def create_event(
    name: str, value: float, step: int, context: dict | None = None
) -> RawMetricEvent:
    """Creates a basic RawMetricEvent for testing."""
    return RawMetricEvent(
        name=name,
        value=value,
        global_step=step,
        timestamp=time.time(),
        context=context or {},
    )


# --- Helper to get internal state ---
def get_actor_internal_state(actor_handle) -> dict[str, Any]:
    """Gets internal state using the test-only method."""
    # Use type ignore as the method is intended for testing
    state_ref = actor_handle._get_internal_state_for_testing.remote()  # type: ignore [attr-defined]
    # Cast the result of ray.get to satisfy MyPy
    return cast("dict[str, Any]", ray.get(state_ref))


# --- Tests ---


def test_actor_initialization(stats_actor):
    """Test if the actor initializes correctly."""
    state = ray.get(stats_actor.get_state.remote())
    assert state["last_processed_step"] == -1
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}
    internal_state = get_actor_internal_state(stats_actor)
    assert not internal_state["raw_data_buffer"]  # Buffer should be empty


def test_log_single_event(stats_actor):
    """Test logging a single valid event adds it to the internal buffer."""
    event = create_event("test_metric", 10.5, 1)
    ray.get(stats_actor.log_event.remote(event))

    internal_state = get_actor_internal_state(stats_actor)
    assert 1 in internal_state["raw_data_buffer"]
    assert "test_metric" in internal_state["raw_data_buffer"][1]
    assert internal_state["raw_data_buffer"][1]["test_metric"] == [10.5]


def test_log_batch_events(stats_actor):
    """Test logging a batch of events adds them to the internal buffer."""
    events = [
        create_event("metric_a", 1.0, 1),
        create_event("metric_b", 2.5, 1),
        create_event("metric_a", 1.1, 2),
    ]
    ray.get(stats_actor.log_batch_events.remote(events))

    internal_state = get_actor_internal_state(stats_actor)
    assert 1 in internal_state["raw_data_buffer"]
    assert 2 in internal_state["raw_data_buffer"]
    assert "metric_a" in internal_state["raw_data_buffer"][1]
    assert "metric_b" in internal_state["raw_data_buffer"][1]
    assert "metric_a" in internal_state["raw_data_buffer"][2]
    assert internal_state["raw_data_buffer"][1]["metric_a"] == [1.0]
    assert internal_state["raw_data_buffer"][1]["metric_b"] == [2.5]
    assert internal_state["raw_data_buffer"][2]["metric_a"] == [1.1]


def test_log_non_finite_event(stats_actor):
    """Test that non-finite values are ignored and valid ones are kept."""
    event_inf = create_event("non_finite", float("inf"), 1)
    event_nan = create_event("non_finite", float("nan"), 2)
    event_valid = create_event("non_finite", 10.0, 3)

    ray.get(stats_actor.log_event.remote(event_inf))
    ray.get(stats_actor.log_event.remote(event_nan))
    ray.get(stats_actor.log_event.remote(event_valid))

    internal_state_before = get_actor_internal_state(stats_actor)
    # Check buffer before processing
    assert (
        1 not in internal_state_before["raw_data_buffer"]
    )  # Inf should be skipped on log_event
    assert (
        2 not in internal_state_before["raw_data_buffer"]
    )  # NaN should be skipped on log_event
    assert 3 in internal_state_before["raw_data_buffer"]
    assert internal_state_before["raw_data_buffer"][3]["non_finite"] == [10.0]

    # Call the force method to bypass time check and process step 3
    # Use type ignore as the method is intended for testing
    ray.get(stats_actor.force_process_and_log.remote(current_global_step=3))  # type: ignore [attr-defined]

    # Check buffer after processing
    internal_state_after = get_actor_internal_state(stats_actor)
    assert 3 not in internal_state_after["raw_data_buffer"]  # Step 3 should be cleared
    assert internal_state_after["last_processed_step"] == 3


def test_process_and_log_trigger(stats_actor):
    """Test that force_process_and_log processes buffered data and updates state."""
    event = create_event("metric_c", 5.0, 10)
    ray.get(stats_actor.log_event.remote(event))

    internal_state_before = get_actor_internal_state(stats_actor)
    assert 10 in internal_state_before["raw_data_buffer"]
    assert internal_state_before["last_processed_step"] == -1  # Assuming initial state

    # Call the force method to bypass time check
    # Use type ignore as the method is intended for testing
    ray.get(stats_actor.force_process_and_log.remote(current_global_step=10))  # type: ignore [attr-defined]

    # Check state after processing
    internal_state_after = get_actor_internal_state(stats_actor)
    assert (
        10 not in internal_state_after["raw_data_buffer"]
    )  # Step 10 should be cleared
    assert internal_state_after["last_processed_step"] == 10


def test_get_set_state(stats_actor):
    """Test saving and restoring the actor's minimal state."""
    # Log an event and process it to change the state
    ray.get(stats_actor.log_event.remote(create_event("s_metric", 1.0, 5)))
    # Use type ignore as the method is intended for testing
    ray.get(stats_actor.force_process_and_log.remote(current_global_step=5))  # type: ignore [attr-defined]

    # Get the minimal state
    state = ray.get(stats_actor.get_state.remote())
    assert isinstance(state, dict)
    assert state["last_processed_step"] == 5
    assert "last_processed_time" in state

    # Pickle and unpickle
    pickled_state = cloudpickle.dumps(state)
    unpickled_state = cloudpickle.loads(pickled_state)

    # Get config etc. from original actor using remote calls
    # Use type ignore as ActorHandle doesn't expose these directly
    original_config = ray.get(stats_actor.get_config.remote())  # type: ignore [attr-defined]
    original_run_name = ray.get(stats_actor.get_run_name.remote())  # type: ignore [attr-defined]
    original_tb_dir = ray.get(stats_actor.get_tb_log_dir.remote())  # type: ignore [attr-defined]

    # Create a new actor instance
    new_actor = StatsCollectorActor.remote(
        stats_config=original_config,
        run_name=original_run_name,
        tb_log_dir=original_tb_dir,
    )
    ray.get(new_actor.get_state.remote())  # Ensure actor is ready

    # Restore state into the new actor
    ray.get(new_actor.set_state.remote(unpickled_state))

    # Verify restored state
    restored_state = ray.get(new_actor.get_state.remote())
    assert restored_state["last_processed_step"] == 5
    assert restored_state["last_processed_time"] == pytest.approx(
        state["last_processed_time"]
    )

    # Verify internal state after restore (buffer should be cleared)
    restored_internal_state = get_actor_internal_state(new_actor)
    assert not restored_internal_state["raw_data_buffer"]

    # Check that processing new events works correctly in the restored actor
    ray.get(new_actor.log_event.remote(create_event("s_metric", 2.0, 6)))
    internal_state_before_process = get_actor_internal_state(new_actor)
    assert 6 in internal_state_before_process["raw_data_buffer"]

    # Use type ignore as the method is intended for testing
    ray.get(new_actor.force_process_and_log.remote(current_global_step=6))  # type: ignore [attr-defined]

    internal_state_after_process = get_actor_internal_state(new_actor)
    assert (
        6 not in internal_state_after_process["raw_data_buffer"]
    )  # Step 6 should be cleared
    assert internal_state_after_process["last_processed_step"] == 6

    ray.kill(new_actor, no_restart=True)


def test_update_and_get_worker_state(stats_actor):
    """Test updating and retrieving worker game states (remains the same)."""
    worker_id = 1
    state1 = MockGameStateForStats(step=10, score=5.0)
    state2 = MockGameStateForStats(step=11, score=6.0)

    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}

    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state1))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 10
    assert latest_states[worker_id].game_score() == 5.0

    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state2))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id].game_score() == 6.0

    worker_id_2 = 2
    state3 = MockGameStateForStats(step=5, score=2.0)
    ray.get(stats_actor.update_worker_game_state.remote(worker_id_2, state3))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert worker_id_2 in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id_2].current_step == 5

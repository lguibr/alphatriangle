# File: tests/stats/test_collector.py
import logging
import time
from typing import Any, cast

import cloudpickle
import pytest
import ray

# Import classes for type hints
# Import new types
# Correct the import path for config
from alphatriangle.config.stats_config import (
    StatsConfig,
    default_stats_config,
)

# Use full path import for mypy compatibility
from alphatriangle.stats.collector import StatsCollectorActor

# Update import to use the new filename
from alphatriangle.stats.stats_types import RawMetricEvent

logger = logging.getLogger(__name__)


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
    cfg = default_stats_config.model_copy(deep=True)
    cfg.processing_interval_seconds = 0.01
    # Ensure some metrics have step/time frequency for testing _should_log
    for mc in cfg.metrics:
        if mc.name == "Loss/Total":
            mc.log_frequency_steps = 1
            mc.log_frequency_seconds = 0
        if mc.name == "Rate/Steps_Per_Sec":
            mc.log_frequency_steps = 0
            mc.log_frequency_seconds = 0.001  # Log frequently by time
    return cast("StatsConfig", cfg)


# Revert to the original fixture without mocks
@pytest.fixture
def stats_actor(mock_stats_config: StatsConfig, tmp_path):
    """Provides a StatsCollectorActor instance with a test MLflow run ID."""
    tb_dir = tmp_path / "tb_logs"
    test_mlflow_run_id = "test_mlflow_run_id_collector"
    actor = StatsCollectorActor.remote(  # type: ignore [attr-defined]
        stats_config=mock_stats_config,
        run_name="test_run",
        tb_log_dir=str(tb_dir),
        mlflow_run_id=test_mlflow_run_id,
    )
    # Ensure actor is initialized before returning
    ray.get(actor.get_state.remote())
    yield actor
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
    state_ref = actor_handle._get_internal_state_for_testing.remote()
    return cast("dict[str, Any]", ray.get(state_ref))


# --- Tests ---


def test_actor_initialization(stats_actor):  # Use original fixture
    """Test if the actor initializes correctly."""
    state = ray.get(stats_actor.get_state.remote())
    assert state["last_processed_step"] == -1
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}
    internal_state = get_actor_internal_state(stats_actor)
    assert not internal_state["raw_data_buffer"]


def test_log_single_event(stats_actor):  # Use original fixture
    """Test logging a single valid event adds it to the internal buffer."""
    event = create_event("test_metric", 10.5, 1)
    ray.get(stats_actor.log_event.remote(event))

    internal_state = get_actor_internal_state(stats_actor)
    assert 1 in internal_state["raw_data_buffer"]
    assert "test_metric" in internal_state["raw_data_buffer"][1]
    # Check the value within the RawMetricEvent object
    assert len(internal_state["raw_data_buffer"][1]["test_metric"]) == 1
    assert isinstance(
        internal_state["raw_data_buffer"][1]["test_metric"][0], RawMetricEvent
    )
    assert internal_state["raw_data_buffer"][1]["test_metric"][0].value == 10.5


def test_log_batch_events(stats_actor):  # Use original fixture
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
    # Check the value within the RawMetricEvent object
    assert len(internal_state["raw_data_buffer"][1]["metric_a"]) == 1
    assert isinstance(
        internal_state["raw_data_buffer"][1]["metric_a"][0], RawMetricEvent
    )
    assert internal_state["raw_data_buffer"][1]["metric_a"][0].value == 1.0
    assert len(internal_state["raw_data_buffer"][1]["metric_b"]) == 1
    assert isinstance(
        internal_state["raw_data_buffer"][1]["metric_b"][0], RawMetricEvent
    )
    assert internal_state["raw_data_buffer"][1]["metric_b"][0].value == 2.5
    assert len(internal_state["raw_data_buffer"][2]["metric_a"]) == 1
    assert isinstance(
        internal_state["raw_data_buffer"][2]["metric_a"][0], RawMetricEvent
    )
    assert internal_state["raw_data_buffer"][2]["metric_a"][0].value == 1.1


def test_log_non_finite_event(stats_actor):  # Use original fixture
    """Test that non-finite values are ignored and valid ones are kept."""
    event_inf = create_event("non_finite", float("inf"), 1)
    event_nan = create_event("non_finite", float("nan"), 2)
    event_valid = create_event("non_finite", 10.0, 3)

    ray.get(stats_actor.log_event.remote(event_inf))
    ray.get(stats_actor.log_event.remote(event_nan))
    ray.get(stats_actor.log_event.remote(event_valid))

    internal_state_before = get_actor_internal_state(stats_actor)
    assert 1 not in internal_state_before["raw_data_buffer"]
    assert 2 not in internal_state_before["raw_data_buffer"]
    assert 3 in internal_state_before["raw_data_buffer"]
    assert "non_finite" in internal_state_before["raw_data_buffer"][3]
    assert len(internal_state_before["raw_data_buffer"][3]["non_finite"]) == 1
    assert isinstance(
        internal_state_before["raw_data_buffer"][3]["non_finite"][0], RawMetricEvent
    )
    assert internal_state_before["raw_data_buffer"][3]["non_finite"][0].value == 10.0

    ray.get(stats_actor.force_process_and_log.remote(current_global_step=3))
    # Add a small delay to allow async processing
    time.sleep(0.1)

    internal_state_after = get_actor_internal_state(stats_actor)
    assert 3 not in internal_state_after["raw_data_buffer"]
    assert internal_state_after["last_processed_step"] == 3


def test_process_and_log_trigger(stats_actor):  # Use original fixture
    """Test that force_process_and_log processes buffered data and updates state."""
    event = create_event("metric_c", 5.0, 10)
    ray.get(stats_actor.log_event.remote(event))

    internal_state_before = get_actor_internal_state(stats_actor)
    assert 10 in internal_state_before["raw_data_buffer"]
    assert internal_state_before["last_processed_step"] == -1

    ray.get(stats_actor.force_process_and_log.remote(current_global_step=10))
    # Add a small delay
    time.sleep(0.1)

    internal_state_after = get_actor_internal_state(stats_actor)
    assert 10 not in internal_state_after["raw_data_buffer"]
    assert internal_state_after["last_processed_step"] == 10


def test_get_set_state(stats_actor):  # Use original fixture
    """Test saving and restoring the actor's minimal state."""
    ray.get(stats_actor.log_event.remote(create_event("s_metric", 1.0, 5)))
    ray.get(stats_actor.force_process_and_log.remote(current_global_step=5))
    # Add a small delay
    time.sleep(0.1)

    state = ray.get(stats_actor.get_state.remote())
    assert isinstance(state, dict)
    assert state["last_processed_step"] == 5
    assert "last_processed_time" in state

    pickled_state = cloudpickle.dumps(state)
    unpickled_state = cloudpickle.loads(pickled_state)

    original_config = ray.get(stats_actor.get_config.remote())
    original_run_name = ray.get(stats_actor.get_run_name.remote())
    original_tb_dir = ray.get(stats_actor.get_tb_log_dir.remote())
    original_mlflow_run_id = get_actor_internal_state(stats_actor).get("mlflow_run_id")

    # Create new actor *without* mocks for restore test
    new_actor = StatsCollectorActor.remote(  # type: ignore [attr-defined]
        stats_config=original_config,
        run_name=original_run_name,
        tb_log_dir=original_tb_dir,
        mlflow_run_id=original_mlflow_run_id,
    )
    ray.get(new_actor.get_state.remote())

    ray.get(new_actor.set_state.remote(unpickled_state))

    restored_state = ray.get(new_actor.get_state.remote())
    assert restored_state["last_processed_step"] == 5
    assert restored_state["last_processed_time"] == pytest.approx(
        state["last_processed_time"]
    )

    restored_internal_state = get_actor_internal_state(new_actor)
    assert not restored_internal_state["raw_data_buffer"]
    assert restored_internal_state["mlflow_run_id"] == original_mlflow_run_id

    ray.get(new_actor.log_event.remote(create_event("s_metric", 2.0, 6)))
    internal_state_before_process = get_actor_internal_state(new_actor)
    assert 6 in internal_state_before_process["raw_data_buffer"]

    ray.get(new_actor.force_process_and_log.remote(current_global_step=6))
    # Add a small delay
    time.sleep(0.1)

    internal_state_after_process = get_actor_internal_state(new_actor)
    assert 6 not in internal_state_after_process["raw_data_buffer"]
    assert internal_state_after_process["last_processed_step"] == 6

    ray.kill(new_actor, no_restart=True)


def test_update_and_get_worker_state(stats_actor):  # Use original fixture
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


# Removed test_all_default_metrics_logged

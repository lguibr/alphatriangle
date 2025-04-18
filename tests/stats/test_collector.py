# File: tests/stats/test_collector.py
import logging
from collections import deque

import cloudpickle
import pytest
import ray

from alphatriangle.stats import StatsCollectorActor
from alphatriangle.utils.types import StepInfo  # Import StepInfo

# --- Fixtures ---


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    if not ray.is_initialized():
        ray.init(logging_level=logging.WARNING, num_cpus=1)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def stats_actor():
    """Provides a fresh StatsCollectorActor instance for each test."""
    actor = StatsCollectorActor.remote(max_history=5)
    # Ensure actor is initialized before returning
    ray.get(actor.clear.remote())  # Use a simple remote call to wait for init
    yield actor
    # Clean up the actor after the test
    ray.kill(actor, no_restart=True)


# --- Helper to create StepInfo ---
def create_step_info(step: int) -> StepInfo:
    """Creates a basic StepInfo dict for testing."""
    return {"global_step": step}


# --- Tests ---


def test_actor_initialization(stats_actor):
    """Test if the actor initializes correctly."""
    assert ray.get(stats_actor.get_data.remote()) == {}
    # Also check initial worker states
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_single_metric(stats_actor):
    """Test logging a single metric."""
    metric_name = "test_metric"
    value = 10.5
    step = 1
    # --- CHANGED: Pass StepInfo ---
    step_info = create_step_info(step)
    ray.get(stats_actor.log.remote(metric_name, value, step_info))
    # --- END CHANGED ---
    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert data[metric_name][0] == (step_info, value)
    # --- END CHANGED ---


def test_log_batch_metrics(stats_actor):
    """Test logging a batch of metrics."""
    # --- CHANGED: Pass StepInfo ---
    step_info_1 = create_step_info(1)
    step_info_2 = create_step_info(2)
    ray.get(stats_actor.log.remote("metric_a", 1.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_b", 2.5, step_info_1))
    ray.get(stats_actor.log.remote("metric_a", 1.1, step_info_2))
    # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert "metric_a" in data
    assert "metric_b" in data
    assert len(data["metric_a"]) == 2, (
        f"Expected 2 entries for metric_a, found {len(data['metric_a'])}"
    )
    assert len(data["metric_b"]) == 1
    # --- CHANGED: Check StepInfo in results ---
    assert data["metric_a"][0] == (step_info_1, 1.0)
    assert data["metric_a"][1] == (step_info_2, 1.1)
    assert data["metric_b"][0] == (step_info_1, 2.5)
    # --- END CHANGED ---


def test_max_history(stats_actor):
    """Test if the max_history constraint is enforced."""
    metric_name = "history_test"
    max_hist = 5  # Matches fixture
    for i in range(max_hist + 3):
        # --- CHANGED: Pass StepInfo ---
        step_info = create_step_info(i)
        ray.get(stats_actor.log.remote(metric_name, float(i), step_info))
        # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == max_hist
    # Check if the first elements were dropped
    # --- CHANGED: Check StepInfo in result ---
    expected_first_step_info = create_step_info(3)
    assert data[metric_name][0] == (expected_first_step_info, 3.0)
    expected_last_step_info = create_step_info(max_hist + 2)
    assert data[metric_name][-1] == (expected_last_step_info, float(max_hist + 2))
    # --- END CHANGED ---


def test_get_metric_data(stats_actor):
    """Test retrieving data for a specific metric."""
    # --- CHANGED: Pass StepInfo ---
    step_info_1 = create_step_info(1)
    step_info_2 = create_step_info(2)
    ray.get(stats_actor.log.remote("metric_1", 10.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_2", 20.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_1", 11.0, step_info_2))
    # --- END CHANGED ---

    metric1_data = ray.get(stats_actor.get_metric_data.remote("metric_1"))
    metric2_data = ray.get(stats_actor.get_metric_data.remote("metric_2"))
    metric3_data = ray.get(stats_actor.get_metric_data.remote("metric_3"))

    assert isinstance(metric1_data, deque)
    assert len(metric1_data) == 2
    # --- CHANGED: Check StepInfo in results ---
    assert list(metric1_data) == [(step_info_1, 10.0), (step_info_2, 11.0)]
    # --- END CHANGED ---

    assert isinstance(metric2_data, deque)
    assert len(metric2_data) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert list(metric2_data) == [(step_info_1, 20.0)]
    # --- END CHANGED ---

    assert metric3_data is None


def test_clear_data(stats_actor):
    """Test clearing the collected data."""
    # --- CHANGED: Pass StepInfo ---
    step_info = create_step_info(1)
    ray.get(stats_actor.log.remote("metric_1", 10.0, step_info))
    # --- END CHANGED ---
    assert len(ray.get(stats_actor.get_data.remote())) == 1
    ray.get(stats_actor.clear.remote())
    assert ray.get(stats_actor.get_data.remote()) == {}
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_non_finite(stats_actor):
    """Test that non-finite values are not logged."""
    metric_name = "non_finite_test"
    # --- CHANGED: Pass StepInfo ---
    ray.get(stats_actor.log.remote(metric_name, float("inf"), create_step_info(1)))
    ray.get(stats_actor.log.remote(metric_name, float("-inf"), create_step_info(2)))
    ray.get(stats_actor.log.remote(metric_name, float("nan"), create_step_info(3)))
    step_info_4 = create_step_info(4)
    ray.get(stats_actor.log.remote(metric_name, 10.0, step_info_4))
    # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert data[metric_name][0] == (step_info_4, 10.0)
    # --- END CHANGED ---


def test_get_set_state(stats_actor):
    """Test saving and restoring the actor's state."""
    # --- CHANGED: Pass StepInfo ---
    step_info_10 = create_step_info(10)
    step_info_11 = create_step_info(11)
    ray.get(stats_actor.log.remote("m1", 1.0, step_info_10))
    ray.get(stats_actor.log.remote("m2", 2.0, step_info_10))
    ray.get(stats_actor.log.remote("m1", 1.5, step_info_11))
    # --- END CHANGED ---

    state = ray.get(stats_actor.get_state.remote())

    # Verify state structure (basic check)
    assert isinstance(state, dict)
    assert "max_history" in state
    assert "_metrics_data_list" in state
    assert isinstance(state["_metrics_data_list"], dict)
    assert "m1" in state["_metrics_data_list"]
    assert isinstance(state["_metrics_data_list"]["m1"], list)
    # --- CHANGED: Check StepInfo in results ---
    assert state["_metrics_data_list"]["m1"] == [
        (step_info_10, 1.0),
        (step_info_11, 1.5),
    ], f"Actual m1 list: {state['_metrics_data_list']['m1']}"
    assert state["_metrics_data_list"]["m2"] == [(step_info_10, 2.0)], (
        f"Actual m2 list: {state['_metrics_data_list']['m2']}"
    )
    # --- END CHANGED ---

    # Use cloudpickle to simulate saving/loading
    pickled_state = cloudpickle.dumps(state)
    unpickled_state = cloudpickle.loads(pickled_state)

    # Create a new actor and restore state
    new_actor = StatsCollectorActor.remote(
        max_history=10
    )  # Different initial max_history
    ray.get(new_actor.set_state.remote(unpickled_state))

    # Verify restored state
    restored_data = ray.get(new_actor.get_data.remote())
    original_data = ray.get(
        stats_actor.get_data.remote()
    )  # Get original data again for comparison

    assert len(restored_data) == len(original_data)
    assert "m1" in restored_data
    assert "m2" in restored_data
    # Compare the deques after converting to lists
    assert list(restored_data["m1"]) == list(original_data["m1"])
    assert list(restored_data["m2"]) == list(original_data["m2"])

    # Check max_history was restored
    # Check behavior by adding more data
    # --- CHANGED: Pass StepInfo ---
    step_info_12 = create_step_info(12)
    step_info_13 = create_step_info(13)
    step_info_14 = create_step_info(14)
    step_info_15 = create_step_info(15)
    ray.get(new_actor.log.remote("m1", 2.0, step_info_12))
    ray.get(new_actor.log.remote("m1", 2.5, step_info_13))
    ray.get(new_actor.log.remote("m1", 3.0, step_info_14))
    ray.get(new_actor.log.remote("m1", 3.5, step_info_15))
    # --- END CHANGED ---

    restored_m1 = ray.get(new_actor.get_metric_data.remote("m1"))
    assert len(restored_m1) == 5  # Max history from loaded state
    # --- CHANGED: Check StepInfo in result ---
    assert restored_m1[0] == (step_info_11, 1.5)  # Check first element is correct
    # --- END CHANGED ---

    # Check that worker states were cleared on restore
    assert ray.get(new_actor.get_latest_worker_states.remote()) == {}

    ray.kill(new_actor, no_restart=True)


# --- Tests for Game State Handling ---
# Mock GameState class for testing state updates
class MockGameStateForStats:
    def __init__(self, step: int, score: float):
        self.current_step = step
        self.game_score = score
        # Add dummy attributes expected by the check in update_worker_game_state
        self.grid_data = True
        self.shapes = True


def test_update_and_get_worker_state(stats_actor):
    """Test updating and retrieving worker game states."""
    worker_id = 1
    state1 = MockGameStateForStats(step=10, score=5.0)
    state2 = MockGameStateForStats(step=11, score=6.0)

    # Initial state should be empty
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}

    # Update state for worker 1
    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state1))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 10
    assert latest_states[worker_id].game_score == 5.0

    # Update state again for worker 1
    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state2))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id].game_score == 6.0

    # Update state for worker 2
    worker_id_2 = 2
    state3 = MockGameStateForStats(step=5, score=2.0)
    ray.get(stats_actor.update_worker_game_state.remote(worker_id_2, state3))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert worker_id_2 in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id_2].current_step == 5

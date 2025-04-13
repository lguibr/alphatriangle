# File: tests/stats/test_collector.py
import pytest
import ray
import time
from collections import deque
import cloudpickle  # Use cloudpickle for state serialization test
import logging  # <-- Import the logging module

from src.stats import StatsCollectorActor


# Fixture to initialize and shutdown Ray for the test session
@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    if not ray.is_initialized():
        # Use logging.WARNING now that logging is imported
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
    ray.get(stats_actor.log.remote(metric_name, value, step))
    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    assert data[metric_name][0] == (step, value)


def test_log_batch_metrics(stats_actor):
    """Test logging a batch of metrics."""
    # Log sequentially to avoid dictionary key overwrite issues in test setup
    ray.get(stats_actor.log.remote("metric_a", 1.0, 1))
    ray.get(stats_actor.log.remote("metric_b", 2.5, 1))
    ray.get(stats_actor.log.remote("metric_a", 1.1, 2))  # Log second point for metric_a

    data = ray.get(stats_actor.get_data.remote())
    assert "metric_a" in data
    assert "metric_b" in data
    # Now metric_a should have 2 entries
    assert (
        len(data["metric_a"]) == 2
    ), f"Expected 2 entries for metric_a, found {len(data['metric_a'])}"
    assert len(data["metric_b"]) == 1
    assert data["metric_a"][0] == (1, 1.0)
    assert data["metric_a"][1] == (2, 1.1)
    assert data["metric_b"][0] == (1, 2.5)


def test_max_history(stats_actor):
    """Test if the max_history constraint is enforced."""
    metric_name = "history_test"
    max_hist = 5  # Matches fixture
    for i in range(max_hist + 3):
        ray.get(stats_actor.log.remote(metric_name, float(i), i))

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == max_hist
    # Check if the first elements were dropped
    assert data[metric_name][0] == (3, 3.0)  # Step 0, 1, 2 should be dropped
    assert data[metric_name][-1] == (max_hist + 2, float(max_hist + 2))


def test_get_metric_data(stats_actor):
    """Test retrieving data for a specific metric."""
    ray.get(stats_actor.log.remote("metric_1", 10.0, 1))
    ray.get(stats_actor.log.remote("metric_2", 20.0, 1))
    ray.get(stats_actor.log.remote("metric_1", 11.0, 2))

    metric1_data = ray.get(stats_actor.get_metric_data.remote("metric_1"))
    metric2_data = ray.get(stats_actor.get_metric_data.remote("metric_2"))
    metric3_data = ray.get(stats_actor.get_metric_data.remote("metric_3"))

    assert isinstance(metric1_data, deque)
    assert len(metric1_data) == 2
    assert list(metric1_data) == [(1, 10.0), (2, 11.0)]

    assert isinstance(metric2_data, deque)
    assert len(metric2_data) == 1
    assert list(metric2_data) == [(1, 20.0)]

    assert metric3_data is None


def test_clear_data(stats_actor):
    """Test clearing the collected data."""
    ray.get(stats_actor.log.remote("metric_1", 10.0, 1))
    assert len(ray.get(stats_actor.get_data.remote())) == 1
    ray.get(stats_actor.clear.remote())
    assert ray.get(stats_actor.get_data.remote()) == {}
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_non_finite(stats_actor):
    """Test that non-finite values are not logged."""
    metric_name = "non_finite_test"
    ray.get(stats_actor.log.remote(metric_name, float("inf"), 1))
    ray.get(stats_actor.log.remote(metric_name, float("-inf"), 2))
    ray.get(stats_actor.log.remote(metric_name, float("nan"), 3))
    ray.get(stats_actor.log.remote(metric_name, 10.0, 4))  # Log a valid one

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    assert data[metric_name][0] == (4, 10.0)


def test_get_set_state(stats_actor):
    """Test saving and restoring the actor's state."""
    # Log some data sequentially
    ray.get(stats_actor.log.remote("m1", 1.0, 10))
    ray.get(stats_actor.log.remote("m2", 2.0, 10))
    ray.get(stats_actor.log.remote("m1", 1.5, 11))

    state = ray.get(stats_actor.get_state.remote())

    # Verify state structure (basic check)
    assert isinstance(state, dict)
    assert "max_history" in state
    # --- Updated Assertion Key ---
    assert "_metrics_data_list" in state
    assert isinstance(state["_metrics_data_list"], dict)
    assert "m1" in state["_metrics_data_list"]
    assert isinstance(state["_metrics_data_list"]["m1"], list)
    # Now check the correct expected list based on sequential logging
    assert state["_metrics_data_list"]["m1"] == [
        (10, 1.0),
        (11, 1.5),
    ], f"Actual m1 list: {state['_metrics_data_list']['m1']}"
    assert state["_metrics_data_list"]["m2"] == [
        (10, 2.0)
    ], f"Actual m2 list: {state['_metrics_data_list']['m2']}"
    # --- End Updated Assertion Key ---

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
    # Need a way to get max_history from actor, let's add a simple getter for test
    # (Alternatively, check behavior by adding more data)
    # Let's check behavior:
    ray.get(new_actor.log.remote("m1", 2.0, 12))
    ray.get(new_actor.log.remote("m1", 2.5, 13))
    ray.get(new_actor.log.remote("m1", 3.0, 14))
    ray.get(new_actor.log.remote("m1", 3.5, 15))  # This should push out (10, 1.0)
    restored_m1 = ray.get(new_actor.get_metric_data.remote("m1"))
    assert len(restored_m1) == 5  # Max history from loaded state
    assert restored_m1[0] == (11, 1.5)  # Check first element is correct

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

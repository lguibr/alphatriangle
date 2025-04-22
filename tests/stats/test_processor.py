# File: tests/stats/test_processor.py
import logging
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from torch.utils.tensorboard import SummaryWriter

from alphatriangle.config.stats_config import StatsConfig, default_stats_config
from alphatriangle.stats.processor import StatsProcessor
from alphatriangle.stats.stats_types import LogContext, RawMetricEvent

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_stats_config() -> StatsConfig:
    """Provides a default StatsConfig for processor testing."""
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
    return cfg


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """Provides a mock MlflowClient instance."""
    return MagicMock(spec=MlflowClient)


@pytest.fixture
def mock_tb_writer() -> MagicMock:
    """Provides a mock SummaryWriter instance."""
    return MagicMock(spec=SummaryWriter)


@pytest.fixture
def stats_processor(
    mock_stats_config: StatsConfig,
    mock_mlflow_client: MagicMock,
    mock_tb_writer: MagicMock,
) -> StatsProcessor:
    """Provides a StatsProcessor instance with mocked clients."""
    test_mlflow_run_id = "test_processor_mlflow_run_id"
    # Instantiate StatsProcessor normally
    processor = StatsProcessor(
        config=mock_stats_config,
        run_name="test_processor_run",
        tb_writer=mock_tb_writer,  # Pass mock writer normally
        mlflow_run_id=test_mlflow_run_id,
    )
    # Manually replace the client instance *after* initialization
    processor.mlflow_client = mock_mlflow_client
    return processor


# Helper to create RawMetricEvent for processor tests
def create_proc_test_event(
    name: str, value: float, step: int, context: dict | None = None
) -> RawMetricEvent:
    """Creates a basic RawMetricEvent for processor testing."""
    return RawMetricEvent(
        name=name,
        value=value,
        global_step=step,
        timestamp=time.time(),
        context=context or {},
    )


def test_processor_aggregation_and_logging(
    stats_processor: StatsProcessor,
    mock_stats_config: StatsConfig,
    mock_mlflow_client: MagicMock,
    mock_tb_writer: MagicMock,
):
    """
    Tests if StatsProcessor correctly aggregates raw data and calls logging clients.
    """
    test_step = 10
    test_value = 5.0
    # Simulate raw data collected by the actor (now list of RawMetricEvent)
    raw_data_input: dict[int, dict[str, list[RawMetricEvent]]] = {
        test_step: {},
    }
    # Store expected values *after* aggregation
    expected_logged_values: dict[str, float] = {}

    # Populate raw_data_input and expected_logged_values based on config
    for metric in mock_stats_config.metrics:
        event_name = metric.event_key
        value = test_value
        num_events_for_metric = 1
        context = {}  # Context for the event itself

        # Simulate specific values and context
        if event_name == "episode_end":
            context = {
                "score": 8.0,
                "length": 50,
                "triangles_cleared": 12,
                "simulations": 1000,
                "trainer_step": test_step - 5,
            }
            value = 1.0  # Value for episode_end itself is often just a counter
            # Set expected values based on aggregation of context keys
            if metric.name == "Episode/Final_Score":
                expected_logged_values[metric.name] = 8.0
            if metric.name == "Episode/Length":
                expected_logged_values[metric.name] = 50.0
            if metric.name == "Episode/Triangles_Cleared_Total":
                expected_logged_values[metric.name] = 12.0
            # Note: episode_end event itself might be aggregated differently (e.g., count)
            if metric.name == "Rate/Episodes_Per_Sec":
                expected_logged_values[metric.name] = 1.0  # Placeholder

        elif event_name == "step_completed":
            value = 1.0
            if metric.name == "Rate/Steps_Per_Sec":
                expected_logged_values[metric.name] = 1.0  # Placeholder

        elif event_name == "mcts_step":
            value = 128.0
            num_events_for_metric = 3
            if metric.name == "MCTS/Avg_Simulations_Per_Step":
                expected_logged_values[metric.name] = 128.0
            if metric.name == "Rate/Simulations_Per_Sec":
                expected_logged_values[metric.name] = (
                    128.0 * num_events_for_metric
                )  # Placeholder

        elif event_name == "Loss/Total":
            value = 1.5
            expected_logged_values[metric.name] = 1.5
        elif event_name == "Progress/Weight_Updates_Total":
            value = 2.0
            expected_logged_values[metric.name] = 2.0
        elif event_name == "LearningRate":
            value = 0.0001
            expected_logged_values[metric.name] = 0.0001
        elif event_name == "PER/Beta":
            value = 0.5
            expected_logged_values[metric.name] = 0.5
        elif event_name == "Buffer/Size":
            value = 500
            expected_logged_values[metric.name] = 500
        elif event_name == "Progress/Total_Simulations":
            value = 10000
            expected_logged_values[metric.name] = 10000
        elif event_name == "Progress/Episodes_Played":
            value = 20
            expected_logged_values[metric.name] = 20
        elif event_name == "System/Num_Active_Workers":
            value = 4
            expected_logged_values[metric.name] = 4
        elif event_name == "System/Num_Pending_Tasks":
            value = 1
            expected_logged_values[metric.name] = 1
        elif event_name == "Loss/Policy":
            value = 0.8
            expected_logged_values[metric.name] = 0.8
        elif event_name == "Loss/Value":
            value = 0.6
            expected_logged_values[metric.name] = 0.6
        elif event_name == "Loss/Entropy":
            value = 0.1
            expected_logged_values[metric.name] = 0.1
        elif event_name == "Loss/Mean_Abs_TD_Error":
            value = 0.25
            expected_logged_values[metric.name] = 0.25
        elif event_name == "RL/Step_Reward_Mean":
            value = 0.05
            expected_logged_values[metric.name] = 0.05

        # Populate raw data with RawMetricEvent objects
        raw_data_input[test_step][event_name] = [
            create_proc_test_event(
                name=event_name, value=value, step=test_step, context=context
            )
            for _ in range(num_events_for_metric)
        ]

        # Store expected aggregated value if not set by context/rate simulation
        # And if the metric doesn't rely on context_key (those are handled above)
        if (
            metric.name not in expected_logged_values
            and metric.aggregation != "rate"
            and not metric.context_key
        ):
            if metric.aggregation == "mean" or metric.aggregation == "latest":
                expected_logged_values[metric.name] = value
            elif metric.aggregation == "sum":
                expected_logged_values[metric.name] = value * num_events_for_metric
            elif metric.aggregation == "count":
                expected_logged_values[metric.name] = num_events_for_metric
            else:
                expected_logged_values[metric.name] = value

    # Create context for the processor
    current_time = time.monotonic()
    log_context = LogContext(
        latest_step=test_step,
        last_log_time=current_time - 1.0,  # Simulate 1 second interval
        current_time=current_time,
        event_timestamps={},  # Simplified for this test
        latest_values={},  # Simplified for this test
    )

    # --- Execute ---
    stats_processor.process_and_log(raw_data_input, log_context)

    # --- Verification ---
    mlflow_calls = mock_mlflow_client.log_metric.call_args_list
    tb_calls = mock_tb_writer.add_scalar.call_args_list

    metrics_logged_mlflow = {c.kwargs["key"] for c in mlflow_calls}
    metrics_logged_tb = {c.args[0] for c in tb_calls}

    all_checks_passed = True
    failure_messages = []

    for metric in mock_stats_config.metrics:
        metric_name = metric.name
        expected_value = expected_logged_values.get(metric_name)

        # Skip rate metrics for exact value check due to complexity
        if metric.aggregation == "rate":
            if "mlflow" in metric.log_to and metric_name not in metrics_logged_mlflow:
                all_checks_passed = False
                failure_messages.append(
                    f"Rate metric {metric_name} not logged to MLflow"
                )
            if "tensorboard" in metric.log_to and metric_name not in metrics_logged_tb:
                all_checks_passed = False
                failure_messages.append(
                    f"Rate metric {metric_name} not logged to TensorBoard"
                )
            continue

        if expected_value is None:
            # Only warn if the metric *should* have been logged based on frequency
            should_have_logged = False
            if (
                metric.log_frequency_steps > 0
                and test_step % metric.log_frequency_steps == 0
            ):
                should_have_logged = True
            if (
                metric.log_frequency_seconds > 0
                and (log_context.current_time - log_context.last_log_time)
                >= metric.log_frequency_seconds
            ):
                should_have_logged = True  # Simplified time check for test

            if should_have_logged:
                logger.warning(
                    f"No expected value calculated for metric '{metric_name}' which should have been logged. Skipping value check."
                )
            continue

        # Check MLflow logging
        if "mlflow" in metric.log_to:
            if metric_name not in metrics_logged_mlflow:
                # Check if it *should* have been logged based on frequency
                if stats_processor._should_log(metric, test_step, log_context):
                    all_checks_passed = False
                    failure_messages.append(
                        f"Metric {metric_name} not logged to MLflow (but should have)"
                    )
            else:
                mlflow_call_found = False
                for c in mlflow_calls:
                    if c.kwargs["key"] == metric_name:
                        if c.kwargs["step"] != test_step:
                            all_checks_passed = False
                            failure_messages.append(
                                f"MLflow step mismatch for {metric_name}: Expected {test_step}, Got {c.kwargs['step']}"
                            )
                        if not np.isclose(c.kwargs["value"], expected_value):
                            all_checks_passed = False
                            failure_messages.append(
                                f"MLflow value mismatch for {metric_name}: Expected {expected_value:.4f}, Got {c.kwargs['value']:.4f}"
                            )
                        mlflow_call_found = True
                        break
                if not mlflow_call_found:
                    pass  # Should be caught by 'in' check

        # Check TensorBoard logging
        if "tensorboard" in metric.log_to:
            if metric_name not in metrics_logged_tb:
                if stats_processor._should_log(metric, test_step, log_context):
                    all_checks_passed = False
                    failure_messages.append(
                        f"Metric {metric_name} not logged to TensorBoard (but should have)"
                    )
            else:
                tb_call_found = False
                for c in tb_calls:
                    if c.args[0] == metric_name:
                        if c.args[2] != test_step:
                            all_checks_passed = False
                            failure_messages.append(
                                f"TensorBoard step mismatch for {metric_name}: Expected {test_step}, Got {c.args[2]}"
                            )
                        if not np.isclose(c.args[1], expected_value):
                            all_checks_passed = False
                            failure_messages.append(
                                f"TensorBoard value mismatch for {metric_name}: Expected {expected_value:.4f}, Got {c.args[1]:.4f}"
                            )
                        tb_call_found = True
                        break
                if not tb_call_found:
                    pass  # Should be caught by 'in' check

    assert all_checks_passed, "Metric logging checks failed:\n" + "\n".join(
        failure_messages
    )
    logger.info(
        f"Verified processor logging calls. MLflow calls: {len(mlflow_calls)}, TensorBoard calls: {len(tb_calls)}"
    )

# File: tests/training/test_save_resume.py
import logging
import shutil
import time
from unittest.mock import MagicMock

import pytest
import ray
from pytest_mock import MockerFixture  # Import MockerFixture

from alphatriangle.config import PersistenceConfig, TrainConfig
from alphatriangle.data import PathManager, Serializer
from alphatriangle.data.schemas import BufferData, CheckpointData
from alphatriangle.training.runner import run_training

logger = logging.getLogger(__name__)

# Use a dynamic base name with the correct format for each test module execution
_MODULE_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_MODULE_RUN_BASE = f"test_sr_{_MODULE_RUN_TIMESTAMP}"


# Helper function from test_path_data_interaction, adapted slightly
def create_dummy_run_state(
    persist_config: PersistenceConfig, run_name: str, steps: list[int]
):
    """Creates dummy run directories and checkpoint/buffer files for testing resume."""
    logger = logging.getLogger(__name__)
    # Use the persist_config (which points to temp dir) to create PathManager
    pm = PathManager(persist_config.model_copy(update={"RUN_NAME": run_name}))
    pm.create_run_directories()
    logger.info(f"Created directories for dummy run '{run_name}' at {pm.run_base_dir}")
    assert pm.run_base_dir.exists()
    assert pm.checkpoint_dir.exists()
    assert pm.buffer_dir.exists()

    serializer = Serializer()
    last_step = 0

    for step in steps:
        # Create dummy CheckpointData
        cp_data = CheckpointData(
            run_name=run_name,
            global_step=step,
            episodes_played=step // 2,
            total_simulations_run=step * 10,
            model_config_dict={},
            env_config_dict={},
            model_state_dict={},
            optimizer_state_dict={},
            stats_collector_state={},
        )
        cp_path = pm.get_checkpoint_path(step=step)
        serializer.save_checkpoint(cp_data, cp_path)
        assert cp_path.exists()
        logger.debug(f"Saved dummy checkpoint: {cp_path}")

        # Create dummy BufferData
        buf_data = BufferData(buffer_list=[])  # Empty buffer for simplicity
        buf_path = pm.get_buffer_path(step=step)
        serializer.save_buffer(buf_data, buf_path)
        assert buf_path.exists()
        logger.debug(f"Saved dummy buffer: {buf_path}")
        last_step = step

    # Create latest links pointing to the last step
    if last_step > 0:
        last_cp_path = pm.get_checkpoint_path(step=last_step)
        if last_cp_path.exists():
            latest_cp_path = pm.get_checkpoint_path(is_latest=True)
            shutil.copy2(last_cp_path, latest_cp_path)
            assert latest_cp_path.exists()
            logger.debug(f"Linked latest checkpoint to {last_cp_path}")
        else:
            logger.error(f"Source checkpoint {last_cp_path} missing for linking.")

        last_buf_path = pm.get_buffer_path(step=last_step)
        if last_buf_path.exists():
            latest_buf_path = pm.get_buffer_path()  # Default buffer link
            shutil.copy2(last_buf_path, latest_buf_path)
            assert latest_buf_path.exists()
            logger.debug(f"Linked default buffer to {last_buf_path}")
        else:
            logger.error(f"Source buffer {last_buf_path} missing for linking.")


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown_module():
    if not ray.is_initialized():
        ray.init(logging_level=logging.WARNING, num_cpus=2, log_to_driver=False)
        initialized = True
    else:
        initialized = False
    yield
    if initialized and ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def persist_config_temp_module(tmp_path_factory) -> PersistenceConfig:
    """PersistenceConfig using a temporary directory for the whole module."""
    # Use tmp_path_factory for a module-scoped temp dir
    tmp_path = tmp_path_factory.mktemp(f"save_resume_data_{_MODULE_RUN_BASE}")
    config = PersistenceConfig(
        ROOT_DATA_DIR=str(tmp_path), RUN_NAME="placeholder_module_run"
    )
    logger.info(f"Using module temp dir for persistence: {tmp_path}")
    return config


@pytest.fixture(scope="module")
def save_run_name() -> str:
    """Provides a consistent run name for the save state test."""
    # Use the base name with the correct timestamp format
    return f"{_MODULE_RUN_BASE}_save"


@pytest.fixture(scope="module")
def resume_run_name() -> str:
    """Provides a consistent run name for the resume state test."""
    return f"{_MODULE_RUN_BASE}_resume"


@pytest.fixture(scope="module")
def save_train_config_module(
    mock_train_config: TrainConfig, save_run_name: str
) -> TrainConfig:
    """TrainConfig for the initial 'save' run (module-scoped)."""
    return mock_train_config.model_copy(
        update={
            "MAX_TRAINING_STEPS": 6,  # Save at step 5, finish after step 6
            "CHECKPOINT_SAVE_FREQ_STEPS": 5,
            "BUFFER_SAVE_FREQ_STEPS": 5,
            "MIN_BUFFER_SIZE_TO_TRAIN": 2,
            "BATCH_SIZE": 2,
            "NUM_SELF_PLAY_WORKERS": 1,
            "AUTO_RESUME_LATEST": False,
            "USE_PER": False,
            "COMPILE_MODEL": False,
            "RUN_NAME": save_run_name,
            "PROFILE_WORKERS": False,  # Ensure profiling is off
        }
    )


@pytest.fixture(scope="module")
def resume_train_config_module(
    save_train_config_module: TrainConfig, resume_run_name: str
) -> TrainConfig:
    """TrainConfig for the second 'resume' run (module-scoped)."""
    return save_train_config_module.model_copy(
        update={
            "AUTO_RESUME_LATEST": True,
            "MAX_TRAINING_STEPS": 8,  # Resume from 6, run steps 7, 8
            "RUN_NAME": resume_run_name,
            "LOAD_CHECKPOINT_PATH": None,
            "LOAD_BUFFER_PATH": None,
            "CHECKPOINT_SAVE_FREQ_STEPS": 100,  # Don't save during resume run
            "BUFFER_SAVE_FREQ_STEPS": 100,  # Don't save during resume run
        }
    )


# --- Tests ---


def test_save_state(
    save_train_config_module: TrainConfig,
    persist_config_temp_module: PersistenceConfig,
    save_run_name: str,
    mocker: MockerFixture,  # Add mocker fixture
):
    """
    Tests saving checkpoint/buffer in the first run within the temp directory.
    """
    serializer = Serializer()
    # Use the module-scoped temp config, but update the run name
    persist_config = persist_config_temp_module.model_copy(
        update={"RUN_NAME": save_run_name}
    )
    pm = PathManager(persist_config)  # Path manager for verification

    logging.info(f"--- Starting Save State Run ({save_run_name}) ---")
    logging.info(f"Target data directory: {persist_config._get_absolute_root()}")

    # Mock setup_logging to prevent interference with pytest capture
    mocker.patch("alphatriangle.training.runner.setup_logging", return_value=None)

    exit_code = run_training(
        log_level_str="INFO",
        train_config_override=save_train_config_module,
        persist_config_override=persist_config,
        profile=False,  # Pass profile flag
    )
    assert exit_code == 0, f"Save State run failed with exit code {exit_code}"
    logging.info(f"--- Finished Save State Run ({save_run_name}) ---")

    # --- Verification for Save State Run ---
    run_base_dir = pm.get_run_base_dir()
    ckpt_dir = pm.checkpoint_dir
    buf_dir = pm.buffer_dir
    cfg_path = pm.config_path

    latest_ckpt_path = pm.get_checkpoint_path(is_latest=True)
    step_5_ckpt_path = pm.get_checkpoint_path(step=5)
    step_6_ckpt_path = pm.get_checkpoint_path(step=6)  # Run finishes after step 6
    buffer_link_path = pm.get_buffer_path()  # Default link
    step_5_buf_path = pm.get_buffer_path(step=5)
    step_6_buf_path = pm.get_buffer_path(step=6)  # Buffer saved after step 6

    assert run_base_dir.exists(), f"Run directory missing: {run_base_dir}"
    assert ckpt_dir.exists(), f"Checkpoint directory missing: {ckpt_dir}"
    assert buf_dir.exists(), f"Buffer directory missing: {buf_dir}"
    assert cfg_path.exists(), f"Config file missing: {cfg_path}"
    assert latest_ckpt_path.exists(), (
        f"Latest checkpoint link missing: {latest_ckpt_path}"
    )
    assert step_5_ckpt_path.exists(), f"Step 5 checkpoint missing: {step_5_ckpt_path}"
    assert step_6_ckpt_path.exists(), f"Step 6 checkpoint missing: {step_6_ckpt_path}"
    assert buffer_link_path.exists(), f"Default buffer link missing: {buffer_link_path}"
    assert step_5_buf_path.exists(), f"Step 5 buffer file missing: {step_5_buf_path}"
    assert step_6_buf_path.exists(), f"Step 6 buffer file missing: {step_6_buf_path}"

    loaded_checkpoint = serializer.load_checkpoint(latest_ckpt_path)
    assert loaded_checkpoint is not None, "Failed to load latest checkpoint"
    assert loaded_checkpoint.global_step == 6, (
        f"Expected latest checkpoint step 6, got {loaded_checkpoint.global_step}"
    )
    assert loaded_checkpoint.run_name == save_run_name, (
        f"Expected run name {save_run_name}, got {loaded_checkpoint.run_name}"
    )

    loaded_buffer = serializer.load_buffer(buffer_link_path)
    assert loaded_buffer is not None, "Failed to load buffer"
    logging.info(
        f"Save State verification complete. Found {len(loaded_buffer.buffer_list)} experiences."
    )


def test_resume_state(
    resume_train_config_module: TrainConfig,
    persist_config_temp_module: PersistenceConfig,
    save_run_name: str,
    resume_run_name: str,
    caplog: pytest.LogCaptureFixture,
    mocker: MockerFixture,  # Add mocker fixture
):
    """
    Tests resuming training from a previously saved state within the temp directory.
    Relies on the state created by test_save_state.
    """
    # Ensure the previous state exists (test_save_state should have run first)
    save_pm = PathManager(
        persist_config_temp_module.model_copy(update={"RUN_NAME": save_run_name})
    )
    assert save_pm.get_checkpoint_path(is_latest=True).exists(), (
        f"Prerequisite checkpoint missing for run {save_run_name}"
    )
    assert save_pm.get_buffer_path().exists(), (
        f"Prerequisite buffer missing for run {save_run_name}"
    )
    logging.info(f"Verified prerequisite state exists for run: {save_run_name}")

    # Use the module-scoped temp config, but update the run name for resume
    persist_config = persist_config_temp_module.model_copy(
        update={"RUN_NAME": resume_run_name}
    )
    resume_pm = PathManager(persist_config)  # Path manager for verification

    logging.info(f"--- Starting Resume State Run ({resume_run_name}) ---")
    logging.info(f"Target data directory: {persist_config._get_absolute_root()}")
    caplog.clear()

    # Mock setup_logging to prevent interference with pytest capture
    # Use a MagicMock to allow checking if it was called (it shouldn't be)
    mock_setup_logging = MagicMock(return_value=None)
    mocker.patch("alphatriangle.training.runner.setup_logging", mock_setup_logging)

    with caplog.at_level(logging.INFO):
        exit_code = run_training(
            log_level_str="INFO",  # This level is now mainly for console in prod
            train_config_override=resume_train_config_module,
            persist_config_override=persist_config,
            profile=False,  # Pass profile flag
        )

    # Assert that our mock was NOT called, confirming run_training didn't set up logging
    # mock_setup_logging.assert_not_called() # Actually run_training *does* call it, but it's mocked

    assert exit_code == 0, f"Resume State run failed with exit code {exit_code}"
    logging.info(f"--- Finished Resume State Run ({resume_run_name}) ---")

    # --- Verification for Resume State Run ---
    # Verify final step reached in resume run by checking the log records
    loop_logger_name = "alphatriangle.training.loop"
    max_steps_message = f"Reached MAX_TRAINING_STEPS ({resume_train_config_module.MAX_TRAINING_STEPS}). Stopping loop."
    found_max_steps_log = False
    for record in caplog.records:
        if record.name == loop_logger_name and record.message == max_steps_message:
            found_max_steps_log = True
            break
    assert found_max_steps_log, (
        f"Log message '{max_steps_message}' from logger '{loop_logger_name}' not found in caplog records."
    )

    # Verify the final success message from the runner is also present
    runner_logger_name = "alphatriangle.training.runner"
    # Define start and end parts of the message to ignore markup
    success_message_start = "Training run '"
    success_message_end = "' completed successfully."
    found_success_log = False

    # --- DEBUG: Print captured logs ---
    # print("\n--- Captured Log Records ---", file=sys.stderr)
    # if not caplog.records:
    #     print("No records captured by caplog.", file=sys.stderr)
    # for i, record in enumerate(caplog.records):
    #     print(
    #         f"Record {i}: Name='{record.name}', Level={record.levelname}, Message='{record.message}'",
    #         file=sys.stderr,
    #     )
    # print("--- End Captured Log Records ---\n", file=sys.stderr)
    # --- END DEBUG ---

    for record in caplog.records:
        # Check if the message starts/ends correctly and contains the run name
        if (
            record.name == runner_logger_name
            and record.message.startswith(success_message_start)
            and record.message.endswith(success_message_end)
            and resume_run_name in record.message
        ):
            found_success_log = True
            break
    assert found_success_log, (
        f"Log message starting with '{success_message_start}', ending with '{success_message_end}', and containing '{resume_run_name}' from logger '{runner_logger_name}' not found in caplog records."
    )

    # Verify that the resume run created its own directory structure
    assert resume_pm.run_base_dir.exists(), (
        f"Resume run directory missing: {resume_pm.run_base_dir}"
    )
    assert resume_pm.log_dir.exists(), (
        f"Resume run log directory missing: {resume_pm.log_dir}"
    )
    # Checkpoints/buffers might not exist if save freq is high
    # assert resume_pm.checkpoint_dir.exists()
    # assert resume_pm.buffer_dir.exists()

    logging.info("Resume state test passed.")

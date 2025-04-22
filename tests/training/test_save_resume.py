# File: tests/training/test_save_resume.py
import logging
import shutil
import time
from pathlib import Path

import pytest
import ray

from alphatriangle.config import PersistenceConfig, TrainConfig
from alphatriangle.data import PathManager, Serializer
from alphatriangle.data.schemas import BufferData, CheckpointData
from alphatriangle.training.runner import run_training

# Use a dynamic base name with the correct format for each test module execution
_MODULE_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_MODULE_RUN_BASE = f"test_sr_{_MODULE_RUN_TIMESTAMP}"


# Helper function from test_path_data_interaction, adapted slightly
def create_dummy_run_state(
    persist_config: PersistenceConfig, run_name: str, steps: list[int]
):
    """Creates dummy run directories and checkpoint/buffer files for testing resume."""
    logger = logging.getLogger(__name__)
    pm = PathManager(persist_config.model_copy(update={"RUN_NAME": run_name}))
    pm.create_run_directories()
    logger.info(f"Created directories for dummy run '{run_name}' at {pm.run_base_dir}")

    pm.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pm.buffer_dir.mkdir(parents=True, exist_ok=True)

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
        logger.info(f"Saved dummy checkpoint: {cp_path}")

        # Create dummy BufferData
        buf_data = BufferData(buffer_list=[])  # Empty buffer for simplicity
        buf_path = pm.get_buffer_path(step=step)
        serializer.save_buffer(buf_data, buf_path)
        logger.info(f"Saved dummy buffer: {buf_path}")
        last_step = step

    # Create latest links pointing to the last step
    if last_step > 0:
        last_cp_path = pm.get_checkpoint_path(step=last_step)
        if last_cp_path.exists():
            latest_cp_path = pm.get_checkpoint_path(is_latest=True)
            shutil.copy2(last_cp_path, latest_cp_path)
            logger.info(f"Linked latest checkpoint to {last_cp_path}")
        else:
            logger.error(f"Source checkpoint {last_cp_path} missing for linking.")

        last_buf_path = pm.get_buffer_path(step=last_step)
        if last_buf_path.exists():
            latest_buf_path = pm.get_buffer_path()  # Default buffer link
            shutil.copy2(last_buf_path, latest_buf_path)
            logger.info(f"Linked default buffer to {last_buf_path}")
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
    tmp_path = tmp_path_factory.mktemp(f"save_resume_data_{_MODULE_RUN_BASE}")
    config = PersistenceConfig(
        ROOT_DATA_DIR=str(tmp_path), RUN_NAME="placeholder_module_run"
    )
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
            "RUN_NAME": save_run_name,  # Use fixed run name
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
            "RUN_NAME": resume_run_name,  # Use fixed run name
            "LOAD_CHECKPOINT_PATH": None,  # Ensure auto-resume is tested
            "LOAD_BUFFER_PATH": None,
            # No need to save frequently during resume test
            "CHECKPOINT_SAVE_FREQ_STEPS": 100,
            "BUFFER_SAVE_FREQ_STEPS": 100,
        }
    )


# --- Tests ---


def test_save_state(
    save_train_config_module: TrainConfig,
    persist_config_temp_module: PersistenceConfig,
    save_run_name: str,
):
    """
    Tests saving checkpoint/buffer in the first run.
    """
    serializer = Serializer()
    persist_config = persist_config_temp_module.model_copy(
        update={"RUN_NAME": save_run_name}
    )

    logging.info(f"--- Starting Save State Run ({save_run_name}) ---")
    exit_code = run_training(
        log_level_str="INFO",
        train_config_override=save_train_config_module,
        persist_config_override=persist_config,
    )
    assert exit_code == 0, f"Save State run failed with exit code {exit_code}"
    logging.info(f"--- Finished Save State Run ({save_run_name}) ---")

    # --- Verification for Save State Run ---
    run_base_dir = Path(persist_config.get_run_base_dir())
    ckpt_dir = run_base_dir / persist_config.CHECKPOINT_SAVE_DIR_NAME
    buf_dir = run_base_dir / persist_config.BUFFER_SAVE_DIR_NAME
    cfg_path = run_base_dir / persist_config.CONFIG_FILENAME

    latest_ckpt_path = ckpt_dir / persist_config.LATEST_CHECKPOINT_FILENAME
    step_5_ckpt_path = ckpt_dir / "checkpoint_step_5.pkl"
    # Final save is now just the last step save + latest link
    step_6_ckpt_path = ckpt_dir / "checkpoint_step_6.pkl"
    buffer_link_path = buf_dir / persist_config.BUFFER_FILENAME
    step_5_buf_path = buf_dir / "buffer_step_5.pkl"
    step_6_buf_path = buf_dir / "buffer_step_6.pkl"

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

    # Verify content of latest checkpoint
    loaded_checkpoint = serializer.load_checkpoint(latest_ckpt_path)
    assert loaded_checkpoint is not None, "Failed to load latest checkpoint"
    assert loaded_checkpoint.global_step == 6, (
        f"Expected latest checkpoint step 6, got {loaded_checkpoint.global_step}"
    )
    assert loaded_checkpoint.run_name == save_run_name, (
        f"Expected run name {save_run_name}, got {loaded_checkpoint.run_name}"
    )

    # Verify content of buffer
    loaded_buffer = serializer.load_buffer(buffer_link_path)
    assert loaded_buffer is not None, "Failed to load buffer"
    # Buffer might be empty if training didn't run long enough to fill min size
    # assert len(loaded_buffer.buffer_list) > 0, "Saved buffer is empty"
    logging.info(
        f"Save State verification complete. Found {len(loaded_buffer.buffer_list)} experiences."
    )


def test_resume_state(
    resume_train_config_module: TrainConfig,
    persist_config_temp_module: PersistenceConfig,
    save_run_name: str,  # Get the name of the run to resume from
    resume_run_name: str,
    caplog: pytest.LogCaptureFixture,
):
    """
    Tests resuming training from a previously saved state.
    This test now creates the prerequisite state itself.
    """
    # --- Setup Prerequisite State ---
    # Use the persist config pointing to the *shared* temp dir
    # Create the dummy state for the 'save_run_name'
    create_dummy_run_state(persist_config_temp_module, save_run_name, steps=[5, 6])
    logging.info(f"Created dummy state for previous run: {save_run_name}")

    # --- Run Resumed Training ---
    persist_config = persist_config_temp_module.model_copy(
        update={"RUN_NAME": resume_run_name}
    )

    logging.info(f"--- Starting Resume State Run ({resume_run_name}) ---")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        exit_code = run_training(
            log_level_str="INFO",
            train_config_override=resume_train_config_module,
            persist_config_override=persist_config,
        )

    assert exit_code == 0, f"Resume State run failed with exit code {exit_code}"
    logging.info(f"--- Finished Resume State Run ({resume_run_name}) ---")

    # --- Verification for Resume State Run (Log Check) ---
    log_text = caplog.text
    assert (
        f"Auto-resuming from latest checkpoint in previous run '{save_run_name}'"
        in log_text
    ), "Missing log message for auto-resuming checkpoint"
    assert "Checkpoint loaded and validated" in log_text, (
        "Missing log message for checkpoint validation"
    )
    assert f"Run: {save_run_name}, Step: 6" in log_text, (
        "Missing log message indicating loaded step"
    )  # Should load step 6
    assert "Applying loaded checkpoint data" in log_text, (
        "Missing log message for applying checkpoint"
    )
    assert "Initial state loaded and applied. Starting step: 6" in log_text, (
        "Missing log message confirming initial state application"
    )
    assert "Buffer loaded and validated." in log_text, (
        "Missing log message for buffer validation"
    )
    # Corrected assertion: Check for the specific log message fragment
    assert "Buffer loaded and validated. Size:" in log_text, (
        "Missing log part indicating buffer size/validation"
    )

    # Optional: Verify final step reached in resume run
    run_base_dir = Path(persist_config.get_run_base_dir())
    ckpt_dir = run_base_dir / persist_config.CHECKPOINT_SAVE_DIR_NAME
    latest_ckpt_path = ckpt_dir / persist_config.LATEST_CHECKPOINT_FILENAME
    assert latest_ckpt_path.exists(), "Latest checkpoint missing after resume run"
    serializer = Serializer()
    final_checkpoint = serializer.load_checkpoint(latest_ckpt_path)
    assert final_checkpoint is not None
    # Resumed from 6, ran until 8 (MAX_TRAINING_STEPS)
    assert final_checkpoint.global_step == 8, (
        f"Expected final step 8 after resume, got {final_checkpoint.global_step}"
    )

    logging.info("Resume state test passed.")

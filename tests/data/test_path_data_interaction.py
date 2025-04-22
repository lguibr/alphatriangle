import logging
import shutil
import time
from pathlib import Path

import pytest

from alphatriangle.config import PersistenceConfig, TrainConfig
from alphatriangle.data import DataManager, PathManager, Serializer
from alphatriangle.data.schemas import BufferData, CheckpointData

# Add logger for test debugging
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Creates a temporary root data directory for a single test."""
    # Use tmp_path directly provided by pytest for test isolation
    data_dir = tmp_path / ".test_alphatriangle_data"
    data_dir.mkdir()
    logger.info(f"Created temporary data dir for test: {data_dir}")
    return data_dir


@pytest.fixture
def persist_config(temp_data_dir: Path) -> PersistenceConfig:
    """PersistenceConfig using the temporary directory for a single test."""
    # Use the test-specific temp dir
    return PersistenceConfig(ROOT_DATA_DIR=str(temp_data_dir))


@pytest.fixture
def train_config() -> TrainConfig:
    """Basic TrainConfig."""
    # Minimal config needed for DataManager init
    return TrainConfig(RUN_NAME="dummy_run")


@pytest.fixture
def serializer() -> Serializer:
    """Serializer instance."""
    return Serializer()


def create_dummy_run(
    persist_config: PersistenceConfig, run_name: str, steps: list[int]
):
    """Creates dummy run directories and checkpoint/buffer files within the temp dir."""
    # Create a PathManager instance specifically for this dummy run setup
    # It will use the ROOT_DATA_DIR from the provided persist_config (temp dir)
    pm = PathManager(persist_config.model_copy(update={"RUN_NAME": run_name}))
    # Create the base run directory and standard subdirs first
    pm.create_run_directories()
    logger.info(f"Created directories for run '{run_name}' at {pm.run_base_dir}")
    assert pm.run_base_dir.exists()
    assert pm.checkpoint_dir.exists()
    assert pm.buffer_dir.exists()

    # Create dummy checkpoint files
    for step in steps:
        cp_path = pm.get_checkpoint_path(step=step)
        logger.debug(f"Attempting to touch checkpoint file: {cp_path}")
        cp_path.touch()
        assert cp_path.exists()
        logger.debug(f"Touched checkpoint file: {cp_path}")

        # Create dummy buffer files if needed
        buf_path = pm.get_buffer_path(step=step)
        logger.debug(f"Attempting to touch buffer file: {buf_path}")
        buf_path.touch()
        assert buf_path.exists()
        logger.debug(f"Touched buffer file: {buf_path}")

    # Create latest links pointing to the last step
    if steps:
        last_step = steps[-1]
        last_cp_path = pm.get_checkpoint_path(step=last_step)
        if last_cp_path.exists():  # Check if source exists before linking
            latest_cp_path = pm.get_checkpoint_path(is_latest=True)
            shutil.copy2(last_cp_path, latest_cp_path)
            assert latest_cp_path.exists()
            logger.debug(f"Linked latest checkpoint to {last_cp_path}")
        else:
            pytest.fail(
                f"Source checkpoint file {last_cp_path} does not exist for linking."
            )

        last_buf_path = pm.get_buffer_path(step=last_step)
        if last_buf_path.exists():  # Check if source exists before linking
            latest_buf_path = pm.get_buffer_path()  # Default buffer link
            shutil.copy2(last_buf_path, latest_buf_path)
            assert latest_buf_path.exists()
            logger.debug(f"Linked default buffer to {last_buf_path}")
        else:
            pytest.fail(
                f"Source buffer file {last_buf_path} does not exist for linking."
            )


def test_find_latest_run_dir(persist_config: PersistenceConfig):
    """Tests the logic for finding the most recent previous run."""
    # Use YYYYMMDD_HHMMSS format for timestamps
    ts_current = time.strftime("%Y%m%d_%H%M%S")
    time.sleep(1.1)  # Ensure timestamps are different
    ts_latest_prev = time.strftime("%Y%m%d_%H%M%S")
    time.sleep(1.1)
    ts_older = time.strftime("%Y%m%d_%H%M%S")  # This is actually the latest timestamp

    run_name_current = f"run_{ts_current}_current"
    run_name_older = (
        f"run_{ts_older}_older"  # This run has the latest timestamp among previous runs
    )
    run_name_latest_previous = (
        f"run_{ts_latest_prev}_latest_prev"  # This run has the middle timestamp
    )
    run_name_no_timestamp = "run_no_timestamp"  # This one won't be matched by regex

    # Create dummy directories using the corrected helper (within temp dir)
    create_dummy_run(persist_config, run_name_current, [])
    create_dummy_run(persist_config, run_name_older, [])
    create_dummy_run(persist_config, run_name_latest_previous, [])
    create_dummy_run(persist_config, run_name_no_timestamp, [])

    # Use a PathManager instance configured for the *current* run to find previous ones
    pm = PathManager(persist_config.model_copy(update={"RUN_NAME": run_name_current}))
    latest_found = pm.find_latest_run_dir(current_run_name=run_name_current)

    # The latest *previous* run should be the one with the latest timestamp (ts_older)
    assert latest_found == run_name_older, (
        f"Expected '{run_name_older}' (latest timestamp) but found '{latest_found}'"
    )


def test_determine_checkpoint_to_load_auto_resume(
    persist_config: PersistenceConfig, serializer: Serializer
):
    """Tests auto-resuming the latest checkpoint."""
    ts_current = time.strftime("%Y%m%d_%H%M%S")
    time.sleep(1.1)
    ts_previous = time.strftime("%Y%m%d_%H%M%S")

    run_name_current = f"run_{ts_current}_current_auto"
    run_name_previous = f"run_{ts_previous}_previous_auto"

    # Create previous run with checkpoints using the corrected helper
    create_dummy_run(persist_config, run_name_previous, steps=[5, 10])

    # Create a dummy CheckpointData for step 10 to make latest.pkl valid
    dummy_cp_data = CheckpointData(
        run_name=run_name_previous,
        global_step=10,
        episodes_played=1,
        total_simulations_run=100,
        model_config_dict={},
        env_config_dict={},
        model_state_dict={},
        optimizer_state_dict={},
        stats_collector_state={},
    )
    # Use a PM instance configured for the previous run to get the correct path
    pm_prev = PathManager(
        persist_config.model_copy(update={"RUN_NAME": run_name_previous})
    )
    latest_cp_path = pm_prev.get_checkpoint_path(is_latest=True)
    # Ensure the directory exists before saving (create_dummy_run should handle this)
    latest_cp_path.parent.mkdir(parents=True, exist_ok=True)
    serializer.save_checkpoint(dummy_cp_data, latest_cp_path)

    # Setup DataManager for the current run
    current_persist_config = persist_config.model_copy(
        update={"RUN_NAME": run_name_current}
    )
    current_train_config = TrainConfig(
        RUN_NAME=run_name_current, AUTO_RESUME_LATEST=True
    )
    dm = DataManager(current_persist_config, current_train_config)

    # Determine path to load
    load_path = dm.path_manager.determine_checkpoint_to_load(
        load_path_config=None, auto_resume=True
    )

    assert load_path is not None
    assert load_path.name == persist_config.LATEST_CHECKPOINT_FILENAME
    # Check parent structure relative to the temp dir
    assert load_path.parent.parent.name == run_name_previous
    assert load_path.parent.parent.parent.name == persist_config.RUNS_DIR_NAME
    assert load_path.parent.parent.parent.parent == persist_config._get_absolute_root()


def test_determine_checkpoint_to_load_specific_path(
    persist_config: PersistenceConfig, serializer: Serializer
):
    """Tests loading a checkpoint from a specific path."""
    ts_current = time.strftime("%Y%m%d_%H%M%S")
    time.sleep(1.1)
    ts_other = time.strftime("%Y%m%d_%H%M%S")

    run_name_current = f"run_{ts_current}_current_spec"
    run_name_other = f"run_{ts_other}_other_spec"
    # Use corrected helper
    create_dummy_run(persist_config, run_name_other, steps=[5])

    # Create a dummy CheckpointData for step 5
    dummy_cp_data = CheckpointData(
        run_name=run_name_other,
        global_step=5,
        episodes_played=1,
        total_simulations_run=50,
        model_config_dict={},
        env_config_dict={},
        model_state_dict={},
        optimizer_state_dict={},
        stats_collector_state={},
    )
    # Use PM configured for the 'other' run
    pm_other = PathManager(
        persist_config.model_copy(update={"RUN_NAME": run_name_other})
    )
    specific_cp_path = pm_other.get_checkpoint_path(step=5)
    # Ensure directory exists (create_dummy_run should handle this)
    specific_cp_path.parent.mkdir(parents=True, exist_ok=True)
    serializer.save_checkpoint(dummy_cp_data, specific_cp_path)

    # Setup DataManager
    current_persist_config = persist_config.model_copy(
        update={"RUN_NAME": run_name_current}
    )
    current_train_config = TrainConfig(
        RUN_NAME=run_name_current,
        AUTO_RESUME_LATEST=False,
        LOAD_CHECKPOINT_PATH=str(specific_cp_path),  # Provide specific path
    )
    dm = DataManager(current_persist_config, current_train_config)

    load_path = dm.path_manager.determine_checkpoint_to_load(
        load_path_config=current_train_config.LOAD_CHECKPOINT_PATH, auto_resume=False
    )

    assert load_path is not None
    assert load_path.resolve() == specific_cp_path.resolve()


def test_determine_buffer_to_load_from_checkpoint_run(
    persist_config: PersistenceConfig, serializer: Serializer
):
    """Tests loading buffer associated with the loaded checkpoint."""
    ts_current = time.strftime("%Y%m%d_%H%M%S")
    time.sleep(1.1)
    ts_previous = time.strftime("%Y%m%d_%H%M%S")

    run_name_current = f"run_{ts_current}_current_buf_chk"
    run_name_previous = f"run_{ts_previous}_previous_buf_chk"
    # Use corrected helper
    create_dummy_run(
        persist_config, run_name_previous, steps=[5]
    )  # Creates buffer_step_5.pkl and buffer.pkl link

    # Create dummy BufferData
    dummy_buf_data = BufferData(buffer_list=[])
    # Use PM configured for previous run
    pm_prev = PathManager(
        persist_config.model_copy(update={"RUN_NAME": run_name_previous})
    )
    default_buf_path = pm_prev.get_buffer_path()
    # Ensure directory exists (create_dummy_run should handle this)
    default_buf_path.parent.mkdir(parents=True, exist_ok=True)
    serializer.save_buffer(dummy_buf_data, default_buf_path)

    # Setup DataManager
    current_persist_config = persist_config.model_copy(
        update={"RUN_NAME": run_name_current}
    )
    current_train_config = TrainConfig(
        RUN_NAME=run_name_current, AUTO_RESUME_LATEST=False
    )
    dm = DataManager(current_persist_config, current_train_config)

    # Simulate that checkpoint_run_name was determined to be run_name_previous
    load_path = dm.path_manager.determine_buffer_to_load(
        load_path_config=None, auto_resume=False, checkpoint_run_name=run_name_previous
    )

    assert load_path is not None
    assert load_path.name == persist_config.BUFFER_FILENAME
    # Check parent structure relative to the temp dir
    assert load_path.parent.parent.name == run_name_previous
    assert load_path.parent.parent.parent.name == persist_config.RUNS_DIR_NAME
    assert load_path.parent.parent.parent.parent == persist_config._get_absolute_root()

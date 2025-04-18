# File: alphatriangle/data/path_manager.py
import datetime
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PersistenceConfig

logger = logging.getLogger(__name__)


class PathManager:
    """Manages file paths, directory creation, and discovery for training runs."""

    def __init__(self, persist_config: "PersistenceConfig"):
        self.persist_config = persist_config
        self.root_data_dir = Path(self.persist_config.ROOT_DATA_DIR)
        self._update_paths()  # Initialize paths based on config

    def _update_paths(self):
        """Updates paths based on the current RUN_NAME in persist_config."""
        self.run_base_dir = Path(self.persist_config.get_run_base_dir())
        self.checkpoint_dir = (
            self.run_base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        self.buffer_dir = self.run_base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        self.log_dir = self.run_base_dir / self.persist_config.LOG_DIR_NAME
        self.config_path = self.run_base_dir / self.persist_config.CONFIG_FILENAME

    def create_run_directories(self):
        """Creates necessary directories for the current run."""
        self.root_data_dir.mkdir(parents=True, exist_ok=True)
        self.run_base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.persist_config.SAVE_BUFFER:
            self.buffer_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """Constructs the path for a checkpoint file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        checkpoint_dir = base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        if is_latest:
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        elif is_best:
            filename = self.persist_config.BEST_CHECKPOINT_FILENAME
        elif is_final and step is not None:
            filename = f"checkpoint_final_step_{step}.pkl"
        elif step is not None:
            filename = f"checkpoint_step_{step}.pkl"
        else:
            # Default to latest if no specific type is given
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        # Ensure filename ends with .pkl
        filename_pkl = Path(filename).with_suffix(".pkl")
        return checkpoint_dir / filename_pkl

    def get_buffer_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_final: bool = False,
    ) -> Path:
        """Constructs the path for the replay buffer file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        buffer_dir = base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        if is_final and step is not None:
            filename = f"buffer_final_step_{step}.pkl"
        elif step is not None and self.persist_config.BUFFER_SAVE_FREQ_STEPS > 0:
            # Use default name for intermediate saves if frequency is set
            filename = self.persist_config.BUFFER_FILENAME
        else:
            # Default name for initial load or if frequency is not set
            filename = self.persist_config.BUFFER_FILENAME
        return buffer_dir / Path(filename).with_suffix(".pkl")

    def get_config_path(self, run_name: str | None = None) -> Path:
        """Constructs the path for the config JSON file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        return base_dir / self.persist_config.CONFIG_FILENAME

    def find_latest_run_dir(self, current_run_name: str) -> str | None:
        """
        Finds the most recent *previous* run directory based on timestamp parsing.
        Assumes run names follow a pattern like 'prefix_YYYYMMDD_HHMMSS'.
        """
        runs_root_dir = self.root_data_dir / self.persist_config.RUNS_DIR_NAME
        potential_runs: list[tuple[datetime.datetime, str]] = []
        run_name_pattern = re.compile(r"^(?:test_run|train)_(\d{8}_\d{6})$")

        try:
            if not runs_root_dir.exists():
                return None

            for d in runs_root_dir.iterdir():
                if d.is_dir() and d.name != current_run_name:
                    match = run_name_pattern.match(d.name)
                    if match:
                        timestamp_str = match.group(1)
                        try:
                            run_time = datetime.datetime.strptime(
                                timestamp_str, "%Y%m%d_%H%M%S"
                            )
                            potential_runs.append((run_time, d.name))
                        except ValueError:
                            logger.warning(
                                f"Could not parse timestamp from directory name: {d.name}"
                            )
                    else:
                        logger.debug(
                            f"Directory name {d.name} does not match expected pattern."
                        )

            if not potential_runs:
                logger.info("No previous run directories found matching the pattern.")
                return None

            potential_runs.sort(key=lambda item: item[0], reverse=True)
            latest_run_name = potential_runs[0][1]
            logger.debug(
                f"Found potential previous runs (sorted): {[name for _, name in potential_runs]}. Latest: {latest_run_name}"
            )
            return latest_run_name

        except Exception as e:
            logger.error(f"Error finding latest run directory: {e}", exc_info=True)
            return None

    def determine_checkpoint_to_load(
        self, load_path_config: str | None, auto_resume: bool
    ) -> Path | None:
        """Determines the absolute path of the checkpoint file to load."""
        current_run_name = self.persist_config.RUN_NAME
        checkpoint_to_load: Path | None = None

        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                checkpoint_to_load = load_path.resolve()
                logger.info(f"Using specified checkpoint path: {checkpoint_to_load}")
            else:
                logger.warning(
                    f"Specified checkpoint path not found: {load_path_config}"
                )

        if not checkpoint_to_load and auto_resume:
            latest_run_name = self.find_latest_run_dir(current_run_name)
            if latest_run_name:
                potential_latest_path = self.get_checkpoint_path(
                    run_name=latest_run_name, is_latest=True
                )
                if potential_latest_path.exists():
                    checkpoint_to_load = potential_latest_path.resolve()
                    logger.info(
                        f"Auto-resuming from latest checkpoint in previous run '{latest_run_name}': {checkpoint_to_load}"
                    )
                else:
                    logger.info(
                        f"Latest checkpoint file not found in latest run directory '{latest_run_name}'."
                    )
            else:
                logger.info("Auto-resume enabled, but no previous run directory found.")

        if not checkpoint_to_load:
            logger.info("No checkpoint found to load. Starting training from scratch.")

        return checkpoint_to_load

    def determine_buffer_to_load(
        self,
        load_path_config: str | None,
        auto_resume: bool,
        checkpoint_run_name: str | None,
    ) -> Path | None:
        """Determines the buffer file path to load."""
        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                logger.info(f"Using specified buffer path: {load_path_config}")
                return load_path.resolve()
            else:
                logger.warning(f"Specified buffer path not found: {load_path_config}")

        if checkpoint_run_name:
            potential_buffer_path = self.get_buffer_path(run_name=checkpoint_run_name)
            if potential_buffer_path.exists():
                logger.info(
                    f"Loading buffer from checkpoint run '{checkpoint_run_name}': {potential_buffer_path}"
                )
                return potential_buffer_path.resolve()
            else:
                logger.info(
                    f"Default buffer file not found in checkpoint run directory '{checkpoint_run_name}'."
                )

        if auto_resume and not checkpoint_run_name:
            latest_previous_run_name = self.find_latest_run_dir(
                self.persist_config.RUN_NAME
            )
            if latest_previous_run_name:
                potential_buffer_path = self.get_buffer_path(
                    run_name=latest_previous_run_name
                )
                if potential_buffer_path.exists():
                    logger.info(
                        f"Auto-resuming buffer from latest previous run '{latest_previous_run_name}' (no checkpoint loaded): {potential_buffer_path}"
                    )
                    return potential_buffer_path.resolve()
                else:
                    logger.info(
                        f"Default buffer file not found in latest run directory '{latest_previous_run_name}'."
                    )

        logger.info("No suitable buffer file found to load.")
        return None

    def update_checkpoint_links(self, step_checkpoint_path: Path, is_best: bool):
        """Updates the 'latest' and optionally 'best' checkpoint links."""
        if not step_checkpoint_path.exists():
            logger.error(
                f"Source checkpoint path does not exist: {step_checkpoint_path}"
            )
            return

        latest_path = self.get_checkpoint_path(is_latest=True)
        best_path = self.get_checkpoint_path(is_best=True)
        try:
            shutil.copy2(step_checkpoint_path, latest_path)
            logger.debug(f"Updated latest checkpoint link to {step_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to update latest checkpoint link: {e}")
        if is_best:
            try:
                shutil.copy2(step_checkpoint_path, best_path)
                logger.info(
                    f"Updated best checkpoint link to step {step_checkpoint_path.stem}"
                )
            except Exception as e:
                logger.error(f"Failed to update best checkpoint link: {e}")

    def update_buffer_link(self, step_buffer_path: Path):
        """Updates the default buffer link."""
        if not step_buffer_path.exists():
            logger.error(f"Source buffer path does not exist: {step_buffer_path}")
            return

        default_buffer_path = self.get_buffer_path()
        try:
            shutil.copy2(step_buffer_path, default_buffer_path)
            logger.debug(f"Updated default buffer file: {default_buffer_path}")
        except Exception as e_default:
            logger.error(
                f"Error updating default buffer file {default_buffer_path}: {e_default}"
            )

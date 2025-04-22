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
        # Resolve root data dir immediately
        self.root_data_dir = self._resolve_root_data_dir()
        self._update_paths()  # Initialize paths based on config

    def _resolve_root_data_dir(self) -> Path:
        """Resolves ROOT_DATA_DIR to an absolute path relative to the project root."""
        project_root = Path.cwd()  # Assume running from project root
        root_path = project_root / self.persist_config.ROOT_DATA_DIR
        return root_path.resolve()

    def _update_paths(self):
        """Updates paths based on the current RUN_NAME in persist_config."""
        self.run_base_dir = self.get_run_base_dir()  # Use method to get absolute path
        self.checkpoint_dir = (
            self.run_base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        self.buffer_dir = self.run_base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        self.log_dir = self.run_base_dir / self.persist_config.LOG_DIR_NAME
        self.tb_log_dir = self.run_base_dir / self.persist_config.TENSORBOARD_DIR_NAME
        self.profile_dir = self.run_base_dir / self.persist_config.PROFILE_DIR_NAME
        self.config_path = self.run_base_dir / self.persist_config.CONFIG_FILENAME

    def get_runs_root_dir(self) -> Path:
        """Gets the absolute path to the directory containing all runs."""
        return self.root_data_dir / self.persist_config.RUNS_DIR_NAME

    def get_run_base_dir(self, run_name: str | None = None) -> Path:
        """Gets the absolute base directory path for a specific run."""
        runs_root = self.get_runs_root_dir()
        name = run_name if run_name else self.persist_config.RUN_NAME
        return runs_root / name

    def create_run_directories(self):
        """Creates necessary directories for the current run."""
        self.root_data_dir.mkdir(parents=True, exist_ok=True)
        self.run_base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)  # Create profile dir
        if self.persist_config.SAVE_BUFFER:
            self.buffer_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
    ) -> Path:
        """Constructs the absolute path for a checkpoint file."""
        target_run_base_dir = self.get_run_base_dir(run_name)
        checkpoint_dir = (
            target_run_base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )

        if is_latest:
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        elif is_best:
            filename = self.persist_config.BEST_CHECKPOINT_FILENAME
        elif step is not None:
            filename = f"checkpoint_step_{step}.pkl"
        else:
            # Default to latest if no specific type is given
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME

        filename_pkl = Path(filename).with_suffix(".pkl")
        return checkpoint_dir / filename_pkl

    def get_buffer_path(
        self, run_name: str | None = None, step: int | None = None
    ) -> Path:
        """Constructs the absolute path for the replay buffer file."""
        target_run_base_dir = self.get_run_base_dir(run_name)
        buffer_dir = target_run_base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME

        if step is not None:
            filename = f"buffer_step_{step}.pkl"
        else:
            # Default name for the main buffer link/file
            filename = self.persist_config.BUFFER_FILENAME

        return buffer_dir / Path(filename).with_suffix(".pkl")

    def get_config_path(self, run_name: str | None = None) -> Path:
        """Constructs the absolute path for the config JSON file."""
        target_run_base_dir = self.get_run_base_dir(run_name)
        return target_run_base_dir / self.persist_config.CONFIG_FILENAME

    def get_profile_path(
        self, worker_id: int, episode_seed: int, run_name: str | None = None
    ) -> Path:
        """Constructs the absolute path for a profile data file."""
        target_run_base_dir = self.get_run_base_dir(run_name)
        profile_dir = target_run_base_dir / self.persist_config.PROFILE_DIR_NAME
        filename = f"worker_{worker_id}_ep_{episode_seed}.prof"
        return profile_dir / filename

    def find_latest_run_dir(self, current_run_name: str) -> str | None:
        """
        Finds the most recent *previous* run directory based on timestamp parsing.
        Assumes run names contain a 'YYYYMMDD_HHMMSS' pattern, potentially with prefixes/suffixes.
        """
        runs_root_dir = self.get_runs_root_dir()
        potential_runs: list[tuple[datetime.datetime, str]] = []
        # Regex: Find YYYYMMDD_HHMMSS pattern anywhere in the name
        run_name_pattern = re.compile(r"(\d{8}_\d{6})")
        logger.info(f"Searching for previous runs in: {runs_root_dir}")
        logger.info(f"Current run name to exclude: {current_run_name}")
        logger.info(f"Using regex pattern: {run_name_pattern.pattern}")

        try:
            if not runs_root_dir.exists():
                logger.info("Runs root directory does not exist.")
                return None

            found_dirs = list(runs_root_dir.iterdir())
            logger.debug(
                f"Found {len(found_dirs)} items in runs directory: {[d.name for d in found_dirs]}"
            )

            for d in found_dirs:
                logger.debug(f"Checking item: {d.name}")
                if d.is_dir() and d.name != current_run_name:
                    logger.debug(
                        f"  '{d.name}' is a directory and not the current run."
                    )
                    match = run_name_pattern.search(d.name)
                    if match:
                        timestamp_str = match.group(1)
                        logger.debug(
                            f"  Regex matched! Found timestamp '{timestamp_str}' in '{d.name}'"
                        )
                        try:
                            run_time = datetime.datetime.strptime(
                                timestamp_str, "%Y%m%d_%H%M%S"
                            )
                            potential_runs.append((run_time, d.name))
                            logger.debug(
                                f"  Successfully parsed timestamp and added '{d.name}' to potential runs."
                            )
                        except ValueError:
                            logger.warning(
                                f"Could not parse timestamp '{timestamp_str}' from directory name: {d.name}"
                            )
                    else:
                        logger.debug(
                            f"  Directory name {d.name} did not match timestamp pattern."
                        )
                elif not d.is_dir():
                    logger.debug(f"  '{d.name}' is not a directory.")
                else:
                    logger.debug(f"  '{d.name}' is the current run directory.")

            if not potential_runs:
                logger.info(
                    "No previous run directories found matching the pattern after filtering."
                )
                return None

            potential_runs.sort(key=lambda item: item[0], reverse=True)
            logger.debug(
                f"Sorted potential runs (most recent first): {[(dt.strftime('%Y%m%d_%H%M%S'), name) for dt, name in potential_runs]}"
            )

            latest_run_name = potential_runs[0][1]
            logger.info(f"Selected latest previous run: {latest_run_name}")
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
            load_path = Path(load_path_config).resolve()  # Resolve immediately
            if load_path.exists():
                checkpoint_to_load = load_path
                logger.info(f"Using specified checkpoint path: {checkpoint_to_load}")
            else:
                logger.warning(
                    f"Specified checkpoint path not found: {load_path_config}"
                )

        if not checkpoint_to_load and auto_resume:
            logger.info(
                f"Attempting to find latest run directory to auto-resume from (current: {current_run_name})."
            )
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
                        f"Latest checkpoint file ('{self.persist_config.LATEST_CHECKPOINT_FILENAME}') not found in latest run directory '{latest_run_name}'."
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
        buffer_to_load: Path | None = None

        if load_path_config:
            load_path = Path(load_path_config).resolve()  # Resolve immediately
            if load_path.exists():
                logger.info(f"Using specified buffer path: {load_path_config}")
                buffer_to_load = load_path
            else:
                logger.warning(f"Specified buffer path not found: {load_path_config}")

        if not buffer_to_load and checkpoint_run_name:
            potential_buffer_path = self.get_buffer_path(run_name=checkpoint_run_name)
            if potential_buffer_path.exists():
                logger.info(
                    f"Loading buffer from checkpoint run '{checkpoint_run_name}' (using default link): {potential_buffer_path}"
                )
                buffer_to_load = potential_buffer_path.resolve()
            else:
                logger.info(
                    f"Default buffer file ('{self.persist_config.BUFFER_FILENAME}') not found in checkpoint run directory '{checkpoint_run_name}'."
                )

        if not buffer_to_load and auto_resume and not checkpoint_run_name:
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
                    buffer_to_load = potential_buffer_path.resolve()
                else:
                    logger.info(
                        f"Default buffer file not found in latest run directory '{latest_previous_run_name}'."
                    )

        if not buffer_to_load:
            logger.info("No suitable buffer file found to load.")

        return buffer_to_load

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
        """Updates the default buffer link ('buffer.pkl')."""
        if not step_buffer_path.exists():
            logger.error(f"Source buffer path does not exist: {step_buffer_path}")
            return

        default_buffer_path = self.get_buffer_path()
        try:
            shutil.copy2(step_buffer_path, default_buffer_path)
            logger.debug(f"Updated default buffer file link: {default_buffer_path}")
        except Exception as e_default:
            logger.error(
                f"Error updating default buffer file {default_buffer_path}: {e_default}"
            )

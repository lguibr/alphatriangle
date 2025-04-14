import json
import logging
import shutil
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import mlflow
import numpy as np
import ray
import torch
from pydantic import ValidationError

from ..utils.sumtree import SumTree
from .schemas import BufferData, CheckpointData, LoadedTrainingState

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..config import PersistenceConfig, TrainConfig
    from ..nn import NeuralNetwork
    from ..rl.core.buffer import (
        ExperienceBuffer,
    )
    from ..stats import StatsCollectorActor

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages loading and saving of training artifacts using Pydantic schemas
    and cloudpickle for serialization. Handles MLflow artifact logging.
    """

    def __init__(
        self, persist_config: "PersistenceConfig", train_config: "TrainConfig"
    ):
        self.persist_config = persist_config
        self.train_config = train_config
        if self.train_config.RUN_NAME != "default_run":
            self.persist_config.RUN_NAME = self.train_config.RUN_NAME
        elif self.persist_config.RUN_NAME == "default_run":
            logger.warning(
                "DataManager RUN_NAME not set in TrainConfig. Using default."
            )

        self.root_data_dir = Path(self.persist_config.ROOT_DATA_DIR)
        self.root_data_dir.mkdir(parents=True, exist_ok=True)
        self._update_paths()
        self._create_directories()
        logger.info(
            f"DataManager initialized. Current Run Name: {self.persist_config.RUN_NAME}. Run directory: {self.run_base_dir}"
        )

    def _update_paths(self):
        """Updates paths based on the current RUN_NAME."""
        self.run_base_dir = Path(self.persist_config.get_run_base_dir())
        self.checkpoint_dir = (
            self.run_base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        self.buffer_dir = self.run_base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        self.log_dir = self.run_base_dir / self.persist_config.LOG_DIR_NAME
        self.config_path = self.run_base_dir / self.persist_config.CONFIG_FILENAME

    def _create_directories(self):
        """Creates necessary temporary directories for the current run."""
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
            filename = self.persist_config.BUFFER_FILENAME
        else:
            filename = self.persist_config.BUFFER_FILENAME
        return buffer_dir / Path(filename).with_suffix(".pkl")

    def find_latest_run_dir(self, current_run_name: str) -> str | None:
        """Finds the most recent *previous* run directory based on name sorting."""
        runs_root_dir = self.root_data_dir / self.persist_config.RUNS_DIR_NAME
        try:
            if not runs_root_dir.exists():
                return None
            potential_dirs = [
                d.name
                for d in runs_root_dir.iterdir()
                if d.is_dir() and d.name != current_run_name
            ]
            if not potential_dirs:
                return None

            potential_dirs.sort(reverse=True)
            latest_run_name = potential_dirs[0]
            logger.debug(
                f"Found potential previous run directories: {potential_dirs}. Latest: {latest_run_name}"
            )
            return latest_run_name
        except Exception as e:
            logger.error(f"Error finding latest run directory: {e}", exc_info=True)
            return None

    def _determine_checkpoint_to_load(self) -> Path | None:
        """Determines the absolute path of the checkpoint file to load."""
        load_path_config = self.train_config.LOAD_CHECKPOINT_PATH
        auto_resume = self.train_config.AUTO_RESUME_LATEST
        current_run_name = self.persist_config.RUN_NAME
        checkpoint_to_load: Path | None = None

        # 1. Priority: Explicit path from config
        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                checkpoint_to_load = load_path.resolve()
                logger.info(f"Using specified checkpoint path: {checkpoint_to_load}")
            else:
                logger.warning(
                    f"Specified checkpoint path not found: {load_path_config}"
                )

        # 2. Fallback: Auto-resume from latest *previous* run
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

    def _determine_buffer_to_load(self, checkpoint_run_name: str | None) -> Path | None:
        """
        Determines the buffer file path to load.
        Prioritizes explicit path, then the run corresponding to the loaded checkpoint,
        then the latest previous run if auto-resuming and no checkpoint was loaded.
        """
        # 1. Priority: Explicit path from config
        if self.train_config.LOAD_BUFFER_PATH:
            load_path = Path(self.train_config.LOAD_BUFFER_PATH)
            if load_path.exists():
                logger.info(
                    f"Using specified buffer path: {self.train_config.LOAD_BUFFER_PATH}"
                )
                return load_path.resolve()
            else:
                logger.warning(
                    f"Specified buffer path not found: {self.train_config.LOAD_BUFFER_PATH}"
                )

        # 2. Fallback: Use run from loaded checkpoint (if any)
        if checkpoint_run_name:
            potential_buffer_path = self.get_buffer_path(
                run_name=checkpoint_run_name
            )  # Use default buffer name
            if potential_buffer_path.exists():
                logger.info(
                    f"Loading buffer from checkpoint run '{checkpoint_run_name}': {potential_buffer_path}"
                )
                return potential_buffer_path.resolve()
            else:
                logger.info(
                    f"Default buffer file not found in checkpoint run directory '{checkpoint_run_name}'."
                )

        # 3. Fallback: Auto-resume from latest *previous* run (only if no checkpoint was loaded)
        if self.train_config.AUTO_RESUME_LATEST and not checkpoint_run_name:
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

    def load_initial_state(self) -> LoadedTrainingState:
        """
        Loads the initial training state using Pydantic models for validation.
        Returns a LoadedTrainingState object containing the deserialized data.
        Handles AUTO_RESUME_LATEST logic for checkpoint and buffer.
        """
        loaded_state = LoadedTrainingState()
        checkpoint_path = self._determine_checkpoint_to_load()
        checkpoint_run_name: str | None = None

        # --- Load Checkpoint (Model + Optimizer + Stats) ---
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                with checkpoint_path.open("rb") as f:
                    loaded_checkpoint_model = cloudpickle.load(f)
                if isinstance(loaded_checkpoint_model, CheckpointData):
                    loaded_state.checkpoint_data = loaded_checkpoint_model
                    checkpoint_run_name = (
                        loaded_state.checkpoint_data.run_name
                    )  # Store run name
                    logger.info(
                        f"Checkpoint loaded and validated (Run: {loaded_state.checkpoint_data.run_name}, Step: {loaded_state.checkpoint_data.global_step})"
                    )
                else:
                    logger.error(
                        f"Loaded checkpoint file {checkpoint_path} did not contain a CheckpointData object (type: {type(loaded_checkpoint_model)})."
                    )
            except ValidationError as e:
                logger.error(
                    f"Pydantic validation failed for checkpoint {checkpoint_path}: {e}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(
                    f"Error loading/validating checkpoint from {checkpoint_path}: {e}",
                    exc_info=True,
                )

        # --- Load Buffer ---
        if self.persist_config.SAVE_BUFFER:
            # Pass the run name from the loaded checkpoint (if any)
            buffer_path = self._determine_buffer_to_load(checkpoint_run_name)
            if buffer_path:
                logger.info(f"Loading buffer: {buffer_path}")
                try:
                    with buffer_path.open("rb") as f:
                        loaded_buffer_model = cloudpickle.load(f)
                    if isinstance(loaded_buffer_model, BufferData):
                        # Basic validation of experience structure
                        valid_experiences = []
                        invalid_count = 0
                        for i, exp in enumerate(loaded_buffer_model.buffer_list):
                            if (
                                isinstance(exp, tuple)
                                and len(exp) == 3
                                and isinstance(exp[0], dict)
                                and "grid" in exp[0]
                                and "other_features" in exp[0]
                                and isinstance(exp[0]["grid"], np.ndarray)
                                and isinstance(exp[0]["other_features"], np.ndarray)
                                and isinstance(exp[1], dict)
                                # Use isinstance with | for multiple types
                                and isinstance(exp[2], float | int)
                            ):
                                valid_experiences.append(exp)
                            else:
                                invalid_count += 1
                                logger.warning(
                                    f"Skipping invalid experience structure at index {i} in loaded buffer: {type(exp)}"
                                )
                        if invalid_count > 0:
                            logger.warning(
                                f"Found {invalid_count} invalid experience structures in loaded buffer."
                            )

                        loaded_buffer_model.buffer_list = valid_experiences
                        loaded_state.buffer_data = loaded_buffer_model
                        logger.info(
                            f"Buffer loaded and validated. Size: {len(loaded_state.buffer_data.buffer_list)}"
                        )
                    else:
                        logger.error(
                            f"Loaded buffer file {buffer_path} did not contain a BufferData object (type: {type(loaded_buffer_model)})."
                        )
                except ValidationError as e:
                    logger.error(
                        f"Pydantic validation failed for buffer {buffer_path}: {e}",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load/validate experience buffer from {buffer_path}: {e}",
                        exc_info=True,
                    )

        if not loaded_state.checkpoint_data and not loaded_state.buffer_data:
            logger.info("No checkpoint or buffer loaded. Starting fresh.")

        return loaded_state

    def save_training_state(
        self,
        nn: "NeuralNetwork",
        optimizer: "Optimizer",
        stats_collector_actor: "StatsCollectorActor",
        buffer: "ExperienceBuffer",
        global_step: int,
        episodes_played: int,
        total_simulations_run: int,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Saves the training state using Pydantic models and cloudpickle."""
        run_name = self.persist_config.RUN_NAME
        logger.info(
            f"Saving training state for run '{run_name}' at step {global_step}. Final={is_final}, Best={is_best}"
        )

        stats_collector_state = {}
        if stats_collector_actor:
            try:
                # Correctly call remote method
                stats_state_ref = stats_collector_actor.get_state.remote()  # type: ignore
                stats_collector_state = ray.get(stats_state_ref, timeout=5.0)
            except Exception as e:
                logger.error(
                    f"Error fetching state from StatsCollectorActor for saving: {e}",
                    exc_info=True,
                )

        optimizer_state_cpu = {}
        try:
            optimizer_state_dict = optimizer.state_dict()

            def move_to_cpu(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, dict):
                    return {k: move_to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [move_to_cpu(elem) for elem in item]
                else:
                    return item

            optimizer_state_cpu = move_to_cpu(optimizer_state_dict)
        except Exception as e:
            logger.error(f"Could not prepare optimizer state for saving: {e}")

        try:
            checkpoint_data = CheckpointData(
                run_name=run_name,
                global_step=global_step,
                episodes_played=episodes_played,
                total_simulations_run=total_simulations_run,
                model_config_dict=nn.model_config.model_dump(),
                env_config_dict=nn.env_config.model_dump(),
                model_state_dict=nn.get_weights(),
                optimizer_state_dict=optimizer_state_cpu,
                stats_collector_state=stats_collector_state,
            )
        except ValidationError as e:
            logger.error(f"Failed to create CheckpointData model: {e}", exc_info=True)
            return

        step_checkpoint_path = self.get_checkpoint_path(
            run_name=run_name, step=global_step, is_final=is_final
        )
        saved_checkpoint_path: Path | None = None
        try:
            step_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with step_checkpoint_path.open("wb") as f:
                cloudpickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint temporarily saved to {step_checkpoint_path}")
            saved_checkpoint_path = step_checkpoint_path
            latest_path = self.get_checkpoint_path(run_name=run_name, is_latest=True)
            best_path = self.get_checkpoint_path(run_name=run_name, is_best=True)
            try:
                shutil.copy2(step_checkpoint_path, latest_path)
            except Exception as e:
                logger.error(f"Failed to update latest checkpoint link: {e}")
            if is_best:
                try:
                    shutil.copy2(step_checkpoint_path, best_path)
                    logger.info(f"Updated best checkpoint link to step {global_step}")
                except Exception as e:
                    logger.error(f"Failed to update best checkpoint link: {e}")
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint file to {step_checkpoint_path}: {e}",
                exc_info=True,
            )

        saved_buffer_path: Path | None = None
        if self.persist_config.SAVE_BUFFER:
            buffer_path = self.get_buffer_path(
                run_name=run_name, step=global_step, is_final=is_final
            )
            default_buffer_path = self.get_buffer_path(run_name=run_name)
            try:
                # --- Access buffer data correctly based on PER ---
                if buffer.use_per:
                    if hasattr(buffer, "tree") and isinstance(buffer.tree, SumTree):
                        buffer_list = [
                            buffer.tree.data[i]
                            for i in range(buffer.tree.n_entries)
                            if buffer.tree.data[i] != 0
                        ]
                    else:
                        logger.error(
                            "PER buffer tree is missing or invalid during save."
                        )
                        buffer_list = []
                else:
                    buffer_list = list(buffer.buffer)

                # Basic validation before saving
                valid_experiences = []
                invalid_count = 0
                for i, exp in enumerate(buffer_list):
                    if (
                        isinstance(exp, tuple)
                        and len(exp) == 3
                        and isinstance(exp[0], dict)
                        and "grid" in exp[0]
                        and "other_features" in exp[0]
                        and isinstance(exp[0]["grid"], np.ndarray)
                        and isinstance(exp[0]["other_features"], np.ndarray)
                        and isinstance(exp[1], dict)
                        # Use isinstance with | for multiple types
                        and isinstance(exp[2], float | int)
                    ):
                        valid_experiences.append(exp)
                    else:
                        invalid_count += 1
                        logger.warning(
                            f"Skipping invalid experience structure at index {i} during save: {type(exp)}"
                        )
                if invalid_count > 0:
                    logger.warning(
                        f"Found {invalid_count} invalid experience structures before saving buffer."
                    )

                buffer_data = BufferData(buffer_list=valid_experiences)
                buffer_path.parent.mkdir(parents=True, exist_ok=True)
                with buffer_path.open("wb") as f:
                    cloudpickle.dump(buffer_data, f)
                logger.info(f"Experience buffer temporarily saved to {buffer_path}")
                saved_buffer_path = buffer_path
                try:
                    # Always update the default buffer file
                    with default_buffer_path.open("wb") as f_default:
                        cloudpickle.dump(buffer_data, f_default)
                    logger.debug(f"Updated default buffer file: {default_buffer_path}")
                except Exception as e_default:
                    logger.error(
                        f"Error updating default buffer file {default_buffer_path}: {e_default}"
                    )
            except ValidationError as e:
                logger.error(f"Failed to create BufferData model: {e}", exc_info=True)
            except Exception as e:
                logger.error(
                    f"Error saving experience buffer to {buffer_path}: {e}",
                    exc_info=True,
                )

        self._log_artifacts(saved_checkpoint_path, saved_buffer_path, run_name, is_best)

    def _log_artifacts(
        self,
        checkpoint_path: Path | None,
        buffer_path: Path | None,
        run_name: str,
        is_best: bool,
    ):
        """Logs saved checkpoint and buffer files to MLflow."""
        try:
            if checkpoint_path and checkpoint_path.exists():
                ckpt_artifact_path = self.persist_config.CHECKPOINT_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(checkpoint_path), artifact_path=ckpt_artifact_path
                )
                latest_path = self.get_checkpoint_path(
                    run_name=run_name, is_latest=True
                )
                if latest_path.exists():
                    mlflow.log_artifact(
                        str(latest_path), artifact_path=ckpt_artifact_path
                    )
                if is_best:
                    best_path = self.get_checkpoint_path(
                        run_name=run_name, is_best=True
                    )
                    if best_path.exists():
                        mlflow.log_artifact(
                            str(best_path), artifact_path=ckpt_artifact_path
                        )
                logger.info(
                    f"Logged checkpoint artifacts to MLflow path: {ckpt_artifact_path}"
                )
            if buffer_path and buffer_path.exists():
                buffer_artifact_path = self.persist_config.BUFFER_SAVE_DIR_NAME
                # Log the step-specific buffer if it was created
                mlflow.log_artifact(
                    str(buffer_path), artifact_path=buffer_artifact_path
                )
                # Always log the default buffer file as well
                default_buffer_path = self.get_buffer_path(run_name=run_name)
                if default_buffer_path.exists():
                    mlflow.log_artifact(
                        str(default_buffer_path), artifact_path=buffer_artifact_path
                    )
                logger.info(
                    f"Logged buffer artifacts to MLflow path: {buffer_artifact_path}"
                )
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}", exc_info=True)

    def save_run_config(self, configs: dict[str, Any]):
        """Saves the combined configuration dictionary as a JSON artifact."""
        try:
            config_path = self.config_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w") as f:

                def default_serializer(obj):
                    if isinstance(obj, torch.Tensor | np.ndarray):  # Use | for types
                        return "<tensor/array>"
                    if isinstance(obj, deque):
                        return list(obj)
                    try:
                        # Attempt standard JSON serialization first
                        return obj.__dict__ if hasattr(obj, "__dict__") else str(obj)
                    except TypeError:
                        # Fallback for objects that are not directly serializable
                        return f"<object of type {type(obj).__name__}>"

                json.dump(configs, f, indent=4, default=default_serializer)
            mlflow.log_artifact(str(config_path), artifact_path="config")
        except Exception as e:
            logger.error(f"Failed to save/log run config JSON: {e}", exc_info=True)

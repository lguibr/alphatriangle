# File: alphatriangle/data/data_manager.py
# No changes needed, already expects ActorHandle | None
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import ray  # Import ray
from pydantic import ValidationError

from .path_manager import PathManager
from .schemas import CheckpointData, LoadedTrainingState
from .serializer import Serializer

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..config import PersistenceConfig, TrainConfig
    from ..nn import NeuralNetwork
    from ..rl.core.buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates loading and saving of training artifacts using PathManager and Serializer.
    Handles MLflow artifact logging.
    """

    def __init__(
        self, persist_config: "PersistenceConfig", train_config: "TrainConfig"
    ):
        self.persist_config = persist_config
        self.train_config = train_config
        # Ensure PersistenceConfig reflects the current run name from TrainConfig
        self.persist_config.RUN_NAME = self.train_config.RUN_NAME

        self.path_manager = PathManager(self.persist_config)
        self.serializer = Serializer()

        self.path_manager.create_run_directories()
        logger.info(
            f"DataManager initialized. Current Run Name: {self.persist_config.RUN_NAME}. Run directory: {self.path_manager.run_base_dir}"
        )

    def load_initial_state(self) -> LoadedTrainingState:
        """
        Loads the initial training state using PathManager and Serializer.
        Returns a LoadedTrainingState object containing the deserialized data.
        Handles AUTO_RESUME_LATEST logic.
        """
        loaded_state = LoadedTrainingState()
        checkpoint_path = self.path_manager.determine_checkpoint_to_load(
            self.train_config.LOAD_CHECKPOINT_PATH,
            self.train_config.AUTO_RESUME_LATEST,
        )
        checkpoint_run_name: str | None = None

        # --- Load Checkpoint (Model + Optimizer + Stats) ---
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                loaded_checkpoint_model = self.serializer.load_checkpoint(
                    checkpoint_path
                )
                if loaded_checkpoint_model:
                    loaded_state.checkpoint_data = loaded_checkpoint_model
                    checkpoint_run_name = (
                        loaded_state.checkpoint_data.run_name
                    )  # Store run name
                    logger.info(
                        f"Checkpoint loaded and validated (Run: {loaded_state.checkpoint_data.run_name}, Step: {loaded_state.checkpoint_data.global_step})"
                    )
                else:
                    logger.error(
                        f"Loading checkpoint from {checkpoint_path} failed or returned None."
                    )
            except (ValidationError, Exception) as e:
                logger.error(
                    f"Error loading/validating checkpoint from {checkpoint_path}: {e}",
                    exc_info=True,
                )

        # --- Load Buffer ---
        if self.persist_config.SAVE_BUFFER:
            buffer_path = self.path_manager.determine_buffer_to_load(
                self.train_config.LOAD_BUFFER_PATH,
                self.train_config.AUTO_RESUME_LATEST,
                checkpoint_run_name,  # Pass run name from loaded checkpoint
            )
            if buffer_path:
                logger.info(f"Loading buffer: {buffer_path}")
                try:
                    loaded_buffer_model = self.serializer.load_buffer(buffer_path)
                    if loaded_buffer_model:
                        loaded_state.buffer_data = loaded_buffer_model
                        logger.info(
                            f"Buffer loaded and validated. Size: {len(loaded_state.buffer_data.buffer_list)}"
                        )
                    else:
                        logger.error(
                            f"Loading buffer from {buffer_path} failed or returned None."
                        )
                except (ValidationError, Exception) as e:
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
        stats_collector_actor: ray.actor.ActorHandle | None,
        buffer: "ExperienceBuffer",
        global_step: int,
        episodes_played: int,
        total_simulations_run: int,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Saves the training state using Serializer and PathManager."""
        run_name = self.persist_config.RUN_NAME
        logger.info(
            f"Saving training state for run '{run_name}' at step {global_step}. Final={is_final}, Best={is_best}"
        )

        stats_collector_state = {}
        if stats_collector_actor is not None:
            try:
                # Call remote method on the handle
                stats_state_ref = stats_collector_actor.get_state.remote()
                stats_collector_state = ray.get(stats_state_ref, timeout=5.0)
            except Exception as e:
                logger.error(
                    f"Error fetching state from StatsCollectorActor for saving: {e}",
                    exc_info=True,
                )

        # --- Save Checkpoint ---
        saved_checkpoint_path: Path | None = None
        try:
            checkpoint_data = CheckpointData(
                run_name=run_name,
                global_step=global_step,
                episodes_played=episodes_played,
                total_simulations_run=total_simulations_run,
                model_config_dict=nn.model_config.model_dump(),
                env_config_dict=nn.env_config.model_dump(),
                model_state_dict=nn.get_weights(),
                optimizer_state_dict=self.serializer.prepare_optimizer_state(optimizer),
                stats_collector_state=stats_collector_state,
            )
            step_checkpoint_path = self.path_manager.get_checkpoint_path(
                step=global_step, is_final=is_final
            )
            self.serializer.save_checkpoint(checkpoint_data, step_checkpoint_path)
            saved_checkpoint_path = step_checkpoint_path  # Store path if save succeeded

            # Update latest/best links
            self.path_manager.update_checkpoint_links(
                step_checkpoint_path, is_best=is_best
            )

        except ValidationError as e:
            logger.error(f"Failed to create CheckpointData model: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

        # --- Save Buffer ---
        saved_buffer_path: Path | None = None
        if self.persist_config.SAVE_BUFFER:
            try:
                buffer_data = self.serializer.prepare_buffer_data(buffer)
                if buffer_data:
                    buffer_path = self.path_manager.get_buffer_path(
                        step=global_step, is_final=is_final
                    )
                    self.serializer.save_buffer(buffer_data, buffer_path)
                    saved_buffer_path = buffer_path  # Store path if save succeeded
                    # Update default buffer link
                    self.path_manager.update_buffer_link(buffer_path)
                else:
                    logger.warning("Buffer data preparation failed, buffer not saved.")
            except ValidationError as e:
                logger.error(f"Failed to create BufferData model: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Failed to save buffer: {e}", exc_info=True)

        # --- Log Artifacts to MLflow ---
        self._log_artifacts(saved_checkpoint_path, saved_buffer_path, is_best)

    def _log_artifacts(
        self,
        checkpoint_path: Path | None,
        buffer_path: Path | None,
        is_best: bool,
    ):
        """Logs saved checkpoint and buffer files to MLflow."""
        try:
            if checkpoint_path and checkpoint_path.exists():
                ckpt_artifact_path = self.persist_config.CHECKPOINT_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(checkpoint_path), artifact_path=ckpt_artifact_path
                )
                latest_path = self.path_manager.get_checkpoint_path(is_latest=True)
                if latest_path.exists():
                    mlflow.log_artifact(
                        str(latest_path), artifact_path=ckpt_artifact_path
                    )
                if is_best:
                    best_path = self.path_manager.get_checkpoint_path(is_best=True)
                    if best_path.exists():
                        mlflow.log_artifact(
                            str(best_path), artifact_path=ckpt_artifact_path
                        )
                logger.info(
                    f"Logged checkpoint artifacts to MLflow path: {ckpt_artifact_path}"
                )
            if buffer_path and buffer_path.exists():
                buffer_artifact_path = self.persist_config.BUFFER_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(buffer_path), artifact_path=buffer_artifact_path
                )
                default_buffer_path = self.path_manager.get_buffer_path()
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
            config_path = self.path_manager.get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            self.serializer.save_config_json(configs, config_path)
            mlflow.log_artifact(str(config_path), artifact_path="config")
        except Exception as e:
            logger.error(f"Failed to save/log run config JSON: {e}", exc_info=True)

    # --- Expose PathManager methods if needed ---
    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        return self.path_manager.get_checkpoint_path(
            run_name, step, is_latest, is_best, is_final
        )

    def get_buffer_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_final: bool = False,
    ) -> Path:
        return self.path_manager.get_buffer_path(run_name, step, is_final)

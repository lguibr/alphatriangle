# File: alphatriangle/training/runner.py
import logging
import sys
import time
import traceback
from typing import TYPE_CHECKING

import mlflow
import ray
import torch

from ..config import APP_NAME, PersistenceConfig, TrainConfig
from ..logging_config import setup_logging  # Import centralized setup
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import log_configs_to_mlflow  # Keep MLflow helper
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _initialize_mlflow(
    persist_config: PersistenceConfig, run_name: str, log_file_path: "Path | None"
) -> bool:
    """Sets up MLflow tracking and starts a run. Optionally logs the log file."""
    try:
        # Use the computed property which resolves the path and creates the dir
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        logger.info(f"Resolved MLflow absolute path: {mlflow_abs_path}")
        logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")

        # Ensure the directory exists (PersistenceConfig property handles this now)
        # Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True) # No longer needed here

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow experiment to: {APP_NAME}")

        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
            # Log the log file artifact *if* it exists
            if log_file_path and log_file_path.exists():
                try:
                    # Log the file into an 'logs' artifact directory
                    mlflow.log_artifact(str(log_file_path), artifact_path="logs")
                    logger.info(f"Logged log file artifact: {log_file_path.name}")
                except Exception as log_artifact_err:
                    logger.error(
                        f"Failed to log log file to MLflow: {log_artifact_err}"
                    )
            elif log_file_path:
                logger.warning(
                    f"Log file path provided but does not exist, skipping MLflow artifact logging: {log_file_path}"
                )
            else:
                logger.info("No log file path provided, skipping MLflow artifact log.")

            return True
        else:
            logger.error("MLflow run failed to start.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        return False


def _load_and_apply_initial_state(components: TrainingComponents) -> TrainingLoop:
    """Loads initial state using DataManager and applies it to components, returning an initialized TrainingLoop."""
    loaded_state = components.data_manager.load_initial_state()
    training_loop = TrainingLoop(components)
    initial_step = 0

    if loaded_state.checkpoint_data:
        cp_data = loaded_state.checkpoint_data
        logger.info(
            f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
        )

        if cp_data.model_state_dict:
            components.nn.set_weights(cp_data.model_state_dict)
        if cp_data.optimizer_state_dict:
            try:
                components.trainer.optimizer.load_state_dict(
                    cp_data.optimizer_state_dict
                )
                for state in components.trainer.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(components.nn.device)
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            try:
                set_state_ref = components.stats_collector_actor.set_state.remote(
                    cp_data.stats_collector_state
                )
                ray.get(set_state_ref, timeout=5.0)
                logger.info("StatsCollectorActor state restored.")
            except Exception as e:
                logger.error(
                    f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                )

        training_loop.set_initial_state(
            cp_data.global_step,
            cp_data.episodes_played,
            cp_data.total_simulations_run,
        )
        initial_step = cp_data.global_step
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    loaded_buffer_size = 0
    if loaded_state.buffer_data:
        if components.train_config.USE_PER:
            logger.info("Rebuilding PER SumTree from loaded buffer data...")
            if not hasattr(components.buffer, "tree") or components.buffer.tree is None:
                components.buffer.tree = SumTree(components.buffer.capacity)
            else:
                # Re-initialize SumTree to ensure correct state
                components.buffer.tree = SumTree(components.buffer.capacity)

            max_p = 1.0
            for exp in loaded_state.buffer_data.buffer_list:
                components.buffer.tree.add(max_p, exp)
            loaded_buffer_size = len(components.buffer)
            logger.info(f"PER buffer loaded. Size: {loaded_buffer_size}")
        else:
            from collections import deque

            components.buffer.buffer = deque(
                loaded_state.buffer_data.buffer_list,
                maxlen=components.buffer.capacity,
            )
            loaded_buffer_size = len(components.buffer)
            logger.info(f"Uniform buffer loaded. Size: {loaded_buffer_size}")
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    logger.info(
        f"Initial state loaded and applied. Starting step: {initial_step}, Buffer size: {loaded_buffer_size}"
    )
    return training_loop


def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    if not training_loop:
        logger.warning("Cannot save final state: TrainingLoop not available.")
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_best=False,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


def run_training(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
    profile: bool,  # Added profile flag
) -> int:
    """Runs the training pipeline (headless)."""
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    log_file_path: Path | None = None
    file_handler: logging.FileHandler | None = None
    ray_initialized_by_setup = False
    mlflow_run_active = False
    tb_log_dir: Path | None = None  # Use Path object

    try:
        # --- Setup File Logging Path ---
        # Determine log file path before setting up logging
        log_dir = (
            persist_config_override.get_run_base_dir()
            / persist_config_override.LOG_DIR_NAME
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"{train_config_override.RUN_NAME}_train.log"

        # --- Setup Centralized Logging ---
        # Pass the determined log file path
        file_handler = setup_logging(log_level_str, log_file_path)
        logger.info(
            f"Logging {log_level_str.upper()} and higher messages to console and: {log_file_path}"
        )

        # --- TensorBoard Directory Setup ---
        tb_log_dir = persist_config_override.get_tensorboard_log_dir()
        if tb_log_dir:
            try:
                tb_log_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured TensorBoard log directory exists: {tb_log_dir}")
            except Exception as e:
                logger.error(
                    f"Failed to create TensorBoard directory '{tb_log_dir}': {e}"
                )
                tb_log_dir = None
        else:
            logger.warning("Could not determine TensorBoard log directory path.")

        # --- Setup Components (includes Ray init, StatsActor init) ---
        # Pass tb_log_dir as string and profile flag
        components, ray_initialized_by_setup = setup_training_components(
            train_config_override,
            persist_config_override,
            str(tb_log_dir) if tb_log_dir else None,
            profile,  # Pass profile flag
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize MLflow ---
        # Pass log_file_path to _initialize_mlflow
        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME, log_file_path
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)

            # Logging of log file artifact moved inside _initialize_mlflow

            if tb_log_dir and mlflow_run_active:
                try:
                    # Log the relative path to the TB dir as a parameter
                    relative_tb_path = tb_log_dir.relative_to(
                        components.persist_config.get_runs_root_dir()
                    )
                    mlflow.log_param("tensorboard_log_dir", str(relative_tb_path))
                except Exception as tb_param_err:
                    logger.error(
                        f"Failed to log TensorBoard path to MLflow: {tb_param_err}"
                    )
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        # --- Load State & Initialize Loop ---
        training_loop = _load_and_apply_initial_state(components)

        # --- Run Training Loop ---
        training_loop.initialize_workers()
        training_loop.run()

        # --- Determine Exit Code ---
        if training_loop.training_complete:
            exit_code = 0
            logger.info(
                f"Training run '[bold cyan]{components.train_config.RUN_NAME}[/]' completed successfully.",
                extra={"markup": True},
            )
        elif training_loop.training_exception:
            exit_code = 1
            logger.error(
                f"Training run '[bold cyan]{components.train_config.RUN_NAME}[/]' failed due to exception: {training_loop.training_exception}",
                extra={"markup": True},
            )
        else:
            # If stopped manually or for other reasons without exception/completion
            exit_code = 0  # Consider manual stop successful
            logger.warning(
                f"Training run '[bold cyan]{components.train_config.RUN_NAME}[/]' stopped before completion.",
                extra={"markup": True},
            )

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during training setup or execution: {e}"
        )
        traceback.print_exc()
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        # --- Cleanup ---
        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop and components:  # Check components exist too
            _save_final_state(training_loop)
            training_loop.cleanup_actors()
            if training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(training_loop.training_exception)
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        elif not components:
            final_status = "SETUP_FAILED"
            error_msg = "Component setup failed"
        else:  # components exist but training_loop doesn't (error during loop init)
            final_status = "SETUP_FAILED"
            error_msg = "Training loop initialization failed"

        # Log final TensorBoard artifacts if the directory exists
        if (
            mlflow_run_active
            and tb_log_dir
            and tb_log_dir.exists()
            and components is not None
        ):
            try:
                time.sleep(1)  # Allow final writes
                # Log the entire TB directory relative to the run base dir
                mlflow.log_artifacts(
                    str(tb_log_dir),
                    artifact_path=components.persist_config.TENSORBOARD_DIR_NAME,
                )
                logger.info(
                    f"Logged TensorBoard directory to MLflow artifacts: {components.persist_config.TENSORBOARD_DIR_NAME}"
                )
            except Exception as tb_artifact_err:
                logger.error(
                    f"Failed to log TensorBoard directory to MLflow: {tb_artifact_err}"
                )

        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        # Close file logger handler if it was created
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                logging.getLogger().removeHandler(file_handler)
            except Exception as e_close:
                # Use print for final cleanup errors as logging might be compromised
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Training finished with exit code {exit_code}.")
    return exit_code

# File: alphatriangle/training/runner.py
import logging
import sys
import time
import traceback
from pathlib import Path

import mlflow
import ray
import torch

from ..config import APP_NAME, PersistenceConfig, TrainConfig
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import (
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    """Sets up MLflow tracking and starts a run."""
    try:
        # Get the absolute path and URI from PersistenceConfig
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        logger.info(
            f"Resolved MLflow absolute path: {mlflow_abs_path}"
        )  # Log resolved path
        logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")  # Log URI

        # Ensure the directory exists
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        # logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}") # Redundant log
        logger.info(f"Set MLflow experiment to: {APP_NAME}")

        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
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
    training_loop = TrainingLoop(components)  # Instantiate loop first
    initial_step = 0  # Default start step

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
                # Ensure optimizer state is on the correct device
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
                # Use type ignore for remote call if needed
                set_state_ref = components.stats_collector_actor.set_state.remote(  # type: ignore [attr-defined]
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
        initial_step = cp_data.global_step  # Store loaded step
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
                # Re-initialize SumTree to ensure clean state
                components.buffer.tree = SumTree(components.buffer.capacity)

            max_p = 1.0  # Default priority for loaded experiences
            for exp in loaded_state.buffer_data.buffer_list:
                # Add with default max priority, actual priorities are lost on save/load
                components.buffer.tree.add(max_p, exp)
            loaded_buffer_size = len(components.buffer)
            logger.info(f"PER buffer loaded. Size: {loaded_buffer_size}")
        else:
            from collections import deque  # Import locally if needed

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
            stats_collector_actor=components.stats_collector_actor,  # Pass handle
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
) -> int:
    """Runs the training pipeline (headless)."""
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    log_file_path = None
    file_handler = None
    ray_initialized_by_setup = False
    mlflow_run_active = False
    tb_log_dir: str | None = None

    try:
        # --- Setup File Logging ---
        log_file_path = setup_file_logging(
            persist_config_override,
            train_config_override.RUN_NAME,
            "train",
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
        )

        # --- TensorBoard Directory Setup (Moved Before Component Setup) ---
        # Ensure the directory exists *before* the StatsCollectorActor tries to use it
        tb_log_dir = persist_config_override.get_tensorboard_log_dir()
        if tb_log_dir:
            try:
                Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured TensorBoard log directory exists: {tb_log_dir}")
            except Exception as e:
                logger.error(
                    f"Failed to create TensorBoard directory '{tb_log_dir}': {e}"
                )
                tb_log_dir = None  # Prevent passing invalid path to actor
        else:
            logger.warning("Could not determine TensorBoard log directory path.")

        # --- Setup Components (includes Ray init, StatsActor init) ---
        # Pass the potentially created tb_log_dir path to setup
        components, ray_initialized_by_setup = setup_training_components(
            train_config_override,
            persist_config_override,
            tb_log_dir,  # Pass tb_log_dir
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize MLflow ---
        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
            if log_file_path:
                try:
                    mlflow.log_artifact(log_file_path, artifact_path="logs")
                except Exception as log_artifact_err:
                    logger.error(
                        f"Failed to log log file to MLflow: {log_artifact_err}"
                    )
            # Log TB path if available
            if tb_log_dir and mlflow_run_active:
                try:
                    # Log relative path for portability
                    relative_tb_path = Path(tb_log_dir).relative_to(
                        Path(components.persist_config.get_run_base_dir()).parent
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
        elif training_loop.training_exception:
            exit_code = 1
        else:
            exit_code = 1  # Loop stopped unexpectedly

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
        if training_loop:
            _save_final_state(training_loop)
            # Cleanup actors (workers + stats collector, including its TB writer)
            training_loop.cleanup_actors()
            if training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(training_loop.training_exception)
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"

        # Log final TB artifacts if possible (writer is closed in cleanup_actors)
        if (
            mlflow_run_active
            and tb_log_dir  # Use the path determined earlier
            and Path(tb_log_dir).exists()
            and components is not None  # Check components exist
        ):
            try:
                # Ensure files are flushed before logging
                time.sleep(1)  # Short delay to allow potential final writes
                mlflow.log_artifacts(
                    tb_log_dir,
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

        # Close file logger
        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Training finished with exit code {exit_code}.")
    return exit_code

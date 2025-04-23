# File: alphatriangle/training/runner.py
import logging
import sys
import time
import traceback

import ray
import torch
from trieye import (  # Import from Trieye
    LoadedTrainingState,
    TrieyeConfig,
)

from ..config import TrainConfig
from ..logging_config import setup_logging  # Import centralized setup
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import log_configs_to_mlflow  # Keep MLflow helper for AT configs
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)


# Removed _initialize_mlflow


def _load_and_apply_initial_state(
    components: TrainingComponents,
) -> tuple[TrainingLoop, LoadedTrainingState]:
    """
    Loads initial state using TrieyeActor and applies it to components.
    Returns an initialized TrainingLoop and the loaded state object.
    """
    logger.info("Requesting initial state load from TrieyeActor...")
    loaded_state: LoadedTrainingState = ray.get(
        components.trieye_actor.load_initial_state.remote()
    )
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
                # Move optimizer state to the correct device
                for state in components.trainer.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(components.nn.device)
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        # Actor state is restored internally by TrieyeActor's load_initial_state

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
        buffer_list = loaded_state.buffer_data.buffer_list
        if components.train_config.USE_PER:
            logger.info("Rebuilding PER SumTree from loaded buffer data...")
            if not hasattr(components.buffer, "tree") or components.buffer.tree is None:
                components.buffer.tree = SumTree(components.buffer.capacity)
            else:
                components.buffer.tree = SumTree(components.buffer.capacity)

            max_p = 1.0
            for exp in buffer_list:
                # Basic validation before adding to tree
                if isinstance(exp, tuple) and len(exp) == 3:
                    components.buffer.tree.add(max_p, exp)
                else:
                    logger.warning(
                        f"Skipping invalid item from loaded buffer: {type(exp)}"
                    )
            loaded_buffer_size = len(components.buffer)
            logger.info(f"PER buffer loaded. Size: {loaded_buffer_size}")
        else:
            from collections import deque

            # Basic validation before adding to deque
            valid_buffer_list = [
                exp for exp in buffer_list if isinstance(exp, tuple) and len(exp) == 3
            ]
            if len(valid_buffer_list) < len(buffer_list):
                logger.warning(
                    f"Filtered {len(buffer_list) - len(valid_buffer_list)} invalid items from loaded buffer."
                )
            components.buffer.buffer = deque(
                valid_buffer_list, maxlen=components.buffer.capacity
            )
            loaded_buffer_size = len(components.buffer)
            logger.info(f"Uniform buffer loaded. Size: {loaded_buffer_size}")
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    logger.info(
        f"Initial state loaded and applied. Starting step: {initial_step}, Buffer size: {loaded_buffer_size}"
    )
    return training_loop, loaded_state


def _save_final_state(training_loop: TrainingLoop):
    """Triggers the Trieye actor to save the final training state."""
    if not training_loop or not training_loop.trieye_actor:
        logger.warning("Cannot save final state: TrainingLoop or TrieyeActor missing.")
        return
    components = training_loop.components
    logger.info("Requesting final training state save via TrieyeActor...")
    try:
        # Prepare data using the local serializer from components
        nn_state = components.nn.get_weights()
        opt_state = components.serializer.prepare_optimizer_state(
            components.trainer.optimizer.state_dict()  # Pass state_dict
        )
        buffer_data = components.serializer.prepare_buffer_data(
            list(components.buffer.buffer)
            if not components.buffer.use_per
            else [
                components.buffer.tree.data[i]
                for i in range(components.buffer.tree.n_entries)
                if components.buffer.tree.data[i] != 0
            ]  # Pass buffer content list
        )

        # Fire-and-forget call to the actor
        components.trieye_actor.save_training_state.remote(
            nn_state_dict=nn_state,
            optimizer_state_dict=opt_state,
            buffer_content=buffer_data.buffer_list if buffer_data else [],
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_best=False,  # Final save is not 'best'
            save_buffer=components.trieye_config.persistence.SAVE_BUFFER,
            model_config_dict=components.model_config.model_dump(),
            env_config_dict=components.env_config.model_dump(),
        )
        # Allow some time for the async save call to potentially start
        time.sleep(0.5)
    except Exception as e_save:
        logger.error(f"Failed to trigger final state save: {e_save}", exc_info=True)


def run_training(
    log_level_str: str,
    train_config_override: TrainConfig,
    trieye_config_override: TrieyeConfig,  # Use TrieyeConfig
    profile: bool,
) -> int:
    """Runs the training pipeline (headless)."""
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    # log_file_path: Path | None = None # Path determined by Trieye
    file_handler: logging.FileHandler | None = None
    ray_initialized_by_setup = False
    trieye_actor_handle: ray.actor.ActorHandle | None = None

    try:
        # --- Setup Console Logging ---
        # File logging is handled by TrieyeActor
        file_handler = setup_logging(log_level_str, log_file=None)  # Pass log_file=None
        logger.info(f"Console logging level set to: {log_level_str.upper()}")

        # --- Setup Components (includes Ray init, TrieyeActor init) ---
        components, ray_initialized_by_setup = setup_training_components(
            train_config_override,
            trieye_config_override,  # Pass TrieyeConfig
            profile,
        )
        if not components or not components.trieye_actor:
            raise RuntimeError(
                "Failed to initialize training components or TrieyeActor."
            )

        trieye_actor_handle = components.trieye_actor  # Store handle for cleanup

        # --- Log Configs and Params to MLflow (if run started by Trieye) ---
        mlflow_run_id = ray.get(trieye_actor_handle.get_mlflow_run_id.remote())
        if mlflow_run_id:
            log_configs_to_mlflow(components)  # Log AlphaTriangle configs
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            # Parameter logging removed as TrieyeActor doesn't expose log_param
        else:
            logger.warning(
                "MLflow run ID not available from TrieyeActor, skipping MLflow param logging."
            )

        # --- Load State & Initialize Loop ---
        training_loop, _ = _load_and_apply_initial_state(components)

        # --- Run Training Loop ---
        training_loop.initialize_workers()
        training_loop.run()

        # --- Determine Exit Code ---
        if training_loop.training_complete:
            exit_code = 0
            logger.info(
                f"Training run '[bold cyan]{components.trieye_config.run_name}[/]' completed successfully.",
                extra={"markup": True},
            )
        elif training_loop.training_exception:
            exit_code = 1
            logger.error(
                f"Training run '[bold cyan]{components.trieye_config.run_name}[/]' failed due to exception: {training_loop.training_exception}",
                extra={"markup": True},
            )
        else:
            exit_code = 0
            logger.warning(
                f"Training run '[bold cyan]{components.trieye_config.run_name}[/]' stopped before completion.",
                extra={"markup": True},
            )

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during training setup or execution: {e}"
        )
        traceback.print_exc()
        # Attempt to log failure status via actor if possible
        if trieye_actor_handle:
            try:
                # Use log_event to log status if log_param is unavailable
                from trieye.schemas import RawMetricEvent

                ray.get(
                    trieye_actor_handle.log_event.remote(
                        RawMetricEvent(
                            name="training_status_code", value=-1, global_step=0
                        )
                    )
                )  # Example event
                ray.get(
                    trieye_actor_handle.log_event.remote(
                        RawMetricEvent(
                            name="error_message_flag",
                            value=1,
                            global_step=0,
                            context={"error": str(e)},
                        )
                    )
                )
            except Exception as log_err:
                logger.error(
                    f"Failed to log setup error status via TrieyeActor log_event: {log_err}"
                )
        exit_code = 1

    finally:
        # --- Cleanup ---
        if training_loop and components:
            _save_final_state(training_loop)  # Trigger final save via actor
            training_loop.cleanup_actors()  # Cleans up workers

        # Shutdown TrieyeActor gracefully
        if trieye_actor_handle:
            logger.info("Shutting down TrieyeActor...")
            try:
                # Force final processing and shutdown
                final_step = training_loop.global_step if training_loop else 0
                ray.get(
                    trieye_actor_handle.force_process_and_log.remote(final_step),
                    timeout=15,
                )
                ray.get(trieye_actor_handle.shutdown.remote(), timeout=15)
                logger.info("TrieyeActor shutdown complete.")
            except Exception as actor_shutdown_err:
                logger.error(
                    f"Error during TrieyeActor shutdown: {actor_shutdown_err}. Attempting kill."
                )
                try:
                    ray.kill(trieye_actor_handle)
                except Exception as kill_err:
                    logger.error(f"Failed to kill TrieyeActor: {kill_err}")

        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        # Close console logger handler if it was created
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                logging.getLogger().removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Training finished with exit code {exit_code}.")
    return exit_code

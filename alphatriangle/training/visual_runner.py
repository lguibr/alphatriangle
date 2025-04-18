# File: alphatriangle/training/visual_runner.py
import logging
import queue
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import mlflow
import pygame
import ray
import torch

from .. import config, environment, visualization
from ..config import APP_NAME, PersistenceConfig, TrainConfig
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import (
    Tee,
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop  # Import TrainingLoop here
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)

# Queue for loop to send combined state dict {worker_id: state, -1: global_stats}
visual_state_queue: queue.Queue[dict[int, Any] | None] = queue.Queue(maxsize=5)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    """Sets up MLflow tracking and starts a run."""
    try:
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
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
    # Pass visual queue to TrainingLoop constructor
    training_loop = TrainingLoop(components, visual_state_queue=visual_state_queue)

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
        # --- CHANGED: Removed isinstance check, rely on ActorHandle type hint ---
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            # --- END CHANGED ---
            try:
                # MyPy should now know this is an ActorHandle
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
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    if loaded_state.buffer_data:
        if components.train_config.USE_PER:
            logger.info("Rebuilding PER SumTree from loaded buffer data...")
            if not hasattr(components.buffer, "tree") or components.buffer.tree is None:
                components.buffer.tree = SumTree(components.buffer.capacity)
            else:
                components.buffer.tree = SumTree(components.buffer.capacity)
            max_p = 1.0
            for exp in loaded_state.buffer_data.buffer_list:
                components.buffer.tree.add(max_p, exp)
            logger.info(f"PER buffer loaded. Size: {len(components.buffer)}")
        else:
            components.buffer.buffer = deque(
                loaded_state.buffer_data.buffer_list,
                maxlen=components.buffer.capacity,
            )
            logger.info(f"Uniform buffer loaded. Size: {len(components.buffer)}")
        if training_loop.buffer_fill_progress:
            training_loop.buffer_fill_progress.set_current_steps(len(components.buffer))
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    return training_loop


def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    if not training_loop:
        logger.warning("Cannot save final state: TrainingLoop not available.")
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        # Pass the actor handle directly
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_final=True,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


def _training_loop_thread_func(training_loop: TrainingLoop):
    """Function to run the training loop in a separate thread."""
    logger = logging.getLogger(__name__)  # Get logger within thread
    try:
        logger.info("Training loop thread started.")
        training_loop.initialize_workers()
        training_loop.run()
        logger.info("Training loop thread finished.")
    except Exception as e:
        logger.critical(f"Error in training loop thread: {e}", exc_info=True)
        training_loop.training_exception = e
    finally:
        # Signal the main visualization loop to exit
        try:
            while not visual_state_queue.empty():
                try:
                    visual_state_queue.get_nowait()
                except queue.Empty:
                    break
            visual_state_queue.put(None, timeout=1.0)
        except queue.Full:
            logger.error("Visual queue still full during shutdown.")
        except Exception as e_q:
            logger.error(f"Error putting None signal into visual queue: {e_q}")


def run_training_visual_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline in visual mode."""
    main_thread_exception = None
    train_thread = None
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    file_handler = None
    tee_stdout = None
    tee_stderr = None
    ray_initialized_by_setup = False
    mlflow_run_active = False
    total_params: int | None = None
    trainable_params: int | None = None

    try:
        # --- Setup File Logging & Redirection ---
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "visual"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to: {log_file_path}"
        )
        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )

        if file_handler and hasattr(file_handler, "stream") and file_handler.stream:
            tee_stdout = Tee(
                original_stdout,
                file_handler.stream,
                main_stream_for_fileno=original_stdout,
            )
            tee_stderr = Tee(
                original_stderr,
                file_handler.stream,
                main_stream_for_fileno=original_stderr,
            )
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            print("--- Stdout/Stderr redirected to console and log file ---")
            logger.info("Stdout/Stderr redirected to console and log file.")
        else:
            logger.error(
                "Could not redirect stdout/stderr: File handler stream not available."
            )

        # --- Setup Components (includes Ray init) ---
        components, ray_initialized_by_setup = setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize MLflow ---
        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)  # Log configs after run starts
            # Log parameter counts after MLflow run starts
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        # --- Load State & Initialize Loop ---
        training_loop = _load_and_apply_initial_state(components)

        # --- Start Training Thread ---
        train_thread = threading.Thread(
            target=_training_loop_thread_func, args=(training_loop,), daemon=True
        )
        train_thread.start()
        logger.info("Training loop thread launched.")

        # --- Initialize Visualization ---
        vis_config = config.VisConfig()
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode(
            (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"{config.APP_NAME} - Training Visual Mode ({components.train_config.RUN_NAME})"
        )
        clock = pygame.time.Clock()
        fonts = visualization.load_fonts()
        # Pass the actor handle directly
        dashboard_renderer = visualization.DashboardRenderer(
            screen,
            vis_config,
            components.env_config,
            fonts,
            components.stats_collector_actor,
            components.model_config,
            total_params=total_params,  # Pass param counts
            trainable_params=trainable_params,
        )

        current_worker_states: dict[int, environment.GameState] = {}
        current_global_stats: dict[str, Any] = {}
        has_received_data = False

        # --- Visualization Loop (Main Thread) ---
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.VIDEORESIZE:
                    try:
                        w, h = max(640, event.w), max(480, event.h)
                        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        dashboard_renderer.screen = screen
                        dashboard_renderer.layout_rects = None
                    except pygame.error as e:
                        logger.error(f"Error resizing window: {e}")

            # Process Visual Queue
            try:
                visual_data = visual_state_queue.get(timeout=0.05)
                if visual_data is None:
                    if train_thread and not train_thread.is_alive():
                        running = False
                        logger.info("Received exit signal from training thread.")
                elif isinstance(visual_data, dict):
                    has_received_data = True
                    global_stats_update = visual_data.pop(-1, {})
                    if isinstance(global_stats_update, dict):
                        if not isinstance(current_global_stats, dict):
                            current_global_stats = {}
                        current_global_stats.update(global_stats_update)
                    else:
                        logger.warning(
                            f"Received non-dict global stats update: {type(global_stats_update)}"
                        )

                    current_worker_states = {
                        k: v
                        for k, v in visual_data.items()
                        if isinstance(k, int)
                        and k >= 0
                        and isinstance(v, environment.GameState)
                    }
                    remaining_items = {
                        k: v
                        for k, v in visual_data.items()
                        if k != -1 and k not in current_worker_states
                    }
                    if remaining_items:
                        logger.warning(
                            f"Unexpected items remaining in visual_data after processing: {remaining_items.keys()}"
                        )
                else:
                    logger.warning(
                        f"Received unexpected item from visual queue: {type(visual_data)}"
                    )
            except queue.Empty:
                pass
            except Exception as q_get_err:
                logger.error(f"Error getting from visual queue: {q_get_err}")
                time.sleep(0.1)

            # Rendering Logic
            screen.fill(visualization.colors.DARK_GRAY)
            if has_received_data:
                try:
                    dashboard_renderer.render(
                        current_worker_states, current_global_stats
                    )
                except Exception as render_err:
                    logger.error(f"Error during rendering: {render_err}", exc_info=True)
                    err_font = fonts.get("help")
                    if err_font:
                        err_surf = err_font.render(
                            f"Render Error: {render_err}",
                            True,
                            visualization.colors.RED,
                        )
                        screen.blit(err_surf, (10, screen.get_height() // 2))
            else:
                help_font = fonts.get("help")
                if help_font:
                    wait_surf = help_font.render(
                        "Waiting for first data from training...",
                        True,
                        visualization.colors.LIGHT_GRAY,
                    )
                    wait_rect = wait_surf.get_rect(
                        center=(screen.get_width() // 2, screen.get_height() // 2)
                    )
                    screen.blit(wait_surf, wait_rect)

            pygame.display.flip()

            # Check Training Thread Status
            if train_thread and not train_thread.is_alive() and running:
                logger.warning("Training loop thread terminated unexpectedly.")
                if training_loop and training_loop.training_exception:
                    logger.error(
                        f"Training thread terminated due to exception: {training_loop.training_exception}"
                    )
                    main_thread_exception = training_loop.training_exception
                running = False

            clock.tick(vis_config.FPS)

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred in visual training script (main thread): {e}"
        )
        traceback.print_exc()
        main_thread_exception = e
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "VIS_FAILED")
                mlflow.log_param("error_message", f"MainThread: {str(e)}")
            except Exception as mlf_err:
                logger.error(f"Failed to log main thread error to MLflow: {mlf_err}")

    finally:
        # Restore stdout/stderr
        if tee_stdout:
            sys.stdout = original_stdout
        if tee_stderr:
            sys.stderr = original_stderr
        print("--- Restored stdout/stderr ---")

        logger.info("Initiating shutdown sequence...")
        if training_loop and not training_loop.stop_requested.is_set():
            training_loop.request_stop()

        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training loop thread to join...")
            train_thread.join(timeout=15.0)
            if train_thread.is_alive():
                logger.error("Training loop thread did not exit gracefully.")

        # --- Cleanup ---
        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            # Save final state
            _save_final_state(training_loop)
            # Cleanup loop actors
            training_loop.cleanup_actors()
            # Determine final status
            if main_thread_exception or training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(
                    main_thread_exception or training_loop.training_exception
                )
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"

        # End MLflow run
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        pygame.quit()
        logger.info("Pygame quit.")

        # Shutdown Ray ONLY if initialized by the setup function in this process
        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by visual runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        # Close file handler
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger = get_root_logger()
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Visual training finished with exit code {exit_code}.")
    return exit_code

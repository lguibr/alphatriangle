# File: src/training/runners.py
# File: src/training/runners.py
import logging
import queue
import sys
import threading
import time
import traceback
from typing import Any

import mlflow
import pygame

# Use relative imports
from .. import config, environment, utils, visualization
from ..data import DataManager
from ..nn import NeuralNetwork
from ..rl import ExperienceBuffer, Trainer
from ..stats import StatsCollectorActor
from . import TrainingComponents, TrainingPipeline
from .logging_utils import (
    Tee,
    get_root_logger,
    setup_file_logging,
)

# Queue for pipeline to send combined state dict {worker_id: state, -1: global_stats}
# Define it here so it's accessible by the visual runner function
visual_state_queue: queue.Queue[dict[int, Any] | None] = queue.Queue(maxsize=5)


def _setup_training_components(
    train_config_override: config.TrainConfig,
    persist_config_override: config.PersistenceConfig,
) -> TrainingComponents | None:
    """Initializes and returns the TrainingComponents bundle."""
    logger = logging.getLogger(__name__)
    try:
        # --- Initialize Configurations ---
        # Use the potentially overridden configs passed in
        train_config = train_config_override
        persist_config = persist_config_override
        # Instantiate Pydantic models using defaults
        # --- CHANGE: Removed type ignore ---
        env_config = config.EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = config.MCTSConfig()
        # --- END CHANGE ---

        # --- Validate Configs ---
        config.print_config_info_and_validate(mcts_config)

        # --- Setup ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)

        # --- Initialize Core Components ---
        # Note: Ray initialization is handled within TrainingPipeline
        # Correctly call remote method
        stats_collector_actor = StatsCollectorActor.remote(max_history=100_000)  # type: ignore
        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)
        data_manager = DataManager(persist_config, train_config)

        # --- Create Components Bundle ---
        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,
            model_config=model_config,
            mcts_config=mcts_config,
            persist_config=persist_config,
        )
        return components
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        return None


def run_training_headless_mode(
    log_level_str: str,
    train_config_override: config.TrainConfig,
    persist_config_override: config.PersistenceConfig,
) -> int:
    """Runs the training pipeline in headless mode."""
    logger = logging.getLogger(__name__)
    pipeline = None
    exit_code = 1
    log_file_path = None
    file_handler = None

    try:
        # --- Setup File Logging ---
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "headless"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
        )

        # --- Setup Components ---
        components = _setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize and Run Pipeline ---
        pipeline = TrainingPipeline(components, visual_mode=False)
        pipeline.run()

        # --- Determine Exit Code ---
        if pipeline.training_loop and pipeline.training_loop.training_complete:
            exit_code = 0
        elif pipeline.training_loop and pipeline.training_loop.training_exception:
            exit_code = 1  # Failed
        else:
            exit_code = 1  # Interrupted or other issue

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during headless training setup or execution: {e}"
        )
        traceback.print_exc()
        # Attempt to log failure status if MLflow run was started by pipeline
        if pipeline and pipeline.mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        # Pipeline handles its own cleanup (Ray shutdown, MLflow end run, saving state)
        if pipeline:
            pipeline.cleanup()

        # Close file handler if it was set up
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
                # Use original stderr if redirection failed or was undone
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Headless training finished with exit code {exit_code}.")
    return exit_code


# --- Visual Mode Runner ---


def _training_pipeline_thread_func(pipeline: TrainingPipeline):
    """Function to run the training pipeline in a separate thread (for visual mode)."""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Training pipeline thread started.")
        pipeline.run()
        logger.info("Training pipeline thread finished.")
    except Exception as e:
        logger.critical(f"Error in training pipeline thread: {e}", exc_info=True)
        if pipeline.training_loop:
            pipeline.training_loop.training_exception = e
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
    train_config_override: config.TrainConfig,
    persist_config_override: config.PersistenceConfig,
) -> int:
    """Runs the training pipeline in visual mode."""
    logger = logging.getLogger(__name__)
    main_thread_exception = None
    train_thread = None
    pipeline: TrainingPipeline | None = None
    exit_code = 1
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    file_handler = None
    tee_stdout = None
    tee_stderr = None

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
        # --- End Redirection ---

        # --- Setup Components ---
        components = _setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize Pipeline ---
        pipeline = TrainingPipeline(
            components, visual_mode=True, visual_state_queue=visual_state_queue
        )

        # --- Start Training Thread ---
        train_thread = threading.Thread(
            target=_training_pipeline_thread_func, args=(pipeline,), daemon=True
        )
        train_thread.start()
        logger.info("Training pipeline thread launched.")

        # --- Initialize Visualization ---
        # Instantiate Pydantic models using defaults
        # --- CHANGE: Removed type ignore ---
        vis_config = config.VisConfig()
        # --- END CHANGE ---
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
        dashboard_renderer = visualization.DashboardRenderer(
            screen,
            vis_config,
            components.env_config,
            fonts,
            components.stats_collector_actor,
            components.model_config,
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
                    # Use update for global stats to handle potential overlaps
                    global_stats_update = visual_data.pop(-1, {})
                    if isinstance(global_stats_update, dict):
                        # Ensure current_global_stats is a dict before updating
                        if not isinstance(current_global_stats, dict):
                            current_global_stats = {}
                        current_global_stats.update(global_stats_update)
                    else:
                        logger.warning(
                            f"Received non-dict global stats update: {type(global_stats_update)}"
                        )

                    # Extract worker states (keys >= 0)
                    current_worker_states = {
                        k: v
                        for k, v in visual_data.items()
                        if isinstance(k, int)
                        and k >= 0
                        and isinstance(v, environment.GameState)
                    }
                    # Check for any remaining items (should ideally be empty)
                    remaining_items = {
                        k: v
                        for k, v in visual_data.items()
                        if k != -1 and k not in current_worker_states
                    }
                    if remaining_items:
                        # Log remaining items but DO NOT merge into global_stats
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
                # Check font exists before using it
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
                logger.warning("Training pipeline thread terminated unexpectedly.")
                if (
                    pipeline
                    and pipeline.training_loop
                    and pipeline.training_loop.training_exception
                ):
                    logger.error(
                        f"Training thread terminated due to exception: {pipeline.training_loop.training_exception}"
                    )
                    main_thread_exception = pipeline.training_loop.training_exception
                running = False

            clock.tick(vis_config.FPS)

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred in visual training script (main thread): {e}"
        )
        traceback.print_exc()
        main_thread_exception = e
        if pipeline and pipeline.mlflow_run_active:
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
        if (
            pipeline
            and pipeline.training_loop
            and not pipeline.training_loop.stop_requested.is_set()
        ):
            pipeline.request_stop()

        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training pipeline thread to join...")
            train_thread.join(timeout=15.0)
            if train_thread.is_alive():
                logger.error("Training pipeline thread did not exit gracefully.")

        if pipeline:
            pipeline.cleanup()

        # Determine final exit code
        if pipeline and pipeline.training_loop:
            if main_thread_exception or pipeline.training_loop.training_exception:
                exit_code = 1
            elif pipeline.training_loop.training_complete:
                exit_code = 0
            else:
                exit_code = 1  # Interrupted
        else:
            exit_code = 1  # Pipeline failed

        pygame.quit()
        logger.info("Pygame quit.")

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

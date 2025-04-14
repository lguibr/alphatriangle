# File: src/training/pipeline.py
# File: src/training/pipeline.py
import logging
import queue
import time
from collections import deque
from pathlib import Path
from typing import Any

import mlflow
import ray
import torch

# Use relative imports
from ..config import (
    APP_NAME,
)
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import log_configs_to_mlflow
from .loop import TrainingLoop

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Manages the overall training setup, execution, and teardown.
    Initializes components, handles Ray/MLflow, loads/saves state,
    and runs the core TrainingLoop.
    """

    def __init__(
        self,
        components: TrainingComponents,
        visual_mode: bool = False,
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None = None,
    ):
        self.components = components
        self.visual_mode = visual_mode
        self.visual_state_queue = visual_state_queue

        self.ray_initialized = False
        self.mlflow_run_active = False
        self.training_loop: TrainingLoop | None = None
        self.start_time = time.time()

        logger.info(f"TrainingPipeline initialized. Visual Mode: {visual_mode}")

    def _initialize_ray(self):
        """Initializes Ray."""
        if not ray.is_initialized():
            try:
                # Keep log_to_driver=True for visibility, but set level higher
                ray.init(logging_level=logging.WARNING, log_to_driver=True)
                self.ray_initialized = True
                logger.info(
                    f"Ray initialized. Cluster resources: {ray.cluster_resources()}"
                )
            except Exception as e:
                logger.critical(f"Failed to initialize Ray: {e}", exc_info=True)
                raise RuntimeError("Ray initialization failed") from e
        else:
            self.ray_initialized = True
            logger.info("Ray already initialized.")

    def _initialize_mlflow(self):
        """Sets up MLflow tracking."""
        try:
            persist_config = self.components.persist_config
            mlflow_abs_path = persist_config.get_mlflow_abs_path()
            # Use Path.mkdir instead of os.makedirs
            Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
            mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(APP_NAME)
            logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
            logger.info(f"Set MLflow experiment to: {APP_NAME}")

            # Start MLflow run
            mlflow.start_run(run_name=self.components.train_config.RUN_NAME)
            self.mlflow_run_active = True
            logger.info(f"MLflow Run started (ID: {mlflow.active_run().info.run_id}).")
            log_configs_to_mlflow(self.components)  # Log configs after run starts
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
            # Continue without MLflow if initialization fails? Or raise error?
            # Let's raise for now, as tracking is important.
            raise RuntimeError("MLflow initialization failed") from e

    def _load_initial_state(self):
        """Loads initial state using DataManager and applies it to components."""
        loaded_state = self.components.data_manager.load_initial_state()

        if loaded_state.checkpoint_data:
            cp_data = loaded_state.checkpoint_data
            logger.info(
                f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
            )

            # Apply state to NN, Trainer (optimizer), StatsCollector
            if cp_data.model_state_dict:
                self.components.nn.set_weights(cp_data.model_state_dict)
            if cp_data.optimizer_state_dict:
                try:
                    self.components.trainer.optimizer.load_state_dict(
                        cp_data.optimizer_state_dict
                    )
                    # Ensure optimizer state tensors are on the correct device
                    for state in self.components.trainer.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.components.nn.device)
                    logger.info("Optimizer state loaded and moved to device.")
                except Exception as opt_load_err:
                    logger.error(
                        f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                    )
            # --- CHANGE: Apply stats collector state ---
            if cp_data.stats_collector_state and self.components.stats_collector_actor:
                try:
                    set_state_ref = (
                        self.components.stats_collector_actor.set_state.remote(
                            cp_data.stats_collector_state
                        )
                    )
                    ray.get(set_state_ref, timeout=5.0)
                    logger.info("StatsCollectorActor state restored.")
                except Exception as e:
                    logger.error(
                        f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                    )
            # --- END CHANGE ---

            # Set initial state in the loop object
            self.training_loop = TrainingLoop(self.components, self.visual_state_queue)
            self.training_loop.set_initial_state(
                cp_data.global_step,
                cp_data.episodes_played,
                cp_data.total_simulations_run,
            )

        else:
            logger.info("No checkpoint data loaded. Starting fresh.")
            self.training_loop = TrainingLoop(self.components, self.visual_state_queue)
            self.training_loop.set_initial_state(0, 0, 0)  # Start from zero

        # Load buffer data
        if loaded_state.buffer_data:
            if self.components.train_config.USE_PER:
                logger.info("Rebuilding PER SumTree from loaded buffer data...")
                if (
                    not hasattr(self.components.buffer, "tree")
                    or self.components.buffer.tree is None
                ):
                    self.components.buffer.tree = SumTree(
                        self.components.buffer.capacity
                    )
                else:
                    # Re-initialize the tree to ensure correct state
                    self.components.buffer.tree = SumTree(
                        self.components.buffer.capacity
                    )
                max_p = 1.0
                for exp in loaded_state.buffer_data.buffer_list:
                    # Add with default max priority, actual priorities are lost
                    self.components.buffer.tree.add(max_p, exp)
                logger.info(f"PER buffer loaded. Size: {len(self.components.buffer)}")
            else:
                self.components.buffer.buffer = deque(
                    loaded_state.buffer_data.buffer_list,
                    maxlen=self.components.buffer.capacity,
                )
                logger.info(
                    f"Uniform buffer loaded. Size: {len(self.components.buffer)}"
                )
            # Update buffer progress bar after loading
            if self.training_loop and self.training_loop.buffer_fill_progress:
                self.training_loop.buffer_fill_progress.set_current_steps(
                    len(self.components.buffer)
                )
        else:
            logger.info("No buffer data loaded.")

        self.components.nn.model.train()  # Ensure model is in train mode

    def run(self):
        """Executes the full training pipeline."""
        self.start_time = time.time()
        try:
            self._initialize_ray()
            self._initialize_mlflow()
            self._load_initial_state()  # This also creates the TrainingLoop instance

            if not self.training_loop:
                raise RuntimeError(
                    "TrainingLoop was not initialized after loading state."
                )

            self.training_loop.initialize_workers()  # Initialize workers after state is loaded
            self.training_loop.run()  # Run the core loop

        except Exception as e:
            logger.critical(f"Training pipeline failed: {e}", exc_info=True)
            if self.training_loop:
                self.training_loop.training_exception = e  # Propagate exception
            # Ensure cleanup is called via the main script's finally block

    def request_stop(self):
        """Requests the training loop to stop."""
        if self.training_loop:
            self.training_loop.request_stop()
        else:
            logger.warning("Cannot request stop: TrainingLoop not initialized.")

    def _save_final_state(self):
        """Saves the final training state."""
        if not self.training_loop:
            logger.warning("Cannot save final state: TrainingLoop not initialized.")
            return

        logger.info("Saving final training state via Pipeline...")
        try:
            self.components.data_manager.save_training_state(
                nn=self.components.nn,
                optimizer=self.components.trainer.optimizer,
                stats_collector_actor=self.components.stats_collector_actor,
                buffer=self.components.buffer,
                global_step=self.training_loop.global_step,
                episodes_played=self.training_loop.episodes_played,
                total_simulations_run=self.training_loop.total_simulations_run,
                is_final=True,
            )
        except Exception as e_save:
            logger.error(
                f"Failed to save final training state: {e_save}", exc_info=True
            )

    def cleanup(self):
        """Performs cleanup of resources."""
        logger.info("Starting TrainingPipeline cleanup...")

        # 1. Save final state (if loop ran)
        if self.training_loop:
            self._save_final_state()
            # 2. Cleanup loop actors
            self.training_loop.cleanup_actors()

        # 3. End MLflow run
        final_status = "UNKNOWN"
        error_msg = ""
        if self.training_loop:
            if self.training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(self.training_loop.training_exception)
            elif self.training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"  # If loop never initialized

        if self.mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
                self.mlflow_run_active = False
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        # 4. Shutdown Ray
        if self.ray_initialized:
            try:
                ray.shutdown()
                logger.info("Ray shut down.")
                self.ray_initialized = False
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        logger.info("TrainingPipeline cleanup finished.")


import logging
import queue # Keep queue for type hint check, but remove usage
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..rl import SelfPlayResult
# REMOVE ProgressBar import
from .loop_helpers import LoopHelpers
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    from ..utils.types import PERBatchSample
    # REMOVE ProgressBar import
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

class TrainingLoop:
    """
    Manages the core asynchronous training loop logic: coordinating worker tasks,
    processing results, triggering training steps. Runs headless.
    """

    def __init__(
        self,
        components: "TrainingComponents",
        # REMOVE visual_state_queue parameter
    ):
        self.components = components
        # REMOVE self.visual_state_queue = visual_state_queue
        self.train_config = components.train_config
        self.buffer = components.buffer
        self.trainer = components.trainer

        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.worker_weight_updates_count = 0
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.training_exception: Exception | None = None

        # REMOVE Progress Bars
        # self.train_step_progress: ProgressBar | None = None
        # self.buffer_fill_progress: ProgressBar | None = None

        self.worker_manager = WorkerManager(components)
        # Pass None for visual_state_queue
        self.loop_helpers = LoopHelpers(
            components,
            None, # Pass None for visual_state_queue
            self._get_loop_state,
        )

        logger.info("TrainingLoop initialized (Headless).")

    def _get_loop_state(self) -> dict[str, Any]:
        """Provides current loop state to helpers."""
        return {
            "global_step": self.global_step,
            "episodes_played": self.episodes_played,
            "total_simulations_run": self.total_simulations_run,
            "worker_weight_updates": self.worker_weight_updates_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "num_active_workers": self.worker_manager.get_num_active_workers(),
            "num_pending_tasks": self.worker_manager.get_num_pending_tasks(),
            # REMOVE progress bars from state
            # "train_progress": self.train_step_progress,
            # "buffer_progress": self.buffer_fill_progress,
            "start_time": self.start_time,
            "num_workers": self.train_config.NUM_SELF_PLAY_WORKERS,
        }

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        self.worker_weight_updates_count = (
            global_step // self.train_config.WORKER_UPDATE_FREQ_STEPS
        )
        # REMOVE progress bar initialization
        # self.train_step_progress, self.buffer_fill_progress = (
        #     self.loop_helpers.initialize_progress_bars(
        #         global_step, len(self.buffer), self.start_time
        #     )
        # )
        self.loop_helpers.reset_rate_counters(
            global_step, episodes_played, total_simulations
        )
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}, WeightUpdates={self.worker_weight_updates_count}"
        )

    def initialize_workers(self):
        """Initializes self-play workers using WorkerManager."""
        self.worker_manager.initialize_workers()

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _process_self_play_result(self, result: SelfPlayResult, worker_id: int):
        """Processes a validated result from a worker."""
        logger.debug(
            f"Processing result from worker {worker_id} (Ep Steps: {result.episode_steps}, Score: {result.final_score:.2f})"
        )

        valid_experiences, invalid_count = self.loop_helpers.validate_experiences(
            result.episode_experiences
        )
        if invalid_count > 0:
            logger.warning(
                f"Worker {worker_id}: {invalid_count} invalid experiences were filtered out before adding to buffer."
            )

        if valid_experiences:
            try:
                self.buffer.add_batch(valid_experiences)
                logger.debug(
                    f"Added {len(valid_experiences)} experiences from worker {worker_id} to buffer (Buffer size: {len(self.buffer)})."
                )
            except Exception as e:
                logger.error(
                    f"Error adding batch to buffer from worker {worker_id}: {e}",
                    exc_info=True,
                )
                return

            # REMOVE progress bar update
            # if self.buffer_fill_progress:
            #     self.buffer_fill_progress.set_current_steps(len(self.buffer))
            self.episodes_played += 1
            self.total_simulations_run += result.total_simulations
        else:
            logger.error(
                f"Worker {worker_id}: Self-play episode produced NO valid experiences (Steps: {result.episode_steps}, Score: {result.final_score:.2f}). This prevents buffer filling and training."
            )

    def _run_training_step(self) -> bool:
        """Runs one training step."""
        if not self.buffer.is_ready():
            return False
        per_sample: PERBatchSample | None = self.buffer.sample(
            self.train_config.BATCH_SIZE, current_train_step=self.global_step
        )
        if not per_sample:
            return False

        train_result: tuple[dict[str, float], np.ndarray] | None = (
            self.trainer.train_step(per_sample)
        )
        if train_result:
            loss_info, td_errors = train_result
            self.global_step += 1
            # REMOVE progress bar update
            # if self.train_step_progress:
            #     self.train_step_progress.set_current_steps(self.global_step)
            if self.train_config.USE_PER:
                self.buffer.update_priorities(per_sample["indices"], td_errors)
            self.loop_helpers.log_training_results_async(
                loss_info, self.global_step, self.total_simulations_run
            )

            if self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS == 0:
                try:
                    self.worker_manager.update_worker_networks(self.global_step)
                    self.worker_weight_updates_count += 1
                    self.loop_helpers.log_weight_update_event(self.global_step)
                except Exception as update_err:
                    logger.error(
                        f"Failed to update worker networks at step {self.global_step}: {update_err}"
                    )

            if self.global_step % 50 == 0: # Keep periodic logging
                logger.info(
                    f"Step {self.global_step}: P Loss={loss_info['policy_loss']:.4f}, V Loss={loss_info['value_loss']:.4f}, Ent={loss_info['entropy']:.4f}, TD Err={loss_info['mean_td_error']:.4f}"
                )
            return True
        else:
            logger.warning(f"Training step {self.global_step + 1} failed.")
            return False

    def run(self):
        """Main training loop."""
        max_steps_info = (
            f"Target steps: {self.train_config.MAX_TRAINING_STEPS}"
            if self.train_config.MAX_TRAINING_STEPS is not None
            else "Running indefinitely (no MAX_TRAINING_STEPS)"
        )
        logger.info(f"Starting TrainingLoop run... {max_steps_info}")
        self.start_time = time.time()

        try:
            self.worker_manager.submit_initial_tasks()

            while not self.stop_requested.is_set():
                if (
                    self.train_config.MAX_TRAINING_STEPS is not None
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached MAX_TRAINING_STEPS ({self.train_config.MAX_TRAINING_STEPS}). Stopping loop."
                    )
                    self.training_complete = True
                    self.request_stop()
                    break

                if self.buffer.is_ready():
                    _ = self._run_training_step()
                else:
                    time.sleep(0.01) # Short sleep if not training

                if self.stop_requested.is_set():
                    break

                wait_timeout = 0.1 if self.buffer.is_ready() else 0.5
                completed_tasks = self.worker_manager.get_completed_tasks(wait_timeout)

                for worker_id, result_or_error in completed_tasks:
                    if isinstance(result_or_error, SelfPlayResult):
                        try:
                            self._process_self_play_result(result_or_error, worker_id)
                        except Exception as proc_err:
                            logger.error(
                                f"Error processing result from worker {worker_id}: {proc_err}",
                                exc_info=True,
                            )
                    elif isinstance(result_or_error, Exception):
                        logger.error(
                            f"Worker {worker_id} task failed with exception: {result_or_error}"
                        )
                    else:
                        logger.error(
                            f"Received unexpected item from completed tasks for worker {worker_id}: {type(result_or_error)}"
                        )

                    self.worker_manager.submit_task(worker_id)

                if self.stop_requested.is_set():
                    break

                # REMOVE visual queue update
                # self.loop_helpers.update_visual_queue()
                self.loop_helpers.log_progress_eta() # Keep ETA logging
                self.loop_helpers.calculate_and_log_rates()

                if not completed_tasks and not self.buffer.is_ready():
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received in TrainingLoop. Stopping.")
            self.request_stop()
        except Exception as e:
            logger.critical(f"Unhandled exception in TrainingLoop: {e}", exc_info=True)
            self.training_exception = e
            self.request_stop()
        finally:
            if (
                self.training_exception
                or self.stop_requested.is_set()
                and not self.training_complete
            ):
                self.training_complete = False
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Cleans up worker actors using WorkerManager."""
        self.worker_manager.cleanup_actors()
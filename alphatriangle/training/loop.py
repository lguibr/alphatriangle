# File: alphatriangle/training/loop.py
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from trieye.schemas import RawMetricEvent  # Import from trieye

from ..rl import SelfPlayResult
from .loop_helpers import LoopHelpers
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    import numpy as np

    from ..utils.types import PERBatchSample
    from .components import TrainingComponents


logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Manages the core asynchronous training loop logic: coordinating worker tasks,
    processing results, triggering training steps, and interacting with TrieyeActor.
    Runs headless.
    """

    def __init__(
        self,
        components: "TrainingComponents",
    ):
        self.components = components
        self.train_config = components.train_config
        self.trieye_config = components.trieye_config
        self.buffer = components.buffer
        self.trainer = components.trainer
        self.trieye_actor = components.trieye_actor
        self.serializer = components.serializer  # Get serializer from components

        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.weight_update_count = 0
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.training_exception: Exception | None = None
        self._buffer_ready_logged = False

        self.worker_manager = WorkerManager(components)
        self.loop_helpers = LoopHelpers(components, self._get_loop_state)

        logger.info("TrainingLoop initialized (Headless, using Trieye).")

    def _get_loop_state(self) -> dict[str, Any]:
        """Provides current loop state to helpers."""
        return {
            "global_step": self.global_step,
            "episodes_played": self.episodes_played,
            "total_simulations_run": self.total_simulations_run,
            "weight_update_count": self.weight_update_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "num_active_workers": self.worker_manager.get_num_active_workers(),
            "num_pending_tasks": self.worker_manager.get_num_pending_tasks(),
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
        self.weight_update_count = (
            global_step // self.train_config.WORKER_UPDATE_FREQ_STEPS
            if self.train_config.WORKER_UPDATE_FREQ_STEPS > 0
            else 0
        )
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}, WeightUpdates={self.weight_update_count}"
        )

    def initialize_workers(self):
        """Initializes self-play workers using WorkerManager."""
        self.worker_manager.initialize_workers()

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _send_event_async(
        self, name: str, value: float | int, context: dict | None = None
    ):
        """Helper to send a raw metric event to the Trieye actor asynchronously."""
        if self.trieye_actor:
            event = RawMetricEvent(
                name=name,
                value=value,
                global_step=self.global_step,
                timestamp=time.time(),
                context=context or {},
            )
            try:
                self.trieye_actor.log_event.remote(event)
            except Exception as e:
                logger.error(f"Failed to send event '{name}' to Trieye actor: {e}")

    def _send_batch_events_async(self, events: list[RawMetricEvent]):
        """Helper to send a batch of raw metric events asynchronously."""
        if self.trieye_actor and events:
            try:
                self.trieye_actor.log_batch_events.remote(events)
            except Exception as e:
                logger.error(f"Failed to send batch events to Trieye actor: {e}")

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
                buffer_size = len(self.buffer)
                logger.debug(
                    f"Added {len(valid_experiences)} experiences from worker {worker_id} to buffer (Buffer size: {buffer_size})."
                )
                if (
                    not self._buffer_ready_logged
                    and buffer_size >= self.buffer.min_size_to_train
                ):
                    logger.info(
                        f"--- Buffer ready for training (Size: {buffer_size}/{self.buffer.min_size_to_train}) ---"
                    )
                    self._buffer_ready_logged = True

            except Exception as e:
                logger.error(
                    f"Error adding batch to buffer from worker {worker_id}: {e}",
                    exc_info=True,
                )
                return

            self.episodes_played += 1
            self.total_simulations_run += result.total_simulations

            self._send_event_async("Progress/Episodes_Played", self.episodes_played)
            self._send_event_async(
                "Progress/Total_Simulations", self.total_simulations_run
            )

        else:
            logger.error(
                f"Worker {worker_id}: Self-play episode produced NO valid experiences (Steps: {result.episode_steps}, Score: {result.final_score:.2f}). This prevents buffer filling and training."
            )

    def _trigger_save_state(self, is_best: bool = False, save_buffer: bool = False):
        """Triggers the Trieye actor to save the current training state."""
        if not self.trieye_actor:
            logger.error("Cannot save state: TrieyeActor handle is missing.")
            return

        logger.info(
            f"Requesting state save via Trieye at step {self.global_step}. Best={is_best}, SaveBuffer={save_buffer}"
        )
        try:
            # Prepare data using the local serializer
            nn_state = self.components.nn.get_weights()
            opt_state = self.serializer.prepare_optimizer_state(self.trainer.optimizer)
            buffer_data = self.serializer.prepare_buffer_data(self.buffer)

            if buffer_data is None and save_buffer:
                logger.error("Failed to prepare buffer data, cannot save buffer.")
                save_buffer = False  # Don't attempt to save None

            # Fire-and-forget call to the actor
            self.trieye_actor.save_training_state.remote(
                nn_state_dict=nn_state,
                optimizer_state_dict=opt_state,
                buffer_content=(
                    buffer_data.buffer_list if buffer_data else []
                ),  # Pass the list
                global_step=self.global_step,
                episodes_played=self.episodes_played,
                total_simulations_run=self.total_simulations_run,
                is_best=is_best,
                save_buffer=save_buffer,
                model_config_dict=self.components.model_config.model_dump(),
                env_config_dict=self.components.env_config.model_dump(),
            )
        except Exception as e_save:
            logger.error(
                f"Failed to trigger save state via Trieye at step {self.global_step}: {e_save}",
                exc_info=True,
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
            prev_step = self.global_step
            self.global_step += 1
            if prev_step == 0:
                logger.info(
                    f"--- First training step completed (Global Step: {self.global_step}) ---"
                )
            self._send_event_async("step_completed", 1.0)

            if self.train_config.USE_PER:
                self.buffer.update_priorities(per_sample["indices"], td_errors)
                per_beta = self.buffer._calculate_beta(self.global_step)
                self._send_event_async("PER/Beta", per_beta)

            events_batch = []
            loss_name_map = {
                "total_loss": "Loss/Total",
                "policy_loss": "Loss/Policy",
                "value_loss": "Loss/Value",
                "entropy": "Loss/Entropy",
                "mean_td_error": "Loss/Mean_Abs_TD_Error",
            }
            for key, value in loss_info.items():
                metric_name = loss_name_map.get(key)
                if metric_name:
                    events_batch.append(
                        RawMetricEvent(
                            name=metric_name,
                            value=value,
                            global_step=self.global_step,
                        )
                    )
                else:
                    logger.warning(f"Unmapped loss key from trainer: {key}")

            current_lr = self.trainer.get_current_lr()
            events_batch.append(
                RawMetricEvent(
                    name="LearningRate", value=current_lr, global_step=self.global_step
                )
            )

            self._send_batch_events_async(events_batch)

            if (
                self.train_config.WORKER_UPDATE_FREQ_STEPS > 0
                and self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS == 0
            ):
                logger.info(
                    f"Step {self.global_step}: Triggering worker network update (Frequency: {self.train_config.WORKER_UPDATE_FREQ_STEPS})."
                )
                try:
                    self.worker_manager.update_worker_networks(self.global_step)
                    self.weight_update_count += 1
                    self._send_event_async(
                        "Progress/Weight_Updates_Total", self.weight_update_count
                    )
                except Exception as update_err:
                    logger.error(
                        f"Failed to update worker networks at step {self.global_step}: {update_err}"
                    )

            if self.global_step % 50 == 0:
                logger.info(
                    f"Step {self.global_step}: P Loss={loss_info.get('policy_loss', 0.0):.4f}, V Loss={loss_info.get('value_loss', 0.0):.4f}"
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

                trained_this_step = False
                if self.buffer.is_ready():
                    trained_this_step = self._run_training_step()
                else:
                    if not self._buffer_ready_logged and self.global_step % 100 == 0:
                        logger.info(
                            f"Waiting for buffer... Size: {len(self.buffer)}/{self.buffer.min_size_to_train}"
                        )
                    time.sleep(0.01)

                if (
                    trained_this_step
                    and self.train_config.CHECKPOINT_SAVE_FREQ_STEPS > 0
                    and self.global_step % self.train_config.CHECKPOINT_SAVE_FREQ_STEPS
                    == 0
                ):
                    self._trigger_save_state(is_best=False, save_buffer=False)

                if (
                    trained_this_step
                    and self.trieye_config.persistence.SAVE_BUFFER
                    and self.trieye_config.persistence.BUFFER_SAVE_FREQ_STEPS > 0
                    and self.global_step
                    % self.trieye_config.persistence.BUFFER_SAVE_FREQ_STEPS
                    == 0
                ):
                    self._trigger_save_state(is_best=False, save_buffer=True)

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

                self.loop_helpers.log_progress_eta()
                loop_state = self._get_loop_state()
                self._send_event_async("Buffer/Size", loop_state["buffer_size"])
                self._send_event_async(
                    "System/Num_Active_Workers", loop_state["num_active_workers"]
                )
                self._send_event_async(
                    "System/Num_Pending_Tasks", loop_state["num_pending_tasks"]
                )

                if self.trieye_actor:
                    self.trieye_actor.process_and_log.remote(self.global_step)

                if (
                    not completed_tasks
                    and not trained_this_step
                    and not self.buffer.is_ready()
                ):
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
        """Cleans up worker actors. TrieyeActor cleanup handled by runner."""
        self.worker_manager.cleanup_actors()

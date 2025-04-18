# File: alphatriangle/training/loop.py
# Changes:
# - Added tracking for time and counts (steps, episodes, sims) over intervals.
# - Calculate rates (steps/sec, episodes/sec, sims/sec).
# - Log these rates to StatsCollectorActor.

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import ray
from pydantic import ValidationError

from ..environment import GameState
from ..rl import SelfPlayResult, SelfPlayWorker
from ..stats.plotter import WEIGHT_UPDATE_METRIC_KEY  # Import the key
from ..utils import format_eta
from ..utils.types import Experience, PERBatchSample, StatsCollectorData
from ..visualization.core import colors
from ..visualization.ui import ProgressBar

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

VISUAL_UPDATE_INTERVAL = 0.2  # How often to push state to the visual queue
STATS_FETCH_INTERVAL = 0.5
VIS_STATE_FETCH_TIMEOUT = 0.1  # Timeout for getting states from collector
# --- ADDED: Interval for calculating and logging rates ---
RATE_CALCULATION_INTERVAL = 5.0  # Calculate rates every 5 seconds
# --- END ADDED ---


class TrainingLoop:
    """
    Manages the core asynchronous training loop logic: worker management,
    data collection, triggering training steps, and updating visual queue.
    Receives initialized components via TrainingComponents. Runs indefinitely
    until stop_requested is set.
    """

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None = None,
    ):
        self.nn = components.nn
        self.buffer = components.buffer
        self.trainer = components.trainer
        self.data_manager = components.data_manager
        self.stats_collector_actor = components.stats_collector_actor
        self.train_config = components.train_config
        self.env_config = components.env_config
        self.mcts_config = components.mcts_config
        self.model_config = components.model_config
        self.persist_config = components.persist_config
        self.visual_state_queue = visual_state_queue

        self.device = self.nn.device
        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.best_eval_score = -float("inf")
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False  # Will be set based on stop_requested
        self.training_exception: Exception | None = None
        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.train_step_progress: ProgressBar | None = None
        self.buffer_fill_progress: ProgressBar | None = None

        self.workers: list[ray.actor.ActorHandle | None] = []
        self.worker_tasks: dict[ray.ObjectRef, int] = {}
        self.active_worker_indices: set[int] = set()

        # --- ADDED: Rate calculation tracking ---
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = 0
        self.last_rate_calc_episodes = 0
        self.last_rate_calc_sims = 0
        # --- END ADDED ---

        logger.info("TrainingLoop initialized.")

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        self._initialize_progress_bars()
        # --- ADDED: Initialize rate tracking state ---
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = global_step
        self.last_rate_calc_episodes = episodes_played
        self.last_rate_calc_sims = total_simulations
        # --- END ADDED ---
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}"
        )

    def initialize_workers(self):
        """Creates the pool of SelfPlayWorker Ray actors."""
        logger.info(
            f"Initializing {self.train_config.NUM_SELF_PLAY_WORKERS} self-play workers..."
        )
        initial_weights = self.nn.get_weights()
        weights_ref = ray.put(initial_weights)
        self.workers = [None] * self.train_config.NUM_SELF_PLAY_WORKERS

        for i in range(self.train_config.NUM_SELF_PLAY_WORKERS):
            try:
                worker = SelfPlayWorker.options(num_cpus=1).remote(
                    actor_id=i,
                    env_config=self.env_config,
                    mcts_config=self.mcts_config,
                    model_config=self.model_config,
                    train_config=self.train_config,
                    stats_collector_actor=self.stats_collector_actor,
                    initial_weights=weights_ref,
                    seed=self.train_config.RANDOM_SEED + i,
                    worker_device_str=self.train_config.WORKER_DEVICE,
                )
                self.workers[i] = worker
                self.active_worker_indices.add(i)
            except Exception as e:
                logger.error(f"Failed to initialize worker {i}: {e}", exc_info=True)

        logger.info(
            f"Initialized {len(self.active_worker_indices)} active self-play workers."
        )
        del weights_ref

    def _initialize_progress_bars(self):
        """Initializes progress bars based on current state."""
        train_total_steps = self.train_config.MAX_TRAINING_STEPS
        if train_total_steps is None:
            logger.info(
                "MAX_TRAINING_STEPS is None, progress bar will show total steps."
            )
            train_total_steps_for_bar = 1  # Set to 1 for immediate overflow display
        else:
            train_total_steps_for_bar = train_total_steps

        self.train_step_progress = ProgressBar(
            "Training Steps",
            total_steps=train_total_steps_for_bar,
            start_time=self.start_time,
            initial_steps=self.global_step,
            initial_color=colors.GREEN,  # Start with green
        )
        self.buffer_fill_progress = ProgressBar(
            "Buffer Fill",
            self.train_config.BUFFER_CAPACITY,
            start_time=self.start_time,
            initial_steps=len(self.buffer),
            initial_color=colors.ORANGE,  # Start with orange
        )

    def _update_worker_networks(self):
        """Sends the latest network weights to all active workers."""
        active_workers = [
            w
            for i, w in enumerate(self.workers)
            if i in self.active_worker_indices and w is not None
        ]
        if not active_workers:
            return
        logger.debug("Updating worker networks...")
        current_weights = self.nn.get_weights()
        weights_ref = ray.put(current_weights)
        update_tasks = [
            worker.set_weights.remote(weights_ref) for worker in active_workers
        ]
        if not update_tasks:
            del weights_ref
            return
        try:
            ray.get(update_tasks, timeout=15.0)
            logger.debug(f"Worker networks updated for {len(active_workers)} workers.")
            # Log weight update event
            if self.stats_collector_actor:
                update_step_metric = {WEIGHT_UPDATE_METRIC_KEY: (1.0, self.global_step)}
                self.stats_collector_actor.log_batch.remote(update_step_metric)  # type: ignore
        except ray.exceptions.RayActorError as e:
            logger.error(
                f"A worker actor failed during weight update: {e}", exc_info=True
            )
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout waiting for workers to update weights.")
        except Exception as e:
            logger.error(
                f"Unexpected error updating worker networks: {e}", exc_info=True
            )
        finally:
            del weights_ref

    def _fetch_latest_stats(self):
        """Fetches the latest stats data from the actor."""
        current_time = time.time()
        if current_time - self.last_stats_fetch_time < STATS_FETCH_INTERVAL:
            return
        self.last_stats_fetch_time = current_time
        if self.stats_collector_actor:
            try:
                data_ref = self.stats_collector_actor.get_data.remote()  # type: ignore
                self.latest_stats_data = ray.get(data_ref, timeout=1.0)
            except Exception as e:
                logger.warning(f"Failed to fetch latest stats: {e}")

    # --- ADDED: Rate calculation and logging ---
    def _calculate_and_log_rates(self):
        """Calculates and logs steps/sec, episodes/sec, sims/sec."""
        current_time = time.time()
        time_delta = current_time - self.last_rate_calc_time
        if time_delta < RATE_CALCULATION_INTERVAL:
            return

        steps_delta = self.global_step - self.last_rate_calc_step
        episodes_delta = self.episodes_played - self.last_rate_calc_episodes
        sims_delta = self.total_simulations_run - self.last_rate_calc_sims

        steps_per_sec = steps_delta / time_delta if time_delta > 0 else 0.0
        episodes_per_sec = episodes_delta / time_delta if time_delta > 0 else 0.0
        sims_per_sec = sims_delta / time_delta if time_delta > 0 else 0.0

        if self.stats_collector_actor:
            rate_stats = {
                "Rate/Steps_Per_Sec": (steps_per_sec, self.global_step),
                "Rate/Episodes_Per_Sec": (episodes_per_sec, self.global_step),
                "Rate/Simulations_Per_Sec": (sims_per_sec, self.global_step),
            }
            try:
                self.stats_collector_actor.log_batch.remote(rate_stats)  # type: ignore
                logger.debug(
                    f"Logged rates at step {self.global_step}: "
                    f"Steps/s={steps_per_sec:.2f}, Eps/s={episodes_per_sec:.2f}, Sims/s={sims_per_sec:.1f}"
                )
            except Exception as e:
                logger.error(f"Failed to log rate stats to collector: {e}")

        # Update last values for next calculation
        self.last_rate_calc_time = current_time
        self.last_rate_calc_step = self.global_step
        self.last_rate_calc_episodes = self.episodes_played
        self.last_rate_calc_sims = self.total_simulations_run

    # --- END ADDED ---

    def _log_progress_eta(self):
        """Logs progress and ETA."""
        if self.global_step % 50 != 0:  # Log every 50 steps
            return
        if not self.train_step_progress:
            return

        elapsed_time = time.time() - self.start_time
        steps_since_load = self.global_step - self.train_step_progress.initial_steps
        steps_per_sec = steps_since_load / elapsed_time if elapsed_time > 1 else 0
        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = f"{target_steps:,}" if target_steps else "Infinite"
        progress_str = f"Step {self.global_step:,}/{target_steps_str}"
        eta_str = format_eta(self.train_step_progress.get_eta_seconds())

        buffer_fill_perc = (
            (len(self.buffer) / self.buffer.capacity) * 100
            if self.buffer.capacity > 0
            else 0.0
        )
        total_sims_str = (
            f"{self.total_simulations_run / 1e6:.2f}M"
            if self.total_simulations_run >= 1e6
            else (
                f"{self.total_simulations_run / 1e3:.1f}k"
                if self.total_simulations_run >= 1000
                else str(self.total_simulations_run)
            )
        )
        num_pending_tasks = len(self.worker_tasks)
        logger.info(
            f"Progress: {progress_str}, Episodes: {self.episodes_played:,}, Total Sims: {total_sims_str}, "
            f"Buffer: {len(self.buffer):,}/{self.buffer.capacity:,} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending_tasks}, Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _update_visual_queue(self):
        """
        Fetches latest worker states and global stats from the StatsCollectorActor
        and puts the combined data onto the visual queue.
        """
        if not self.visual_state_queue or not self.stats_collector_actor:
            return
        current_time = time.time()
        if current_time - self.last_visual_update_time < VISUAL_UPDATE_INTERVAL:
            return
        self.last_visual_update_time = current_time

        latest_worker_states: dict[int, GameState] = {}
        try:
            states_ref = self.stats_collector_actor.get_latest_worker_states.remote()  # type: ignore
            latest_worker_states = ray.get(states_ref, timeout=VIS_STATE_FETCH_TIMEOUT)
            if not isinstance(latest_worker_states, dict):
                logger.warning(
                    f"StatsCollectorActor returned invalid type for states: {type(latest_worker_states)}"
                )
                latest_worker_states = {}
        except Exception as e:
            logger.warning(
                f"Failed to fetch latest worker states for visualization: {e}"
            )
            latest_worker_states = {}

        self._fetch_latest_stats()

        worker_step_stats: dict[int, dict[str, Any]] = {}
        active_worker_ids_copy = self.active_worker_indices.copy()
        for worker_id in active_worker_ids_copy:
            worker_stats = {}
            visits_key = "MCTS/Step_Visits"
            depth_key = "MCTS/Step_Depth"
            score_key = "RL/Current_Score"
            if (
                visits_key in self.latest_stats_data
                and self.latest_stats_data[visits_key]
            ):
                worker_stats["mcts_visits"] = self.latest_stats_data[visits_key][-1][1]
            if (
                depth_key in self.latest_stats_data
                and self.latest_stats_data[depth_key]
            ):
                worker_stats["mcts_depth"] = self.latest_stats_data[depth_key][-1][1]
            if (
                score_key in self.latest_stats_data
                and self.latest_stats_data[score_key]
            ):
                worker_stats["current_score"] = self.latest_stats_data[score_key][-1][1]

            if worker_stats:
                worker_step_stats[worker_id] = worker_stats

        visual_data: dict[int, Any] = {}
        for worker_id, state in latest_worker_states.items():
            if isinstance(state, GameState):
                visual_data[worker_id] = state
            else:
                logger.warning(
                    f"Received invalid state type for worker {worker_id} from collector: {type(state)}"
                )

        global_stats_for_vis = {
            "global_step": self.global_step,
            "total_episodes": self.episodes_played,
            "total_simulations": self.total_simulations_run,
            "train_progress": self.train_step_progress,
            "buffer_progress": self.buffer_fill_progress,
            "stats_data": self.latest_stats_data,
            "worker_step_stats": worker_step_stats,
            "num_workers": len(self.active_worker_indices),
            "pending_tasks": len(self.worker_tasks),
        }
        visual_data[-1] = global_stats_for_vis

        if not visual_data or len(visual_data) == 1:
            logger.debug(
                "No worker states available from collector to send to visual queue."
            )
            return

        worker_keys = [k for k in visual_data if k != -1]
        logger.debug(
            f"Putting visual data on queue. Worker IDs with states: {worker_keys}"
        )

        try:
            while self.visual_state_queue.qsize() > 2:
                try:
                    self.visual_state_queue.get_nowait()
                except queue.Empty:
                    break
            self.visual_state_queue.put_nowait(visual_data)
        except queue.Full:
            logger.warning("Visual state queue full, dropping state dictionary.")
        except Exception as qe:
            logger.error(f"Error putting state dict in visual queue: {qe}")

    def _validate_experiences(
        self, experiences: list[Experience]
    ) -> tuple[list[Experience], int]:
        """Validates the structure and content of experiences."""
        valid_experiences = []
        invalid_count = 0
        for i, exp in enumerate(experiences):
            is_valid = False
            try:
                if isinstance(exp, tuple) and len(exp) == 3:
                    state_type, policy_map, value = exp
                    if (
                        isinstance(state_type, dict)
                        and "grid" in state_type
                        and "other_features" in state_type
                        and isinstance(state_type["grid"], np.ndarray)
                        and isinstance(state_type["other_features"], np.ndarray)
                        and isinstance(policy_map, dict)
                        and isinstance(value, float | int)
                    ):
                        if np.all(np.isfinite(state_type["grid"])) and np.all(
                            np.isfinite(state_type["other_features"])
                        ):
                            is_valid = True
                        else:
                            logger.warning(
                                f"Experience {i} contains non-finite features."
                            )
                    else:
                        logger.warning(
                            f"Experience {i} has incorrect types: state={type(state_type)}, policy={type(policy_map)}, value={type(value)}"
                        )
                else:
                    logger.warning(
                        f"Experience {i} is not a tuple of length 3: type={type(exp)}, len={len(exp) if isinstance(exp, tuple) else 'N/A'}"
                    )
            except Exception as e:
                logger.error(
                    f"Unexpected error validating experience {i}: {e}", exc_info=True
                )
                is_valid = False
            if is_valid:
                valid_experiences.append(exp)
            else:
                invalid_count += 1
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid experiences.")
        return valid_experiences, invalid_count

    def _process_self_play_result(self, result: SelfPlayResult, worker_id: int):
        """Processes a validated result from a worker."""
        logger.debug(
            f"Processing result from worker {worker_id} (Ep Steps: {result.episode_steps}, Score: {result.final_score:.2f})"
        )

        valid_experiences, invalid_count = self._validate_experiences(
            result.episode_experiences
        )
        # --- ADDED: Log if experiences were filtered ---
        if invalid_count > 0:
            logger.warning(
                f"Worker {worker_id}: {invalid_count} invalid experiences were filtered out before adding to buffer."
            )
        # --- END ADDED ---

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
                return  # Don't update counters if add failed

            if self.buffer_fill_progress:
                self.buffer_fill_progress.set_current_steps(len(self.buffer))
            self.episodes_played += 1
            self.total_simulations_run += result.total_simulations
        else:
            # --- CHANGED: Log more prominently if NO valid experiences ---
            logger.error(
                f"Worker {worker_id}: Self-play episode produced NO valid experiences (Steps: {result.episode_steps}, Score: {result.final_score:.2f}). This prevents buffer filling and training."
            )
            # --- END CHANGED ---

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
            if self.train_step_progress:
                self.train_step_progress.set_current_steps(self.global_step)
            if self.train_config.USE_PER:
                self.buffer.update_priorities(per_sample["indices"], td_errors)
            self._log_training_results_async(loss_info)
            if self.global_step % 50 == 0:
                logger.info(
                    f"Step {self.global_step}: P Loss={loss_info['policy_loss']:.4f}, V Loss={loss_info['value_loss']:.4f}, Ent={loss_info['entropy']:.4f}, TD Err={loss_info['mean_td_error']:.4f}"
                )
            return True
        else:
            logger.warning(f"Training step {self.global_step + 1} failed.")
            return False

    def _log_training_results_async(self, loss_info: dict):
        """Logs training results asynchronously."""
        current_lr = self.trainer.get_current_lr()
        step = self.global_step
        train_step_perc = (
            (
                self.train_step_progress.get_progress_fraction() * 100
            )  # Use fraction for display
            if self.train_step_progress
            else 0.0
        )
        per_beta = (
            self.buffer._calculate_beta(step) if self.train_config.USE_PER else None
        )

        if self.stats_collector_actor:
            stats_batch = {
                "Loss/Total": (loss_info["total_loss"], step),
                "Loss/Policy": (loss_info["policy_loss"], step),
                "Loss/Value": (loss_info["value_loss"], step),
                "Loss/Entropy": (loss_info["entropy"], step),
                "Loss/Mean_TD_Error": (loss_info["mean_td_error"], step),
                "LearningRate": (current_lr, step),
                "Progress/Train_Step_Percent": (train_step_perc, step),
                "Buffer/Size": (len(self.buffer), step),
                "Progress/Total_Simulations": (self.total_simulations_run, step),
            }
            if per_beta is not None:
                stats_batch["PER/Beta"] = (per_beta, step)
            try:
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
                logger.debug(
                    f"Logged training batch to StatsCollectorActor for Step {step}."
                )
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

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
            # Initial task submission
            for worker_idx in self.active_worker_indices:
                worker = self.workers[worker_idx]
                if worker:
                    task_ref = worker.run_episode.remote()
                    self.worker_tasks[task_ref] = worker_idx

            while not self.stop_requested.is_set():
                # Check if max steps reached
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

                # Training Step
                if self.buffer.is_ready():
                    trained_this_cycle = self._run_training_step()
                    if trained_this_cycle and (
                        self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS
                        == 0
                    ):
                        self._update_worker_networks()
                else:
                    # If buffer not ready, sleep briefly to avoid busy-waiting
                    time.sleep(0.01)

                if self.stop_requested.is_set():
                    break

                # Handle Completed Worker Tasks
                wait_timeout = 0.1 if self.buffer.is_ready() else 0.5
                ready_refs, _ = ray.wait(
                    list(self.worker_tasks.keys()), num_returns=1, timeout=wait_timeout
                )

                if ready_refs:
                    for ref in ready_refs:
                        worker_idx = self.worker_tasks.pop(ref, -1)
                        if (
                            worker_idx == -1
                            or worker_idx not in self.active_worker_indices
                        ):
                            continue

                        result = None
                        processing_error = None
                        try:
                            logger.debug(
                                f"Attempting ray.get for worker {worker_idx} task {ref}"
                            )
                            result_raw = ray.get(ref)
                            logger.debug(f"ray.get succeeded for worker {worker_idx}")
                            try:
                                result = SelfPlayResult.model_validate(result_raw)
                                logger.debug(
                                    f"Pydantic validation passed for worker {worker_idx} result."
                                )
                            except ValidationError as e_val:
                                processing_error = f"Pydantic validation failed for result from worker {worker_idx}: {e_val}"
                                logger.error(processing_error, exc_info=False)
                                logger.debug(
                                    f"Invalid data structure received: {result_raw}"
                                )
                                result = None
                            except Exception as e_other_val:
                                processing_error = f"Unexpected error during result validation for worker {worker_idx}: {e_other_val}"
                                logger.error(processing_error, exc_info=True)
                                result = None

                            if result:
                                logger.debug(
                                    f"Processing validated result for worker {worker_idx}"
                                )
                                self._process_self_play_result(result, worker_idx)
                                logger.debug(
                                    f"Finished processing result for worker {worker_idx}"
                                )
                            # --- ADDED: Log if result was None after validation ---
                            elif not processing_error:
                                logger.warning(
                                    f"Worker {worker_idx} returned a result that became None after validation (but no validation error raised)."
                                )
                            # --- END ADDED ---

                        except ray.exceptions.RayActorError as e:
                            processing_error = f"Worker {worker_idx} actor failed: {e}"
                            logger.error(processing_error, exc_info=True)
                            self.workers[worker_idx] = None
                            self.active_worker_indices.discard(worker_idx)
                        except Exception as e:
                            processing_error = (
                                f"Error processing result from worker {worker_idx}: {e}"
                            )
                            logger.error(processing_error, exc_info=True)
                            if not isinstance(e, ValidationError):
                                self.workers[worker_idx] = None
                                self.active_worker_indices.discard(worker_idx)

                        # Relaunch task only if the worker is still active
                        if (
                            worker_idx in self.active_worker_indices
                            and self.workers[worker_idx] is not None
                        ):
                            worker = self.workers[worker_idx]
                            if worker:
                                logger.debug(
                                    f"Relaunching task for worker {worker_idx}"
                                )
                                new_task_ref = worker.run_episode.remote()
                                self.worker_tasks[new_task_ref] = worker_idx
                            else:
                                # This case should ideally not happen if active_worker_indices is correct
                                logger.error(
                                    f"Worker {worker_idx} is None during relaunch despite being in active set."
                                )
                                self.active_worker_indices.discard(worker_idx)
                        elif processing_error:
                            logger.warning(
                                f"Not relaunching task for worker {worker_idx} due to error: {processing_error}"
                            )
                            # Worker already removed or marked as None above

                if self.stop_requested.is_set():
                    break

                # Periodic Tasks
                self._update_visual_queue()
                self._log_progress_eta()
                self._calculate_and_log_rates()  # <-- ADDED

                # Checkpointing (handled by TrainingPipeline)

                if not ready_refs and not self.buffer.is_ready():
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received in TrainingLoop. Stopping.")
            self.request_stop()
        except Exception as e:
            logger.critical(f"Unhandled exception in TrainingLoop: {e}", exc_info=True)
            self.training_exception = e
            self.request_stop()
        finally:
            if self.training_exception:
                self.training_complete = False  # Failed
            elif self.stop_requested.is_set() and not self.training_complete:
                self.training_complete = False  # Interrupted
            # If loop finished naturally by reaching max steps, training_complete is already True
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Kills Ray actors associated with this loop."""
        logger.info("Cleaning up TrainingLoop actors...")
        # Cancel pending tasks first
        for task_ref in list(self.worker_tasks.keys()):
            try:
                ray.cancel(task_ref, force=True)
            except Exception as cancel_e:
                logger.warning(f"Error cancelling task {task_ref}: {cancel_e}")
        self.worker_tasks = {}

        # Kill actors
        for i, worker in enumerate(self.workers):
            if worker:
                try:
                    ray.kill(worker, no_restart=True)
                    logger.debug(f"Killed worker {i}.")
                except Exception as kill_e:
                    logger.warning(f"Error killing worker {i}: {kill_e}")
        self.workers = []
        self.active_worker_indices = set()
        logger.info("TrainingLoop actors cleaned up.")

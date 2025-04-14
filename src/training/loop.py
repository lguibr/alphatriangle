# File: src/training/loop.py
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np  # Import numpy
import ray
from pydantic import ValidationError

from src.environment import GameState
from src.rl import SelfPlayResult, SelfPlayWorker
from src.utils import format_eta
from src.utils.types import Experience, PERBatchSample, StatsCollectorData
from src.visualization.ui import ProgressBar

# Import TrainingComponents only for type hinting
if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

LARGE_STEP_COUNT = 10_000_000
VISUAL_UPDATE_INTERVAL = 0.2  # How often to push state to the visual queue
STATS_FETCH_INTERVAL = 0.5
VIS_STATE_FETCH_TIMEOUT = 0.1  # Timeout for getting states from collector


class TrainingLoop:
    """
    Manages the core asynchronous training loop logic: worker management,
    data collection, triggering training steps, and updating visual queue.
    Receives initialized components via TrainingComponents.
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
        self.training_complete = False
        self.target_steps_reached = False
        self.training_exception: Exception | None = None
        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.train_step_progress: ProgressBar | None = None
        self.buffer_fill_progress: ProgressBar | None = None

        self.workers: list[ray.actor.ActorHandle | None] = []
        self.worker_tasks: dict[ray.ObjectRef, int] = {}
        self.active_worker_indices: set[int] = set()

        logger.info("TrainingLoop initialized.")

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        self._initialize_progress_bars()
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
                    # Pass the stats collector handle to the worker
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
        del weights_ref  # Ensure the reference is deleted

    def _initialize_progress_bars(self):
        """Initializes progress bars based on current state."""
        train_total_steps = self.train_config.MAX_TRAINING_STEPS or LARGE_STEP_COUNT
        self.train_step_progress = ProgressBar(
            "Training Steps",
            train_total_steps,
            start_time=self.start_time,
            initial_steps=self.global_step,
        )
        self.buffer_fill_progress = ProgressBar(
            "Buffer Fill",
            self.train_config.BUFFER_CAPACITY,
            start_time=self.start_time,
            initial_steps=len(self.buffer),
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
        except ray.exceptions.RayActorError as e:
            logger.error(
                f"A worker actor failed during weight update: {e}", exc_info=True
            )
            # Identify and remove the failed worker
            # This requires more complex tracking or handling the error when the task result is retrieved
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout waiting for workers to update weights.")
        except Exception as e:
            logger.error(
                f"Unexpected error updating worker networks: {e}", exc_info=True
            )
        finally:
            del weights_ref  # Ensure the reference is deleted

    def _fetch_latest_stats(self):
        """Fetches the latest stats data from the actor."""
        current_time = time.time()
        if current_time - self.last_stats_fetch_time < STATS_FETCH_INTERVAL:
            return
        self.last_stats_fetch_time = current_time
        if self.stats_collector_actor:
            try:
                # Correctly call remote method
                data_ref = self.stats_collector_actor.get_data.remote()  # type: ignore
                self.latest_stats_data = ray.get(data_ref, timeout=1.0)
            except Exception as e:
                logger.warning(f"Failed to fetch latest stats: {e}")

    def _log_progress_eta(self):
        """Logs progress and ETA."""
        if self.global_step % 50 != 0 and not self.target_steps_reached:
            return
        if not self.train_step_progress:
            return

        elapsed_time = time.time() - self.start_time
        steps_since_load = self.global_step - self.train_step_progress.initial_steps
        steps_per_sec = steps_since_load / elapsed_time if elapsed_time > 1 else 0
        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = str(target_steps) if target_steps else "inf"
        eta_str = (
            format_eta(self.train_step_progress.get_eta_seconds())
            if not self.target_steps_reached and target_steps is not None
            else "N/A"
        )
        progress_str = f"Step {self.global_step}/{target_steps_str}"
        if self.target_steps_reached:
            progress_str += (
                f" (TARGET REACHED +{self.global_step - (target_steps or 0)} extra)"
            )
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
            f"Progress: {progress_str}, Episodes: {self.episodes_played}, Total Sims: {total_sims_str}, "
            f"Buffer: {len(self.buffer)}/{self.buffer.capacity} ({buffer_fill_perc:.1f}%), "
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

        # Fetch latest worker states from the collector actor
        latest_worker_states: dict[int, GameState] = {}
        try:
            # Correctly call remote method
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
            latest_worker_states = {}  # Use empty dict on error

        # Fetch latest metrics data (needed for plots and per-step worker stats)
        self._fetch_latest_stats()

        # Extract per-worker step stats from the fetched metrics data
        worker_step_stats: dict[int, dict[str, Any]] = {}
        active_worker_ids_copy = self.active_worker_indices.copy()
        for worker_id in active_worker_ids_copy:
            worker_stats = {}
            visits_key = f"Worker_{worker_id}/Step_MCTS_Visits"
            depth_key = f"Worker_{worker_id}/Step_MCTS_Depth"
            score_key = f"Worker_{worker_id}/Step_Score"

            if (
                visits_key in self.latest_stats_data
                and self.latest_stats_data[visits_key]
            ):
                worker_stats["mcts_visits"] = self.latest_stats_data[visits_key][-1][
                    1
                ]  # Get last value
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

        # Assemble the data dictionary for the queue
        visual_data: dict[int, Any] = {}

        # Add worker states (already copies from the actor)
        for worker_id, state in latest_worker_states.items():
            if isinstance(state, GameState):  # Basic type check
                visual_data[worker_id] = state
            else:
                logger.warning(
                    f"Received invalid state type for worker {worker_id} from collector: {type(state)}"
                )

        # Add global stats and per-worker step stats
        global_stats_for_vis = {
            "global_step": self.global_step,
            "target_steps_reached": self.target_steps_reached,
            "total_episodes": self.episodes_played,
            "total_simulations": self.total_simulations_run,
            "train_progress": self.train_step_progress,
            "buffer_progress": self.buffer_fill_progress,
            "stats_data": self.latest_stats_data,  # Full stats data for plotting
            "worker_step_stats": worker_step_stats,  # Add the extracted per-step stats
            "num_workers": len(self.active_worker_indices),
            "pending_tasks": len(self.worker_tasks),
        }
        visual_data[-1] = global_stats_for_vis

        if not visual_data or len(visual_data) == 1:  # Only global stats present
            logger.debug(
                "No worker states available from collector to send to visual queue."
            )
            return

        worker_keys = [k for k in visual_data if k != -1]
        logger.debug(
            f"Putting visual data on queue. Worker IDs with states: {worker_keys}"
        )

        try:
            # Clear older states if queue is getting full
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
                        # Use isinstance with | for multiple types
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
        # Removed: self.latest_final_worker_states[worker_id] = result.final_game_state

        valid_experiences, _ = self._validate_experiences(result.episode_experiences)
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
                return  # Don't update counters if buffer add fails

            if self.buffer_fill_progress:
                self.buffer_fill_progress.set_current_steps(len(self.buffer))
            self.episodes_played += 1
            self.total_simulations_run += result.total_simulations
            self._log_self_play_results_async(result, worker_id)
        else:
            logger.warning(
                f"Self-play episode from worker {worker_id} produced no valid experiences."
            )

    def _log_self_play_results_async(self, result: SelfPlayResult, worker_id: int):
        """Logs self-play results asynchronously."""
        episode_num = (
            self.episodes_played
        )  # Use episode number as step for episode stats
        global_step = self.global_step  # Use global step for buffer/sim stats
        buffer_size = len(self.buffer)
        total_sims = self.total_simulations_run
        buffer_fill_perc = (
            (self.buffer_fill_progress.get_progress() * 100)
            if self.buffer_fill_progress
            else 0.0
        )

        logger.info(
            f"[W{worker_id}] Ep {episode_num} ({result.episode_steps} steps, Score: {result.final_score:.2f}, "
            f"Visits: {result.avg_root_visits:.1f}, Depth: {result.avg_tree_depth:.1f}). Buffer: {buffer_size}"
        )

        if self.stats_collector_actor:
            stats_batch = {
                # Log episode stats against episode number
                "SelfPlay/Episode_Score": (result.final_score, episode_num),
                "SelfPlay/Episode_Length": (result.episode_steps, episode_num),
                "MCTS/Avg_Root_Visits": (result.avg_root_visits, episode_num),
                "MCTS/Avg_Tree_Depth": (result.avg_tree_depth, episode_num),
                # Log buffer/sim stats against global step
                "Buffer/Size": (buffer_size, global_step),
                "Progress/Total_Simulations": (total_sims, global_step),
                "Buffer/Fill_Percent": (buffer_fill_perc, global_step),
            }
            try:
                # Correctly call remote method
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
                logger.debug(
                    f"Logged self-play batch to StatsCollectorActor for Ep {episode_num} / Step {global_step}."
                )
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

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
        step = self.global_step  # Log training stats against global step
        train_step_perc = (
            (self.train_step_progress.get_progress() * 100)
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
            }
            if per_beta is not None:
                stats_batch["PER/Beta"] = (per_beta, step)
            try:
                # Correctly call remote method
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
                logger.debug(
                    f"Logged training batch to StatsCollectorActor for Step {step}."
                )
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

    def run(self):
        """Main training loop."""
        logger.info(
            f"Starting TrainingLoop run... Target steps: {self.train_config.MAX_TRAINING_STEPS or 'Infinite'}"
        )
        self.start_time = time.time()

        try:
            # Initial task submission
            for worker_idx in self.active_worker_indices:
                worker = self.workers[worker_idx]
                if worker:
                    task_ref = worker.run_episode.remote()
                    self.worker_tasks[task_ref] = worker_idx

            while not self.stop_requested.is_set():
                # Check Target Steps
                if (
                    not self.target_steps_reached
                    and self.train_config.MAX_TRAINING_STEPS
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached target training steps ({self.train_config.MAX_TRAINING_STEPS}). Training continues..."
                    )
                    self.target_steps_reached = True

                # Training Step
                if self.buffer.is_ready():
                    trained_this_cycle = self._run_training_step()
                    if trained_this_cycle and (
                        self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS
                        == 0
                    ):
                        self._update_worker_networks()

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

                        # Relaunch task
                        if (
                            processing_error is None
                            and worker_idx in self.active_worker_indices
                        ):
                            worker = self.workers[worker_idx]
                            if worker:
                                logger.debug(
                                    f"Relaunching task for worker {worker_idx}"
                                )
                                new_task_ref = worker.run_episode.remote()
                                self.worker_tasks[new_task_ref] = worker_idx
                            else:
                                logger.error(
                                    f"Worker {worker_idx} is None during relaunch."
                                )
                                self.active_worker_indices.discard(worker_idx)
                        elif processing_error:
                            logger.warning(
                                f"Not relaunching task for worker {worker_idx} due to error: {processing_error}"
                            )
                            self.workers[worker_idx] = None
                            self.active_worker_indices.discard(worker_idx)
                            logger.info(
                                f"Removed worker {worker_idx} from active pool due to processing error."
                            )

                if self.stop_requested.is_set():
                    break

                # Periodic Tasks
                self._update_visual_queue()  # Now fetches states from collector
                self._log_progress_eta()

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
                self.training_complete = False
            elif self.stop_requested.is_set():
                self.training_complete = self.target_steps_reached
            else:
                self.training_complete = self.target_steps_reached or (
                    self.train_config.MAX_TRAINING_STEPS is None
                )
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Kills Ray actors associated with this loop."""
        logger.info("Cleaning up TrainingLoop actors...")
        for i, worker in enumerate(self.workers):
            if worker:
                try:
                    ray.kill(worker, no_restart=True)
                except Exception as kill_e:
                    logger.warning(f"Error killing worker {i}: {kill_e}")
        self.workers = []
        self.active_worker_indices = set()
        self.worker_tasks = {}
        logger.info("TrainingLoop actors cleaned up.")

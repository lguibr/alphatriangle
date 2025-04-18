# File: alphatriangle/training/loop_helpers.py
import logging
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import ray

from ..environment import GameState
from ..stats.plot_definitions import WEIGHT_UPDATE_METRIC_KEY
from ..utils import format_eta
from ..utils.types import Experience, StatsCollectorData, StepInfo
from ..visualization.core import colors
from ..visualization.ui import ProgressBar

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

VISUAL_UPDATE_INTERVAL = 0.2
STATS_FETCH_INTERVAL = 0.5
VIS_STATE_FETCH_TIMEOUT = 0.1
RATE_CALCULATION_INTERVAL = 5.0  # Check rates every 5 seconds


class LoopHelpers:
    """Provides helper functions for the TrainingLoop."""

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None,
        get_loop_state_func: Callable[[], dict[str, Any]],
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.get_loop_state = get_loop_state_func  # Function to get current loop state

        self.stats_collector_actor = components.stats_collector_actor
        self.train_config = components.train_config
        self.trainer = components.trainer  # Needed for LR

        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = 0
        self.last_rate_calc_episodes = 0
        self.last_rate_calc_sims = 0

    def reset_rate_counters(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Resets counters used for rate calculation."""
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = global_step
        self.last_rate_calc_episodes = episodes_played
        self.last_rate_calc_sims = total_simulations

    def initialize_progress_bars(
        self, global_step: int, buffer_size: int, start_time: float
    ) -> tuple[ProgressBar, ProgressBar]:
        """Initializes and returns progress bars."""
        train_total_steps = self.train_config.MAX_TRAINING_STEPS
        train_total_steps_for_bar = (
            train_total_steps if train_total_steps is not None else 1
        )

        train_step_progress = ProgressBar(
            "Training Steps",
            total_steps=train_total_steps_for_bar,
            start_time=start_time,
            initial_steps=global_step,
            initial_color=colors.GREEN,
        )
        buffer_fill_progress = ProgressBar(
            "Buffer Fill",
            self.train_config.BUFFER_CAPACITY,
            start_time=start_time,
            initial_steps=buffer_size,
            initial_color=colors.ORANGE,
        )
        return train_step_progress, buffer_fill_progress

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

    def calculate_and_log_rates(self):
        """
        Calculates and logs steps/sec, episodes/sec, sims/sec, and buffer size.
        Ensures buffer-related rates are logged against buffer size.
        Logs metrics with StepInfo containing global_step and buffer_size.
        """
        current_time = time.time()
        time_delta = current_time - self.last_rate_calc_time
        if time_delta < RATE_CALCULATION_INTERVAL:
            return

        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        episodes_played = loop_state["episodes_played"]
        total_simulations = loop_state["total_simulations_run"]
        current_buffer_size = int(loop_state["buffer_size"])  # Use int for step info

        steps_delta = global_step - self.last_rate_calc_step
        episodes_delta = episodes_played - self.last_rate_calc_episodes
        sims_delta = total_simulations - self.last_rate_calc_sims

        steps_per_sec = steps_delta / time_delta if time_delta > 0 else 0.0
        episodes_per_sec = episodes_delta / time_delta if time_delta > 0 else 0.0
        sims_per_sec = sims_delta / time_delta if time_delta > 0 else 0.0

        if self.stats_collector_actor:
            step_info_buffer: StepInfo = {
                "global_step": global_step,
                "buffer_size": current_buffer_size,
            }
            step_info_global: StepInfo = {"global_step": global_step}

            rate_stats: dict[str, tuple[float, StepInfo]] = {
                "Rate/Episodes_Per_Sec": (episodes_per_sec, step_info_buffer),
                "Rate/Simulations_Per_Sec": (sims_per_sec, step_info_buffer),
                "Buffer/Size": (float(current_buffer_size), step_info_buffer),
            }
            log_msg_steps = "Steps/s=N/A"
            if steps_delta > 0:
                rate_stats["Rate/Steps_Per_Sec"] = (steps_per_sec, step_info_global)
                log_msg_steps = f"Steps/s={steps_per_sec:.2f}"

            try:
                self.stats_collector_actor.log_batch.remote(rate_stats)  # type: ignore
                logger.debug(
                    f"Logged rates/buffer at step {global_step} / buffer {current_buffer_size}: "
                    f"{log_msg_steps}, Eps/s={episodes_per_sec:.2f}, Sims/s={sims_per_sec:.1f}, "
                    f"Buffer={current_buffer_size}"
                )
            except Exception as e:
                logger.error(f"Failed to log rate/buffer stats to collector: {e}")

        self.reset_rate_counters(global_step, episodes_played, total_simulations)

    def log_progress_eta(self):
        """Logs progress and ETA."""
        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        train_progress = loop_state["train_progress"]

        if global_step == 0 or global_step % 100 != 0 or not train_progress:
            return

        elapsed_time = time.time() - loop_state["start_time"]
        steps_since_load = global_step - train_progress.initial_steps
        steps_per_sec = 0.0
        self._fetch_latest_stats()  # Fetch stats to get latest rate

        rate_dq = self.latest_stats_data.get("Rate/Steps_Per_Sec")
        if rate_dq:
            # Get the value from the last tuple (step_info, value)
            steps_per_sec = rate_dq[-1][1]
        elif elapsed_time > 1 and steps_since_load > 0:
            # Fallback calculation if rate not in stats yet
            steps_per_sec = steps_since_load / elapsed_time

        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = f"{target_steps:,}" if target_steps else "Infinite"
        progress_str = f"Step {global_step:,}/{target_steps_str}"
        eta_str = format_eta(train_progress.get_eta_seconds())

        buffer_fill_perc = (
            (loop_state["buffer_size"] / loop_state["buffer_capacity"]) * 100
            if loop_state["buffer_capacity"] > 0
            else 0.0
        )
        total_sims = loop_state["total_simulations_run"]
        total_sims_str = (
            f"{total_sims / 1e6:.2f}M"
            if total_sims >= 1e6
            else (f"{total_sims / 1e3:.1f}k" if total_sims >= 1000 else str(total_sims))
        )
        num_pending_tasks = loop_state["num_pending_tasks"]
        logger.info(
            f"Progress: {progress_str}, Episodes: {loop_state['episodes_played']:,}, Total Sims: {total_sims_str}, "
            f"Buffer: {loop_state['buffer_size']:,}/{loop_state['buffer_capacity']:,} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending_tasks}, Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def update_visual_queue(self):
        """Fetches latest states/stats and puts them onto the visual queue."""
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

        self._fetch_latest_stats()  # Fetch latest metric data

        visual_data: dict[int, Any] = {}
        for worker_id, state in latest_worker_states.items():
            if isinstance(state, GameState):
                visual_data[worker_id] = state
            else:
                logger.warning(
                    f"Received invalid state type for worker {worker_id} from collector: {type(state)}"
                )

        visual_data[-1] = {
            **self.get_loop_state(),
            "stats_data": self.latest_stats_data,
        }

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

    def validate_experiences(
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

    def log_training_results_async(
        self, loss_info: dict[str, float], global_step: int, total_simulations: int
    ) -> None:
        """
        Logs training results asynchronously to the StatsCollectorActor.
        Logs metrics with StepInfo containing global_step.
        """
        current_lr = self.trainer.get_current_lr()
        loop_state = self.get_loop_state()
        train_progress = loop_state.get("train_progress")
        buffer = self.components.buffer

        train_step_perc = (
            (train_progress.get_progress_fraction() * 100) if train_progress else 0.0
        )
        per_beta = (
            buffer._calculate_beta(global_step) if self.train_config.USE_PER else None
        )

        if self.stats_collector_actor:
            step_info: StepInfo = {"global_step": global_step}
            stats_batch: dict[str, tuple[float, StepInfo]] = {
                "Loss/Total": (loss_info["total_loss"], step_info),
                "Loss/Policy": (loss_info["policy_loss"], step_info),
                "Loss/Value": (loss_info["value_loss"], step_info),
                "Loss/Entropy": (loss_info["entropy"], step_info),
                "Loss/Mean_TD_Error": (loss_info["mean_td_error"], step_info),
                "LearningRate": (current_lr, step_info),
                "Progress/Train_Step_Percent": (train_step_perc, step_info),
                "Progress/Total_Simulations": (float(total_simulations), step_info),
            }
            if per_beta is not None:
                stats_batch["PER/Beta"] = (per_beta, step_info)
            try:
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
                logger.debug(
                    f"Logged training batch to StatsCollectorActor for Step {global_step}."
                )
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

    def log_weight_update_event(self, global_step: int) -> None:
        """Logs the event of a worker weight update with StepInfo."""
        if self.stats_collector_actor:
            try:
                # Log with value 1.0 at the current global step
                step_info: StepInfo = {"global_step": global_step}
                update_metric = {WEIGHT_UPDATE_METRIC_KEY: (1.0, step_info)}
                self.stats_collector_actor.log_batch.remote(update_metric)  # type: ignore
                logger.info(f"Logged worker weight update event at step {global_step}.")
            except Exception as e:
                logger.error(f"Failed to log weight update event: {e}")

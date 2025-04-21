# File: alphatriangle/training/loop_helpers.py
import logging
import queue  # Keep for type hint check
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any  # Add cast

import numpy as np
import ray

# --- ADD TensorBoard ---
from torch.utils.tensorboard import SummaryWriter

# --- END ADD ---
# Import GameState from trianglengin
# Keep alphatriangle imports
# REMOVED WEIGHT_UPDATE_METRIC_KEY import (no longer plotting)
from ..utils import format_eta
from ..utils.types import Experience, StatsCollectorData, StepInfo

# REMOVE ProgressBar import
# REMOVE colors import

if TYPE_CHECKING:
    from collections import deque

    from .components import TrainingComponents

logger = logging.getLogger(__name__)

# REMOVE VISUAL_UPDATE_INTERVAL
STATS_FETCH_INTERVAL = 0.5
# REMOVE VIS_STATE_FETCH_TIMEOUT
RATE_CALCULATION_INTERVAL = 5.0
WEIGHT_UPDATE_EVENT_KEY = "Events/Weight_Update"  # Define key for logging


class LoopHelpers:
    """Provides helper functions for the TrainingLoop."""

    def __init__(
        self,
        components: "TrainingComponents",
        # REMOVE visual_state_queue parameter
        _visual_state_queue: (
            queue.Queue[dict[int, Any] | None] | None
        ),  # Keep param name but mark unused
        get_loop_state_func: Callable[[], dict[str, Any]],
    ):
        self.components = components
        # REMOVE self.visual_state_queue = visual_state_queue
        self.get_loop_state = get_loop_state_func

        self.stats_collector_actor = components.stats_collector_actor
        self.train_config = components.train_config
        self.trainer = components.trainer

        # REMOVE self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = 0
        self.last_rate_calc_episodes = 0
        self.last_rate_calc_sims = 0

        # --- ADD TensorBoard Writer ---
        self.tb_writer: SummaryWriter | None = None
        # --- END ADD ---

    # --- ADD Method to set writer ---
    def set_tensorboard_writer(self, writer: SummaryWriter):
        self.tb_writer = writer

    # --- END ADD ---

    def reset_rate_counters(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Resets counters used for rate calculation."""
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = global_step
        self.last_rate_calc_episodes = episodes_played
        self.last_rate_calc_sims = total_simulations

    # REMOVE initialize_progress_bars method

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
        Logs to StatsCollectorActor and TensorBoard.
        """
        current_time = time.time()
        time_delta = current_time - self.last_rate_calc_time
        if time_delta < RATE_CALCULATION_INTERVAL:
            return

        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        episodes_played = loop_state["episodes_played"]
        total_simulations = loop_state["total_simulations_run"]
        current_buffer_size = int(loop_state["buffer_size"])

        steps_delta = global_step - self.last_rate_calc_step
        episodes_delta = episodes_played - self.last_rate_calc_episodes
        sims_delta = total_simulations - self.last_rate_calc_sims

        steps_per_sec = steps_delta / time_delta if time_delta > 0 else 0.0
        episodes_per_sec = episodes_delta / time_delta if time_delta > 0 else 0.0
        sims_per_sec = sims_delta / time_delta if time_delta > 0 else 0.0

        # Log to StatsCollectorActor
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
            if steps_delta > 0:
                rate_stats["Rate/Steps_Per_Sec"] = (steps_per_sec, step_info_global)

            try:
                self.stats_collector_actor.log_batch.remote(rate_stats)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to log rate/buffer stats to collector: {e}")

        # --- ADD Log to TensorBoard ---
        if self.tb_writer:
            try:
                self.tb_writer.add_scalar(
                    "Rates/Episodes_Per_Sec", episodes_per_sec, global_step
                )
                self.tb_writer.add_scalar(
                    "Rates/Simulations_Per_Sec", sims_per_sec, global_step
                )
                self.tb_writer.add_scalar(
                    "Buffer/Size", float(current_buffer_size), global_step
                )
                if steps_delta > 0:
                    self.tb_writer.add_scalar(
                        "Rates/Steps_Per_Sec", steps_per_sec, global_step
                    )
            except Exception as tb_err:
                logger.error(f"Failed to log rates to TensorBoard: {tb_err}")
        # --- END ADD ---

        log_msg_steps = (
            f"Steps/s={steps_per_sec:.2f}" if steps_delta > 0 else "Steps/s=N/A"
        )
        logger.debug(
            f"Logged rates/buffer at step {global_step} / buffer {current_buffer_size}: "
            f"{log_msg_steps}, Eps/s={episodes_per_sec:.2f}, Sims/s={sims_per_sec:.1f}, "
            f"Buffer={current_buffer_size}"
        )

        self.reset_rate_counters(global_step, episodes_played, total_simulations)

    def log_progress_eta(self):
        """Logs progress and ETA."""
        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]

        if global_step == 0 or global_step % 100 != 0:
            return

        elapsed_time = time.time() - loop_state["start_time"]
        steps_since_start = global_step

        steps_per_sec = 0.0
        self._fetch_latest_stats()
        # Safely access latest steps_per_sec from stats
        rate_dq: deque[tuple[StepInfo, float]] | None = self.latest_stats_data.get(
            "Rate/Steps_Per_Sec"
        )
        if rate_dq and len(rate_dq) > 0:
            # Ensure the last item is a tuple with expected structure
            last_item = rate_dq[-1]
            if (
                isinstance(last_item, tuple)
                and len(last_item) == 2
                and isinstance(last_item[1], float | int)
            ):
                steps_per_sec = float(last_item[1])
            else:
                logger.warning(
                    f"Unexpected structure in Rate/Steps_Per_Sec deque: {last_item}"
                )

        # Fallback if no stats available or invalid
        if steps_per_sec <= 0 and elapsed_time > 1 and steps_since_start > 0:
            steps_per_sec = steps_since_start / elapsed_time

        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = f"{target_steps:,}" if target_steps else "Infinite"
        progress_str = f"Step {global_step:,}/{target_steps_str}"

        eta_str = "--"
        if target_steps and steps_per_sec > 1e-6:
            remaining_steps = target_steps - global_step
            if remaining_steps > 0:
                eta_seconds = remaining_steps / steps_per_sec
                eta_str = format_eta(eta_seconds)

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
                        and isinstance(policy_map, dict)  # Check for dict specifically
                        and isinstance(value, float | int)
                    ):
                        if np.all(np.isfinite(state_type["grid"])) and np.all(
                            np.isfinite(state_type["other_features"])
                        ):
                            is_valid = True
                        else:
                            logger.warning(
                                f"Experience {i} contains non-finite features (grid_finite={np.all(np.isfinite(state_type['grid']))}, other_finite={np.all(np.isfinite(state_type['other_features']))})."
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
        Logs training results asynchronously to StatsCollectorActor and TensorBoard.
        """
        current_lr = self.trainer.get_current_lr()
        buffer = self.components.buffer

        per_beta: float | None = None  # Ensure per_beta is typed
        if self.train_config.USE_PER and hasattr(buffer, "_calculate_beta"):
            per_beta = buffer._calculate_beta(global_step)

        # Log to StatsCollectorActor
        if self.stats_collector_actor:
            step_info: StepInfo = {"global_step": global_step}
            stats_batch: dict[str, tuple[float, StepInfo]] = {
                "Loss/Total": (loss_info.get("total_loss", 0.0), step_info),
                "Loss/Policy": (loss_info.get("policy_loss", 0.0), step_info),
                "Loss/Value": (loss_info.get("value_loss", 0.0), step_info),
                "Loss/Entropy": (loss_info.get("entropy", 0.0), step_info),
                "Loss/Mean_TD_Error": (loss_info.get("mean_td_error", 0.0), step_info),
                "LearningRate": (current_lr, step_info),
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

        # --- ADD Log to TensorBoard ---
        if self.tb_writer:
            try:
                self.tb_writer.add_scalar(
                    "Loss/Total", loss_info.get("total_loss", 0.0), global_step
                )
                self.tb_writer.add_scalar(
                    "Loss/Policy", loss_info.get("policy_loss", 0.0), global_step
                )
                self.tb_writer.add_scalar(
                    "Loss/Value", loss_info.get("value_loss", 0.0), global_step
                )
                self.tb_writer.add_scalar(
                    "Loss/Entropy", loss_info.get("entropy", 0.0), global_step
                )
                self.tb_writer.add_scalar(
                    "Loss/Mean_TD_Error",
                    loss_info.get("mean_td_error", 0.0),
                    global_step,
                )
                self.tb_writer.add_scalar("LearningRate", current_lr, global_step)
                self.tb_writer.add_scalar(
                    "Progress/Total_Simulations", float(total_simulations), global_step
                )
                if per_beta is not None:
                    self.tb_writer.add_scalar("PER/Beta", per_beta, global_step)
            except Exception as tb_err:
                logger.error(f"Failed to log training results to TensorBoard: {tb_err}")
        # --- END ADD ---

    def log_weight_update_event(self, global_step: int) -> None:
        """Logs the event of a worker weight update with StepInfo."""
        if self.stats_collector_actor:
            try:
                step_info: StepInfo = {"global_step": global_step}
                # Use the defined key for the event
                update_metric = {WEIGHT_UPDATE_EVENT_KEY: (1.0, step_info)}
                self.stats_collector_actor.log_batch.remote(update_metric)  # type: ignore
                logger.info(f"Logged worker weight update event at step {global_step}.")
            except Exception as e:
                logger.error(f"Failed to log weight update event: {e}")
        # --- ADD Log to TensorBoard ---
        if self.tb_writer:
            try:
                # Log as a scalar event (value 1 indicates update occurred)
                self.tb_writer.add_scalar(WEIGHT_UPDATE_EVENT_KEY, 1.0, global_step)
            except Exception as tb_err:
                logger.error(
                    f"Failed to log weight update event to TensorBoard: {tb_err}"
                )

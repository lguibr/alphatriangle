import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils import format_eta
from ..utils.types import Experience

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

# Removed constants related to old logging intervals and keys


class LoopHelpers:
    """Provides helper functions for the TrainingLoop (simplified)."""

    def __init__(
        self,
        components: "TrainingComponents",
        get_loop_state_func: Callable[[], dict[str, Any]],
    ):
        self.components = components
        self.get_loop_state = get_loop_state_func
        self.train_config = components.train_config
        # No longer needs direct access to stats_collector or trainer for logging

        # Keep state for ETA calculation if needed, or rely on StatsProcessor rates
        # self.last_rate_calc_time = time.time()
        # self.last_rate_calc_step = 0

        logger.info("LoopHelpers initialized (Simplified).")

    # Removed set_tensorboard_writer
    # Removed reset_rate_counters
    # Removed _fetch_latest_stats
    # Removed calculate_and_log_rates
    # Removed log_training_results_async
    # Removed log_weight_update_event
    # Removed log_additional_stats

    def log_progress_eta(self):
        """Logs progress and ETA to console."""
        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]

        # Log less frequently to console
        if global_step == 0 or global_step % 100 != 0:
            return

        elapsed_time = time.time() - loop_state["start_time"]
        steps_since_start = global_step

        # --- Get rate from StatsProcessor (requires fetching or passing) ---
        # This part is tricky without direct access. For simplicity,
        # we might just use the average rate since start for the console log.
        # A more robust way would involve fetching the latest rate from the stats actor,
        # but that adds complexity back. Let's use the simple average for now.
        steps_per_sec = 0.0
        if elapsed_time > 1 and steps_since_start > 0:
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

        # Simplified console log - detailed rates come from StatsProcessor via MLflow/TB
        logger.info(
            f"Progress: {progress_str}, Episodes: {loop_state['episodes_played']:,}, Total Sims: {total_sims_str}, "
            f"Buffer: {loop_state['buffer_size']:,}/{loop_state['buffer_capacity']:,} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending_tasks}, Avg Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def validate_experiences(
        self, experiences: list[Experience]
    ) -> tuple[list[Experience], int]:
        """Validates the structure and content of experiences."""
        # This function remains largely the same
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

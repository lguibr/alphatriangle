import logging
import random
from collections import deque

import numpy as np

from ...config import TrainConfig
from ...utils.sumtree import SumTree
from ...utils.types import (
    Experience,
    ExperienceBatch,
    PERBatchSample,
)

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience Replay Buffer storing (StateType, PolicyTarget, Value).
    Supports both uniform sampling and Prioritized Experience Replay (PER)
    based on TrainConfig.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.capacity = config.BUFFER_CAPACITY
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN
        self.use_per = config.USE_PER

        if self.use_per:
            self.tree = SumTree(self.capacity)
            self.per_alpha = config.PER_ALPHA
            self.per_beta_initial = config.PER_BETA_INITIAL
            self.per_beta_final = config.PER_BETA_FINAL
            # Ensure anneal steps is at least 1 to avoid division by zero
            self.per_beta_anneal_steps = max(
                1, config.PER_BETA_ANNEAL_STEPS or config.MAX_TRAINING_STEPS or 1
            )
            self.per_epsilon = config.PER_EPSILON
            logger.info(
                f"Experience buffer initialized with PER (alpha={self.per_alpha}, beta_init={self.per_beta_initial}). Capacity: {self.capacity}"
            )
        else:
            self.buffer: deque[Experience] = deque(maxlen=self.capacity)
            logger.info(
                f"Experience buffer initialized with uniform sampling. Capacity: {self.capacity}"
            )

    def _get_priority(self, error: float) -> float:
        """Calculates priority from TD error."""
        # Ensure return type is float
        return float((np.abs(error) + self.per_epsilon) ** self.per_alpha)

    def add(self, experience: Experience):
        """Adds a single experience. Uses max priority if PER is enabled."""
        if self.use_per:
            max_p = self.tree.max_priority
            self.tree.add(max_p, experience)
        else:
            self.buffer.append(experience)

    def add_batch(self, experiences: list[Experience]):
        """Adds a batch of experiences. Uses max priority if PER is enabled."""
        if self.use_per:
            max_p = self.tree.max_priority
            for exp in experiences:
                self.tree.add(max_p, exp)
        else:
            self.buffer.extend(experiences)

    def _calculate_beta(self, current_step: int) -> float:
        """Linearly anneals beta from initial to final value."""
        fraction = min(1.0, current_step / self.per_beta_anneal_steps)
        beta = self.per_beta_initial + fraction * (
            self.per_beta_final - self.per_beta_initial
        )
        return beta

    def sample(
        self, batch_size: int, current_train_step: int | None = None
    ) -> PERBatchSample | None:
        """
        Samples a batch of experiences.
        Uses prioritized sampling if PER is enabled, otherwise uniform.
        Requires current_train_step if PER is enabled to calculate beta.
        """
        current_size = len(self)
        if current_size < batch_size or current_size < self.min_size_to_train:
            return None

        if self.use_per:
            if current_train_step is None:
                raise ValueError("current_train_step is required for PER sampling.")

            batch: ExperienceBatch = []
            idxs = np.empty((batch_size,), dtype=np.int32)
            is_weights = np.empty((batch_size,), dtype=np.float32)
            beta = self._calculate_beta(current_train_step)

            priority_segment = self.tree.total_priority / batch_size
            max_weight = 0.0

            for i in range(batch_size):
                a = priority_segment * i
                b = priority_segment * (i + 1)
                value = random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(value)

                if not isinstance(data, tuple):
                    logger.warning(
                        f"PER sampling encountered non-experience data at index {idx}. Resampling."
                    )
                    # Resample with a random value across the entire range
                    value = random.uniform(0, self.tree.total_priority)
                    idx, p, data = self.tree.get_leaf(value)
                    if not isinstance(data, tuple):
                        logger.error(f"PER resampling failed. Skipping sample {i}.")
                        # Fallback: sample a random valid index if possible
                        if self.tree.n_entries > 0:
                            rand_data_idx = random.randint(0, self.tree.n_entries - 1)
                            rand_tree_idx = rand_data_idx + self.capacity - 1
                            idx, p, data = self.tree.get_leaf(
                                self.tree.tree[rand_tree_idx]
                            )
                            if not isinstance(data, tuple):
                                continue  # Give up on this sample if fallback fails
                        else:
                            continue  # Cannot sample if tree is empty

                sampling_prob = p / self.tree.total_priority
                weight = (
                    (current_size * sampling_prob) ** (-beta)
                    if sampling_prob > 1e-9
                    else 0.0
                )
                is_weights[i] = weight
                max_weight = max(max_weight, weight)
                idxs[i] = idx
                batch.append(data)

            if max_weight > 1e-9:
                is_weights /= max_weight
            else:
                logger.warning(
                    "Max importance sampling weight is near zero. Weights might be invalid."
                )
                is_weights.fill(1.0)

            return {"batch": batch, "indices": idxs, "weights": is_weights}

        else:
            uniform_batch = random.sample(self.buffer, batch_size)
            dummy_indices = np.zeros(batch_size, dtype=np.int32)
            uniform_weights = np.ones(batch_size, dtype=np.float32)
            return {
                "batch": uniform_batch,
                "indices": dummy_indices,
                "weights": uniform_weights,
            }

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Updates the priorities of sampled experiences based on TD errors."""
        if not self.use_per:
            return

        if len(tree_indices) != len(td_errors):
            logger.error(
                f"Mismatch between tree_indices ({len(tree_indices)}) and td_errors ({len(td_errors)}) lengths."
            )
            return

        # Calculate priorities for each error
        priorities = np.array([self._get_priority(err) for err in td_errors])

        if not np.all(np.isfinite(priorities)):
            logger.warning("Non-finite priorities calculated. Clamping.")
            priorities = np.nan_to_num(
                priorities,
                nan=self.per_epsilon,
                posinf=self.tree.max_priority,
                neginf=self.per_epsilon,
            )
            priorities = np.maximum(priorities, self.per_epsilon)

        # Use strict=False for zip, although lengths should match after check above
        for idx, p in zip(tree_indices, priorities, strict=False):
            if not (0 <= idx < len(self.tree.tree)):
                logger.error(f"Invalid tree index {idx} provided for priority update.")
                continue
            self.tree.update(idx, p)

        # Update the overall max priority tracked by the tree
        if len(priorities) > 0:
            self.tree._max_priority = max(self.tree.max_priority, np.max(priorities))

    def __len__(self) -> int:
        """Returns the current number of experiences in the buffer."""
        return self.tree.n_entries if self.use_per else len(self.buffer)

    def is_ready(self) -> bool:
        """Checks if the buffer has enough samples to start training."""
        return len(self) >= self.min_size_to_train

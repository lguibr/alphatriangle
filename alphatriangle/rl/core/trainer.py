import logging
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig

# Keep alphatriangle imports
from ...config import TrainConfig
from ...nn import NeuralNetwork
from ...utils.types import ExperienceBatch, PERBatchSample

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the neural network training process, including loss calculation
    and optimizer steps. Supports Distributional RL (C51) value loss.
    """

    def __init__(
        self,
        nn_interface: NeuralNetwork,
        train_config: TrainConfig,
        env_config: EnvConfig,
    ):
        self.nn = nn_interface
        self.model = nn_interface.model
        self.train_config = train_config
        self.env_config = env_config
        self.model_config = nn_interface.model_config
        self.device = nn_interface.device
        self.optimizer = self._create_optimizer()
        self.scheduler: _LRScheduler | None = self._create_scheduler(self.optimizer)

        self.num_atoms = self.nn.num_atoms
        self.v_min = self.nn.v_min
        self.v_max = self.nn.v_max
        self.delta_z = self.nn.delta_z
        self.support = self.nn.support.to(self.device)

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer based on TrainConfig."""
        lr = self.train_config.LEARNING_RATE
        wd = self.train_config.WEIGHT_DECAY
        params = self.model.parameters()
        opt_type = self.train_config.OPTIMIZER_TYPE.lower()
        logger.info(f"Creating optimizer: {opt_type}, LR: {lr}, WD: {wd}")
        if opt_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}"
            )

    def _create_scheduler(self, optimizer: optim.Optimizer) -> _LRScheduler | None:
        """Creates the learning rate scheduler based on TrainConfig."""
        scheduler_type_config = self.train_config.LR_SCHEDULER_TYPE
        scheduler_type: str | None = None
        if scheduler_type_config:
            scheduler_type = scheduler_type_config.lower()

        if not scheduler_type or scheduler_type == "none":
            logger.info("No LR scheduler configured.")
            return None

        logger.info(f"Creating LR scheduler: {scheduler_type}")
        if scheduler_type == "steplr":
            step_size = getattr(self.train_config, "LR_SCHEDULER_STEP_SIZE", 100000)
            gamma = getattr(self.train_config, "LR_SCHEDULER_GAMMA", 0.1)
            logger.info(f"  StepLR params: step_size={step_size}, gamma={gamma}")
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            )
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            if t_max is None:
                logger.warning(
                    "LR_SCHEDULER_T_MAX is None for CosineAnnealingLR. Scheduler might not work as expected."
                )
                t_max = self.train_config.MAX_TRAINING_STEPS or 1_000_000
            logger.info(f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}")
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                ),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type_config}")

    def _prepare_batch(
        self, batch: ExperienceBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a batch of experiences into tensors.
        The 4th tensor is now the n-step return G (scalar).
        """
        batch_size = len(batch)
        grids = []
        other_features = []
        n_step_returns = []
        action_dim_int = int(
            self.env_config.NUM_SHAPE_SLOTS
            * self.env_config.ROWS
            * self.env_config.COLS
        )
        policy_target_tensor = torch.zeros(
            (batch_size, action_dim_int),
            dtype=torch.float32,
            device=self.device,
        )

        for i, (state_features, policy_target_map, n_step_return) in enumerate(batch):
            grids.append(state_features["grid"])
            other_features.append(state_features["other_features"])
            n_step_returns.append(n_step_return)
            for action, prob in policy_target_map.items():
                if 0 <= action < action_dim_int:
                    policy_target_tensor[i, action] = prob
                else:
                    logger.warning(
                        f"Action {action} out of bounds in policy target map for sample {i}."
                    )

        grid_tensor = torch.from_numpy(np.stack(grids)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(other_features)).to(
            self.device
        )
        n_step_return_tensor = torch.tensor(
            n_step_returns, dtype=torch.float32, device=self.device
        )

        expected_other_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if batch_size > 0 and other_features_tensor.shape[1] != expected_other_dim:
            raise ValueError(
                f"Unexpected other_features tensor shape: {other_features_tensor.shape}, expected dim {expected_other_dim}"
            )

        return (
            grid_tensor,
            other_features_tensor,
            policy_target_tensor,
            n_step_return_tensor,
        )

    def _calculate_target_distribution(
        self, n_step_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Projects the n-step returns onto the fixed support atoms (z).
        Args:
            n_step_returns: Tensor of shape (batch_size,) containing scalar n-step returns (G).
        Returns:
            Tensor of shape (batch_size, num_atoms) representing the target distribution.
        """
        batch_size = n_step_returns.size(0)
        m = torch.zeros(
            (batch_size, self.num_atoms), dtype=torch.float32, device=self.device
        )
        target_returns = n_step_returns.clamp(self.v_min, self.v_max)
        b = (target_returns - self.v_min) / self.delta_z
        lower_idx = b.floor().long()
        u = b.ceil().long()
        # Ensure indices are within bounds [0, num_atoms - 1]
        lower_idx = torch.clamp(lower_idx, 0, self.num_atoms - 1)
        u = torch.clamp(u, 0, self.num_atoms - 1)

        # Handle cases where lower and upper indices are the same
        eq_mask = lower_idx == u
        ne_mask = ~eq_mask

        # Calculate weights for non-equal indices
        m_l = u[ne_mask].float() - b[ne_mask]
        m_u = b[ne_mask] - lower_idx[ne_mask].float()

        # Distribute probability mass
        batch_indices = torch.arange(batch_size, device=self.device)
        m[batch_indices[ne_mask], lower_idx[ne_mask]] += m_l
        m[batch_indices[ne_mask], u[ne_mask]] += m_u

        # Handle equal indices (assign full probability mass)
        m[batch_indices[eq_mask], lower_idx[eq_mask]] += 1.0

        # Normalize rows just in case of floating point issues near boundaries
        # This might not be strictly necessary but adds robustness
        # row_sums = m.sum(dim=1, keepdim=True)
        # m = m / (row_sums + 1e-9) # Add epsilon for stability

        return m

    def train_step(
        self, per_sample: PERBatchSample
    ) -> tuple[dict[str, float], np.ndarray] | None:
        """
        Performs a single training step on the given batch from PER buffer.
        Uses distributional cross-entropy loss for the value head.
        Returns raw loss info dictionary and TD errors for priority updates.
        """
        batch = per_sample["batch"]
        is_weights = per_sample["weights"]

        if not batch:
            logger.warning("train_step called with empty batch.")
            return None

        self.model.train()
        try:
            grid_t, other_t, policy_target_t, n_step_return_t = self._prepare_batch(
                batch
            )
            is_weights_t = torch.from_numpy(is_weights).to(self.device)
        except Exception as e:
            logger.error(f"Error preparing batch for training: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        policy_logits, value_logits = self.model(grid_t, other_t)

        # --- Value Loss (Distributional Cross-Entropy) ---
        with torch.no_grad():
            target_distribution = self._calculate_target_distribution(n_step_return_t)
        log_pred_dist = F.log_softmax(value_logits, dim=1)
        # Ensure target_distribution sums to 1 for valid cross-entropy calculation
        # target_distribution = target_distribution / (target_distribution.sum(dim=1, keepdim=True) + 1e-9)
        value_loss_elementwise = -torch.sum(target_distribution * log_pred_dist, dim=1)
        value_loss = (value_loss_elementwise * is_weights_t).mean()

        # --- Policy Loss (Cross-Entropy) ---
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_target_t = torch.nan_to_num(policy_target_t, nan=0.0)
        # Ensure policy target sums to 1
        # policy_target_t = policy_target_t / (policy_target_t.sum(dim=1, keepdim=True) + 1e-9)
        policy_loss_elementwise = -torch.sum(policy_target_t * log_probs, dim=1)
        policy_loss = (policy_loss_elementwise * is_weights_t).mean()

        # --- Entropy Bonus ---
        entropy_scalar: float = 0.0
        entropy_loss_term = torch.tensor(0.0, device=self.device)
        if self.train_config.ENTROPY_BONUS_WEIGHT > 0:
            policy_probs = F.softmax(policy_logits, dim=1)
            # Add epsilon for numerical stability in log
            entropy_term_elementwise: torch.Tensor = -torch.sum(
                policy_probs * torch.log(policy_probs + 1e-9), dim=1
            )
            # Calculate mean entropy across batch BEFORE applying weights
            entropy_scalar = float(entropy_term_elementwise.mean().item())
            # Apply entropy bonus loss (weighted mean if needed, but usually mean is fine)
            entropy_loss_term = (
                -self.train_config.ENTROPY_BONUS_WEIGHT
                * entropy_term_elementwise.mean()  # Use mean entropy, not weighted
            )

        # --- Total Loss ---
        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + entropy_loss_term
        )

        # --- Backpropagation and Optimization ---
        total_loss.backward()

        if (
            self.train_config.GRADIENT_CLIP_VALUE is not None
            and self.train_config.GRADIENT_CLIP_VALUE > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # --- Calculate TD Errors for PER ---
        with torch.no_grad():
            # Use the element-wise value loss as the TD error proxy for distributional RL
            # This is a common heuristic. Alternatively, calculate expected value difference.
            # Using value_loss_elementwise directly aligns priorities with the loss being minimized.
            td_errors = value_loss_elementwise.detach().cpu().numpy()
            # Alternative: Calculate expected value difference
            # expected_value_pred = self.nn._logits_to_expected_value(value_logits)
            # td_errors = (n_step_return_t - expected_value_pred.squeeze(1)).detach().cpu().numpy()

        # --- Prepare Raw Loss Info ---
        # Return individual loss components for logging by the stats system
        loss_info = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_scalar,  # Report the mean entropy scalar
            "mean_td_error": float(
                np.mean(np.abs(td_errors))
            ),  # Keep reporting mean TD error
        }

        return loss_info, td_errors

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer."""
        try:
            # Ensure optimizer and param_groups exist
            if self.optimizer and self.optimizer.param_groups:
                return float(self.optimizer.param_groups[0]["lr"])
            else:
                logger.warning("Optimizer or param_groups not available.")
                return 0.0
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(f"Could not retrieve learning rate from optimizer: {e}")
            return 0.0

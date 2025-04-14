import logging
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from ...config import EnvConfig, TrainConfig
from ...nn import NeuralNetwork
from ...utils.types import (
    ExperienceBatch,
    PERBatchSample,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the neural network training process, including loss calculation and optimizer steps."""

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
            # Cast return type
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
        Converts a batch of experiences (containing StateType) into tensors.
        No feature extraction needed here anymore.
        """
        batch_size = len(batch)
        grids = []
        other_features = []
        value_targets = []
        action_dim_int = int(self.env_config.ACTION_DIM)  # type: ignore[call-overload]
        policy_target_tensor = torch.zeros(
            (batch_size, action_dim_int),
            dtype=torch.float32,
            device=self.device,
        )

        for i, (state_features, policy_target_map, value_target) in enumerate(batch):
            grids.append(state_features["grid"])
            other_features.append(state_features["other_features"])
            value_targets.append(value_target)
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
        value_target_tensor = torch.tensor(
            value_targets, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        expected_other_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if batch_size > 0 and other_features_tensor.shape[1] != expected_other_dim:
            raise ValueError(
                f"Unexpected other_features tensor shape: {other_features_tensor.shape}, expected dim {expected_other_dim}"
            )

        return (
            grid_tensor,
            other_features_tensor,
            policy_target_tensor,
            value_target_tensor,
        )

    def train_step(
        self, per_sample: PERBatchSample
    ) -> tuple[dict[str, float], np.ndarray] | None:
        """
        Performs a single training step on the given batch from PER buffer.
        Returns loss info dictionary and TD errors for priority updates.
        """
        batch = per_sample["batch"]
        is_weights = per_sample["weights"]

        if not batch:
            logger.warning("train_step called with empty batch.")
            return None

        self.model.train()
        try:
            grid_t, other_t, policy_target_t, value_target_t = self._prepare_batch(
                batch
            )
            is_weights_t = torch.from_numpy(is_weights).to(self.device)
        except Exception as e:
            logger.error(f"Error preparing batch for training: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        policy_logits, value_pred = self.model(grid_t, other_t)

        value_loss_elementwise = F.mse_loss(
            value_pred, value_target_t, reduction="none"
        )
        value_loss = (value_loss_elementwise * is_weights_t.unsqueeze(1)).mean()

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_target_t = torch.nan_to_num(policy_target_t, nan=0.0)
        policy_loss_elementwise = -torch.sum(policy_target_t * log_probs, dim=1)
        policy_loss = (policy_loss_elementwise * is_weights_t).mean()

        entropy_scalar: float = 0.0  # Initialize as float
        entropy_loss_term = torch.tensor(
            0.0, device=self.device
        )  # Initialize as tensor
        if self.train_config.ENTROPY_BONUS_WEIGHT > 0:
            policy_probs = F.softmax(policy_logits, dim=1)
            # Calculate entropy term: -Sum(p * log(p))
            entropy_term_elementwise: torch.Tensor = -torch.sum(
                policy_probs * torch.log(policy_probs + 1e-9), dim=1
            )
            # Calculate mean entropy across batch for logging
            entropy_scalar = float(
                entropy_term_elementwise.mean().item()
            )  # Cast result to float
            # Calculate the loss term (negative entropy bonus)
            entropy_loss_term = (
                -self.train_config.ENTROPY_BONUS_WEIGHT
                * entropy_term_elementwise.mean()
            )

        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + entropy_loss_term  # Use the calculated term
        )

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

        td_errors = value_loss_elementwise.squeeze(1).detach().cpu().numpy()

        loss_info = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_scalar,
            "mean_td_error": float(np.mean(np.abs(td_errors))),
        }

        return loss_info, td_errors

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer."""
        try:
            # Ensure return type is float
            return float(self.optimizer.param_groups[0]["lr"])
        except (IndexError, KeyError):
            logger.warning("Could not retrieve learning rate from optimizer.")
            return 0.0

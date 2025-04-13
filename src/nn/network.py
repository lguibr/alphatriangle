import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Mapping, Dict, Any

from src.config import ModelConfig, EnvConfig, TrainConfig
from src.environment import GameState
from src.utils.types import ActionType, PolicyValueOutput, StateType
from .model import AlphaTriangleNet
from src.features import extract_state_features

logger = logging.getLogger(__name__)


class NetworkEvaluationError(Exception):
    """Custom exception for errors during network evaluation."""

    pass


class NeuralNetwork:
    """Wrapper for the PyTorch model providing evaluation and state management."""

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,
        train_config: TrainConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.env_config = env_config
        self.train_config = train_config
        self.device = device
        self.model = AlphaTriangleNet(model_config, env_config).to(device)
        self.action_dim = env_config.ACTION_DIM
        self.model.eval()

    def _state_to_tensors(self, state: GameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from GameState and converts them to tensors."""
        state_dict: StateType = extract_state_features(state, self.model_config)
        grid_tensor = torch.from_numpy(state_dict["grid"]).unsqueeze(0).to(self.device)
        other_features_tensor = (
            torch.from_numpy(state_dict["other_features"]).unsqueeze(0).to(self.device)
        )
        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError(
                f"Non-finite values found in input grid_tensor for state {state}"
            )
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                f"Non-finite values found in input other_features_tensor for state {state}"
            )
        return grid_tensor, other_features_tensor

    def _batch_states_to_tensors(
        self, states: List[GameState]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from a batch of GameStates and converts to batched tensors."""
        if not states:
            grid_shape = (
                0,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            )
            other_shape = (0, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
            return torch.empty(grid_shape, device=self.device), torch.empty(
                other_shape, device=self.device
            )

        batch_grid = []
        batch_other = []
        for state in states:
            state_dict: StateType = extract_state_features(state, self.model_config)
            batch_grid.append(state_dict["grid"])
            batch_other.append(state_dict["other_features"])

        grid_tensor = torch.from_numpy(np.stack(batch_grid)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(batch_other)).to(self.device)

        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values found in batched input grid_tensor"
            )
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values found in batched input other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """Evaluates a single state. Raises NetworkEvaluationError on issues."""
        self.model.eval()
        try:
            grid_tensor, other_features_tensor = self._state_to_tensors(state)
            policy_logits, value = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected for state {state}. Logits: {policy_logits}"
                )
            if not torch.all(torch.isfinite(value)):
                raise NetworkEvaluationError(
                    f"Non-finite value detected for state {state}: {value.item()}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax for state {state}. Logits were: {policy_logits}"
                )

            policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
            policy_probs = np.maximum(policy_probs, 0)
            prob_sum = np.sum(policy_probs)
            if abs(prob_sum - 1.0) > 1e-5:
                logger.warning(
                    f"Evaluate: Policy probabilities sum to {prob_sum:.6f} (not 1.0) for state {state.current_step}. Re-normalizing."
                )
                if prob_sum <= 1e-9:
                    raise NetworkEvaluationError(
                        f"Policy probability sum is near zero ({prob_sum}) for state {state.current_step}. Cannot normalize."
                    )
                policy_probs /= prob_sum

            value_scalar = value.squeeze(0).item()

            action_policy: Mapping[ActionType, float] = {
                i: float(p) for i, p in enumerate(policy_probs)
            }

            num_non_zero = sum(1 for p in action_policy.values() if p > 1e-6)
            logger.debug(
                f"Evaluate Final Policy Dict (State {state.current_step}): {num_non_zero}/{self.action_dim} non-zero probs. Example: {list(action_policy.items())[:5]}"
            )

            return action_policy, value_scalar

        except Exception as e:
            logger.error(
                f"Exception during single evaluation for state {state}: {e}",
                exc_info=True,
            )
            raise NetworkEvaluationError(
                f"Evaluation failed for state {state}: {e}"
            ) from e

    @torch.inference_mode()
    def evaluate_batch(self, states: List[GameState]) -> List[PolicyValueOutput]:
        """Evaluates a batch of states. Raises NetworkEvaluationError on issues."""
        if not states:
            return []
        self.model.eval()
        logger.debug(f"Evaluating batch of {len(states)} states...")
        try:
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)
            policy_logits, value = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected in batch evaluation. Logits shape: {policy_logits.shape}"
                )
            if not torch.all(torch.isfinite(value)):
                raise NetworkEvaluationError(
                    f"Non-finite values detected in batch value output. Value shape: {value.shape}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax in batch. Logits shape: {policy_logits.shape}"
                )

            policy_probs = policy_probs_tensor.cpu().numpy()
            values = value.squeeze(1).cpu().numpy()

            results: List[PolicyValueOutput] = []
            for batch_idx in range(len(states)):
                probs_i = np.maximum(policy_probs[batch_idx], 0)
                prob_sum_i = np.sum(probs_i)
                if abs(prob_sum_i - 1.0) > 1e-5:
                    logger.warning(
                        f"EvaluateBatch: Policy probabilities sum to {prob_sum_i:.6f} (not 1.0) for sample {batch_idx}. Re-normalizing."
                    )
                    if prob_sum_i <= 1e-9:
                        raise NetworkEvaluationError(
                            f"Policy probability sum is near zero ({prob_sum_i}) for batch sample {batch_idx}. Cannot normalize."
                        )
                    probs_i /= prob_sum_i

                policy_i: Mapping[ActionType, float] = {
                    i: float(p) for i, p in enumerate(probs_i)
                }
                value_i = float(values[batch_idx])
                results.append((policy_i, value_i))

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(f"Batch evaluation failed: {e}") from e

        logger.debug(f"  Batch evaluation finished. Returning {len(results)} results.")
        return results

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Loads the model's state dictionary from the provided weights."""
        try:
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            self.model.load_state_dict(weights_on_device)
            self.model.eval()
            logger.debug("NN weights set successfully.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise

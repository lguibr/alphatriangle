# File: alphatriangle/nn/network.py
import logging
import sys  # Import sys for platform check
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..config import EnvConfig, ModelConfig, TrainConfig
from ..environment import GameState
from ..features import extract_state_features
from ..utils.types import ActionType, PolicyValueOutput, StateType
from .model import AlphaTriangleNet

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class NetworkEvaluationError(Exception):
    """Custom exception for errors during network evaluation."""

    pass


class NeuralNetwork:
    """
    Wrapper for the PyTorch model providing evaluation and state management.
    Handles distributional value head (C51) by calculating expected value for MCTS.
    Optionally compiles the model using torch.compile().
    """

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

        self.num_atoms = model_config.NUM_VALUE_ATOMS
        self.v_min = model_config.VALUE_MIN
        self.v_max = model_config.VALUE_MAX
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_atoms, device=self.device
        )

        # --- ADDED: Check for Windows/MPS before attempting compile ---
        if self.train_config.COMPILE_MODEL:
            # --- ADDED: Skip compilation entirely on Windows due to Triton dependency ---
            if sys.platform == "win32":
                logger.warning(
                    "Model compilation requested but running on Windows. "
                    "Skipping torch.compile() as the default CUDA backend (Inductor) requires Triton, "
                    "which is not officially supported on Windows. Proceeding with eager execution."
                )
            # --- END ADDED ---
            elif self.device.type == "mps":
                logger.warning(
                    "Model compilation requested but device is 'mps'. "
                    "Skipping torch.compile() due to known compatibility issues with this backend. "
                    "Proceeding with eager execution."
                )
            elif hasattr(torch, "compile"):
                try:
                    logger.info(
                        f"Attempting to compile model with torch.compile() on device '{self.device}'..."
                    )
                    self.model = torch.compile(self.model)  # type: ignore
                    logger.info(
                        f"Model compiled successfully on device '{self.device}'."
                    )
                except Exception as e:
                    logger.warning(
                        f"torch.compile() failed on device '{self.device}': {e}. "
                        f"Proceeding without compilation (using eager mode). "
                        f"Compilation might not be supported for this model/backend combination.",
                        exc_info=False,
                    )
            else:
                logger.warning(
                    "torch.compile() requested but not available (requires PyTorch 2.0+). Proceeding without compilation."
                )
        else:
            logger.info(
                "Model compilation skipped (COMPILE_MODEL=False in TrainConfig)."
            )
        # --- END ADDED ---

    def _state_to_tensors(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
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
        self, states: list[GameState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _logits_to_expected_value(self, value_logits: torch.Tensor) -> torch.Tensor:
        """Calculates the expected value from the value distribution logits."""
        value_probs = F.softmax(value_logits, dim=1)
        # Expand support to match batch size for broadcasting
        support_expanded = self.support.expand_as(value_probs)
        expected_value = torch.sum(value_probs * support_expanded, dim=1, keepdim=True)
        return expected_value

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """
        Evaluates a single state.
        Returns policy mapping and EXPECTED value from the distribution.
        Raises NetworkEvaluationError on issues.
        """
        self.model.eval()
        try:
            grid_tensor, other_features_tensor = self._state_to_tensors(state)
            policy_logits, value_logits = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected for state {state}. Logits: {policy_logits}"
                )
            if not torch.all(torch.isfinite(value_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite value_logits detected for state {state}: {value_logits}"
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

            expected_value_tensor = self._logits_to_expected_value(value_logits)
            expected_value_scalar = expected_value_tensor.squeeze(
                0
            ).item()  # Squeeze batch and atom dim, get scalar

            action_policy: Mapping[ActionType, float] = {
                i: float(p) for i, p in enumerate(policy_probs)
            }

            num_non_zero = sum(1 for p in action_policy.values() if p > 1e-6)
            logger.debug(
                f"Evaluate Final Policy Dict (State {state.current_step}): {num_non_zero}/{self.action_dim} non-zero probs. Example: {list(action_policy.items())[:5]}"
            )

            return action_policy, expected_value_scalar

        except Exception as e:
            logger.error(
                f"Exception during single evaluation for state {state}: {e}",
                exc_info=True,
            )
            raise NetworkEvaluationError(
                f"Evaluation failed for state {state}: {e}"
            ) from e

    @torch.inference_mode()
    def evaluate_batch(self, states: list[GameState]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of states.
        Returns a list of (policy mapping, EXPECTED value).
        Raises NetworkEvaluationError on issues.
        """
        if not states:
            return []
        self.model.eval()
        logger.debug(f"Evaluating batch of {len(states)} states...")
        try:
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)
            policy_logits, value_logits = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected in batch evaluation. Logits shape: {policy_logits.shape}"
                )
            if not torch.all(torch.isfinite(value_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite value_logits detected in batch value output. Value shape: {value_logits.shape}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax in batch. Logits shape: {policy_logits.shape}"
                )

            policy_probs = policy_probs_tensor.cpu().numpy()
            expected_values_tensor = self._logits_to_expected_value(value_logits)
            expected_values = (
                expected_values_tensor.squeeze(1).cpu().numpy()
            )  # Squeeze the atom dim

            results: list[PolicyValueOutput] = []
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
                value_i = float(expected_values[batch_idx])  # This is now a scalar
                results.append((policy_i, value_i))

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(f"Batch evaluation failed: {e}") from e

        logger.debug(f"  Batch evaluation finished. Returning {len(results)} results.")
        return results

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        # If model is compiled, access the original model for state_dict
        model_to_save = getattr(self.model, "_orig_mod", self.model)
        return {k: v.cpu() for k, v in model_to_save.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]):
        """Loads the model's state dictionary from the provided weights."""
        try:
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            # If model is compiled, load into the original model
            model_to_load = getattr(self.model, "_orig_mod", self.model)
            model_to_load.load_state_dict(weights_on_device)
            self.model.eval()  # Ensure the main (potentially compiled) model is in eval mode
            logger.debug("NN weights set successfully.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise

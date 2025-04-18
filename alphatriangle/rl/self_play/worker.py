# File: alphatriangle/rl/self_play/worker.py
import logging
import random
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import ray
import torch  # Import torch

from ...config import MCTSConfig, ModelConfig, TrainConfig
from ...environment import EnvConfig, GameState
from ...features import extract_state_features
from ...mcts import (
    MCTSExecutionError,
    Node,
    get_policy_target,
    run_mcts_simulations,
    select_action_based_on_visits,
)
from ...nn import NeuralNetwork
from ...utils import get_device, set_random_seeds

# --- REMOVED: Type imports moved below ---
# from ...utils.types import Experience, PolicyTargetMapping, StateType, StepInfo
# --- END REMOVED ---

if TYPE_CHECKING:
    from ...stats import StatsCollectorActor

    # --- ADDED: Type imports moved here ---
    from ...utils.types import Experience, PolicyTargetMapping, StateType, StepInfo

    # --- END ADDED ---


from ..types import SelfPlayResult

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running self-play episodes using MCTS and a NN.
    Implements MCTS tree reuse between steps.
    Stores extracted features (StateType) and the N-STEP RETURN in the experience buffer.
    Returns a SelfPlayResult Pydantic model including aggregated stats.
    Reports current state and step stats asynchronously using StepInfo including game_step and trainer_step.
    """

    def __init__(
        self,
        actor_id: int,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        stats_collector_actor: "StatsCollectorActor",
        initial_weights: dict | None = None,
        seed: int | None = None,
        worker_device_str: str = "cpu",
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.stats_collector_actor = stats_collector_actor
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str

        # --- N-Step Config ---
        self.n_step = self.train_config.N_STEP_RETURNS
        self.gamma = self.train_config.GAMMA

        # Store the global step of the current weights
        self.current_trainer_step = 0

        # Configure logging for the worker process
        worker_log_level = logging.INFO
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        global logger
        logger = logging.getLogger(__name__)

        mcts_log_level = logging.WARNING
        nn_log_level = logging.WARNING
        logging.getLogger("alphatriangle.mcts").setLevel(mcts_log_level)
        logging.getLogger("alphatriangle.nn").setLevel(nn_log_level)

        set_random_seeds(self.seed)

        self.device = get_device(self.worker_device_str)

        if self.device.type == "cuda":
            try:
                torch.cuda.set_device(self.device)
                logger.info(
                    f"Successfully set default CUDA device for worker {self.actor_id} to {self.device} (Index: {torch.cuda.current_device()})."
                )
                count = torch.cuda.device_count()
                if count != 1:
                    logger.warning(
                        f"Worker {self.actor_id} sees {count} CUDA devices, expected 1 after Ray assignment. This might indicate an issue."
                    )
                else:
                    logger.info(
                        f"Worker {self.actor_id} sees 1 CUDA device as expected."
                    )

            except Exception as cuda_set_err:
                logger.error(
                    f"Failed to set default CUDA device for worker {self.actor_id} to {self.device}: {cuda_set_err}. "
                    f"Compilation or CUDA operations might fail.",
                    exc_info=True,
                )

        self.nn_evaluator = NeuralNetwork(
            model_config=self.model_config,
            env_config=self.env_config,
            train_config=self.train_config,
            device=self.device,
        )

        if initial_weights:
            self.set_weights(initial_weights)
        else:
            self.nn_evaluator.model.eval()

        logger.debug(f"INIT: MCTS Config: {self.mcts_config.model_dump()}")
        logger.info(
            f"Worker initialized on device {self.device}. Seed: {self.seed}. LogLevel: {logging.getLevelName(logger.getEffectiveLevel())}"
        )
        logger.debug("Worker init complete.")

    def set_weights(self, weights: dict):
        """Updates the neural network weights."""
        try:
            # Removed attempt to get step from weights dict
            self.nn_evaluator.set_weights(weights)
            logger.debug("Weights updated.")
        except Exception as e:
            logger.error(f"Failed to set weights: {e}", exc_info=True)

    def set_current_trainer_step(self, global_step: int):
        """Sets the global step corresponding to the current network weights."""
        self.current_trainer_step = global_step
        logger.debug(f"Worker {self.actor_id} trainer step set to {global_step}")

    def _report_current_state(self, game_state: GameState):
        """Asynchronously sends the current game state to the collector."""
        if self.stats_collector_actor:
            try:
                state_copy = game_state.copy()
                self.stats_collector_actor.update_worker_game_state.remote(  # type: ignore
                    self.actor_id, state_copy
                )
                logger.debug(
                    f"Reported state step {state_copy.current_step} to collector."
                )
            except Exception as e:
                logger.error(f"Failed to report game state to collector: {e}")

    def _log_step_stats_async(
        self,
        game_state: GameState,
        mcts_visits: int,
        mcts_depth: int,
        step_reward: float,
    ):
        """
        Asynchronously logs per-step stats to the collector using StepInfo,
        including the current game_step_index and the stored current_trainer_step.
        """
        if self.stats_collector_actor:
            try:
                # Include current_trainer_step
                step_info: StepInfo = {
                    "game_step_index": game_state.current_step,
                    "global_step": self.current_trainer_step,  # Add trainer step context
                }
                step_stats: dict[str, tuple[float, StepInfo]] = {
                    "RL/Current_Score": (game_state.game_score, step_info),
                    "MCTS/Step_Visits": (float(mcts_visits), step_info),
                    "MCTS/Step_Depth": (float(mcts_depth), step_info),
                    "RL/Step_Reward": (step_reward, step_info),
                }
                logger.debug(f"Sending step stats to collector: {step_stats}")
                self.stats_collector_actor.log_batch.remote(step_stats)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to log step stats to collector: {e}")

    def run_episode(self) -> SelfPlayResult:
        """
        Runs a single episode of self-play using MCTS and the internal neural network.
        Implements MCTS tree reuse.
        Stores extracted features (StateType) and the N-STEP RETURN in the experience buffer.
        Returns a SelfPlayResult Pydantic model including aggregated stats.
        Reports current state and step stats asynchronously.
        """
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)

        if game.is_over():
            logger.error(
                f"Game is over immediately after reset with seed {episode_seed}. Returning empty result."
            )
            return SelfPlayResult(
                episode_experiences=[],
                final_score=0.0,
                episode_steps=0,
                total_simulations=0,
                avg_root_visits=0.0,
                avg_tree_depth=0.0,
            )

        n_step_state_policy_buffer: deque[tuple[StateType, PolicyTargetMapping]] = (
            deque(maxlen=self.n_step)
        )
        n_step_reward_buffer: deque[float] = deque(maxlen=self.n_step)
        episode_experiences: list[Experience] = []

        step_root_visits: list[int] = []
        step_tree_depths: list[int] = []
        step_simulations: list[int] = []

        logger.info(f"Starting episode with seed {episode_seed}")
        self._report_current_state(game)

        root_node: Node | None = Node(state=game.copy())

        while not game.is_over():
            step_start_time = time.monotonic()
            if root_node is None:
                logger.error(
                    "MCTS root node became None unexpectedly. Aborting episode."
                )
                break

            if root_node.state.is_over():
                logger.warning(
                    f"MCTS root node state (Step {root_node.state.current_step}) is already terminal before running simulations. Ending episode."
                )
                break

            logger.info(
                f"Step {game.current_step}: Running MCTS simulations ({self.mcts_config.num_simulations}) on state from step {root_node.state.current_step}..."
            )
            mcts_start_time = time.monotonic()
            mcts_max_depth = 0
            try:
                mcts_max_depth = run_mcts_simulations(
                    root_node, self.mcts_config, self.nn_evaluator
                )
            except MCTSExecutionError as mcts_err:
                logger.error(
                    f"Step {game.current_step}: MCTS simulation failed critically: {mcts_err}",
                    exc_info=False,
                )
                break
            except Exception as mcts_err:
                logger.error(
                    f"Step {game.current_step}: MCTS simulation failed unexpectedly: {mcts_err}",
                    exc_info=True,
                )
                break

            mcts_duration = time.monotonic() - mcts_start_time
            logger.info(
                f"Step {game.current_step}: MCTS finished ({mcts_duration:.3f}s). Max Depth: {mcts_max_depth}, Root Visits: {root_node.visit_count}"
            )

            # Log stats *before* taking the step
            self._log_step_stats_async(
                game, root_node.visit_count, mcts_max_depth, step_reward=0.0
            )

            action_selection_start_time = time.monotonic()
            temp = (
                self.mcts_config.temperature_initial
                if game.current_step < self.mcts_config.temperature_anneal_steps
                else self.mcts_config.temperature_final
            )
            try:
                policy_target = get_policy_target(root_node, temperature=1.0)
                action = select_action_based_on_visits(root_node, temperature=temp)
            except Exception as policy_err:
                logger.error(
                    f"Step {game.current_step}: MCTS policy/action selection failed: {policy_err}",
                    exc_info=True,
                )
                break

            action_selection_duration = time.monotonic() - action_selection_start_time

            logger.info(
                f"Step {game.current_step}: Selected Action {action} (Temp={temp:.3f}). Selection time: {action_selection_duration:.4f}s"
            )

            feature_start_time = time.monotonic()
            try:
                state_features: StateType = extract_state_features(
                    game, self.model_config
                )
            except Exception as e:
                logger.error(
                    f"Error extracting features at step {game.current_step}: {e}",
                    exc_info=True,
                )
                break

            feature_duration = time.monotonic() - feature_start_time
            logger.debug(
                f"Step {game.current_step}: Feature extraction time: {feature_duration:.4f}s"
            )

            n_step_state_policy_buffer.append((state_features, policy_target))

            step_simulations.append(self.mcts_config.num_simulations)
            step_root_visits.append(root_node.visit_count)
            step_tree_depths.append(mcts_max_depth)

            game_step_start_time = time.monotonic()
            step_reward = 0.0
            try:
                step_reward, done = game.step(action)
            except Exception as step_err:
                logger.error(
                    f"Error executing game step for action {action}: {step_err}",
                    exc_info=True,
                )
                break

            game_step_duration = time.monotonic() - game_step_start_time
            logger.info(
                f"Step {game.current_step}: Action {action} taken. Reward: {step_reward:.3f}, Done: {done}. Game step time: {game_step_duration:.4f}s"
            )

            n_step_reward_buffer.append(step_reward)

            if len(n_step_reward_buffer) == self.n_step:
                discounted_reward_sum = 0.0
                for i in range(self.n_step):
                    discounted_reward_sum += (self.gamma**i) * n_step_reward_buffer[i]

                bootstrap_value = 0.0
                if not done:
                    try:
                        _, bootstrap_value = self.nn_evaluator.evaluate(game)
                    except Exception as eval_err:
                        logger.error(
                            f"Error evaluating bootstrap state S_{game.current_step}: {eval_err}",
                            exc_info=True,
                        )
                        bootstrap_value = 0.0

                n_step_return = (
                    discounted_reward_sum + (self.gamma**self.n_step) * bootstrap_value
                )

                state_features_t_minus_n, policy_target_t_minus_n = (
                    n_step_state_policy_buffer[0]
                )

                episode_experiences.append(
                    (
                        state_features_t_minus_n,
                        policy_target_t_minus_n,
                        n_step_return,
                    )
                )

            self._report_current_state(game)
            # Log stats *after* taking the step
            self._log_step_stats_async(
                game,
                root_node.visit_count if root_node else 0,
                mcts_max_depth,
                step_reward=step_reward,
            )

            tree_reuse_start_time = time.monotonic()
            if not done:
                if root_node and action in root_node.children:  # Check root_node exists
                    root_node = root_node.children[action]
                    root_node.parent = None
                    logger.debug(
                        f"Reused MCTS subtree for action {action}. New root step: {root_node.state.current_step}"
                    )
                else:
                    logger.error(
                        f"Child node for selected action {action} not found in MCTS tree children: {list(root_node.children.keys()) if root_node else 'No Root'}. Resetting MCTS root to current game state."
                    )
                    root_node = Node(state=game.copy())
            else:
                root_node = None

            tree_reuse_duration = time.monotonic() - tree_reuse_start_time
            logger.debug(
                f"Step {game.current_step}: Tree reuse/reset time: {tree_reuse_duration:.4f}s"
            )

            step_duration = time.monotonic() - step_start_time
            logger.info(
                f"Step {game.current_step} total duration: {step_duration:.3f}s"
            )

            if done:
                break

        final_score = game.game_score
        logger.info(
            f"Episode finished. Final Score: {final_score:.2f}, Steps: {game.current_step}"
        )

        remaining_steps = len(n_step_reward_buffer)
        for k in range(remaining_steps):
            discounted_reward_sum = 0.0
            for i in range(remaining_steps - k):
                discounted_reward_sum += (self.gamma**i) * n_step_reward_buffer[k + i]

            n_step_return = discounted_reward_sum
            state_features_t, policy_target_t = n_step_state_policy_buffer[k]
            episode_experiences.append(
                (state_features_t, policy_target_t, n_step_return)
            )

        total_sims_episode = sum(step_simulations)
        avg_visits_episode = np.mean(step_root_visits) if step_root_visits else 0.0
        avg_depth_episode = np.mean(step_tree_depths) if step_tree_depths else 0.0

        if not episode_experiences:
            logger.warning(
                f"Episode finished with 0 experiences collected. Final score: {final_score}, Steps: {game.current_step}"
            )

        return SelfPlayResult(
            episode_experiences=episode_experiences,
            final_score=final_score,
            episode_steps=game.current_step,
            total_simulations=total_sims_episode,
            avg_root_visits=float(avg_visits_episode),
            avg_tree_depth=float(avg_depth_episode),
        )

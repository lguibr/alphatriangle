# File: src/rl/self_play/worker.py
import logging
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import ray

from src.config import MCTSConfig, ModelConfig, TrainConfig
from src.environment import EnvConfig, GameState
from src.features import extract_state_features
from src.mcts import (
    MCTSExecutionError,
    Node,
    get_policy_target,
    run_mcts_simulations,
    select_action_based_on_visits,
)
from src.nn import NeuralNetwork
from src.utils import get_device, set_random_seeds

# Move application type imports into TYPE_CHECKING block
if TYPE_CHECKING:
    from src.stats import StatsCollectorActor  # Import for type hinting
    from src.utils.types import Experience, PolicyTargetMapping, StateType

from ..types import SelfPlayResult

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running self-play episodes using MCTS and a NN.
    Implements MCTS tree reuse between steps.
    Stores extracted features (StateType) in the experience buffer.
    Returns a SelfPlayResult Pydantic model including aggregated stats.
    Asynchronously reports its current game state and per-step stats
    to the StatsCollectorActor.
    """

    def __init__(
        self,
        actor_id: int,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        # Add stats_collector_actor handle
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
        # Store the handle to the stats collector
        self.stats_collector_actor = stats_collector_actor
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str

        # Configure logging for the worker process
        worker_log_level = logging.INFO
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        global logger
        logger = logging.getLogger(__name__)

        mcts_log_level = logging.INFO
        nn_log_level = logging.INFO
        logging.getLogger("src.mcts").setLevel(mcts_log_level)
        logging.getLogger("src.nn").setLevel(nn_log_level)

        set_random_seeds(self.seed)

        self.device = get_device(self.worker_device_str)
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
            self.nn_evaluator.set_weights(weights)
            logger.debug("Weights updated.")
        except Exception as e:
            logger.error(f"Failed to set weights: {e}", exc_info=True)

    def _report_current_state(self, game_state: GameState):
        """Asynchronously sends the current game state to the collector."""
        if self.stats_collector_actor:
            try:
                # Send a copy to avoid potential issues with shared state
                state_copy = game_state.copy()
                # Correctly call remote method
                self.stats_collector_actor.update_worker_game_state.remote(  # type: ignore
                    self.actor_id, state_copy
                )
                logger.debug(
                    f"Reported state step {state_copy.current_step} to collector."
                )
            except Exception as e:
                logger.error(f"Failed to report game state to collector: {e}")

    def _log_step_stats_async(
        self, game_state: GameState, mcts_visits: int, mcts_depth: int
    ):
        """Asynchronously logs per-step stats to the collector."""
        if self.stats_collector_actor:
            try:
                # Use distinct keys for per-step worker stats
                step_stats = {
                    f"Worker_{self.actor_id}/Step_Score": (
                        game_state.game_score,
                        game_state.current_step,
                    ),
                    f"Worker_{self.actor_id}/Step_MCTS_Visits": (
                        mcts_visits,
                        game_state.current_step,
                    ),
                    f"Worker_{self.actor_id}/Step_MCTS_Depth": (
                        mcts_depth,
                        game_state.current_step,
                    ),
                }
                # Correctly call remote method
                self.stats_collector_actor.log_batch.remote(step_stats)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to log step stats to collector: {e}")

    def run_episode(self) -> SelfPlayResult:
        """
        Runs a single episode of self-play using MCTS and the internal neural network.
        Implements MCTS tree reuse.
        Stores extracted features (StateType) in the experience buffer.
        Returns a SelfPlayResult Pydantic model including aggregated stats.
        Reports current state and step stats asynchronously.
        """
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)

        # Use TYPE_CHECKING import for Experience type hint
        raw_experiences: list[tuple["StateType", "PolicyTargetMapping", float]] = []
        step_root_visits: list[int] = []
        step_tree_depths: list[int] = []
        step_simulations: list[int] = []

        logger.info(f"Starting episode with seed {episode_seed}")
        self._report_current_state(game)  # Report initial state

        root_node: Node | None = Node(state=game.copy())

        while not game.is_over():
            step_start_time = time.monotonic()
            if root_node is None:
                logger.error(
                    "MCTS root node became None unexpectedly. Aborting episode."
                )
                break

            logger.info(
                f"Step {game.current_step}: Running MCTS simulations ({self.mcts_config.num_simulations}) on state from step {root_node.state.current_step}..."
            )
            mcts_start_time = time.monotonic()
            mcts_max_depth = 0  # Default value
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

            # Log per-step MCTS stats (before selecting action)
            self._log_step_stats_async(game, root_node.visit_count, mcts_max_depth)

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
                # Extract features from the state *before* taking the action
                state_features: "StateType" = extract_state_features(
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

            raw_experiences.append((state_features, policy_target, 0.0))
            step_simulations.append(self.mcts_config.num_simulations)
            step_root_visits.append(root_node.visit_count)
            step_tree_depths.append(mcts_max_depth)

            game_step_start_time = time.monotonic()
            try:
                _, done = game.step(action)  # Updates the local 'game' instance
            except Exception as step_err:
                logger.error(
                    f"Error executing game step for action {action}: {step_err}",
                    exc_info=True,
                )
                break

            game_step_duration = time.monotonic() - game_step_start_time
            logger.info(
                f"Step {game.current_step}: Action {action} taken. Done: {done}. Game step time: {game_step_duration:.4f}s"
            )

            # Report the new state after the step
            self._report_current_state(game)
            # Also log current score/step (after the step is taken)
            self._log_step_stats_async(
                game, root_node.visit_count, mcts_max_depth
            )  # Log again with updated game step/score

            tree_reuse_start_time = time.monotonic()
            if not done:
                if action in root_node.children:
                    root_node = root_node.children[action]
                    root_node.parent = None
                    logger.debug(
                        f"Reused MCTS subtree for action {action}. New root step: {root_node.state.current_step}"
                    )
                else:
                    logger.error(
                        f"Child node for selected action {action} not found in MCTS tree children: {list(root_node.children.keys())}. Resetting MCTS root to current game state."
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

        # --- Episode End ---
        final_outcome = game.get_outcome() if game.is_over() else 0.0
        logger.info(
            f"Episode finished. Outcome: {final_outcome}, Steps: {game.current_step}"
        )

        # Use TYPE_CHECKING import for Experience type hint
        processed_experiences: list["Experience"] = [
            (state_type, policy, final_outcome)
            for state_type, policy, _ in raw_experiences
        ]

        total_sims_episode = sum(step_simulations)
        avg_visits_episode = np.mean(step_root_visits) if step_root_visits else 0.0
        avg_depth_episode = np.mean(step_tree_depths) if step_tree_depths else 0.0

        # No longer need to return final_game_state
        # Cast numpy floats to standard floats for Pydantic model
        return SelfPlayResult(
            episode_experiences=processed_experiences,
            final_score=final_outcome,
            episode_steps=game.current_step,
            total_simulations=total_sims_episode,
            avg_root_visits=float(avg_visits_episode),
            avg_tree_depth=float(avg_depth_episode),
        )

# File: alphatriangle/rl/self_play/worker.py
import cProfile
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray
from trianglengin import EnvConfig, GameState
from trimcts import SearchConfiguration, run_mcts

from ...config import ModelConfig, TrainConfig
from ...features import extract_state_features
from ...nn import NeuralNetwork
from ...stats.stats_types import RawMetricEvent
from ...utils import get_device, set_random_seeds
from ..types import SelfPlayResult
from .mcts_helpers import (
    PolicyGenerationError,
    get_policy_target_from_visits,
    select_action_from_visits,
)

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from ...utils.types import Experience, PolicyTargetMapping, StateType

    MctsTreeHandle = Any

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running self-play episodes using trimcts and a NN.
    Uses trianglengin.GameState and supports MCTS tree reuse.
    Sends raw metric events to the StatsCollectorActor.
    """

    def __init__(
        self,
        actor_id: int,
        env_config: EnvConfig,
        mcts_config: SearchConfiguration,
        model_config: ModelConfig,
        train_config: TrainConfig,
        stats_collector_actor: "ActorHandle",
        run_base_dir: str,
        initial_weights: dict | None = None,
        seed: int | None = None,
        worker_device_str: str = "cpu",
        profile_this_worker: bool = False,
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.stats_collector = stats_collector_actor
        self.run_base_dir = Path(run_base_dir)
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str
        self.profile_this_worker = profile_this_worker

        self.n_step = self.train_config.N_STEP_RETURNS
        self.gamma = self.train_config.GAMMA
        self.current_trainer_step = 0

        global logger
        logger = logging.getLogger(__name__)

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
            f"Worker {self.actor_id} initialized on device {self.device}. Seed: {self.seed}. LogLevel: {logging.getLevelName(logger.getEffectiveLevel())}"
        )
        logger.debug(f"Worker {self.actor_id} init complete.")

        self.profiler: cProfile.Profile | None = None
        if self.profile_this_worker:
            self.profiler = cProfile.Profile()
            logger.warning(f"Worker {self.actor_id}: Profiling ENABLED.")
        else:
            logger.info(f"Worker {self.actor_id}: Profiling DISABLED.")

    def set_weights(self, weights: dict):
        """Updates the neural network weights."""
        try:
            self.nn_evaluator.set_weights(weights)
            logger.info(f"Worker {self.actor_id}: Weights updated.")
        except Exception as e:
            logger.error(
                f"Worker {self.actor_id}: Failed to set weights: {e}", exc_info=True
            )

    def set_current_trainer_step(self, global_step: int):
        """Sets the global step corresponding to the current network weights."""
        self.current_trainer_step = global_step
        logger.info(f"Worker {self.actor_id}: Trainer step set to {global_step}")

    def _send_event(self, name: str, value: float | int, context: dict | None = None):
        """Helper to send a raw metric event to the collector."""
        if self.stats_collector:
            event = RawMetricEvent(
                name=name,
                value=value,
                global_step=self.current_trainer_step,
                timestamp=time.time(),
                context=context or {},
            )
            try:
                self.stats_collector.log_event.remote(event)
            except Exception as e:
                logger.error(f"Failed to send event '{name}' to stats collector: {e}")

    def run_episode(self) -> SelfPlayResult:
        """Runs a single episode of self-play with MCTS tree reuse."""
        step_at_start = self.current_trainer_step
        logger.info(
            f"Worker {self.actor_id}: Starting run_episode. Seed: {self.seed}. Using network from trainer step: {step_at_start}"
        )
        if self.profiler:
            self.profiler.enable()

        result: SelfPlayResult | None = None
        episode_seed = self.seed + random.randint(0, 1000)
        game: GameState | None = None
        mcts_tree_handle: MctsTreeHandle | None = None
        last_action: int = -1
        final_score = 0.0
        final_step = 0
        total_sims_episode = 0
        total_triangles_cleared_episode = 0

        try:
            self.nn_evaluator.model.eval()
            logger.debug(
                f"Worker {self.actor_id}: Initializing GameState with seed {episode_seed}..."
            )
            game = GameState(self.env_config, initial_seed=episode_seed)
            logger.debug(
                f"Worker {self.actor_id}: GameState initialized. Initial step: {game.current_step}, Initial score: {game.game_score()}"
            )

            if game.is_over():
                reason = game.get_game_over_reason() or "Unknown reason at start"
                logger.error(
                    f"Worker {self.actor_id}: Game over immediately after reset (Seed: {episode_seed}). Reason: {reason}"
                )
                episode_end_context = {
                    "score": game.game_score(),
                    "length": 0,
                    "simulations": 0,
                    "triangles_cleared": 0,
                    "trainer_step": step_at_start,
                }
                self._send_event("episode_end", 1.0, context=episode_end_context)
                return SelfPlayResult(
                    episode_experiences=[],
                    final_score=game.game_score(),
                    episode_steps=0,
                    trainer_step_at_episode_start=step_at_start,
                    total_simulations=0,
                    avg_root_visits=0.0,
                    avg_tree_depth=0.0,
                    context=episode_end_context,
                )
            if not game.valid_actions():
                logger.error(
                    f"Worker {self.actor_id}: Game not over, but NO valid actions at start (Seed: {episode_seed})."
                )
                episode_end_context = {
                    "score": game.game_score(),
                    "length": 0,
                    "simulations": 0,
                    "triangles_cleared": 0,
                    "trainer_step": step_at_start,
                }
                self._send_event("episode_end", 1.0, context=episode_end_context)
                return SelfPlayResult(
                    episode_experiences=[],
                    final_score=game.game_score(),
                    episode_steps=0,
                    trainer_step_at_episode_start=step_at_start,
                    total_simulations=0,
                    avg_root_visits=0.0,
                    avg_tree_depth=0.0,
                    context=episode_end_context,
                )

            n_step_state_policy_buffer: deque[tuple[StateType, PolicyTargetMapping]] = (
                deque(maxlen=self.n_step)
            )
            n_step_reward_buffer: deque[float] = deque(maxlen=self.n_step)
            episode_experiences: list[Experience] = []
            last_total_visits = 0

            logger.info(
                f"Worker {self.actor_id}: Starting episode loop with seed {episode_seed}"
            )

            while not game.is_over():
                current_step_in_loop = game.current_step
                logger.debug(
                    f"Worker {self.actor_id}: --- Starting Step {current_step_in_loop} ---"
                )
                step_start_time = time.monotonic()

                if game.is_over():
                    logger.warning(
                        f"Worker {self.actor_id}: State terminal before MCTS (Step {current_step_in_loop}). Exiting loop."
                    )
                    break

                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Running MCTS ({self.mcts_config.max_simulations} sims)..."
                )
                mcts_start_time = time.monotonic()
                visit_counts: dict[int, int] = {}
                new_mcts_tree_handle: MctsTreeHandle | None = None
                try:
                    visit_counts, new_mcts_tree_handle = run_mcts(
                        root_state=game,
                        network_interface=self.nn_evaluator,
                        config=self.mcts_config,
                        previous_tree_handle=mcts_tree_handle,
                        last_action=last_action,
                    )
                    mcts_tree_handle = new_mcts_tree_handle
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: MCTS returned visit_counts: {len(visit_counts)} actions."
                    )
                except Exception as mcts_err:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: trimcts failed: {mcts_err}",
                        exc_info=True,
                    )
                    mcts_tree_handle = None
                    break
                mcts_duration = time.monotonic() - mcts_start_time
                last_total_visits = sum(visit_counts.values())
                total_sims_episode += last_total_visits
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: MCTS finished ({mcts_duration:.3f}s). Total visits: {last_total_visits}"
                )

                self._send_event(
                    "mcts_step",
                    float(last_total_visits),
                    context={"game_step": current_step_in_loop},
                )

                if not visit_counts:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: MCTS returned empty visit counts. Cannot proceed."
                    )
                    break

                action_selection_start_time = time.monotonic()
                temp = 1.0
                selection_temp = 1.0
                if self.train_config.MAX_TRAINING_STEPS is not None:
                    explore_steps = self.train_config.MAX_TRAINING_STEPS * 0.1
                    selection_temp = (
                        1.0 if current_step_in_loop < explore_steps else 0.1
                    )

                action_dim_int = int(
                    self.env_config.NUM_SHAPE_SLOTS
                    * self.env_config.ROWS
                    * self.env_config.COLS
                )
                action: int = -1
                try:
                    policy_target = get_policy_target_from_visits(
                        visit_counts, action_dim_int, temperature=temp
                    )
                    action = select_action_from_visits(
                        visit_counts, temperature=selection_temp
                    )
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Policy target generated. Action selected: {action}"
                    )
                except PolicyGenerationError as policy_err:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Policy/Action selection failed: {policy_err}",
                        exc_info=False,
                    )
                    break
                except Exception as policy_err:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Unexpected policy error: {policy_err}",
                        exc_info=True,
                    )
                    break
                action_selection_duration = (
                    time.monotonic() - action_selection_start_time
                )
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Action selection time: {action_selection_duration:.4f}s"
                )

                feature_start_time = time.monotonic()
                try:
                    state_features: StateType = extract_state_features(
                        game, self.model_config
                    )
                except Exception as e:
                    logger.error(
                        f"Worker {self.actor_id}: Feature extraction error (Step {current_step_in_loop}): {e}",
                        exc_info=True,
                    )
                    break
                feature_duration = time.monotonic() - feature_start_time
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Feature extraction time: {feature_duration:.4f}s"
                )

                n_step_state_policy_buffer.append((state_features, policy_target))

                game_step_start_time = time.monotonic()
                step_reward, done = 0.0, False
                cleared_triangles_this_step = 0
                try:
                    step_reward, done = game.step(action)
                    # Get cleared triangles count using the new method
                    # Add type ignore as this method is new in the dependency
                    cleared_triangles_this_step = game.get_last_cleared_triangles()  # type: ignore [attr-defined]
                    total_triangles_cleared_episode += cleared_triangles_this_step
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: game.step({action}) -> Reward: {step_reward:.3f}, Done: {done}, Cleared: {cleared_triangles_this_step}"
                    )
                    last_action = action
                except Exception as step_err:
                    logger.error(
                        f"Worker {self.actor_id}: Game step error (Action {action}, Step {current_step_in_loop}): {step_err}",
                        exc_info=True,
                    )
                    last_action = -1
                    break
                game_step_duration = time.monotonic() - game_step_start_time
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Game step time: {game_step_duration:.4f}s"
                )

                n_step_reward_buffer.append(step_reward)
                self._send_event(
                    "step_reward",
                    step_reward,
                    context={"game_step": current_step_in_loop},
                )
                if cleared_triangles_this_step > 0:
                    self._send_event(
                        "triangles_cleared_step",
                        cleared_triangles_this_step,
                        context={"game_step": current_step_in_loop},
                    )

                if len(n_step_reward_buffer) == self.n_step:
                    discounted_reward_sum = sum(
                        (self.gamma**i) * n_step_reward_buffer[i]
                        for i in range(self.n_step)
                    )
                    bootstrap_value = 0.0
                    if not done:
                        try:
                            _, bootstrap_value = self.nn_evaluator.evaluate_state(game)
                        except Exception as eval_err:
                            logger.error(
                                f"Worker {self.actor_id}: Bootstrap eval error (Step {game.current_step}): {eval_err}",
                                exc_info=True,
                            )
                            bootstrap_value = 0.0
                    n_step_return = (
                        discounted_reward_sum
                        + (self.gamma**self.n_step) * bootstrap_value
                    )
                    state_features_t_minus_n, policy_target_t_minus_n = (
                        n_step_state_policy_buffer[0]
                    )
                    exp: Experience = (
                        state_features_t_minus_n,
                        policy_target_t_minus_n,
                        n_step_return,
                    )
                    episode_experiences.append(exp)
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Stored experience for step {current_step_in_loop - self.n_step + 1}. N-Step Return: {n_step_return:.4f}"
                    )

                self._send_event(
                    "current_score",
                    game.game_score(),
                    context={"game_step": current_step_in_loop},
                )

                step_duration = time.monotonic() - step_start_time
                logger.debug(
                    f"Worker {self.actor_id}: --- Finished Step {current_step_in_loop}. Duration: {step_duration:.3f}s ---"
                )

                if done:
                    logger.info(
                        f"Worker {self.actor_id}: Game ended naturally at step {game.current_step}. Reason: {game.get_game_over_reason()}"
                    )
                    break

            final_score = game.game_score() if game else 0.0
            final_step = game.current_step if game else 0
            logger.info(
                f"Worker {self.actor_id}: Episode loop finished. Final Score: {final_score:.2f}, Final Step: {final_step}, Total Cleared: {total_triangles_cleared_episode}"
            )

            remaining_steps = len(n_step_reward_buffer)
            logger.debug(
                f"Worker {self.actor_id}: Processing {remaining_steps} remaining n-step items."
            )
            for k in range(remaining_steps):
                discounted_reward_sum = sum(
                    (self.gamma**i) * n_step_reward_buffer[k + i]
                    for i in range(remaining_steps - k)
                )
                n_step_return = discounted_reward_sum
                state_features_t, policy_target_t = n_step_state_policy_buffer[k]
                final_exp: Experience = (
                    state_features_t,
                    policy_target_t,
                    n_step_return,
                )
                episode_experiences.append(final_exp)
                logger.debug(
                    f"Worker {self.actor_id}: Stored final experience for step {final_step - remaining_steps + k + 1}. N-Step Return: {n_step_return:.4f}"
                )

            if not episode_experiences:
                logger.warning(
                    f"Worker {self.actor_id}: Episode finished with 0 experiences collected. Score: {final_score}, Steps: {final_step}"
                )

            episode_end_context = {
                "score": final_score,
                "length": final_step,
                "simulations": total_sims_episode,
                "triangles_cleared": total_triangles_cleared_episode,
                "trainer_step": step_at_start,
            }
            self._send_event(name="episode_end", value=1.0, context=episode_end_context)

            result = SelfPlayResult(
                episode_experiences=episode_experiences,
                final_score=final_score,
                episode_steps=final_step,
                trainer_step_at_episode_start=step_at_start,
                total_simulations=total_sims_episode,
                avg_root_visits=float(last_total_visits),
                avg_tree_depth=0.0,
                context=episode_end_context,
            )
            logger.info(
                f"Worker {self.actor_id}: Episode result created. Experiences: {len(result.episode_experiences)}"
            )

        except Exception as e:
            logger.critical(
                f"Worker {self.actor_id}: Unhandled exception in run_episode: {e}",
                exc_info=True,
            )
            final_score = game.game_score() if game else 0.0
            final_step = game.current_step if game else 0
            episode_end_context = {
                "score": final_score,
                "length": final_step,
                "simulations": total_sims_episode,
                "triangles_cleared": total_triangles_cleared_episode,
                "trainer_step": step_at_start,
                "error": True,
            }
            self._send_event(
                "episode_end",
                1.0,
                context=episode_end_context,
            )
            result = SelfPlayResult(
                episode_experiences=[],
                final_score=final_score,
                episode_steps=final_step,
                trainer_step_at_episode_start=step_at_start,
                total_simulations=total_sims_episode,
                avg_root_visits=0.0,
                avg_tree_depth=0.0,
                context=episode_end_context,
            )

        finally:
            mcts_tree_handle = None
            if self.profiler:
                self.profiler.disable()
                profile_dir = self.run_base_dir / "profile_data"
                profile_dir.mkdir(exist_ok=True)
                profile_filename = (
                    profile_dir / f"worker_{self.actor_id}_ep_{episode_seed}.prof"
                )
                try:
                    self.profiler.dump_stats(str(profile_filename))
                    logger.warning(
                        f"Worker {self.actor_id}: Profiling stats saved to {profile_filename}"
                    )
                except Exception as e:
                    logger.error(
                        f"Worker {self.actor_id}: Failed to save profile stats: {e}"
                    )

        if result is None:
            logger.error(
                f"Worker {self.actor_id}: run_episode finished without setting a result. Returning empty."
            )
            episode_end_context = {
                "score": 0.0,
                "length": 0,
                "simulations": 0,
                "triangles_cleared": 0,
                "trainer_step": step_at_start,
                "error": True,
            }
            self._send_event(
                "episode_end",
                1.0,
                context=episode_end_context,
            )
            result = SelfPlayResult(
                episode_experiences=[],
                final_score=0.0,
                episode_steps=0,
                trainer_step_at_episode_start=step_at_start,
                total_simulations=0,
                avg_root_visits=0.0,
                avg_tree_depth=0.0,
                context=episode_end_context,
            )

        logger.info(
            f"Worker {self.actor_id}: Finished run_episode. Returning result with {len(result.episode_experiences)} experiences."
        )
        return result

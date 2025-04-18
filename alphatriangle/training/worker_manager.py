# File: alphatriangle/training/worker_manager.py
import logging
from typing import TYPE_CHECKING

import ray
from pydantic import ValidationError

from ..rl import SelfPlayResult, SelfPlayWorker

# --- CHANGED: Import from plot_definitions ---
from ..stats.plot_definitions import WEIGHT_UPDATE_METRIC_KEY

# --- END CHANGED ---

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the pool of SelfPlayWorker actors, task submission, and results."""

    def __init__(self, components: "TrainingComponents"):
        self.components = components
        self.train_config = components.train_config
        self.nn = components.nn
        self.stats_collector_actor = components.stats_collector_actor

        self.workers: list[ray.actor.ActorHandle | None] = []
        self.worker_tasks: dict[ray.ObjectRef, int] = {}
        self.active_worker_indices: set[int] = set()

    def initialize_workers(self):
        """Creates the pool of SelfPlayWorker Ray actors."""
        logger.info(
            f"Initializing {self.train_config.NUM_SELF_PLAY_WORKERS} self-play workers..."
        )
        initial_weights = self.nn.get_weights()
        weights_ref = ray.put(initial_weights)
        self.workers = [None] * self.train_config.NUM_SELF_PLAY_WORKERS

        for i in range(self.train_config.NUM_SELF_PLAY_WORKERS):
            try:
                worker = SelfPlayWorker.options(num_cpus=1).remote(
                    actor_id=i,
                    env_config=self.components.env_config,
                    mcts_config=self.components.mcts_config,
                    model_config=self.components.model_config,
                    train_config=self.train_config,
                    stats_collector_actor=self.stats_collector_actor,
                    initial_weights=weights_ref,
                    seed=self.train_config.RANDOM_SEED + i,
                    worker_device_str=self.train_config.WORKER_DEVICE,
                )
                self.workers[i] = worker
                self.active_worker_indices.add(i)
            except Exception as e:
                logger.error(f"Failed to initialize worker {i}: {e}", exc_info=True)

        logger.info(
            f"Initialized {len(self.active_worker_indices)} active self-play workers."
        )
        del weights_ref

    def submit_initial_tasks(self):
        """Submits the first task for each active worker."""
        logger.info("Submitting initial tasks to workers...")
        for worker_idx in self.active_worker_indices:
            self.submit_task(worker_idx)

    def submit_task(self, worker_idx: int):
        """Submits a new run_episode task to a specific worker."""
        if worker_idx not in self.active_worker_indices:
            logger.warning(f"Attempted to submit task to inactive worker {worker_idx}.")
            return
        worker = self.workers[worker_idx]
        if worker:
            try:
                task_ref = worker.run_episode.remote()
                self.worker_tasks[task_ref] = worker_idx
                logger.debug(f"Submitted task to worker {worker_idx}")
            except Exception as e:
                logger.error(
                    f"Failed to submit task to worker {worker_idx}: {e}", exc_info=True
                )
                self.active_worker_indices.discard(worker_idx)
                self.workers[worker_idx] = None
        else:
            logger.error(
                f"Worker {worker_idx} is None during task submission despite being in active set."
            )
            self.active_worker_indices.discard(worker_idx)

    def get_completed_tasks(
        self, timeout: float = 0.1
    ) -> list[tuple[int, SelfPlayResult | Exception]]:
        """
        Checks for completed worker tasks using ray.wait.
        Returns a list of tuples: (worker_id, result_or_exception).
        """
        # --- ADDED: Type annotation ---
        completed_results: list[tuple[int, SelfPlayResult | Exception]] = []
        # --- END ADDED ---
        if not self.worker_tasks:
            return completed_results

        ready_refs, _ = ray.wait(
            list(self.worker_tasks.keys()), num_returns=1, timeout=timeout
        )

        if not ready_refs:
            return completed_results

        for ref in ready_refs:
            worker_idx = self.worker_tasks.pop(ref, -1)
            if worker_idx == -1 or worker_idx not in self.active_worker_indices:
                logger.warning(
                    f"Received result for unknown or inactive worker task: {ref}"
                )
                continue

            try:
                logger.debug(f"Attempting ray.get for worker {worker_idx} task {ref}")
                result_raw = ray.get(ref)
                logger.debug(f"ray.get succeeded for worker {worker_idx}")
                try:
                    result_validated = SelfPlayResult.model_validate(result_raw)
                    completed_results.append((worker_idx, result_validated))
                    logger.debug(
                        f"Pydantic validation passed for worker {worker_idx} result."
                    )
                except ValidationError as e_val:
                    error_msg = f"Pydantic validation failed for result from worker {worker_idx}: {e_val}"
                    logger.error(error_msg, exc_info=False)
                    logger.debug(f"Invalid data structure received: {result_raw}")
                    # --- FIXED: Append exception ---
                    completed_results.append((worker_idx, ValueError(error_msg)))
                    # --- END FIXED ---
                except Exception as e_other_val:
                    error_msg = f"Unexpected error during result validation for worker {worker_idx}: {e_other_val}"
                    logger.error(error_msg, exc_info=True)
                    # --- FIXED: Append exception ---
                    completed_results.append((worker_idx, e_other_val))
                    # --- END FIXED ---

            except ray.exceptions.RayActorError as e_actor:
                logger.error(
                    f"Worker {worker_idx} actor failed: {e_actor}", exc_info=True
                )
                # --- FIXED: Append exception ---
                completed_results.append((worker_idx, e_actor))
                # --- END FIXED ---
                self.workers[worker_idx] = None
                self.active_worker_indices.discard(worker_idx)
            except Exception as e_get:
                logger.error(
                    f"Error getting result from worker {worker_idx} task {ref}: {e_get}",
                    exc_info=True,
                )
                # --- FIXED: Append exception ---
                completed_results.append((worker_idx, e_get))
                # --- END FIXED ---

        # --- FIXED: Return type matches annotation ---
        return completed_results
        # --- END FIXED ---

    def update_worker_networks(self):
        """Sends the latest network weights to all active workers."""
        active_workers = [
            w
            for i, w in enumerate(self.workers)
            if i in self.active_worker_indices and w is not None
        ]
        if not active_workers:
            return
        logger.debug("Updating worker networks...")
        current_weights = self.nn.get_weights()
        weights_ref = ray.put(current_weights)
        update_tasks = [
            worker.set_weights.remote(weights_ref) for worker in active_workers
        ]
        if not update_tasks:
            del weights_ref
            return
        try:
            ray.get(update_tasks, timeout=15.0)
            logger.debug(f"Worker networks updated for {len(active_workers)} workers.")
            if self.stats_collector_actor:
                # TODO: Pass global_step if needed for accurate logging
                update_step_metric = {WEIGHT_UPDATE_METRIC_KEY: (1.0, 0)}
                self.stats_collector_actor.log_batch.remote(update_step_metric)  # type: ignore
        except ray.exceptions.RayActorError as e:
            logger.error(
                f"A worker actor failed during weight update: {e}", exc_info=True
            )
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout waiting for workers to update weights.")
        except Exception as e:
            logger.error(
                f"Unexpected error updating worker networks: {e}", exc_info=True
            )
        finally:
            del weights_ref

    def get_num_active_workers(self) -> int:
        """Returns the number of currently active workers."""
        return len(self.active_worker_indices)

    def get_num_pending_tasks(self) -> int:
        """Returns the number of tasks currently running."""
        return len(self.worker_tasks)

    def cleanup_actors(self):
        """Kills Ray actors associated with this manager."""
        logger.info("Cleaning up WorkerManager actors...")
        for task_ref in list(self.worker_tasks.keys()):
            try:
                ray.cancel(task_ref, force=True)
            except Exception as cancel_e:
                logger.warning(f"Error cancelling task {task_ref}: {cancel_e}")
        self.worker_tasks = {}

        for i, worker in enumerate(self.workers):
            if worker:
                try:
                    ray.kill(worker, no_restart=True)
                    logger.debug(f"Killed worker {i}.")
                except Exception as kill_e:
                    logger.warning(f"Error killing worker {i}: {kill_e}")
        self.workers = []
        self.active_worker_indices = set()
        logger.info("WorkerManager actors cleaned up.")

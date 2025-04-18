# File: alphatriangle/training/setup.py
import logging
from typing import TYPE_CHECKING

import ray
import torch

from .. import config, utils
from ..data import DataManager
from ..nn import NeuralNetwork
from ..rl import ExperienceBuffer, Trainer
from ..stats import StatsCollectorActor
from .components import TrainingComponents

if TYPE_CHECKING:
    from ..config import PersistenceConfig, TrainConfig

logger = logging.getLogger(__name__)


def setup_training_components(
    train_config_override: "TrainConfig",
    persist_config_override: "PersistenceConfig",
) -> tuple[TrainingComponents | None, bool]:
    """
    Initializes Ray (if not already initialized), detects cores, updates config,
    and returns the TrainingComponents bundle and a flag indicating if Ray was initialized here.
    Adjusts worker count based on detected cores.
    """
    ray_initialized_here = False
    detected_cpu_cores: int | None = None

    try:
        # --- Initialize Ray (if needed) and Detect Cores ---
        if not ray.is_initialized():
            try:
                # Attempt initialization
                ray.init(logging_level=logging.WARNING, log_to_driver=True)
                ray_initialized_here = True
                logger.info("Ray initialized by setup_training_components.")
            except Exception as e:
                # Log critical error and re-raise to stop setup
                logger.critical(f"Failed to initialize Ray: {e}", exc_info=True)
                raise RuntimeError("Ray initialization failed") from e
        else:
            logger.info("Ray already initialized.")
            # Ensure flag is False if Ray was already running
            ray_initialized_here = False

        # --- Detect Cores (proceed even if Ray was already initialized) ---
        try:
            resources = ray.cluster_resources()
            detected_cpu_cores = int(resources.get("CPU", 0)) - 2
            logger.info(f"Ray detected {detected_cpu_cores} CPU cores.")
        except Exception as e:
            logger.error(f"Could not get Ray cluster resources: {e}")
            # Continue without detected cores, will use configured value

        # --- Initialize Configurations ---
        train_config = train_config_override
        persist_config = persist_config_override
        env_config = config.EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = config.MCTSConfig()

        # --- Adjust Worker Count based on Detected Cores ---
        requested_workers = train_config.NUM_SELF_PLAY_WORKERS
        actual_workers = requested_workers  # Start with configured value

        if detected_cpu_cores is not None and detected_cpu_cores > 0:
            # --- CHANGED: Prioritize detected cores ---
            actual_workers = detected_cpu_cores  # Use detected cores
            if actual_workers != requested_workers:
                logger.info(
                    f"Overriding configured workers ({requested_workers}) with detected CPU cores ({actual_workers})."
                )
            else:
                logger.info(
                    f"Using {actual_workers} self-play workers (matches detected cores)."
                )
            # --- END CHANGED ---
        else:
            logger.warning(
                f"Could not detect valid CPU cores ({detected_cpu_cores}). Using configured NUM_SELF_PLAY_WORKERS: {requested_workers}"
            )
            actual_workers = requested_workers  # Fallback to configured value

        # Update the config object with the final determined number
        train_config.NUM_SELF_PLAY_WORKERS = actual_workers
        logger.info(f"Final worker count set to: {train_config.NUM_SELF_PLAY_WORKERS}")

        # --- Validate Configs ---
        config.print_config_info_and_validate(mcts_config)

        # --- Setup ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)
        worker_device = utils.get_device(train_config.WORKER_DEVICE)
        logger.info(f"Determined Training Device: {device}")
        logger.info(f"Determined Worker Device: {worker_device}")
        logger.info(f"Model Compilation Enabled: {train_config.COMPILE_MODEL}")

        # --- Initialize Core Components ---
        stats_collector_actor = StatsCollectorActor.remote(max_history=500_000)  # type: ignore
        logger.info("Initialized StatsCollectorActor with max_history=500k.")
        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)
        data_manager = DataManager(persist_config, train_config)

        # --- Create Components Bundle ---
        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,
            model_config=model_config,
            mcts_config=mcts_config,
            persist_config=persist_config,
        )
        # Return components and the flag indicating if Ray was initialized *by this function*
        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        # Return None and the Ray init flag (which might be True if init succeeded before error)
        return None, ray_initialized_here


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

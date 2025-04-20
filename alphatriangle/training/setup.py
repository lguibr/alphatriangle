import logging
from typing import TYPE_CHECKING

import ray
import torch

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Keep alphatriangle imports
from .. import config, utils
from ..data import DataManager
from ..nn import NeuralNetwork
from ..rl import ExperienceBuffer, Trainer
from ..stats import StatsCollectorActor
from .components import TrainingComponents

# REMOVE VisualStateActor import

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
        if not ray.is_initialized():
            try:
                ray.init(logging_level=logging.WARNING, log_to_driver=True)
                ray_initialized_here = True
                logger.info("Ray initialized by setup_training_components.")
            except Exception as e:
                logger.critical(f"Failed to initialize Ray: {e}", exc_info=True)
                raise RuntimeError("Ray initialization failed") from e
        else:
            logger.info("Ray already initialized.")
            ray_initialized_here = False

        try:
            resources = ray.cluster_resources()
            # Reserve 1 core for main process, 1 for stats/other? Be conservative.
            detected_cpu_cores = int(resources.get("CPU", 0)) - 2
            logger.info(
                f"Ray detected {detected_cpu_cores} available CPU cores for workers."
            )
        except Exception as e:
            logger.error(f"Could not get Ray cluster resources: {e}")

        train_config = train_config_override
        persist_config = persist_config_override
        # Instantiate EnvConfig from trianglengin
        env_config = EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = config.MCTSConfig()

        requested_workers = train_config.NUM_SELF_PLAY_WORKERS
        actual_workers = requested_workers

        if detected_cpu_cores is not None and detected_cpu_cores > 0:
            # Cap requested workers by detected cores
            actual_workers = min(requested_workers, detected_cpu_cores)
            if actual_workers != requested_workers:
                logger.info(
                    f"Adjusting requested workers ({requested_workers}) to available cores ({detected_cpu_cores}). Using {actual_workers} workers."
                )
            else:
                logger.info(
                    f"Using {actual_workers} self-play workers (fits within detected cores)."
                )
        else:
            logger.warning(
                f"Could not detect valid CPU cores ({detected_cpu_cores}). Using configured NUM_SELF_PLAY_WORKERS: {requested_workers}"
            )
            actual_workers = requested_workers

        train_config.NUM_SELF_PLAY_WORKERS = actual_workers
        logger.info(f"Final worker count set to: {train_config.NUM_SELF_PLAY_WORKERS}")

        # Pass trianglengin.EnvConfig to validation
        config.print_config_info_and_validate(mcts_config)

        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)
        worker_device = utils.get_device(train_config.WORKER_DEVICE)
        logger.info(f"Determined Training Device: {device}")
        logger.info(f"Determined Worker Device: {worker_device}")
        logger.info(f"Model Compilation Enabled: {train_config.COMPILE_MODEL}")

        stats_collector_actor = StatsCollectorActor.remote(max_history=500_000)  # type: ignore
        logger.info("Initialized StatsCollectorActor with max_history=500k.")
        # REMOVE VisualStateActor instantiation
        # visual_state_actor = VisualStateActor.remote() # type: ignore
        # logger.info("Initialized VisualStateActor.")

        # Pass trianglengin.EnvConfig to NN and Trainer
        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)
        data_manager = DataManager(persist_config, train_config)

        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            # REMOVE visual_state_actor
            train_config=train_config,
            env_config=env_config,  # Store trianglengin.EnvConfig
            model_config=model_config,
            mcts_config=mcts_config,
            persist_config=persist_config,
        )
        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        return None, ray_initialized_here


# ... (count_parameters remains the same) ...
def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

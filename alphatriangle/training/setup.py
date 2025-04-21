import logging
from typing import TYPE_CHECKING

import ray
import torch

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig  # UPDATED IMPORT
from trimcts import SearchConfiguration

# Keep alphatriangle imports
from .. import config, utils
from ..config import AlphaTriangleMCTSConfig
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
        # --- Ray Initialization ---
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

        # --- Resource Detection ---
        try:
            resources = ray.cluster_resources()
            # Subtract 1 for the main process, 1 for potential overhead/OS
            detected_cpu_cores = max(0, int(resources.get("CPU", 0)) - 2)
            logger.info(
                f"Ray detected {detected_cpu_cores} available CPU cores for workers."
            )
        except Exception as e:
            logger.error(f"Could not get Ray cluster resources: {e}")

        # --- Load Configurations ---
        train_config = train_config_override
        persist_config = persist_config_override
        env_config = EnvConfig()  # Instantiate EnvConfig from trianglengin
        model_config = config.ModelConfig()
        alphatriangle_mcts_config = AlphaTriangleMCTSConfig()

        # --- Adjust Worker Count ---
        requested_workers = train_config.NUM_SELF_PLAY_WORKERS
        actual_workers = requested_workers
        if detected_cpu_cores is not None and detected_cpu_cores > 0:
            actual_workers = min(requested_workers, detected_cpu_cores)
            if actual_workers != requested_workers:
                logger.info(
                    f"Adjusting requested workers ({requested_workers}) to available cores ({detected_cpu_cores}). Using {actual_workers} workers."
                )
        else:
            logger.warning(
                f"Could not detect valid CPU cores ({detected_cpu_cores}). Using configured NUM_SELF_PLAY_WORKERS: {requested_workers}"
            )
        train_config.NUM_SELF_PLAY_WORKERS = actual_workers
        logger.info(f"Final worker count set to: {train_config.NUM_SELF_PLAY_WORKERS}")

        # --- Validate Configurations ---
        config.print_config_info_and_validate(alphatriangle_mcts_config)

        # --- Create trimcts SearchConfiguration ---
        trimcts_mcts_config = SearchConfiguration(
            max_simulations=alphatriangle_mcts_config.max_simulations,
            max_depth=alphatriangle_mcts_config.max_depth,
            cpuct=alphatriangle_mcts_config.cpuct,
            dirichlet_alpha=alphatriangle_mcts_config.dirichlet_alpha,
            dirichlet_epsilon=alphatriangle_mcts_config.dirichlet_epsilon,
            discount=alphatriangle_mcts_config.discount,
        )
        logger.info(
            f"Created trimcts.SearchConfiguration with max_simulations={trimcts_mcts_config.max_simulations}"
        )

        # --- Setup Devices and Seeds ---
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

        run_base_dir = data_manager.path_manager.run_base_dir
        logger.info(f"Run base directory for workers: {run_base_dir}")

        # --- Bundle Components ---
        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,  # Pass trianglengin.EnvConfig
            model_config=model_config,
            mcts_config=trimcts_mcts_config,
            persist_config=persist_config,
        )

        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        return None, ray_initialized_here


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

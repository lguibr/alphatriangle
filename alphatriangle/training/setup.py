# File: alphatriangle/training/setup.py
import logging
from typing import TYPE_CHECKING

import ray
import torch

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig

# Keep alphatriangle imports
from .. import config, utils
from ..config import AlphaTriangleMCTSConfig, StatsConfig
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
    tb_log_dir: str | None = None,
    profile: bool = False,  # Added profile flag
    mlflow_run_id: str | None = None,  # Added mlflow_run_id
) -> tuple[TrainingComponents | None, bool]:
    """
    Initializes Ray (if not already initialized), detects cores, updates config,
    and returns the TrainingComponents bundle and a flag indicating if Ray was initialized here.
    Adjusts worker count based on detected cores. Logs expected Ray Dashboard URL.
    Handles minimal Ray installations gracefully regarding the dashboard.
    """
    ray_initialized_here = False
    detected_cpu_cores: int | None = None
    dashboard_started_successfully = False  # Flag to track if dashboard init succeeded

    try:
        # --- Ray Initialization ---
        if not ray.is_initialized():
            try:
                # Attempt to initialize with the dashboard first
                logger.info("Attempting to initialize Ray with dashboard...")
                ray.init(
                    logging_level=logging.WARNING,
                    log_to_driver=False,
                    include_dashboard=True,
                )
                ray_initialized_here = True
                dashboard_started_successfully = True  # Assume success if no exception
                logger.info(
                    "Ray initialized by setup_training_components WITH dashboard attempt."
                )
            except Exception as e_dash:
                # Check if the error is specifically about missing dashboard packages
                if "Cannot include dashboard with missing packages" in str(e_dash):
                    logger.warning(
                        "Ray dashboard dependencies missing. Retrying Ray initialization without dashboard. "
                        "Install 'ray[default]' for dashboard support."
                    )
                    try:
                        ray.init(
                            logging_level=logging.WARNING,
                            log_to_driver=False,
                            include_dashboard=False,  # Retry without dashboard
                        )
                        ray_initialized_here = True
                        dashboard_started_successfully = (
                            False  # Dashboard definitely not started
                        )
                        logger.info(
                            "Ray initialized by setup_training_components WITHOUT dashboard."
                        )
                    except Exception as e_no_dash:
                        logger.critical(
                            f"Failed to initialize Ray even without dashboard: {e_no_dash}",
                            exc_info=True,
                        )
                        raise RuntimeError("Ray initialization failed") from e_no_dash
                else:
                    # Different error during initialization
                    logger.critical(
                        f"Failed to initialize Ray (with dashboard attempt): {e_dash}",
                        exc_info=True,
                    )
                    raise RuntimeError("Ray initialization failed") from e_dash

            # Log dashboard status based on initialization success/failure
            if dashboard_started_successfully:
                logger.info(
                    "Ray Dashboard *should* be running. Check Ray startup logs (console/log file) for the exact URL (usually http://127.0.0.1:8265)."
                )
            elif ray_initialized_here:  # Initialized without dashboard
                logger.info(
                    "Ray Dashboard is NOT running (missing dependencies). Install 'ray[default]' to enable it."
                )

        else:  # Ray already initialized
            logger.info("Ray already initialized.")
            # Cannot reliably check dashboard status of existing session without potentially unstable APIs
            logger.info(
                "Ray Dashboard status in existing session unknown. Check Ray logs or http://127.0.0.1:8265."
            )
            ray_initialized_here = False

        # --- Resource Detection ---
        try:
            resources = ray.cluster_resources()
            # Reserve 1 core for the main process + 1 for overhead/OS
            cores_to_reserve = 2
            available_cores = int(resources.get("CPU", 0))
            detected_cpu_cores = max(0, available_cores - cores_to_reserve)
            logger.info(
                f"Ray detected {available_cores} total CPU cores. Reserving {cores_to_reserve}. Available for workers: {detected_cpu_cores}."
            )
        except Exception as e:
            logger.error(f"Could not get Ray cluster resources: {e}")

        # --- Load Configurations ---
        train_config = train_config_override
        persist_config = persist_config_override
        persist_config.RUN_NAME = train_config.RUN_NAME
        env_config = EnvConfig()
        model_config = config.ModelConfig()
        alphatriangle_mcts_config = AlphaTriangleMCTSConfig()
        stats_config = StatsConfig()

        # --- Adjust Worker Count ---
        requested_workers = train_config.NUM_SELF_PLAY_WORKERS
        actual_workers = requested_workers
        if detected_cpu_cores is not None and detected_cpu_cores > 0:
            actual_workers = min(requested_workers, detected_cpu_cores)
            if actual_workers != requested_workers:
                logger.info(
                    f"Adjusting requested workers ({requested_workers}) to available cores ({detected_cpu_cores}). Using {actual_workers} workers."
                )
        elif detected_cpu_cores == 0:
            logger.warning(
                "Detected 0 available CPU cores after reservation. Setting worker count to 1."
            )
            actual_workers = 1
        else:
            logger.warning(
                f"Could not detect valid CPU cores ({detected_cpu_cores}). Using configured NUM_SELF_PLAY_WORKERS: {requested_workers}"
            )
        train_config.NUM_SELF_PLAY_WORKERS = actual_workers
        logger.info(f"Final worker count set to: {train_config.NUM_SELF_PLAY_WORKERS}")

        # --- Validate Configurations ---
        config.print_config_info_and_validate(alphatriangle_mcts_config)

        # --- Create trimcts SearchConfiguration ---
        trimcts_mcts_config = alphatriangle_mcts_config.to_trimcts_config()
        logger.info(
            f"Created trimcts.SearchConfiguration with max_simulations={trimcts_mcts_config.max_simulations}, "
            f"mcts_batch_size={trimcts_mcts_config.mcts_batch_size}"
        )

        # --- Setup Devices and Seeds ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)
        worker_device = utils.get_device(train_config.WORKER_DEVICE)
        logger.info(f"Determined Training Device: {device}")
        logger.info(f"Determined Worker Device: {worker_device}")
        logger.info(f"Model Compilation Enabled: {train_config.COMPILE_MODEL}")
        logger.info(f"Worker Profiling Enabled: {profile}")  # Log profile status

        # --- Initialize Core Components ---
        data_manager = DataManager(persist_config, train_config)
        run_base_dir = data_manager.path_manager.run_base_dir

        stats_collector_actor = StatsCollectorActor.remote(  # type: ignore [attr-defined]
            stats_config=stats_config,
            run_name=train_config.RUN_NAME,
            tb_log_dir=tb_log_dir,
            mlflow_run_id=mlflow_run_id,  # Pass run_id
        )
        logger.info("Initialized StatsCollectorActor.")

        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)

        logger.info(f"Run base directory for workers: {run_base_dir}")

        # --- Bundle Components ---
        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,
            model_config=model_config,
            mcts_config=trimcts_mcts_config,
            persist_config=persist_config,
            stats_config=stats_config,
            profile_workers=profile,
        )

        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        if ray_initialized_here and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down due to setup error.")
            except Exception as ray_err:
                logger.error(f"Error shutting down Ray during setup cleanup: {ray_err}")
        return None, ray_initialized_here


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

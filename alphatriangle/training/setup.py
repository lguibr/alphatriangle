# File: alphatriangle/training/setup.py
import logging
from typing import TYPE_CHECKING, cast

import ray
import torch

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig
from trieye import (  # Import from Trieye
    Serializer,
    TrieyeActor,
    TrieyeConfig,
)

# Keep alphatriangle imports
from .. import config, utils
from ..config import AlphaTriangleMCTSConfig, ModelConfig  # Import ModelConfig
from ..nn import NeuralNetwork
from ..rl import ExperienceBuffer, Trainer
from .components import TrainingComponents

if TYPE_CHECKING:
    from ..config import TrainConfig

logger = logging.getLogger(__name__)


def setup_training_components(
    train_config_override: "TrainConfig",
    trieye_config_override: TrieyeConfig,
    profile: bool,
    # Add optional overrides
    model_config_override: ModelConfig | None = None,
    mcts_config_override: AlphaTriangleMCTSConfig | None = None,
) -> tuple[TrainingComponents | None, bool]:
    """
    Initializes Ray, detects cores, updates config, initializes TrieyeActor,
    and returns the TrainingComponents bundle. Accepts optional config overrides.
    """
    ray_initialized_here = False  # Initialize before try block
    detected_cpu_cores: int | None = None
    dashboard_started_successfully = False
    trieye_actor_handle: ray.actor.ActorHandle | None = None

    try:
        # --- Ray Initialization ---
        # (Ray init logic remains the same)
        if not ray.is_initialized():
            try:
                logger.info("Attempting to initialize Ray with dashboard...")
                ray.init(
                    logging_level=logging.WARNING,
                    log_to_driver=False,
                    include_dashboard=True,
                )
                ray_initialized_here = True
                dashboard_started_successfully = True
                logger.info(
                    "Ray initialized by setup_training_components WITH dashboard attempt."
                )
            except Exception as e_dash:
                if "Cannot include dashboard with missing packages" in str(e_dash):
                    logger.warning(
                        "Ray dashboard dependencies missing. Retrying Ray initialization without dashboard. Install 'ray[default]' for dashboard support."
                    )
                    try:
                        ray.init(
                            logging_level=logging.WARNING,
                            log_to_driver=False,
                            include_dashboard=False,
                        )
                        ray_initialized_here = True
                        dashboard_started_successfully = False
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
                    logger.critical(
                        f"Failed to initialize Ray (with dashboard attempt): {e_dash}",
                        exc_info=True,
                    )
                    raise RuntimeError("Ray initialization failed") from e_dash

            if dashboard_started_successfully:
                logger.info(
                    "Ray Dashboard *should* be running. Check Ray startup logs for the exact URL (usually http://127.0.0.1:8265)."
                )
            elif ray_initialized_here:
                logger.info(
                    "Ray Dashboard is NOT running (missing dependencies). Install 'ray[default]' to enable it."
                )
        else:
            logger.info("Ray already initialized.")
            logger.info(
                "Ray Dashboard status in existing session unknown. Check Ray logs or http://127.0.0.1:8265."
            )
            ray_initialized_here = False

        # --- Resource Detection ---
        try:
            resources = ray.cluster_resources()
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
        trieye_config = trieye_config_override
        env_config = EnvConfig()
        # Use overrides if provided, otherwise load defaults
        model_config = model_config_override or config.ModelConfig()
        alphatriangle_mcts_config = mcts_config_override or AlphaTriangleMCTSConfig()
        logger.info(
            f"Using Model Config: {'Override' if model_config_override else 'Default'}"
        )
        logger.info(
            f"Using MCTS Config: {'Override' if mcts_config_override else 'Default'}"
        )

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

        # --- Validate AlphaTriangle Configurations ---
        # Pass the potentially overridden MCTS config instance for validation
        config.print_config_info_and_validate(alphatriangle_mcts_config)

        # --- Create trimcts SearchConfiguration ---
        trimcts_mcts_config = alphatriangle_mcts_config.to_trimcts_config()
        logger.info(
            f"Created trimcts.SearchConfiguration with max_simulations={trimcts_mcts_config.max_simulations}, mcts_batch_size={trimcts_mcts_config.mcts_batch_size}"
        )

        # --- Setup Devices and Seeds ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)
        worker_device = utils.get_device(train_config.WORKER_DEVICE)
        logger.info(f"Determined Training Device: {device}")
        logger.info(f"Determined Worker Device: {worker_device}")
        logger.info(f"Model Compilation Enabled: {train_config.COMPILE_MODEL}")
        logger.info(f"Worker Profiling Enabled: {profile}")

        # --- Initialize Trieye Actor ---
        actor_name = f"trieye_actor_{trieye_config.run_name}"
        try:
            trieye_actor_handle = ray.get_actor(actor_name)
            logger.info(f"Reconnected to existing TrieyeActor '{actor_name}'.")
        except ValueError:
            logger.info(f"Creating new TrieyeActor '{actor_name}'.")
            trieye_actor_handle = TrieyeActor.options(
                name=actor_name, lifetime="detached"
            ).remote(config=trieye_config)
            if trieye_actor_handle:
                ray.get(trieye_actor_handle.get_mlflow_run_id.remote(), timeout=10)
                logger.info(f"TrieyeActor '{actor_name}' created and ready.")
            else:
                logger.error(
                    f"TrieyeActor handle is None immediately after creation for '{actor_name}'."
                )
                raise RuntimeError(
                    f"Failed to get handle for TrieyeActor '{actor_name}' after creation."
                ) from None

        if not trieye_actor_handle:
            logger.critical(
                f"Failed to create or connect to TrieyeActor '{actor_name}'. Cannot proceed."
            )
            raise RuntimeError(
                f"TrieyeActor '{actor_name}' handle is invalid."
            ) from None

        # --- Initialize Core AlphaTriangle Components ---
        serializer = Serializer()  # Instantiate Serializer here
        # Pass the potentially overridden model_config
        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)

        # --- Bundle Components ---
        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            trieye_actor=cast("ray.actor.ActorHandle", trieye_actor_handle),
            trieye_config=trieye_config,
            serializer=serializer,
            train_config=train_config,
            env_config=env_config,
            model_config=model_config,  # Store the used model config
            mcts_config=trimcts_mcts_config,  # Store the used MCTS config
            profile_workers=profile,
        )

        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        if trieye_actor_handle:
            try:
                ray.kill(trieye_actor_handle)
            except Exception as kill_err:
                logger.error(
                    f"Error killing TrieyeActor during setup cleanup: {kill_err}"
                )
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

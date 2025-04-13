# File: src/cli.py
import typer
import logging
import sys
import os
from typing_extensions import Annotated

# Ensure the src directory is in the Python path *if running directly*,
# but this shouldn't be necessary when installed as a package.
# Keep it for potential direct script execution during development.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary functions/classes AFTER potentially modifying sys.path
try:
    from src import config
    from src.app import Application
    from src.training.runners import (
        run_training_visual_mode,
        run_training_headless_mode,
    )
    from src.utils import set_random_seeds
    from src.mcts import MCTSConfig  # Import Pydantic MCTSConfig
except ImportError as e:
    print(f"ImportError in cli.py: {e}")
    print("This might happen if the package is not installed correctly or")
    print("if running the script directly without the project root in PYTHONPATH.")
    sys.exit(1)


app = typer.Typer(
    name="alphatriangle",
    help="AlphaZero implementation for a triangle puzzle game.",
    add_completion=False,
)

# Shared options
LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level",
        "-l",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
]

SeedOption = Annotated[
    int,
    typer.Option(
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
]


def setup_logging(log_level_str: str):
    """Configures root logger based on string level."""
    log_level_str = log_level_str.upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override existing config
    )
    logging.getLogger("ray").setLevel(logging.WARNING)  # Keep Ray less verbose
    logging.getLogger("matplotlib").setLevel(
        logging.WARNING
    )  # Keep Matplotlib less verbose
    logging.info(f"Root logger level set to {logging.getLevelName(log_level)}")


def run_interactive_mode(mode: str, seed: int, log_level: str):
    """Runs the interactive application."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)  # Get logger after setup
    logger.info(f"Running in {mode.capitalize()} mode...")
    set_random_seeds(seed)

    # Instantiate MCTSConfig needed for validation function
    mcts_config = MCTSConfig()  # Instantiate Pydantic model
    # Pass MCTSConfig instance to validation
    config.print_config_info_and_validate(mcts_config)

    try:
        app_instance = Application(mode=mode)
        app_instance.run()
    except ImportError as e:
        logger.error(f"ImportError: {e}")
        logger.error("Please ensure:")
        logger.error(
            "1. You are running from the project root directory (if developing)."
        )
        logger.error(
            "2. The package is installed correctly (`pip install .` or `pip install alphatriangle`)."
        )
        logger.error(
            "3. Dependencies are installed (`pip install -r requirements.txt`)."
        )
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Exiting.")
    sys.exit(0)


@app.command()
def play(
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
):
    """Run the game in interactive Play mode."""
    run_interactive_mode(mode="play", seed=seed, log_level=log_level)


@app.command()
def debug(
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
):
    """Run the game in interactive Debug mode."""
    run_interactive_mode(mode="debug", seed=seed, log_level=log_level)


@app.command()
def train(
    headless: Annotated[
        bool,
        typer.Option("--headless", "-H", help="Run training without visualization."),
    ] = False,
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
    # Add options to override specific TrainConfig parameters if desired
    # e.g., run_name: Annotated[Optional[str], typer.Option("--run-name")] = None
):
    """Run the AlphaTriangle training pipeline."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)  # Get logger after setup

    # --- Configuration Overrides ---
    # Create default configs first
    train_config_override = config.TrainConfig()
    persist_config_override = config.PersistenceConfig()

    # Apply overrides from CLI options if they were added
    # if run_name:
    #     train_config_override.RUN_NAME = run_name
    #     persist_config_override.RUN_NAME = run_name

    # Set seed in config
    train_config_override.RANDOM_SEED = seed

    if headless:
        logger.info("Starting training in Headless mode...")
        exit_code = run_training_headless_mode(
            log_level_str=log_level,
            train_config_override=train_config_override,
            persist_config_override=persist_config_override,
        )
    else:
        logger.info("Starting training in Visual mode...")
        exit_code = run_training_visual_mode(
            log_level_str=log_level,
            train_config_override=train_config_override,
            persist_config_override=persist_config_override,
        )

    logger.info(f"Training finished with exit code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    app()

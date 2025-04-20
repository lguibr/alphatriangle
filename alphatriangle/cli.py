# File: alphatriangle/cli.py
import logging
import sys
from typing import Annotated

import typer

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Import alphatriangle specific configs and runner
from alphatriangle import config, utils
from alphatriangle.config import (
    MCTSConfig,
    PersistenceConfig,
    TrainConfig,
)

# Import the renamed runner function
from alphatriangle.training.runners import run_training

app = typer.Typer(
    name="alphatriangle",
    help="AlphaZero training pipeline for a triangle puzzle game (uses trianglengin).",  # Updated help
    add_completion=False,
)

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
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(
        logging.WARNING
    )  # Keep even if removed? Maybe for dev deps.
    # Add trianglengin logger control if needed
    logging.getLogger("trianglengin").setLevel(
        logging.INFO
    )  # Example: Set engine log level
    logging.info(f"Root logger level set to {logging.getLevelName(log_level)}")


# --- REMOVED run_interactive_mode, play, debug commands ---


@app.command()
def train(
    # REMOVE headless option
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
):
    """Run the AlphaTriangle training pipeline (headless)."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Use alphatriangle configs here
    train_config_override = TrainConfig()
    persist_config_override = PersistenceConfig()
    train_config_override.RANDOM_SEED = seed

    logger.info("Starting training...")
    # Call the single runner function directly
    exit_code = run_training(
        log_level_str=log_level,
        train_config_override=train_config_override,
        persist_config_override=persist_config_override,
    )

    logger.info(f"Training finished with exit code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    app()

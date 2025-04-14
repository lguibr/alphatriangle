# File: run_training_headless.py
# This script is now primarily for legacy execution or direct debugging.
# The recommended way to run is via the 'alphatriangle train --headless' command.

import argparse
import logging
import sys
from pathlib import Path  # Import Path

# Ensure the src directory is in the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the function that now contains the core logic
try:
    from src.config import PersistenceConfig, TrainConfig
    from src.training.runners import run_training_headless_mode
except ImportError as e:
    print(f"ImportError: {e}")
    print(
        "Please ensure the package is installed (`pip install .`) or run from the project root."
    )
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaTriangle Headless Training (Legacy Runner)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Override run name")
    # Add other potential overrides if needed

    args = parser.parse_args()

    # Setup logging here just to show the initial message
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    initial_log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=initial_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting headless training via legacy runner...")

    # Create config overrides
    train_config_override = TrainConfig()
    persist_config_override = PersistenceConfig()
    train_config_override.RANDOM_SEED = args.seed
    if args.run_name:
        train_config_override.RUN_NAME = args.run_name
        persist_config_override.RUN_NAME = args.run_name

    # Call the refactored function
    exit_code = run_training_headless_mode(
        log_level_str=args.log_level,
        train_config_override=train_config_override,
        persist_config_override=persist_config_override,
    )

    sys.exit(exit_code)

# File: run_interactive.py
# File: run_interactive.py
# This script is now primarily for legacy execution or direct debugging.
# The recommended way to run is via the 'alphatriangle' command-line tool.

import argparse
import sys

# Use absolute imports assuming the package is installed
try:
    from alphatriangle.cli import run_interactive_mode
except ImportError as e:
    print(f"ImportError: {e}")
    print(
        "Please ensure the package is installed (`pip install -e .`) or run from the project root."
    )
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaTriangle Interactive Modes (Legacy Runner)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="play",
        choices=["play", "debug"],
        help="Interaction mode ('play' or 'debug')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    run_interactive_mode(mode=args.mode, seed=args.seed, log_level=args.log_level)

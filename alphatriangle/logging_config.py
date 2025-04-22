# File: alphatriangle/logging_config.py
import logging
import sys
from pathlib import Path

# Define the triangle emoji
TRIANGLE_EMOJI = "▲"


class CustomFormatter(logging.Formatter):
    """Custom formatter to add emoji and handle different levels."""

    # Define colors for different levels (optional)
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Format strings for different levels
    log_format = f"{TRIANGLE_EMOJI} %(asctime)s [%(levelname)s] %(name)s: %(message)s"
    debug_format = f"{TRIANGLE_EMOJI} %(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"

    FORMATS = {
        logging.DEBUG: grey + debug_format + reset,
        logging.INFO: grey + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.log_format)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(level: str, log_file: Path | None = None):
    """
    Configures the root logger for the entire application.

    Args:
        level: The desired logging level string (e.g., "INFO", "DEBUG").
        log_file: Optional path to a file for logging.
    """
    log_level_str = level.upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)

    # Create file handler if path is provided
    file_handler = None
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(log_level)
            # Use a standard formatter for the file
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            log_file_path_str = str(log_file)
        except Exception as e:
            root_logger.error(f"Failed to set up file logging at {log_file}: {e}")
            log_file_path_str = "None"
    else:
        log_file_path_str = "None"

    # Set levels for third-party libraries
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("trianglengin").setLevel(logging.INFO)
    logging.getLogger("trimcts").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    root_logger.info(
        f"Logging configured. Level: {logging.getLevelName(log_level)}. Console: True. File: {log_file_path_str}"
    )

    # Return file handler for potential closing later (optional)
    return file_handler

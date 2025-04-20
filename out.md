File: .python-version
3.10.13


File: LICENSE
MIT License

Copyright (c) 2025 Luis Guilherme P. M.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File: MANIFEST.in
# File: MANIFEST.in
include README.md
include LICENSE
include requirements.txt
graft alphatriangle
graft tests
include .python-version
include pyproject.toml
global-exclude __pycache__
global-exclude *.py[co]

File: pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "alphatriangle"
version = "0.4.0"
authors = [{ name="Luis Guilherme P. M.", email="lgpelin92@gmail.com" }]
description = "AlphaZero implementation for a triangle puzzle game."
readme = "README.md"
license = { file="LICENSE" }
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Games/Entertainment :: Puzzle Games",
    "Development Status :: 3 - Alpha",
]
dependencies = [
    "pygame>=2.1.0",
    "numpy>=1.20.0",
    "torch>=2.0.0",
    "torchvision>=0.11.0",
    "cloudpickle>=2.0.0",
    "numba>=0.55.0",
    "mlflow>=1.20.0",
    "matplotlib>=3.5.0",
    "ray>=2.8.0",
    "pydantic>=2.0.0",
    "typing_extensions>=4.0.0",
    "typer[all]>=0.9.0", # Added typer for CLI
]

[project.urls]
"Homepage" = "https://github.com/lguibr/alphatriangle"
"Bug Tracker" = "https://github.com/lguibr/alphatriangle/issues"

[project.scripts]
alphatriangle = "alphatriangle.cli:app"

[tool.setuptools.packages.find]
# No 'where' needed, find searches from the project root by default
# It will find the 'alphatriangle' directory now.


[tool.setuptools.package-data]
"*" = ["*.txt", "*.md", "*.json"] # Include non-code files

# --- Tool Configurations (Optional but Recommended) ---

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "C4", "ARG", "SIM", "TCH", "PTH", "NPY"]
ignore = ["E501"] # Ignore line length errors if needed selectively

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true # Start with true, gradually reduce
# Add specific module ignores if necessary
# [[tool.mypy.overrides]]
# module = "some_missing_types_module.*"
# ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=alphatriangle --cov-report=term-missing" # Point coverage to the new package dir
testpaths = [
    "tests",
]

[tool.coverage.run]
omit = [
    "alphatriangle/cli.py", # Exclude CLI from coverage for now
    "alphatriangle/visualization/*", # Exclude visualization for now
    "alphatriangle/app.py",
    "run_*.py",
    "alphatriangle/training/logging_utils.py", # Logging utils can be hard to cover fully
    "alphatriangle/config/*", # Config models are mostly declarative
    "alphatriangle/data/schemas.py",
    "alphatriangle/structs/*",
    "alphatriangle/utils/types.py",
    "alphatriangle/mcts/core/types.py",
    "alphatriangle/rl/types.py",
    "*/__init__.py",
    "*/README.md",
]

[tool.coverage.report]
fail_under = 28 # Set a reasonable initial coverage target
show_missing = true

File: README.md

[![CI/CD Status](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/alphatriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/alphatriangle) - [![PyPI version](https://badge.fury.io/py/alphatriangle.svg)](https://badge.fury.io/py/alphatriangle)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) 


#  AlphaTriangle
<img src="bitmap.png" alt="AlphaTriangle Logo" width="300"/>


## Overview
AlphaTriangle is a project implementing an artificial intelligence agent based on AlphaZero principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through self-play reinforcement learning, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch).

The project includes:
*   A playable version of the triangle puzzle game using Pygame.
*   An implementation of the MCTS algorithm tailored for the game.
*   A deep neural network (policy and value heads) implemented in PyTorch, featuring convolutional layers and **optional Transformer Encoder layers**.
*   A reinforcement learning pipeline coordinating **parallel self-play (using Ray)**, data storage, and network training, managed by the `alphatriangle.training` module.
*   Visualization tools for interactive play, debugging, and monitoring training progress (**with near real-time plot updates**).
*   Experiment tracking using MLflow.
*   Unit tests for core components.
*   A command-line interface for easy execution.

## Core Technologies

*   **Python 3.10+**
*   **Pygame:** For game visualization and interactive modes.
*   **PyTorch:** For the deep learning model (CNNs, **optional Transformers**, Distributional Value Head) and training, with CUDA/MPS support.
*   **NumPy:** For numerical operations, especially state representation.
*   **Ray:** For parallelizing self-play data generation and statistics collection across multiple CPU cores/processes.
*   **Numba:** (Optional, used in `features.grid_features`) For performance optimization of specific grid calculations.
*   **Cloudpickle:** For serializing the experience replay buffer and training checkpoints.
*   **MLflow:** For logging parameters, metrics, and artifacts (checkpoints, buffers) during training runs.
*   **Pydantic:** For configuration management and data validation.
*   **Typer:** For the command-line interface.
*   **Pytest:** For running unit tests.

## Project Structure

```markdown
.
├── .github/workflows/      # GitHub Actions CI/CD
│   └── ci_cd.yml
├── .alphatriangle_data/    # Root directory for ALL persistent data (GITIGNORED)
│   ├── mlruns/             # MLflow tracking data
│   └── runs/               # Stores temporary/local artifacts per run
│       └── <run_name>/
│           ├── checkpoints/
│           ├── buffers/
│           ├── logs/
│           └── configs.json
├── alphatriangle/          # Source code for the project package
│   ├── __init__.py
│   ├── app.py
│   ├── cli.py              # CLI logic
│   ├── config/             # Pydantic configuration models
│   │   └── README.md
│   ├── data/               # Data saving/loading logic
│   │   └── README.md
│   ├── environment/        # Game rules, state, actions
│   │   └── README.md
│   ├── features/           # Feature extraction logic
│   │   └── README.md
│   ├── interaction/        # User input handling
│   │   └── README.md
│   ├── mcts/               # Monte Carlo Tree Search
│   │   └── README.md
│   ├── nn/                 # Neural network definition and wrapper
│   │   └── README.md
│   ├── rl/                 # RL components (Trainer, Buffer, Worker)
│   │   └── README.md
│   ├── stats/              # Statistics collection and plotting
│   │   └── README.md
│   ├── structs/            # Core data structures (Triangle, Shape)
│   │   └── README.md
│   ├── training/           # Training orchestration (Loop, Setup, Runners)
│   │   └── README.md
│   ├── utils/              # Shared utilities and types
│   │   └── README.md
│   └── visualization/      # Pygame rendering components
│       └── README.md
├── tests/                  # Unit tests
│   ├── ...
├── .gitignore
├── .python-version
├── LICENSE                 # License file (MIT)
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration
├── README.md               # This file
├── requirements.txt        # List of dependencies (also in pyproject.toml)
├── run_interactive.py      # Legacy script to run interactive modes
├── run_shape_editor.py     # Script to run the interactive shape definition tool
├── run_training_headless.py # Legacy script for headless training
└── run_training_visual.py  # Legacy script for visual training
```

## Key Modules (`alphatriangle`)

*   **`cli`:** Defines the command-line interface using Typer. ([`alphatriangle/cli.py`](alphatriangle/cli.py))
*   **`config`:** Centralized Pydantic configuration classes. ([`alphatriangle/config/README.md`](alphatriangle/config/README.md))
*   **`structs`:** Defines core, low-level data structures (`Triangle`, `Shape`) and constants. ([`alphatriangle/structs/README.md`](alphatriangle/structs/README.md))
*   **`environment`:** Defines the game rules, `GameState`, action encoding/decoding, and grid/shape *logic*. ([`alphatriangle/environment/README.md`](alphatriangle/environment/README.md))
*   **`features`:** Contains logic to convert `GameState` objects into numerical features (`StateType`). ([`alphatriangle/features/README.md`](alphatriangle/features/README.md))
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`AlphaTriangleNet`) and a wrapper class (`NeuralNetwork`). ([`alphatriangle/nn/README.md`](alphatriangle/nn/README.md))
*   **`mcts`:** Implements the Monte Carlo Tree Search algorithm (`Node`, `run_mcts_simulations`). ([`alphatriangle/mcts/README.md`](alphatriangle/mcts/README.md))
*   **`rl`:** Contains RL components: `Trainer` (network updates), `ExperienceBuffer` (data storage, **supports PER**), and `SelfPlayWorker` (Ray actor for parallel self-play). ([`alphatriangle/rl/README.md`](alphatriangle/rl/README.md))
*   **`training`:** Orchestrates the training process using `TrainingLoop`, managing workers, data flow, logging, and checkpoints. Includes `runners.py` for callable training functions. ([`alphatriangle/training/README.md`](alphatriangle/training/README.md))
*   **`stats`:** Contains the `StatsCollectorActor` (Ray actor) for asynchronous statistics collection and the `Plotter` class for rendering plots. ([`alphatriangle/stats/README.md`](alphatriangle/stats/README.md))
*   **`visualization`:** Uses Pygame to render the game state, previews, HUD, plots, etc. `DashboardRenderer` handles the training visualization layout. ([`alphatriangle/visualization/README.md`](alphatriangle/visualization/README.md))
*   **`interaction`:** Handles keyboard/mouse input for interactive modes via `InputHandler`. ([`alphatriangle/interaction/README.md`](alphatriangle/interaction/README.md))
*   **`data`:** Manages saving and loading of training artifacts (`DataManager`) using Pydantic schemas and `cloudpickle`. ([`alphatriangle/data/README.md`](alphatriangle/data/README.md))
*   **`utils`:** Provides common helper functions, shared type definitions, and geometry helpers. ([`alphatriangle/utils/README.md`](alphatriangle/utils/README.md))
*   **`app`:** Integrates components for interactive modes (`run_interactive.py`). ([`alphatriangle/app.py`](alphatriangle/app.py))

## Setup

1.  **Clone the repository (for development):**
    ```bash
    git clone https://github.com/lguibr/alphatriangle.git
    cd alphatriangle
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the package:**
    *   **For users:**
        ```bash
        pip install alphatriangle # Or pip install git+https://github.com/lguibr/alphatriangle.git
        ```
    *   **For developers (editable install):**
        ```bash
        pip install -e .
        # Install dev dependencies (optional, for running tests/linting)
        pip install pytest pytest-cov pytest-mock ruff mypy codecov twine build
        ```
    *Note: Ensure you have the correct PyTorch version installed for your system (CPU/CUDA/MPS). See [pytorch.org](https://pytorch.org/). Ray may have specific system requirements.*
4.  **(Optional) Add data directory to `.gitignore`:**
    Create or edit the `.gitignore` file in your project root and add the line:
    ```
    .alphatriangle_data/
    ```

## Running the Code (CLI)

Use the `alphatriangle` command:

*   **Show Help:**
    ```bash
    alphatriangle --help
    ```
*   **Interactive Play Mode:**
    ```bash
    alphatriangle play [--seed 42] [--log-level INFO]
    ```
*   **Interactive Debug Mode:**
    ```bash
    alphatriangle debug [--seed 42] [--log-level DEBUG]
    ```
*   **Run Training (Visual Mode):**
    ```bash
    alphatriangle train [--seed 42] [--log-level INFO]
    ```
*   **Run Training (Headless Mode):**
    ```bash
    alphatriangle train --headless [--seed 42] [--log-level INFO]
    # or
    alphatriangle train -H [--seed 42] [--log-level INFO]
    ```
*   **Shape Editor (Run directly):**
    ```bash
    python run_shape_editor.py
    ```
*   **Monitoring Training (MLflow UI):**
    While training (headless or visual), or after runs have completed, open a separate terminal in the project root and run:
    ```bash
    mlflow ui --backend-store-uri file:./.alphatriangle_data/mlruns
    ```
    Then navigate to `http://localhost:5000` (or the specified port) in your browser.
*   **Running Unit Tests (Development):**
    ```bash
    pytest tests/
    ```

## Configuration

All major parameters are defined in the Pydantic classes within the `alphatriangle/config/` directory. Modify these files to experiment with different settings. The `alphatriangle/config/validation.py` script performs basic checks on startup.

## Data Storage

All persistent data, including MLflow tracking data and run-specific artifacts, is stored within the `.alphatriangle_data/` directory in the project root, managed by the `DataManager` and MLflow.

## Maintainability

This project includes README files within each major `alphatriangle` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic.

File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=2.0.0
torchvision>=0.11.0
cloudpickle>=2.0.0
numba>=0.55.0
mlflow>=1.20.0
matplotlib>=3.5.0
ray>=2.8.0
pydantic>=2.0.0
typing_extensions>=4.0.0
pytest>=7.0.0
pytest-mock>=3.0.0
typer[all]>=0.9.0
pytest-cov>=3.0.0

File: alphatriangle\app.py
import logging

import pygame

from . import (
    config,
    environment,
    interaction,
    visualization,
)

logger = logging.getLogger(__name__)


class Application:
    """Main application integrating visualization and interaction."""

    def __init__(self, mode: str = "play"):
        self.vis_config = config.VisConfig()
        self.env_config = config.EnvConfig()
        self.mode = mode

        pygame.init()
        pygame.font.init()
        self.screen = self._setup_screen()
        self.clock = pygame.time.Clock()
        self.fonts = visualization.load_fonts()

        if self.mode in ["play", "debug"]:
            # Create GameState first
            self.game_state = environment.GameState(self.env_config)
            # Create Visualizer
            self.visualizer = visualization.Visualizer(
                self.screen, self.vis_config, self.env_config, self.fonts
            )
            # Create InputHandler, passing GameState and Visualizer
            self.input_handler = interaction.InputHandler(
                self.game_state, self.visualizer, self.mode, self.env_config
            )
        else:
            # Handle other modes or raise error if necessary
            logger.error(f"Unsupported application mode: {self.mode}")
            raise ValueError(f"Unsupported application mode: {self.mode}")

        self.running = True

    def _setup_screen(self) -> pygame.Surface:
        """Initializes the Pygame screen."""
        screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(f"{config.APP_NAME} - {self.mode.capitalize()} Mode")
        return screen

    def run(self):
        """Main application loop."""
        logger.info(f"Starting application in {self.mode} mode.")
        while self.running:
            # dt = ( # Unused variable
            #     self.clock.tick(self.vis_config.FPS) / 1000.0
            # )  # Delta time (unused currently)
            self.clock.tick(self.vis_config.FPS)  # Still tick the clock

            # Handle Input using InputHandler
            if self.input_handler:
                self.running = self.input_handler.handle_input()
                if not self.running:
                    break  # Exit loop if handle_input returns False
            else:
                # Fallback event handling if input_handler is not initialized (should not happen in play/debug)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Basic resize handling needed even without input handler
                    # Combine nested if statements
                    if event.type == pygame.VIDEORESIZE and self.visualizer:
                        try:
                            w, h = max(320, event.w), max(240, event.h)
                            # Update visualizer's screen reference
                            self.visualizer.screen = pygame.display.set_mode(
                                (w, h), pygame.RESIZABLE
                            )
                            # Invalidate visualizer's layout cache
                            self.visualizer.layout_rects = None
                        except pygame.error as e:
                            logger.error(f"Error resizing window: {e}")
                if not self.running:
                    break

            # Render using Visualizer
            if (
                self.mode in ["play", "debug"]
                and self.visualizer
                and self.game_state
                and self.input_handler
            ):
                # Get interaction state needed for rendering from InputHandler
                interaction_render_state = (
                    self.input_handler.get_render_interaction_state()
                )
                # Pass game state, mode, and interaction state to visualizer
                self.visualizer.render(
                    self.game_state,
                    self.mode,
                    **interaction_render_state,  # Unpack the dict as keyword arguments
                )
                pygame.display.flip()  # Update the full display

        logger.info("Application loop finished.")
        pygame.quit()


File: alphatriangle\cli.py
# File: alphatriangle/cli.py
import logging
import sys
from typing import Annotated

import typer

from alphatriangle import config, utils
from alphatriangle.app import Application
from alphatriangle.config import (
    MCTSConfig,
    PersistenceConfig,
    TrainConfig,
)

# --- REVERTED: Import from the re-exporting runners.py ---
from alphatriangle.training.runners import (
    run_training_headless_mode,
    run_training_visual_mode,
)

# --- END REVERTED ---

app = typer.Typer(
    name="alphatriangle",
    help="AlphaZero implementation for a triangle puzzle game.",
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
    utils.set_random_seeds(seed)

    mcts_config = MCTSConfig()
    config.print_config_info_and_validate(mcts_config)

    try:
        app_instance = Application(mode=mode)
        app_instance.run()
    except ImportError as e:
        logger.error(f"Runtime ImportError: {e}")
        logger.error("Please ensure all dependencies are installed.")
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
):
    """Run the AlphaTriangle training pipeline."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    train_config_override = TrainConfig()
    persist_config_override = PersistenceConfig()
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


File: alphatriangle\__init__.py


File: alphatriangle\config\app_config.py
APP_NAME: str = "AlphaTriangle"


File: alphatriangle\config\env_config.py
# File: alphatriangle/config/env_config.py
# No changes needed for this refactoring step. Keep the existing content.
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class EnvConfig(BaseModel):
    """Configuration for the game environment (Pydantic model)."""

    ROWS: int = Field(default=8, gt=0)
    # Provide a default that matches the default ROWS
    COLS_PER_ROW: list[int] = Field(default=[9, 11, 13, 15, 15, 13, 11, 9])
    COLS: int = Field(default=15, gt=0)
    NUM_SHAPE_SLOTS: int = Field(default=3, gt=0)
    MIN_LINE_LENGTH: int = Field(default=3, gt=0)

    # --- Reward System Constants (v3) ---
    REWARD_PER_PLACED_TRIANGLE: float = Field(default=0.01)
    REWARD_PER_CLEARED_TRIANGLE: float = Field(default=0.5)
    REWARD_PER_STEP_ALIVE: float = Field(default=0.005)
    PENALTY_GAME_OVER: float = Field(default=-10.0)
    # --- End Reward System Constants ---

    @field_validator("COLS_PER_ROW")
    @classmethod
    def check_cols_per_row_length(cls, v: list[int], info) -> list[int]:
        data = info.data if info.data else info.values
        rows = data.get("ROWS")
        if rows is None:
            return v
        if len(v) != rows:
            raise ValueError(f"COLS_PER_ROW length ({len(v)}) must equal ROWS ({rows})")
        if any(width <= 0 for width in v):
            raise ValueError("All values in COLS_PER_ROW must be positive.")
        return v

    @model_validator(mode="after")
    def check_cols_match_max_cols_per_row(self) -> "EnvConfig":
        """Ensure COLS is at least the maximum width required by any row."""
        if hasattr(self, "COLS_PER_ROW") and self.COLS_PER_ROW:
            max_row_width = max(self.COLS_PER_ROW, default=0)
            if max_row_width > self.COLS:
                raise ValueError(
                    f"COLS ({self.COLS}) must be >= the maximum value in COLS_PER_ROW ({max_row_width})"
                )
        elif not hasattr(self, "COLS_PER_ROW"):
            pass
        return self

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def ACTION_DIM(self) -> int:
        """Total number of possible actions (shape_slot * row * col)."""
        if (
            hasattr(self, "NUM_SHAPE_SLOTS")
            and hasattr(self, "ROWS")
            and hasattr(self, "COLS")
        ):
            return self.NUM_SHAPE_SLOTS * self.ROWS * self.COLS
        return 0


EnvConfig.model_rebuild(force=True)


File: alphatriangle\config\mcts_config.py
# File: alphatriangle/config/mcts_config.py
from pydantic import BaseModel, Field, field_validator


class MCTSConfig(BaseModel):
    """
    Configuration for Monte Carlo Tree Search (Pydantic model).
    --- TUNED FOR INCREASED EXPLORATION & DEPTH ---
    """

    num_simulations: int = Field(default=2048, ge=1)
    puct_coefficient: float = Field(default=2.0, gt=0)
    temperature_initial: float = Field(default=1.0, ge=0)
    temperature_final: float = Field(default=0.1, ge=0)
    temperature_anneal_steps: int = Field(default=100, ge=0)
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1.0)
    max_search_depth: int = Field(default=64, ge=1)

    @field_validator("temperature_final")
    @classmethod
    def check_temp_final_le_initial(cls, v: float, info) -> float:
        data = info.data if info.data else info.values
        initial_temp = data.get("temperature_initial")
        if initial_temp is not None and v > initial_temp:
            raise ValueError(
                "temperature_final cannot be greater than temperature_initial"
            )
        return v


MCTSConfig.model_rebuild(force=True)


File: alphatriangle\config\model_config.py
# File: alphatriangle/config/model_config.py
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the Neural Network model (Pydantic model).
    --- TUNED FOR SMALLER CAPACITY (~3M Params Target, Laptop Feasible) ---
    """

    # Input channels for the grid (e.g., 1 for occupancy, more for history/colors)
    GRID_INPUT_CHANNELS: int = Field(default=1, gt=0)

    # --- CNN Architecture Parameters ---
    CONV_FILTERS: list[int] = Field(default=[32, 64, 128])  # Smaller CNN
    CONV_KERNEL_SIZES: list[int | tuple[int, int]] = Field(default=[3, 3, 3])
    CONV_STRIDES: list[int | tuple[int, int]] = Field(default=[1, 1, 1])
    CONV_PADDING: list[int | tuple[int, int] | str] = Field(default=[1, 1, 1])

    # --- Residual Block Parameters ---
    NUM_RESIDUAL_BLOCKS: int = Field(default=2, ge=0)  # Fewer blocks
    RESIDUAL_BLOCK_FILTERS: int = Field(default=128, gt=0)  # Match last conv filter

    # --- Transformer Parameters (Optional) ---
    USE_TRANSFORMER: bool = Field(default=True)  # Keep Transformer enabled
    TRANSFORMER_DIM: int = Field(default=128, gt=0)  # Match Res block filters
    TRANSFORMER_HEADS: int = Field(default=4, gt=0)  # Moderate heads
    TRANSFORMER_LAYERS: int = Field(default=2, ge=0)  # Fewer layers
    TRANSFORMER_FC_DIM: int = Field(default=256, gt=0)  # Moderate feedforward dim

    # --- Fully Connected Layers ---
    FC_DIMS_SHARED: list[int] = Field(default=[128])  # Single shared layer

    # --- Policy Head ---
    POLICY_HEAD_DIMS: list[int] = Field(default=[128])  # Single policy layer

    # --- Distributional Value Head Parameters ---
    NUM_VALUE_ATOMS: int = Field(default=51, gt=1)  # Standard C51 atoms
    VALUE_MIN: float = Field(default=-10.0)  # Reasonable expected value range
    VALUE_MAX: float = Field(default=10.0)  # Reasonable expected value range

    # --- Value Head Dims ---
    VALUE_HEAD_DIMS: list[int] = Field(default=[128])  # Single value layer

    # --- Other Hyperparameters ---
    ACTIVATION_FUNCTION: Literal["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"] = Field(
        default="ReLU"
    )
    USE_BATCH_NORM: bool = Field(default=True)

    # --- Input Feature Dimension ---
    # Dimension of the non-grid feature vector concatenated after CNN/Transformer.
    # Must match the output of features.extractor.get_combined_other_features.
    OTHER_NN_INPUT_FEATURES_DIM: int = Field(default=30, gt=0)  # Keep default

    @model_validator(mode="after")
    def check_conv_layers_consistency(self) -> "ModelConfig":
        # Ensure attributes exist before checking lengths
        if (
            hasattr(self, "CONV_FILTERS")
            and hasattr(self, "CONV_KERNEL_SIZES")
            and hasattr(self, "CONV_STRIDES")
            and hasattr(self, "CONV_PADDING")
        ):
            n_filters = len(self.CONV_FILTERS)
            if not (
                len(self.CONV_KERNEL_SIZES) == n_filters
                and len(self.CONV_STRIDES) == n_filters
                and len(self.CONV_PADDING) == n_filters
            ):
                raise ValueError(
                    "Lengths of CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES, and CONV_PADDING must match."
                )
        return self

    @model_validator(mode="after")
    def check_residual_filter_match(self) -> "ModelConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "NUM_RESIDUAL_BLOCKS")
            and self.NUM_RESIDUAL_BLOCKS > 0
            and hasattr(self, "CONV_FILTERS")
            and self.CONV_FILTERS
            and hasattr(self, "RESIDUAL_BLOCK_FILTERS")
            and self.CONV_FILTERS[-1] != self.RESIDUAL_BLOCK_FILTERS
        ):
            # This warning is now handled by the projection layer in the model if needed
            pass  # Model handles projection if needed
        return self

    @model_validator(mode="after")
    def check_transformer_config(self) -> "ModelConfig":
        # Ensure attributes exist before checking
        if hasattr(self, "USE_TRANSFORMER") and self.USE_TRANSFORMER:
            if not hasattr(self, "TRANSFORMER_LAYERS") or self.TRANSFORMER_LAYERS < 0:
                raise ValueError("TRANSFORMER_LAYERS cannot be negative.")
            if self.TRANSFORMER_LAYERS > 0:
                if not hasattr(self, "TRANSFORMER_DIM") or self.TRANSFORMER_DIM <= 0:
                    raise ValueError(
                        "TRANSFORMER_DIM must be positive if TRANSFORMER_LAYERS > 0."
                    )
                if (
                    not hasattr(self, "TRANSFORMER_HEADS")
                    or self.TRANSFORMER_HEADS <= 0
                ):
                    raise ValueError(
                        "TRANSFORMER_HEADS must be positive if TRANSFORMER_LAYERS > 0."
                    )
                if self.TRANSFORMER_DIM % self.TRANSFORMER_HEADS != 0:
                    raise ValueError(
                        f"TRANSFORMER_DIM ({self.TRANSFORMER_DIM}) must be divisible by TRANSFORMER_HEADS ({self.TRANSFORMER_HEADS})."
                    )
                if (
                    not hasattr(self, "TRANSFORMER_FC_DIM")
                    or self.TRANSFORMER_FC_DIM <= 0
                ):
                    raise ValueError(
                        "TRANSFORMER_FC_DIM must be positive if TRANSFORMER_LAYERS > 0."
                    )
        return self

    @model_validator(mode="after")
    def check_transformer_dim_consistency(self) -> "ModelConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "USE_TRANSFORMER")
            and self.USE_TRANSFORMER
            and hasattr(self, "TRANSFORMER_LAYERS")
            and self.TRANSFORMER_LAYERS > 0
            and hasattr(self, "CONV_FILTERS")
            and self.CONV_FILTERS
            and hasattr(self, "TRANSFORMER_DIM")
        ):
            cnn_output_channels = (
                self.RESIDUAL_BLOCK_FILTERS
                if hasattr(self, "NUM_RESIDUAL_BLOCKS") and self.NUM_RESIDUAL_BLOCKS > 0
                else self.CONV_FILTERS[-1]
            )
            if cnn_output_channels != self.TRANSFORMER_DIM:
                # This is handled by an input projection layer in the model now
                pass  # Model handles projection
        return self

    @model_validator(mode="after")
    def check_value_distribution_params(self) -> "ModelConfig":
        if (
            hasattr(self, "VALUE_MIN")
            and hasattr(self, "VALUE_MAX")
            and self.VALUE_MIN >= self.VALUE_MAX
        ):
            raise ValueError("VALUE_MIN must be strictly less than VALUE_MAX.")
        return self


ModelConfig.model_rebuild(force=True)


File: alphatriangle\config\persistence_config.py
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class PersistenceConfig(BaseModel):
    """Configuration for saving/loading artifacts (Pydantic model)."""

    ROOT_DATA_DIR: str = Field(default=".alphatriangle_data")
    RUNS_DIR_NAME: str = Field(default="runs")
    MLFLOW_DIR_NAME: str = Field(default="mlruns")

    CHECKPOINT_SAVE_DIR_NAME: str = Field(default="checkpoints")
    BUFFER_SAVE_DIR_NAME: str = Field(default="buffers")
    GAME_STATE_SAVE_DIR_NAME: str = Field(default="game_states")
    LOG_DIR_NAME: str = Field(default="logs")

    LATEST_CHECKPOINT_FILENAME: str = Field(default="latest.pkl")
    BEST_CHECKPOINT_FILENAME: str = Field(default="best.pkl")
    BUFFER_FILENAME: str = Field(default="buffer.pkl")
    CONFIG_FILENAME: str = Field(default="configs.json")

    RUN_NAME: str = Field(default="default_run")

    SAVE_GAME_STATES: bool = Field(default=False)
    GAME_STATE_SAVE_FREQ_EPISODES: int = Field(default=5, ge=1)

    SAVE_BUFFER: bool = Field(default=True)
    BUFFER_SAVE_FREQ_STEPS: int = Field(default=10, ge=1)

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the file URI for MLflow tracking using pathlib."""
        # Ensure attributes exist before calculating
        if hasattr(self, "ROOT_DATA_DIR") and hasattr(self, "MLFLOW_DIR_NAME"):
            abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
            return abs_path.as_uri()
        return ""

    def get_run_base_dir(self, run_name: str | None = None) -> str:
        """Gets the base directory for a specific run."""
        # Ensure attributes exist before calculating
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "RUNS_DIR_NAME"):
            return ""  # Fallback
        name = run_name if run_name else self.RUN_NAME
        return str(Path(self.ROOT_DATA_DIR).joinpath(self.RUNS_DIR_NAME, name))

    def get_mlflow_abs_path(self) -> str:
        """Gets the absolute OS path to the MLflow directory as a string."""
        # Ensure attributes exist before calculating
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "MLFLOW_DIR_NAME"):
            return ""  # Fallback
        abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
        return str(abs_path)


# Ensure model is rebuilt after changes
PersistenceConfig.model_rebuild(force=True)


File: alphatriangle\config\README.md
# File: alphatriangle/config/README.md
# Configuration Module (`alphatriangle.config`)

## Purpose and Architecture

This module centralizes all configuration parameters for the AlphaTriangle project. It uses separate **Pydantic models** for different aspects of the application (environment, model, training, visualization, persistence) to promote modularity, clarity, and automatic validation.

-   **Modularity:** Separating configurations makes it easier to manage parameters for different components.
-   **Type Safety & Validation:** Using Pydantic models (`BaseModel`) provides strong type hinting, automatic parsing, and validation of configuration values based on defined types and constraints (e.g., `Field(gt=0)`).
-   **Validation Script:** The [`validation.py`](validation.py) script instantiates all configuration models, triggering Pydantic's validation, and prints a summary.
-   **Dynamic Defaults:** Some configurations, like `RUN_NAME` in `TrainConfig`, use `default_factory` for dynamic defaults (e.g., timestamp).
-   **Computed Fields:** Properties like `ACTION_DIM` in `EnvConfig` or `MLFLOW_TRACKING_URI` in `PersistenceConfig` are defined using `@computed_field` for clarity.
-   **Tuned Defaults:** The default values in `TrainConfig` and `ModelConfig` are now tuned for **more substantial learning runs** compared to the previous quick-testing defaults.

## Exposed Interfaces

-   **Pydantic Models:**
    -   [`EnvConfig`](env_config.py): Environment parameters (grid size, shapes).
    -   [`ModelConfig`](model_config.py): Neural network architecture parameters. **Defaults tuned for larger capacity.**
    -   [`TrainConfig`](train_config.py): Training loop hyperparameters (batch size, learning rate, workers, **PER settings**, etc.). **Defaults tuned for longer runs.**
    -   [`VisConfig`](vis_config.py): Visualization parameters (screen size, FPS, layout).
    -   [`PersistenceConfig`](persistence_config.py): Data saving/loading parameters (directories, filenames).
    -   [`MCTSConfig`](mcts_config.py): MCTS parameters (simulations, exploration constants, temperature).
-   **Constants:**
    -   [`APP_NAME`](app_config.py): The name of the application.
-   **Functions:**
    -   `print_config_info_and_validate(mcts_config_instance: MCTSConfig)`: Validates and prints a summary of all configurations by instantiating the Pydantic models.

## Dependencies

This module primarily defines configurations and relies heavily on **Pydantic**.

-   **`pydantic`**: The core library used for defining models and validation.
-   **Standard Libraries:** `typing`, `time`, `os`, `logging`, `pathlib`.

---

**Note:** Please keep this README updated when adding, removing, or significantly modifying configuration parameters or the structure of the Pydantic models. Accurate documentation is crucial for maintainability.

File: alphatriangle\config\train_config.py
# File: alphatriangle/config/train_config.py
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Get logger instance
logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    """
    Configuration for the training process (Pydantic model).
    --- TUNED FOR MORE SUBSTANTIAL LEARNING RUNS ---
    """

    RUN_NAME: str = Field(
        # More descriptive default run name
        default_factory=lambda: f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    LOAD_CHECKPOINT_PATH: str | None = Field(default=None)
    LOAD_BUFFER_PATH: str | None = Field(default=None)
    AUTO_RESUME_LATEST: bool = Field(default=True)  # Resume if possible
    # --- DEVICE: Defaults to 'auto' for automatic detection (CUDA > MPS > CPU) ---
    # This controls the device for the main Trainer process.
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="auto")
    RANDOM_SEED: int = Field(default=42)

    # --- Training Loop ---
    MAX_TRAINING_STEPS: int | None = Field(default=100_000, ge=1)  # Target steps

    # --- Workers & Batching ---
    NUM_SELF_PLAY_WORKERS: int = Field(
        default=8,  # Default workers, capped by cores
        ge=1,
        description="Suggested number of workers. Actual number may be adjusted based on detected CPU cores.",
    )
    # --- WORKER_DEVICE: Defaults to 'cpu' for self-play workers ---
    # Workers run MCTS and NN eval; CPU is often sufficient and avoids GPU contention.
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="cpu")
    BATCH_SIZE: int = Field(default=128, ge=1)  # Moderate batch size
    BUFFER_CAPACITY: int = Field(default=200_000, ge=1)  # Larger buffer
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(
        default=20_000,
        ge=1,  # Start training after 10% fill
    )
    WORKER_UPDATE_FREQ_STEPS: int = Field(default=500, ge=1)

    # --- N-Step Returns ---
    N_STEP_RETURNS: int = Field(default=5, ge=1)  # 5-step returns
    GAMMA: float = Field(default=0.99, gt=0, le=1.0)  # Discount factor

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field(default="AdamW")
    LEARNING_RATE: float = Field(default=2e-4, gt=0)
    WEIGHT_DECAY: float = Field(default=1e-4, ge=0)
    GRADIENT_CLIP_VALUE: float | None = Field(default=1.0)

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Literal["StepLR", "CosineAnnealingLR"] | None = Field(
        default="CosineAnnealingLR"
    )
    # T_MAX will be set automatically based on new MAX_TRAINING_STEPS
    LR_SCHEDULER_T_MAX: int | None = Field(default=None)
    LR_SCHEDULER_ETA_MIN: float = Field(default=1e-6, ge=0)

    # --- Loss Weights ---
    POLICY_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    ENTROPY_BONUS_WEIGHT: float = Field(default=0.001, ge=0)  # Small entropy bonus

    # --- Checkpointing ---
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(default=2500, ge=1)  # Save every 2500 steps

    # --- Prioritized Experience Replay (PER) ---
    USE_PER: bool = Field(default=True)  # Enable PER
    PER_ALPHA: float = Field(default=0.6, ge=0)
    PER_BETA_INITIAL: float = Field(default=0.4, ge=0, le=1.0)
    PER_BETA_FINAL: float = Field(default=1.0, ge=0, le=1.0)
    # Anneal steps will be set automatically based on MAX_TRAINING_STEPS
    PER_BETA_ANNEAL_STEPS: int | None = Field(default=None)
    PER_EPSILON: float = Field(default=1e-5, gt=0)

    # --- Model Compilation ---
    COMPILE_MODEL: bool = Field(
        default=True,
        description=(
            "Enable torch.compile() for potential speedup (Trainer only). Requires PyTorch 2.0+. "
            "May have initial overhead or compatibility issues on some setups/GPUs "
            "(especially non-CUDA backends like MPS). Set to False if encountering problems. "
            "The application will attempt compilation and fall back gracefully if it fails."
        ),
    )

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        # Ensure attributes exist before comparing
        if (
            hasattr(self, "MIN_BUFFER_SIZE_TO_TRAIN")
            and hasattr(self, "BUFFER_CAPACITY")
            and self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY
        ):
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        if (
            hasattr(self, "BATCH_SIZE")
            and hasattr(self, "BUFFER_CAPACITY")
            and self.BATCH_SIZE > self.BUFFER_CAPACITY
        ):
            raise ValueError("BATCH_SIZE cannot be greater than BUFFER_CAPACITY.")
        if (
            hasattr(self, "BATCH_SIZE")
            and hasattr(self, "MIN_BUFFER_SIZE_TO_TRAIN")
            and self.BATCH_SIZE > self.MIN_BUFFER_SIZE_TO_TRAIN
        ):
            # Allow batch size to be larger than min buffer size (will just wait longer)
            pass
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "LR_SCHEDULER_TYPE")
            and self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and hasattr(self, "LR_SCHEDULER_T_MAX")
            and self.LR_SCHEDULER_T_MAX is None  # Only set if not manually specified
        ):
            if (
                hasattr(self, "MAX_TRAINING_STEPS")
                and self.MAX_TRAINING_STEPS is not None
            ):
                # Assign to self.LR_SCHEDULER_T_MAX only if MAX_TRAINING_STEPS is valid
                if self.MAX_TRAINING_STEPS >= 1:
                    self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
                    logger.info(
                        f"Set LR_SCHEDULER_T_MAX to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                    )
                else:
                    # Handle invalid MAX_TRAINING_STEPS case if necessary
                    self.LR_SCHEDULER_T_MAX = 100_000  # Fallback (matches new default)
                    logger.warning(
                        f"Warning: MAX_TRAINING_STEPS is invalid ({self.MAX_TRAINING_STEPS}), setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}"
                    )
            else:
                self.LR_SCHEDULER_T_MAX = 100_000  # Fallback (matches new default)
                logger.warning(
                    f"Warning: MAX_TRAINING_STEPS is None, setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}"
                )

        if (
            hasattr(self, "LR_SCHEDULER_T_MAX")
            and self.LR_SCHEDULER_T_MAX is not None
            and self.LR_SCHEDULER_T_MAX <= 0
        ):
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "USE_PER")
            and self.USE_PER
            and hasattr(self, "PER_BETA_ANNEAL_STEPS")
            and self.PER_BETA_ANNEAL_STEPS is None  # Only set if not manually specified
        ):
            if (
                hasattr(self, "MAX_TRAINING_STEPS")
                and self.MAX_TRAINING_STEPS is not None
            ):
                # Assign to self.PER_BETA_ANNEAL_STEPS only if MAX_TRAINING_STEPS is valid
                if self.MAX_TRAINING_STEPS >= 1:
                    self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
                    logger.info(
                        f"Set PER_BETA_ANNEAL_STEPS to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                    )
                else:
                    # Handle invalid MAX_TRAINING_STEPS case if necessary
                    self.PER_BETA_ANNEAL_STEPS = (
                        100_000  # Fallback (matches new default)
                    )
                    logger.warning(
                        f"Warning: MAX_TRAINING_STEPS is invalid ({self.MAX_TRAINING_STEPS}), setting PER_BETA_ANNEAL_STEPS to default {self.PER_BETA_ANNEAL_STEPS}"
                    )
            else:
                self.PER_BETA_ANNEAL_STEPS = 100_000  # Fallback (matches new default)
                logger.warning(
                    f"Warning: MAX_TRAINING_STEPS is None, setting PER_BETA_ANNEAL_STEPS to default {self.PER_BETA_ANNEAL_STEPS}"
                )

        if (
            hasattr(self, "PER_BETA_ANNEAL_STEPS")
            and self.PER_BETA_ANNEAL_STEPS is not None
            and self.PER_BETA_ANNEAL_STEPS <= 0
        ):
            raise ValueError("PER_BETA_ANNEAL_STEPS must be positive if set.")
        return self

    @field_validator("GRADIENT_CLIP_VALUE")
    @classmethod
    def check_gradient_clip(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v

    @field_validator("PER_BETA_FINAL")
    @classmethod
    def check_per_beta_final(cls, v: float, info) -> float:
        # info.data might not be available during initial validation in Pydantic v2
        # Check 'values' if info.data is empty
        data = info.data if info.data else info.values
        initial_beta = data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v


# Ensure model is rebuilt after changes
TrainConfig.model_rebuild(force=True)


File: alphatriangle\config\validation.py
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .env_config import EnvConfig
from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig
from .vis_config import VisConfig

logger = logging.getLogger(__name__)


def print_config_info_and_validate(mcts_config_instance: MCTSConfig | None):
    """Prints configuration summary and performs validation using Pydantic."""
    print("-" * 40)
    print("Configuration Validation & Summary")
    print("-" * 40)
    all_valid = True
    configs_validated: dict[str, Any] = {}

    config_classes: dict[str, type[BaseModel]] = {
        "Environment": EnvConfig,
        "Model": ModelConfig,
        "Training": TrainConfig,
        "Visualization": VisConfig,
        "Persistence": PersistenceConfig,
        "MCTS": MCTSConfig,
    }

    for name, ConfigClass in config_classes.items():
        instance: BaseModel | None = None
        try:
            if name == "MCTS":
                if mcts_config_instance is not None:
                    instance = MCTSConfig.model_validate(
                        mcts_config_instance.model_dump()
                    )
                    print(f"[{name}] - Instance provided & validated OK")
                else:
                    instance = ConfigClass()
                    print(f"[{name}] - Validated OK (Instantiated Default)")
            else:
                instance = ConfigClass()
                print(f"[{name}] - Validated OK")
            configs_validated[name] = instance
        except ValidationError as e:
            logger.error(f"Validation failed for {name} Config:")
            logger.error(e)
            all_valid = False
            configs_validated[name] = None
        except Exception as e:
            logger.error(
                f"Unexpected error instantiating/validating {name} Config: {e}"
            )
            all_valid = False
            configs_validated[name] = None

    print("-" * 40)
    print("Configuration Values:")
    print("-" * 40)

    for name, instance in configs_validated.items():
        print(f"--- {name} Config ---")
        if instance:
            dump_data = instance.model_dump()
            for field_name, value in dump_data.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {field_name}: [List with {len(value)} items]")
                elif isinstance(value, dict) and len(value) > 5:
                    print(f"  {field_name}: {{Dict with {len(value)} keys}}")
                else:
                    print(f"  {field_name}: {value}")
        else:
            print("  <Validation Failed>")
        print("-" * 20)

    print("-" * 40)
    if not all_valid:
        logger.critical("Configuration validation failed. Please check errors above.")
        raise ValueError("Invalid configuration settings.")
    else:
        logger.info("All configurations validated successfully.")
    print("-" * 40)


File: alphatriangle\config\vis_config.py
from pydantic import BaseModel, Field


class VisConfig(BaseModel):
    """Configuration for visualization (Pydantic model)."""

    FPS: int = Field(default=30, gt=0)
    SCREEN_WIDTH: int = Field(default=1000, gt=0)
    SCREEN_HEIGHT: int = Field(default=800, gt=0)

    # Layout
    GRID_AREA_RATIO: float = Field(default=0.7, gt=0, le=1.0)
    PREVIEW_AREA_WIDTH: int = Field(default=150, gt=0)
    PADDING: int = Field(default=10, ge=0)
    HUD_HEIGHT: int = Field(default=40, ge=0)

    # Fonts (sizes)
    FONT_UI_SIZE: int = Field(default=24, gt=0)
    FONT_SCORE_SIZE: int = Field(default=30, gt=0)
    FONT_HELP_SIZE: int = Field(default=18, gt=0)

    # Preview Area
    PREVIEW_PADDING: int = Field(default=5, ge=0)
    PREVIEW_BORDER_WIDTH: int = Field(default=1, ge=0)
    PREVIEW_SELECTED_BORDER_WIDTH: int = Field(default=3, ge=0)
    PREVIEW_INNER_PADDING: int = Field(default=2, ge=0)


VisConfig.model_rebuild(force=True)


File: alphatriangle\config\__init__.py
from .app_config import APP_NAME
from .env_config import EnvConfig
from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig
from .validation import print_config_info_and_validate
from .vis_config import VisConfig

__all__ = [
    "APP_NAME",
    "EnvConfig",
    "ModelConfig",
    "PersistenceConfig",
    "TrainConfig",
    "VisConfig",
    "MCTSConfig",
    "print_config_info_and_validate",
]


File: alphatriangle\data\data_manager.py
# File: alphatriangle/data/data_manager.py
# No changes needed, already expects ActorHandle | None
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import ray  # Import ray
from pydantic import ValidationError

from .path_manager import PathManager
from .schemas import CheckpointData, LoadedTrainingState
from .serializer import Serializer

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..config import PersistenceConfig, TrainConfig
    from ..nn import NeuralNetwork
    from ..rl.core.buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates loading and saving of training artifacts using PathManager and Serializer.
    Handles MLflow artifact logging.
    """

    def __init__(
        self, persist_config: "PersistenceConfig", train_config: "TrainConfig"
    ):
        self.persist_config = persist_config
        self.train_config = train_config
        # Ensure PersistenceConfig reflects the current run name from TrainConfig
        self.persist_config.RUN_NAME = self.train_config.RUN_NAME

        self.path_manager = PathManager(self.persist_config)
        self.serializer = Serializer()

        self.path_manager.create_run_directories()
        logger.info(
            f"DataManager initialized. Current Run Name: {self.persist_config.RUN_NAME}. Run directory: {self.path_manager.run_base_dir}"
        )

    def load_initial_state(self) -> LoadedTrainingState:
        """
        Loads the initial training state using PathManager and Serializer.
        Returns a LoadedTrainingState object containing the deserialized data.
        Handles AUTO_RESUME_LATEST logic.
        """
        loaded_state = LoadedTrainingState()
        checkpoint_path = self.path_manager.determine_checkpoint_to_load(
            self.train_config.LOAD_CHECKPOINT_PATH,
            self.train_config.AUTO_RESUME_LATEST,
        )
        checkpoint_run_name: str | None = None

        # --- Load Checkpoint (Model + Optimizer + Stats) ---
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                loaded_checkpoint_model = self.serializer.load_checkpoint(
                    checkpoint_path
                )
                if loaded_checkpoint_model:
                    loaded_state.checkpoint_data = loaded_checkpoint_model
                    checkpoint_run_name = (
                        loaded_state.checkpoint_data.run_name
                    )  # Store run name
                    logger.info(
                        f"Checkpoint loaded and validated (Run: {loaded_state.checkpoint_data.run_name}, Step: {loaded_state.checkpoint_data.global_step})"
                    )
                else:
                    logger.error(
                        f"Loading checkpoint from {checkpoint_path} failed or returned None."
                    )
            except (ValidationError, Exception) as e:
                logger.error(
                    f"Error loading/validating checkpoint from {checkpoint_path}: {e}",
                    exc_info=True,
                )

        # --- Load Buffer ---
        if self.persist_config.SAVE_BUFFER:
            buffer_path = self.path_manager.determine_buffer_to_load(
                self.train_config.LOAD_BUFFER_PATH,
                self.train_config.AUTO_RESUME_LATEST,
                checkpoint_run_name,  # Pass run name from loaded checkpoint
            )
            if buffer_path:
                logger.info(f"Loading buffer: {buffer_path}")
                try:
                    loaded_buffer_model = self.serializer.load_buffer(buffer_path)
                    if loaded_buffer_model:
                        loaded_state.buffer_data = loaded_buffer_model
                        logger.info(
                            f"Buffer loaded and validated. Size: {len(loaded_state.buffer_data.buffer_list)}"
                        )
                    else:
                        logger.error(
                            f"Loading buffer from {buffer_path} failed or returned None."
                        )
                except (ValidationError, Exception) as e:
                    logger.error(
                        f"Failed to load/validate experience buffer from {buffer_path}: {e}",
                        exc_info=True,
                    )

        if not loaded_state.checkpoint_data and not loaded_state.buffer_data:
            logger.info("No checkpoint or buffer loaded. Starting fresh.")

        return loaded_state

    def save_training_state(
        self,
        nn: "NeuralNetwork",
        optimizer: "Optimizer",
        stats_collector_actor: ray.actor.ActorHandle | None,
        buffer: "ExperienceBuffer",
        global_step: int,
        episodes_played: int,
        total_simulations_run: int,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Saves the training state using Serializer and PathManager."""
        run_name = self.persist_config.RUN_NAME
        logger.info(
            f"Saving training state for run '{run_name}' at step {global_step}. Final={is_final}, Best={is_best}"
        )

        stats_collector_state = {}
        if stats_collector_actor is not None:
            try:
                # Call remote method on the handle
                stats_state_ref = stats_collector_actor.get_state.remote()
                stats_collector_state = ray.get(stats_state_ref, timeout=5.0)
            except Exception as e:
                logger.error(
                    f"Error fetching state from StatsCollectorActor for saving: {e}",
                    exc_info=True,
                )

        # --- Save Checkpoint ---
        saved_checkpoint_path: Path | None = None
        try:
            checkpoint_data = CheckpointData(
                run_name=run_name,
                global_step=global_step,
                episodes_played=episodes_played,
                total_simulations_run=total_simulations_run,
                model_config_dict=nn.model_config.model_dump(),
                env_config_dict=nn.env_config.model_dump(),
                model_state_dict=nn.get_weights(),
                optimizer_state_dict=self.serializer.prepare_optimizer_state(optimizer),
                stats_collector_state=stats_collector_state,
            )
            step_checkpoint_path = self.path_manager.get_checkpoint_path(
                step=global_step, is_final=is_final
            )
            self.serializer.save_checkpoint(checkpoint_data, step_checkpoint_path)
            saved_checkpoint_path = step_checkpoint_path  # Store path if save succeeded

            # Update latest/best links
            self.path_manager.update_checkpoint_links(
                step_checkpoint_path, is_best=is_best
            )

        except ValidationError as e:
            logger.error(f"Failed to create CheckpointData model: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

        # --- Save Buffer ---
        saved_buffer_path: Path | None = None
        if self.persist_config.SAVE_BUFFER:
            try:
                buffer_data = self.serializer.prepare_buffer_data(buffer)
                if buffer_data:
                    buffer_path = self.path_manager.get_buffer_path(
                        step=global_step, is_final=is_final
                    )
                    self.serializer.save_buffer(buffer_data, buffer_path)
                    saved_buffer_path = buffer_path  # Store path if save succeeded
                    # Update default buffer link
                    self.path_manager.update_buffer_link(buffer_path)
                else:
                    logger.warning("Buffer data preparation failed, buffer not saved.")
            except ValidationError as e:
                logger.error(f"Failed to create BufferData model: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Failed to save buffer: {e}", exc_info=True)

        # --- Log Artifacts to MLflow ---
        self._log_artifacts(saved_checkpoint_path, saved_buffer_path, is_best)

    def _log_artifacts(
        self,
        checkpoint_path: Path | None,
        buffer_path: Path | None,
        is_best: bool,
    ):
        """Logs saved checkpoint and buffer files to MLflow."""
        try:
            if checkpoint_path and checkpoint_path.exists():
                ckpt_artifact_path = self.persist_config.CHECKPOINT_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(checkpoint_path), artifact_path=ckpt_artifact_path
                )
                latest_path = self.path_manager.get_checkpoint_path(is_latest=True)
                if latest_path.exists():
                    mlflow.log_artifact(
                        str(latest_path), artifact_path=ckpt_artifact_path
                    )
                if is_best:
                    best_path = self.path_manager.get_checkpoint_path(is_best=True)
                    if best_path.exists():
                        mlflow.log_artifact(
                            str(best_path), artifact_path=ckpt_artifact_path
                        )
                logger.info(
                    f"Logged checkpoint artifacts to MLflow path: {ckpt_artifact_path}"
                )
            if buffer_path and buffer_path.exists():
                buffer_artifact_path = self.persist_config.BUFFER_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(buffer_path), artifact_path=buffer_artifact_path
                )
                default_buffer_path = self.path_manager.get_buffer_path()
                if default_buffer_path.exists():
                    mlflow.log_artifact(
                        str(default_buffer_path), artifact_path=buffer_artifact_path
                    )
                logger.info(
                    f"Logged buffer artifacts to MLflow path: {buffer_artifact_path}"
                )
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}", exc_info=True)

    def save_run_config(self, configs: dict[str, Any]):
        """Saves the combined configuration dictionary as a JSON artifact."""
        try:
            config_path = self.path_manager.get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            self.serializer.save_config_json(configs, config_path)
            mlflow.log_artifact(str(config_path), artifact_path="config")
        except Exception as e:
            logger.error(f"Failed to save/log run config JSON: {e}", exc_info=True)

    # --- Expose PathManager methods if needed ---
    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        return self.path_manager.get_checkpoint_path(
            run_name, step, is_latest, is_best, is_final
        )

    def get_buffer_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_final: bool = False,
    ) -> Path:
        return self.path_manager.get_buffer_path(run_name, step, is_final)


File: alphatriangle\data\path_manager.py
# File: alphatriangle/data/path_manager.py
import datetime
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PersistenceConfig

logger = logging.getLogger(__name__)


class PathManager:
    """Manages file paths, directory creation, and discovery for training runs."""

    def __init__(self, persist_config: "PersistenceConfig"):
        self.persist_config = persist_config
        self.root_data_dir = Path(self.persist_config.ROOT_DATA_DIR)
        self._update_paths()  # Initialize paths based on config

    def _update_paths(self):
        """Updates paths based on the current RUN_NAME in persist_config."""
        self.run_base_dir = Path(self.persist_config.get_run_base_dir())
        self.checkpoint_dir = (
            self.run_base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        self.buffer_dir = self.run_base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        self.log_dir = self.run_base_dir / self.persist_config.LOG_DIR_NAME
        self.config_path = self.run_base_dir / self.persist_config.CONFIG_FILENAME

    def create_run_directories(self):
        """Creates necessary directories for the current run."""
        self.root_data_dir.mkdir(parents=True, exist_ok=True)
        self.run_base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.persist_config.SAVE_BUFFER:
            self.buffer_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """Constructs the path for a checkpoint file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        checkpoint_dir = base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        if is_latest:
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        elif is_best:
            filename = self.persist_config.BEST_CHECKPOINT_FILENAME
        elif is_final and step is not None:
            filename = f"checkpoint_final_step_{step}.pkl"
        elif step is not None:
            filename = f"checkpoint_step_{step}.pkl"
        else:
            # Default to latest if no specific type is given
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        # Ensure filename ends with .pkl
        filename_pkl = Path(filename).with_suffix(".pkl")
        return checkpoint_dir / filename_pkl

    def get_buffer_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_final: bool = False,
    ) -> Path:
        """Constructs the path for the replay buffer file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        buffer_dir = base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        if is_final and step is not None:
            filename = f"buffer_final_step_{step}.pkl"
        elif step is not None and self.persist_config.BUFFER_SAVE_FREQ_STEPS > 0:
            # Use default name for intermediate saves if frequency is set
            filename = self.persist_config.BUFFER_FILENAME
        else:
            # Default name for initial load or if frequency is not set
            filename = self.persist_config.BUFFER_FILENAME
        return buffer_dir / Path(filename).with_suffix(".pkl")

    def get_config_path(self, run_name: str | None = None) -> Path:
        """Constructs the path for the config JSON file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        return base_dir / self.persist_config.CONFIG_FILENAME

    def find_latest_run_dir(self, current_run_name: str) -> str | None:
        """
        Finds the most recent *previous* run directory based on timestamp parsing.
        Assumes run names follow a pattern like 'prefix_YYYYMMDD_HHMMSS'.
        """
        runs_root_dir = self.root_data_dir / self.persist_config.RUNS_DIR_NAME
        potential_runs: list[tuple[datetime.datetime, str]] = []
        run_name_pattern = re.compile(r"^(?:test_run|train)_(\d{8}_\d{6})$")

        try:
            if not runs_root_dir.exists():
                return None

            for d in runs_root_dir.iterdir():
                if d.is_dir() and d.name != current_run_name:
                    match = run_name_pattern.match(d.name)
                    if match:
                        timestamp_str = match.group(1)
                        try:
                            run_time = datetime.datetime.strptime(
                                timestamp_str, "%Y%m%d_%H%M%S"
                            )
                            potential_runs.append((run_time, d.name))
                        except ValueError:
                            logger.warning(
                                f"Could not parse timestamp from directory name: {d.name}"
                            )
                    else:
                        logger.debug(
                            f"Directory name {d.name} does not match expected pattern."
                        )

            if not potential_runs:
                logger.info("No previous run directories found matching the pattern.")
                return None

            potential_runs.sort(key=lambda item: item[0], reverse=True)
            latest_run_name = potential_runs[0][1]
            logger.debug(
                f"Found potential previous runs (sorted): {[name for _, name in potential_runs]}. Latest: {latest_run_name}"
            )
            return latest_run_name

        except Exception as e:
            logger.error(f"Error finding latest run directory: {e}", exc_info=True)
            return None

    def determine_checkpoint_to_load(
        self, load_path_config: str | None, auto_resume: bool
    ) -> Path | None:
        """Determines the absolute path of the checkpoint file to load."""
        current_run_name = self.persist_config.RUN_NAME
        checkpoint_to_load: Path | None = None

        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                checkpoint_to_load = load_path.resolve()
                logger.info(f"Using specified checkpoint path: {checkpoint_to_load}")
            else:
                logger.warning(
                    f"Specified checkpoint path not found: {load_path_config}"
                )

        if not checkpoint_to_load and auto_resume:
            latest_run_name = self.find_latest_run_dir(current_run_name)
            if latest_run_name:
                potential_latest_path = self.get_checkpoint_path(
                    run_name=latest_run_name, is_latest=True
                )
                if potential_latest_path.exists():
                    checkpoint_to_load = potential_latest_path.resolve()
                    logger.info(
                        f"Auto-resuming from latest checkpoint in previous run '{latest_run_name}': {checkpoint_to_load}"
                    )
                else:
                    logger.info(
                        f"Latest checkpoint file not found in latest run directory '{latest_run_name}'."
                    )
            else:
                logger.info("Auto-resume enabled, but no previous run directory found.")

        if not checkpoint_to_load:
            logger.info("No checkpoint found to load. Starting training from scratch.")

        return checkpoint_to_load

    def determine_buffer_to_load(
        self,
        load_path_config: str | None,
        auto_resume: bool,
        checkpoint_run_name: str | None,
    ) -> Path | None:
        """Determines the buffer file path to load."""
        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                logger.info(f"Using specified buffer path: {load_path_config}")
                return load_path.resolve()
            else:
                logger.warning(f"Specified buffer path not found: {load_path_config}")

        if checkpoint_run_name:
            potential_buffer_path = self.get_buffer_path(run_name=checkpoint_run_name)
            if potential_buffer_path.exists():
                logger.info(
                    f"Loading buffer from checkpoint run '{checkpoint_run_name}': {potential_buffer_path}"
                )
                return potential_buffer_path.resolve()
            else:
                logger.info(
                    f"Default buffer file not found in checkpoint run directory '{checkpoint_run_name}'."
                )

        if auto_resume and not checkpoint_run_name:
            latest_previous_run_name = self.find_latest_run_dir(
                self.persist_config.RUN_NAME
            )
            if latest_previous_run_name:
                potential_buffer_path = self.get_buffer_path(
                    run_name=latest_previous_run_name
                )
                if potential_buffer_path.exists():
                    logger.info(
                        f"Auto-resuming buffer from latest previous run '{latest_previous_run_name}' (no checkpoint loaded): {potential_buffer_path}"
                    )
                    return potential_buffer_path.resolve()
                else:
                    logger.info(
                        f"Default buffer file not found in latest run directory '{latest_previous_run_name}'."
                    )

        logger.info("No suitable buffer file found to load.")
        return None

    def update_checkpoint_links(self, step_checkpoint_path: Path, is_best: bool):
        """Updates the 'latest' and optionally 'best' checkpoint links."""
        if not step_checkpoint_path.exists():
            logger.error(
                f"Source checkpoint path does not exist: {step_checkpoint_path}"
            )
            return

        latest_path = self.get_checkpoint_path(is_latest=True)
        best_path = self.get_checkpoint_path(is_best=True)
        try:
            shutil.copy2(step_checkpoint_path, latest_path)
            logger.debug(f"Updated latest checkpoint link to {step_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to update latest checkpoint link: {e}")
        if is_best:
            try:
                shutil.copy2(step_checkpoint_path, best_path)
                logger.info(
                    f"Updated best checkpoint link to step {step_checkpoint_path.stem}"
                )
            except Exception as e:
                logger.error(f"Failed to update best checkpoint link: {e}")

    def update_buffer_link(self, step_buffer_path: Path):
        """Updates the default buffer link."""
        if not step_buffer_path.exists():
            logger.error(f"Source buffer path does not exist: {step_buffer_path}")
            return

        default_buffer_path = self.get_buffer_path()
        try:
            shutil.copy2(step_buffer_path, default_buffer_path)
            logger.debug(f"Updated default buffer file: {default_buffer_path}")
        except Exception as e_default:
            logger.error(
                f"Error updating default buffer file {default_buffer_path}: {e_default}"
            )


File: alphatriangle\data\README.md
# File: alphatriangle/data/README.md
# Data Management Module (`alphatriangle.data`)

## Purpose and Architecture

This module is responsible for handling the persistence of training artifacts using structured data schemas defined with Pydantic. It manages:

-   Neural network checkpoints (model weights, optimizer state).
-   Experience replay buffers.
-   Statistics collector state.
-   Run configuration files.

The core component is the [`DataManager`](data_manager.py) class, which centralizes file path management and saving/loading logic based on the [`PersistenceConfig`](../config/persistence_config.py) and [`TrainConfig`](../config/train_config.py). It uses `cloudpickle` for robust serialization of complex Python objects, including Pydantic models containing tensors and deques.

-   **Schemas ([`schemas.py`](schemas.py)):** Defines Pydantic models (`CheckpointData`, `BufferData`, `LoadedTrainingState`) to structure the data being saved and loaded, ensuring clarity and enabling validation.
-   **Path Management ([`path_manager.py`](path_manager.py)):** The `PathManager` class handles constructing file paths, creating directories, and finding previous runs.
-   **Serialization ([`serializer.py`](serializer.py)):** The `Serializer` class handles the actual reading/writing of files using `cloudpickle` and JSON, including validation during loading.
-   **Centralization:** `DataManager` provides a single point of control for saving/loading operations.
-   **Configuration-Driven:** Uses `PersistenceConfig` and `TrainConfig` to determine save locations, filenames, and loading behavior (e.g., auto-resume).
-   **Run Management:** Organizes saved artifacts into subdirectories based on the `RUN_NAME`.
-   **State Loading:** `DataManager.load_initial_state` determines the correct files, deserializes them, validates the structure, and returns a `LoadedTrainingState` object.
-   **State Saving:** `DataManager.save_training_state` assembles data into Pydantic models, serializes them, and saves to files.
-   **MLflow Integration:** Logs saved artifacts (checkpoints, buffers, configs) to MLflow after successful local saving.

## Exposed Interfaces

-   **Classes:**
    -   `DataManager`: Orchestrates saving and loading.
        -   `__init__(persist_config: PersistenceConfig, train_config: TrainConfig)`
        -   `load_initial_state() -> LoadedTrainingState`: Loads state, returns Pydantic model.
        -   `save_training_state(...)`: Saves state using Pydantic models and cloudpickle.
        -   `save_run_config(configs: Dict[str, Any])`: Saves config JSON.
        -   `get_checkpoint_path(...) -> Path`
        -   `get_buffer_path(...) -> Path`
    -   `PathManager`: Manages file paths.
    -   `Serializer`: Handles serialization/deserialization.
    -   `CheckpointData` (from `schemas.py`): Pydantic model for checkpoint structure.
    -   `BufferData` (from `schemas.py`): Pydantic model for buffer structure.
    -   `LoadedTrainingState` (from `schemas.py`): Pydantic model wrapping loaded data.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `PersistenceConfig`, `TrainConfig`.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.rl`](../rl/README.md)**: `ExperienceBuffer`.
-   **[`alphatriangle.stats`](../stats/README.md)**: `StatsCollectorActor`.
-   **[`alphatriangle.utils`](../utils/README.md)**: `Experience`.
-   **`torch.optim`**: `Optimizer`.
-   **Standard Libraries:** `os`, `shutil`, `logging`, `glob`, `re`, `json`, `collections.deque`, `pathlib`, `datetime`.
-   **Third-Party:** `pydantic`, `cloudpickle`, `torch`, `ray`, `mlflow`, `numpy`.

---

**Note:** Please keep this README updated when changing the Pydantic schemas, the types of artifacts managed, the saving/loading mechanisms, or the responsibilities of the `DataManager`, `PathManager`, or `Serializer`.

File: alphatriangle\data\schemas.py
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Use relative import
from ..utils.types import Experience

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointData(BaseModel):
    """Pydantic model defining the structure of saved checkpoint data."""

    model_config = arbitrary_types_config

    run_name: str
    global_step: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    total_simulations_run: int = Field(..., ge=0)
    model_config_dict: dict[str, Any]
    env_config_dict: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    stats_collector_state: dict[str, Any]


class BufferData(BaseModel):
    """Pydantic model defining the structure of saved buffer data."""

    model_config = arbitrary_types_config

    buffer_list: list[Experience]


class LoadedTrainingState(BaseModel):
    """Pydantic model representing the fully loaded state."""

    model_config = arbitrary_types_config

    checkpoint_data: CheckpointData | None = None
    buffer_data: BufferData | None = None


BufferData.model_rebuild(force=True)
LoadedTrainingState.model_rebuild(force=True)


File: alphatriangle\data\serializer.py
# File: alphatriangle/data/serializer.py
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
from pydantic import ValidationError

from ..utils.sumtree import SumTree
from .schemas import BufferData, CheckpointData

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..rl.core.buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class Serializer:
    """Handles serialization and deserialization of training data."""

    def load_checkpoint(self, path: Path) -> CheckpointData | None:
        """Loads and validates checkpoint data from a file."""
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, CheckpointData):
                # Pydantic automatically validates on load if type matches
                return loaded_data
            else:
                logger.error(
                    f"Loaded checkpoint file {path} did not contain a CheckpointData object (type: {type(loaded_data)})."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for checkpoint {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {path}")
            return None
        except Exception as e:
            logger.error(
                f"Error loading/deserializing checkpoint from {path}: {e}",
                exc_info=True,
            )
            return None

    def save_checkpoint(self, data: CheckpointData, path: Path):
        """Saves checkpoint data to a file using cloudpickle."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
            logger.info(f"Checkpoint data saved to {path}")
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint file to {path}: {e}", exc_info=True
            )
            raise  # Re-raise the exception

    def load_buffer(self, path: Path) -> BufferData | None:
        """Loads and validates buffer data from a file."""
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, BufferData):
                # Perform basic validation on loaded experiences
                valid_experiences = []
                invalid_count = 0
                for i, exp in enumerate(loaded_data.buffer_list):
                    if (
                        isinstance(exp, tuple)
                        and len(exp) == 3
                        and isinstance(exp[0], dict)
                        and "grid" in exp[0]
                        and "other_features" in exp[0]
                        and isinstance(exp[0]["grid"], np.ndarray)
                        and isinstance(exp[0]["other_features"], np.ndarray)
                        and isinstance(exp[1], dict)
                        and isinstance(exp[2], float | int)
                    ):
                        valid_experiences.append(exp)
                    else:
                        invalid_count += 1
                        logger.warning(
                            f"Skipping invalid experience structure at index {i} in loaded buffer: {type(exp)}"
                        )
                if invalid_count > 0:
                    logger.warning(
                        f"Found {invalid_count} invalid experience structures in loaded buffer."
                    )
                loaded_data.buffer_list = valid_experiences
                return loaded_data
            else:
                logger.error(
                    f"Loaded buffer file {path} did not contain a BufferData object (type: {type(loaded_data)})."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for buffer {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Buffer file not found: {path}")
            return None
        except Exception as e:
            logger.error(
                f"Failed to load/deserialize experience buffer from {path}: {e}",
                exc_info=True,
            )
            return None

    def save_buffer(self, data: BufferData, path: Path):
        """Saves buffer data to a file using cloudpickle."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
            logger.info(f"Buffer data saved to {path}")
        except Exception as e:
            logger.error(
                f"Error saving experience buffer to {path}: {e}", exc_info=True
            )
            raise  # Re-raise the exception

    def prepare_optimizer_state(self, optimizer: "Optimizer") -> dict[str, Any]:
        """Prepares optimizer state dictionary, moving tensors to CPU."""
        optimizer_state_cpu = {}
        try:
            optimizer_state_dict = optimizer.state_dict()

            def move_to_cpu(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, dict):
                    return {k: move_to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [move_to_cpu(elem) for elem in item]
                else:
                    return item

            optimizer_state_cpu = move_to_cpu(optimizer_state_dict)
        except Exception as e:
            logger.error(f"Could not prepare optimizer state for saving: {e}")
        return optimizer_state_cpu

    def prepare_buffer_data(self, buffer: "ExperienceBuffer") -> BufferData | None:
        """Prepares buffer data for saving, extracting experiences."""
        try:
            if buffer.use_per:
                if hasattr(buffer, "tree") and isinstance(buffer.tree, SumTree):
                    buffer_list = [
                        buffer.tree.data[i]
                        for i in range(buffer.tree.n_entries)
                        if buffer.tree.data[i] != 0
                    ]
                else:
                    logger.error("PER buffer tree is missing or invalid during save.")
                    return None
            else:
                buffer_list = list(buffer.buffer)

            # Basic validation before creating BufferData
            valid_experiences = []
            invalid_count = 0
            for i, exp in enumerate(buffer_list):
                if (
                    isinstance(exp, tuple)
                    and len(exp) == 3
                    and isinstance(exp[0], dict)
                    and "grid" in exp[0]
                    and "other_features" in exp[0]
                    and isinstance(exp[0]["grid"], np.ndarray)
                    and isinstance(exp[0]["other_features"], np.ndarray)
                    and isinstance(exp[1], dict)
                    and isinstance(exp[2], float | int)
                ):
                    valid_experiences.append(exp)
                else:
                    invalid_count += 1
                    logger.warning(
                        f"Skipping invalid experience structure at index {i} during save prep: {type(exp)}"
                    )
            if invalid_count > 0:
                logger.warning(
                    f"Found {invalid_count} invalid experience structures before saving buffer."
                )

            return BufferData(buffer_list=valid_experiences)
        except Exception as e:
            logger.error(f"Error preparing buffer data for saving: {e}")
            return None

    def save_config_json(self, configs: dict[str, Any], path: Path):
        """Saves the configuration dictionary as JSON."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:

                def default_serializer(obj):
                    if isinstance(obj, torch.Tensor | np.ndarray):
                        return "<tensor/array>"
                    if isinstance(obj, deque):
                        return list(obj)
                    try:
                        return obj.__dict__ if hasattr(obj, "__dict__") else str(obj)
                    except TypeError:
                        return f"<object of type {type(obj).__name__}>"

                json.dump(configs, f, indent=4, default=default_serializer)
            logger.info(f"Run config saved to {path}")
        except Exception as e:
            logger.error(
                f"Failed to save run config JSON to {path}: {e}", exc_info=True
            )
            raise


File: alphatriangle\data\__init__.py
# File: alphatriangle/data/__init__.py
"""
Data management module for handling checkpoints, buffers, and potentially logs.
Uses Pydantic schemas for data structure definition.
"""

from .data_manager import DataManager
from .path_manager import PathManager
from .schemas import BufferData, CheckpointData, LoadedTrainingState
from .serializer import Serializer

__all__ = [
    "DataManager",
    "PathManager",
    "Serializer",
    "CheckpointData",
    "BufferData",
    "LoadedTrainingState",
]


File: alphatriangle\environment\README.md
# File: alphatriangle/environment/README.md
# Environment Module (`alphatriangle.environment`)

## Purpose and Architecture

This module defines the game world for AlphaTriangle. It encapsulates the rules, state representation, actions, and core game logic. **Crucially, this module is now independent of any feature extraction logic specific to the neural network.** Its sole focus is the simulation of the game itself.

-   **State Representation:** [`GameState`](core/game_state.py) holds the current board ([`GridData`](grid/grid_data.py)), available shapes (`List[Shape]`), score, and game status. It represents the canonical state of the game. It uses core structures like `Shape` and `Triangle` defined in [`alphatriangle.structs`](../structs/README.md).
-   **Core Logic:** Submodules ([`grid`](grid/README.md), [`shapes`](shapes/README.md), [`logic`](logic/README.md)) handle specific aspects like checking valid placements, clearing lines, managing shape generation, and calculating rewards. These logic modules operate on `GridData`, `Shape`, and `Triangle`. **Shape refilling now happens in batches: all slots are refilled only when all slots become empty.**
-   **Action Handling:** [`action_codec`](core/action_codec.py) provides functions to convert between a structured action (shape index, row, column) and a single integer representation used by the RL agent and MCTS.
-   **Modularity:** Separating grid logic, shape logic, and core state makes the code easier to understand and modify.

**Note:** Feature extraction (converting `GameState` to NN input tensors) is handled by the separate [`alphatriangle.features`](../features/README.md) module. Core data structures (`Triangle`, `Shape`) are defined in [`alphatriangle.structs`](../structs/README.md).

## Exposed Interfaces

-   **Core ([`core/README.md`](core/README.md)):**
    -   `GameState`: The main class representing the environment state.
        -   `reset()`
        -   `step(action_index: ActionType) -> Tuple[float, bool]`
        -   `valid_actions() -> List[ActionType]`
        -   `is_over() -> bool`
        -   `get_outcome() -> float`
        -   `copy() -> GameState`
        -   Public attributes like `grid_data`, `shapes`, `game_score`, `current_step`, etc.
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`
-   **Grid ([`grid/README.md`](grid/README.md)):**
    -   `GridData`: Class holding grid triangle data and line information.
    -   `GridLogic`: Namespace containing functions like `link_neighbors`, `initialize_lines_and_index`, `can_place`, `check_and_clear_lines`.
-   **Shapes ([`shapes/README.md`](shapes/README.md)):**
    -   `ShapeLogic`: Namespace containing functions like `refill_shape_slots`, `generate_random_shape`. **Includes `PREDEFINED_SHAPE_TEMPLATES` constant.**
-   **Logic ([`logic/README.md`](logic/README.md)):**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float` **(Triggers batch refill)**
    -   `calculate_reward(...) -> float` (Used internally by `execute_placement`)
-   **Config:**
    -   `EnvConfig`: Configuration class (re-exported for convenience).

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**:
    -   Uses `EnvConfig` extensively to define grid dimensions, shape slots, etc.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `SHAPE_COLORS`, `NO_COLOR_ID`, `DEBUG_COLOR_ID`, `COLOR_TO_ID_MAP`.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   Uses `ActionType`.
-   **`numpy`**:
    -   Used for grid representation (`GridData`).
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`, `copy`.

---

**Note:** Please keep this README updated when changing game rules, state representation, action space, or the module's internal structure (especially the shape refill logic). Accurate documentation is crucial for maintainability.

File: alphatriangle\environment\__init__.py
"""
Environment module defining the game rules, state, actions, and logic.
This module is now independent of feature extraction for the NN.
"""

from alphatriangle.config import EnvConfig

from .core.action_codec import decode_action, encode_action
from .core.game_state import GameState
from .grid import logic as GridLogic
from .grid.grid_data import GridData
from .logic.actions import get_valid_actions
from .logic.step import calculate_reward, execute_placement
from .shapes import logic as ShapeLogic

__all__ = [
    # Core
    "GameState",
    "encode_action",
    "decode_action",
    # Grid
    "GridData",
    "GridLogic",
    # Shapes
    "ShapeLogic",
    # Logic
    "get_valid_actions",
    "execute_placement",
    "calculate_reward",
    # Config
    "EnvConfig",
]


File: alphatriangle\environment\core\action_codec.py
from ...config import EnvConfig
from ...utils.types import ActionType


def encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType:
    """Encodes a (shape_idx, r, c) action into a single integer."""
    if not (0 <= shape_idx < config.NUM_SHAPE_SLOTS):
        raise ValueError(
            f"Invalid shape index: {shape_idx}, must be < {config.NUM_SHAPE_SLOTS}"
        )
    if not (0 <= r < config.ROWS):
        raise ValueError(f"Invalid row index: {r}, must be < {config.ROWS}")
    if not (0 <= c < config.COLS):
        raise ValueError(f"Invalid column index: {c}, must be < {config.COLS}")

    action_index = shape_idx * (config.ROWS * config.COLS) + r * config.COLS + c
    return action_index


def decode_action(action_index: ActionType, config: EnvConfig) -> tuple[int, int, int]:
    """Decodes an integer action into (shape_idx, r, c)."""
    # Cast ACTION_DIM to int for comparison
    action_dim_int = int(config.ACTION_DIM)  # type: ignore[call-overload]
    if not (0 <= action_index < action_dim_int):
        raise ValueError(
            f"Invalid action index: {action_index}, must be < {action_dim_int}"
        )

    grid_size = config.ROWS * config.COLS
    shape_idx = action_index // grid_size
    remainder = action_index % grid_size
    r = remainder // config.COLS
    c = remainder % config.COLS

    return shape_idx, r, c


File: alphatriangle\environment\core\game_state.py
import logging
import random
from typing import TYPE_CHECKING

from ...config import EnvConfig
from ...utils.types import ActionType
from .. import shapes
from ..grid.grid_data import GridData
from ..logic.actions import get_valid_actions
from ..logic.step import execute_placement
from .action_codec import decode_action

if TYPE_CHECKING:
    from ...structs import Shape


logger = logging.getLogger(__name__)


class GameState:
    """
    Represents the mutable state of the game. Does not handle NN feature extraction
    or visualization/interaction-specific state.
    """

    def __init__(
        self, config: EnvConfig | None = None, initial_seed: int | None = None
    ):
        self.env_config = config if config else EnvConfig()  # type: ignore[call-arg]
        self._rng = (
            random.Random(initial_seed) if initial_seed is not None else random.Random()
        )

        self.grid_data: GridData = None  # type: ignore
        self.shapes: list[Shape | None] = []
        self.game_score: float = 0.0
        self.game_over: bool = False
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0
        self.current_step: int = 0

        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.grid_data = GridData(self.env_config)
        self.shapes = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score = 0.0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0
        self.game_over = False
        self.current_step = 0

        # Call refill_shape_slots with the updated signature (no index)
        shapes.refill_shape_slots(self, self._rng)

        if not self.valid_actions():
            logger.warning(
                "Game is over immediately after reset (no valid initial moves)."
            )
            self.game_over = True

    def step(self, action_index: ActionType) -> tuple[float, bool]:
        """
        Performs one game step.
        Returns:
            Tuple[float, bool]: (reward, done)
        """
        if self.is_over():
            logger.warning("Attempted to step in a game that is already over.")
            return 0.0, True

        shape_idx, r, c = decode_action(action_index, self.env_config)
        reward = execute_placement(self, shape_idx, r, c, self._rng)
        self.current_step += 1

        if not self.game_over and not self.valid_actions():
            self.game_over = True
            logger.info(f"Game over detected after step {self.current_step}.")

        return reward, self.game_over

    def valid_actions(self) -> list[ActionType]:
        """Returns a list of valid encoded action indices."""
        return get_valid_actions(self)

    def is_over(self) -> bool:
        """Checks if the game is over."""
        return self.game_over

    def get_outcome(self) -> float:
        """Returns the terminal outcome value (e.g., final score). Used by MCTS."""
        if not self.is_over():
            logger.warning("get_outcome() called on a non-terminal state.")
            # Consider returning a default value or raising an error?
            # Returning current score might be misleading for MCTS if not terminal.
            # Let's return 0.0 as a neutral value if not over.
            return 0.0
        return self.game_score

    def copy(self) -> "GameState":
        """Creates a deep copy for simulations (e.g., MCTS)."""
        new_state = GameState.__new__(GameState)
        new_state.env_config = self.env_config
        new_state._rng = random.Random()
        new_state._rng.setstate(self._rng.getstate())
        new_state.grid_data = self.grid_data.deepcopy()
        new_state.shapes = [s.copy() if s else None for s in self.shapes]
        new_state.game_score = self.game_score
        new_state.game_over = self.game_over
        new_state.triangles_cleared_this_episode = self.triangles_cleared_this_episode
        new_state.pieces_placed_this_episode = self.pieces_placed_this_episode
        new_state.current_step = self.current_step
        return new_state

    def __str__(self) -> str:
        shape_strs = [str(s) if s else "None" for s in self.shapes]
        return f"GameState(Step:{self.current_step}, Score:{self.game_score:.1f}, Over:{self.is_over()}, Shapes:[{', '.join(shape_strs)}])"


File: alphatriangle\environment\core\README.md
# File: alphatriangle/environment/core/README.md
# Environment Core Submodule (`alphatriangle.environment.core`)

## Purpose and Architecture

This submodule contains the most fundamental components of the game environment: the [`GameState`](game_state.py) class and the [`action_codec`](action_codec.py).

-   **`GameState`:** This class acts as the central hub for the environment's state. It holds references to the [`GridData`](../grid/grid_data.py), the current shapes, score, game status, and other relevant information. It provides the primary interface (`reset`, `step`, `valid_actions`, `is_over`, `get_outcome`, `copy`) for agents (like MCTS or self-play workers) to interact with the game. It delegates specific logic (like placement validation, line clearing, shape generation) to other submodules ([`grid`](../grid/README.md), [`shapes`](../shapes/README.md), [`logic`](../logic/README.md)).
-   **`action_codec`:** Provides simple, stateless functions (`encode_action`, `decode_action`) to translate between the agent's integer action representation and the game's internal representation (shape index, row, column). This decouples the agent's action space from the internal game logic.

## Exposed Interfaces

-   **Classes:**
    -   `GameState`: The main state class (see [`alphatriangle/environment/README.md`](../README.md) for methods).
-   **Functions:**
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used by `GameState` and `action_codec`.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`: Used for method signatures and return types.
-   **[`alphatriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used internally by `GameState`.
-   **[`alphatriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used internally by `GameState`.
-   **[`alphatriangle.environment.logic`](../logic/README.md)**:
    -   `get_valid_actions`, `execute_placement`: Used internally by `GameState`.
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`.

---

**Note:** Please keep this README updated when modifying the core `GameState` interface or the action encoding/decoding scheme. Accurate documentation is crucial for maintainability.

File: alphatriangle\environment\core\__init__.py


File: alphatriangle\environment\grid\grid_data.py
# File: alphatriangle/environment/grid/grid_data.py
import copy
import logging

import numpy as np

from ...config import EnvConfig
from ...structs import NO_COLOR_ID

logger = logging.getLogger(__name__)


def _precompute_lines(config: EnvConfig) -> list[list[tuple[int, int]]]:
    """
    Generates all potential horizontal and diagonal lines based on grid geometry.
    Returns a list of lines, where each line is a list of (row, col) tuples.
    This function no longer needs actual Triangle objects.
    """
    lines = []
    rows, cols = config.ROWS, config.COLS
    min_len = config.MIN_LINE_LENGTH

    # --- Determine playable cells based on config ---
    playable_mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        playable_width = config.COLS_PER_ROW[r]
        padding = cols - playable_width
        pad_left = padding // 2
        playable_start_col = pad_left
        playable_end_col = pad_left + playable_width
        for c in range(cols):
            if playable_start_col <= c < playable_end_col:
                playable_mask[r, c] = True
    # --- End Playable Mask ---

    # Helper to check validity and playability
    def is_valid_playable(r, c):
        return 0 <= r < rows and 0 <= c < cols and playable_mask[r, c]

    # --- Trace Lines using Coordinates ---
    visited_in_line: set[tuple[int, int, str]] = set()  # (r, c, direction)

    for r_start in range(rows):
        for c_start in range(cols):
            if not is_valid_playable(r_start, c_start):
                continue

            # --- Trace Horizontal ---
            if (r_start, c_start, "h") not in visited_in_line:
                current_line_h = []
                # Trace left
                cr, cc = r_start, c_start
                while is_valid_playable(cr, cc - 1):
                    cc -= 1
                # Trace right from the start
                while is_valid_playable(cr, cc):
                    if (cr, cc, "h") not in visited_in_line:
                        current_line_h.append((cr, cc))
                        visited_in_line.add((cr, cc, "h"))
                    else:
                        # If we hit a visited cell, the rest of the line was already processed
                        break
                    cc += 1
                if len(current_line_h) >= min_len:
                    lines.append(current_line_h)

            # --- Trace Diagonal TL-BR (Down-Right) ---
            if (r_start, c_start, "d1") not in visited_in_line:
                current_line_d1 = []
                # Trace backwards (Up-Left)
                cr, cc = r_start, c_start
                while True:
                    is_up = (cr + cc) % 2 != 0
                    prev_r, prev_c = (cr, cc - 1) if is_up else (cr - 1, cc)
                    if is_valid_playable(prev_r, prev_c):
                        cr, cc = prev_r, prev_c
                    else:
                        break
                # Trace forwards
                while is_valid_playable(cr, cc):
                    if (cr, cc, "d1") not in visited_in_line:
                        current_line_d1.append((cr, cc))
                        visited_in_line.add((cr, cc, "d1"))
                    else:
                        break
                    is_up = (cr + cc) % 2 != 0
                    next_r, next_c = (cr + 1, cc) if is_up else (cr, cc + 1)
                    cr, cc = next_r, next_c
                if len(current_line_d1) >= min_len:
                    lines.append(current_line_d1)

            # --- Trace Diagonal BL-TR (Up-Right) ---
            if (r_start, c_start, "d2") not in visited_in_line:
                current_line_d2 = []
                # Trace backwards (Down-Left)
                cr, cc = r_start, c_start
                while True:
                    is_up = (cr + cc) % 2 != 0
                    prev_r, prev_c = (cr + 1, cc) if is_up else (cr, cc - 1)
                    if is_valid_playable(prev_r, prev_c):
                        cr, cc = prev_r, prev_c
                    else:
                        break
                # Trace forwards
                while is_valid_playable(cr, cc):
                    if (cr, cc, "d2") not in visited_in_line:
                        current_line_d2.append((cr, cc))
                        visited_in_line.add((cr, cc, "d2"))
                    else:
                        break
                    is_up = (cr + cc) % 2 != 0
                    next_r, next_c = (cr, cc + 1) if is_up else (cr - 1, cc)
                    cr, cc = next_r, next_c
                if len(current_line_d2) >= min_len:
                    lines.append(current_line_d2)
    # --- End Line Tracing ---

    # Remove duplicates (lines traced from different start points)
    unique_lines_tuples = {tuple(sorted(line)) for line in lines}
    unique_lines = [list(line_tuple) for line_tuple in unique_lines_tuples]

    # Final filter by length (should be redundant but safe)
    final_lines = [line for line in unique_lines if len(line) >= min_len]

    return final_lines


class GridData:
    """
    Holds the grid state using NumPy arrays for occupancy, death zones, and color IDs.
    Manages precomputed line information based on coordinates.
    """

    def __init__(self, config: EnvConfig):
        self.rows = config.ROWS
        self.cols = config.COLS
        self.config = config

        # --- NumPy Array State ---
        self._occupied_np: np.ndarray = np.zeros((self.rows, self.cols), dtype=bool)
        self._death_np: np.ndarray = np.zeros((self.rows, self.cols), dtype=bool)
        # Stores color ID, NO_COLOR_ID (-1) means empty/no color
        self._color_id_np: np.ndarray = np.full(
            (self.rows, self.cols), NO_COLOR_ID, dtype=np.int8
        )
        # --- End NumPy Array State ---

        self._initialize_death_zone(config)
        self._occupied_np[self._death_np] = True  # Death cells are considered occupied

        # --- Line Information (Coordinate Based) ---
        # Stores frozensets of (r, c) tuples
        self.potential_lines: set[frozenset[tuple[int, int]]] = set()
        # Maps (r, c) tuple to a set of line frozensets it belongs to
        self._coord_to_lines_map: dict[
            tuple[int, int], set[frozenset[tuple[int, int]]]
        ] = {}
        # --- End Line Information ---

        self._initialize_lines_and_index()
        logger.debug(
            f"GridData initialized ({self.rows}x{self.cols}) using NumPy arrays. Found {len(self.potential_lines)} potential lines."
        )

    def _initialize_death_zone(self, config: EnvConfig):
        """Initializes the death zone numpy array."""
        cols_per_row = config.COLS_PER_ROW
        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"COLS_PER_ROW length mismatch: {len(cols_per_row)} vs {self.rows}"
            )

        for r in range(self.rows):
            playable_width = cols_per_row[r]
            padding = self.cols - playable_width
            pad_left = padding // 2
            playable_start_col = pad_left
            playable_end_col = pad_left + playable_width
            for c in range(self.cols):
                if not (playable_start_col <= c < playable_end_col):
                    self._death_np[r, c] = True

    def _initialize_lines_and_index(self) -> None:
        """
        Precomputes potential lines (as coordinate sets) and creates a map
        from coordinates to the lines they belong to.
        """
        self.potential_lines = set()
        self._coord_to_lines_map = {}

        potential_lines_coords = _precompute_lines(self.config)

        for line_coords in potential_lines_coords:
            # Filter out lines containing death cells
            valid_line = True
            line_coord_set: set[tuple[int, int]] = set()
            for r, c in line_coords:
                # Use self.valid() and self._death_np directly
                if self.valid(r, c) and not self._death_np[r, c]:
                    line_coord_set.add((r, c))
                else:
                    valid_line = False
                    break  # Skip this line if any part is invalid/death

            if valid_line and len(line_coord_set) >= self.config.MIN_LINE_LENGTH:
                frozen_line = frozenset(line_coord_set)
                self.potential_lines.add(frozen_line)
                # Add to the reverse map
                for coord in line_coord_set:
                    if coord not in self._coord_to_lines_map:
                        self._coord_to_lines_map[coord] = set()
                    self._coord_to_lines_map[coord].add(frozen_line)

        logger.debug(
            f"Initialized {len(self.potential_lines)} potential lines and mapping for {len(self._coord_to_lines_map)} coordinates."
        )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_death(self, r: int, c: int) -> bool:
        """Checks if a cell is a death cell."""
        if not self.valid(r, c):
            return True  # Out of bounds is considered death
        # Cast NumPy bool_ to Python bool for type consistency
        return bool(self._death_np[r, c])

    def is_occupied(self, r: int, c: int) -> bool:
        """Checks if a cell is occupied (includes death cells)."""
        if not self.valid(r, c):
            return True  # Out of bounds is considered occupied
        # Cast NumPy bool_ to Python bool for type consistency
        return bool(self._occupied_np[r, c])

    def get_color_id(self, r: int, c: int) -> int:
        """Gets the color ID of a cell."""
        if not self.valid(r, c):
            return NO_COLOR_ID
        # Cast NumPy int8 to Python int for type consistency
        return int(self._color_id_np[r, c])

    def get_occupied_state(self) -> np.ndarray:
        """Returns a copy of the occupancy numpy array."""
        return self._occupied_np.copy()

    def get_death_state(self) -> np.ndarray:
        """Returns a copy of the death zone numpy array."""
        return self._death_np.copy()

    def get_color_id_state(self) -> np.ndarray:
        """Returns a copy of the color ID numpy array."""
        return self._color_id_np.copy()

    def deepcopy(self) -> "GridData":
        """
        Creates a deep copy of the grid data using NumPy array copying
        and standard dictionary/set copying for line data.
        """
        new_grid = GridData.__new__(
            GridData
        )  # Create new instance without calling __init__
        new_grid.rows = self.rows
        new_grid.cols = self.cols
        new_grid.config = self.config  # Config is likely immutable, shallow copy ok

        # 1. Copy NumPy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()
        new_grid._color_id_np = self._color_id_np.copy()

        # 2. Copy Line Data (Set of frozensets and Dict[Tuple, Set[frozenset]])
        # potential_lines contains immutable frozensets, shallow copy is fine
        new_grid.potential_lines = self.potential_lines.copy()
        # _coord_to_lines_map values are sets, need deepcopy
        new_grid._coord_to_lines_map = copy.deepcopy(self._coord_to_lines_map)

        # No Triangle objects or neighbors to handle anymore

        return new_grid

    def __str__(self) -> str:
        # Basic representation, could be enhanced to show grid visually
        occupied_count = np.sum(self._occupied_np & ~self._death_np)
        return f"GridData({self.rows}x{self.cols}, Occupied: {occupied_count})"


File: alphatriangle\environment\grid\logic.py
# File: alphatriangle/environment/grid/logic.py
import logging
from typing import TYPE_CHECKING

# Import NO_COLOR_ID from the structs package directly
from ...structs import NO_COLOR_ID

if TYPE_CHECKING:
    from ...structs import Shape
    from .grid_data import GridData

logger = logging.getLogger(__name__)


# Removed link_neighbors function as it's no longer needed


def can_place(grid_data: "GridData", shape: "Shape", r: int, c: int) -> bool:
    """
    Checks if a shape can be placed at the specified (r, c) top-left position
    on the grid, considering occupancy, death zones, and triangle orientation.
    Reads state from GridData's NumPy arrays.
    """
    if not shape or not shape.triangles:
        return False

    for dr, dc, is_up_shape in shape.triangles:
        tri_r, tri_c = r + dr, c + dc

        # Check bounds and death zone first
        if not grid_data.valid(tri_r, tri_c) or grid_data._death_np[tri_r, tri_c]:
            return False

        # Check occupancy
        if grid_data._occupied_np[tri_r, tri_c]:
            return False

        # Check orientation match
        is_up_grid = (tri_r + tri_c) % 2 != 0
        if is_up_grid != is_up_shape:
            # Log the mismatch for debugging the test failure
            logger.debug(
                f"Orientation mismatch at ({tri_r},{tri_c}): Grid is {'Up' if is_up_grid else 'Down'}, Shape requires {'Up' if is_up_shape else 'Down'}"
            )
            return False

    return True


def check_and_clear_lines(
    grid_data: "GridData", newly_occupied_coords: set[tuple[int, int]]
) -> tuple[int, set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    """
    Checks for completed lines involving the newly occupied coordinates and clears them.
    Operates on GridData's NumPy arrays.

    Args:
        grid_data: The GridData object (will be modified).
        newly_occupied_coords: A set of (r, c) tuples that were just occupied.

    Returns:
        Tuple containing:
            - int: Number of lines cleared.
            - set[tuple[int, int]]: Set of unique (r, c) coordinates cleared.
            - set[frozenset[tuple[int, int]]]: Set containing the frozenset representations
                                                of the actual lines that were cleared.
    """
    lines_to_check: set[frozenset[tuple[int, int]]] = set()
    for coord in newly_occupied_coords:
        if coord in grid_data._coord_to_lines_map:
            lines_to_check.update(grid_data._coord_to_lines_map[coord])

    cleared_lines_set: set[frozenset[tuple[int, int]]] = set()
    unique_coords_cleared: set[tuple[int, int]] = set()

    if not lines_to_check:
        return 0, unique_coords_cleared, cleared_lines_set

    logger.debug(f"Checking {len(lines_to_check)} potential lines for completion.")

    for line_coords_fs in lines_to_check:
        is_complete = True
        for r_line, c_line in line_coords_fs:
            # Check occupancy directly from the NumPy array
            if not grid_data._occupied_np[r_line, c_line]:
                is_complete = False
                break

        if is_complete:
            logger.debug(f"Line completed: {line_coords_fs}")
            cleared_lines_set.add(line_coords_fs)
            # Add coordinates from this cleared line to the set of unique cleared coordinates
            unique_coords_cleared.update(line_coords_fs)

    if unique_coords_cleared:
        logger.info(
            f"Clearing {len(cleared_lines_set)} lines involving {len(unique_coords_cleared)} unique coordinates."
        )
        # Update NumPy arrays for cleared coordinates
        # Convert set to tuple of arrays for advanced indexing
        if unique_coords_cleared:  # Ensure set is not empty
            rows_idx, cols_idx = zip(*unique_coords_cleared, strict=False)
            grid_data._occupied_np[rows_idx, cols_idx] = False
            grid_data._color_id_np[rows_idx, cols_idx] = NO_COLOR_ID

    return len(cleared_lines_set), unique_coords_cleared, cleared_lines_set


File: alphatriangle\environment\grid\README.md
# File: alphatriangle/environment/grid/README.md
# Environment Grid Submodule (`alphatriangle.environment.grid`)

## Purpose and Architecture

This submodule manages the game's grid structure and related logic. It defines the triangular cells, their properties, relationships, and operations like placement validation and line clearing.

-   **Cell Representation:** The `Triangle` class (defined in [`alphatriangle.structs`](../../structs/README.md)) represents a single cell, storing its position and orientation (`is_up`). The actual state (occupied, death, color) is managed within `GridData`.
-   **Grid Data Structure:** The [`GridData`](grid_data.py) class holds the grid state using efficient `numpy` arrays (`_occupied_np`, `_death_np`, `_color_id_np`). It also manages precomputed information about potential lines (sets of coordinates) for efficient clearing checks.
-   **Grid Logic:** The [`logic.py`](logic.py) module (exposed as `GridLogic`) contains functions operating on `GridData`. This includes:
    -   Initializing the grid based on `EnvConfig` (defining death zones).
    -   Precomputing potential lines (`_precompute_lines`) and indexing them (`initialize_lines_and_index`) for efficient checking.
    -   Checking if a shape can be placed (`can_place`), **including matching triangle orientations**.
    -   Checking for and clearing completed lines (`check_and_clear_lines`). **This function does NOT implement gravity.**
-   **Grid Features:** Note: The `grid_features.py` module, which previously provided functions to calculate scalar metrics (heights, holes, bumpiness), has been **moved** to the top-level [`alphatriangle.features`](../../features/README.md) module (`alphatriangle/features/grid_features.py`) as part of decoupling feature extraction from the core environment.

## Exposed Interfaces

-   **Classes:**
    -   `GridData`: Holds the grid state using NumPy arrays.
        -   `__init__(config: EnvConfig)`
        -   `valid(r: int, c: int) -> bool`
        -   `is_death(r: int, c: int) -> bool`
        -   `is_occupied(r: int, c: int) -> bool`
        -   `get_color_id(r: int, c: int) -> int`
        -   `get_occupied_state() -> np.ndarray`
        -   `get_death_state() -> np.ndarray`
        -   `get_color_id_state() -> np.ndarray`
        -   `deepcopy() -> GridData`
-   **Modules/Namespaces:**
    -   `logic` (often imported as `GridLogic`):
        -   `initialize_lines_and_index(grid_data: GridData)`
        -   `can_place(grid_data: GridData, shape: Shape, r: int, c: int) -> bool`
        -   `check_and_clear_lines(grid_data: GridData, newly_occupied_coords: Set[Tuple[int, int]]) -> Tuple[int, Set[Tuple[int, int]], Set[frozenset[Tuple[int, int]]]]` **(Returns: lines_cleared_count, unique_coords_cleared_set, set_of_cleared_lines_coord_sets)**

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used by `GridData` initialization and logic functions.
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `NO_COLOR_ID`.
-   **`numpy`**:
    -   Used extensively in `GridData`.
-   **Standard Libraries:** `typing`, `logging`, `numpy`, `copy`.

---

**Note:** Please keep this README updated when changing the grid structure, cell properties, placement rules, or line clearing logic. Accurate documentation is crucial for maintainability.

File: alphatriangle\environment\grid\triangle.py
class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up
        self.is_death = is_death
        self.is_occupied = is_death
        self.color: tuple[int, int, int] | None = None

        self.neighbor_left: Triangle | None = None
        self.neighbor_right: Triangle | None = None
        self.neighbor_vert: Triangle | None = None

    def get_points(
        self, ox: float, oy: float, cw: float, ch: float
    ) -> list[tuple[float, float]]:
        """Calculates vertex points for drawing, relative to origin (ox, oy)."""
        x = ox + self.col * (cw * 0.75)
        y = oy + self.row * ch
        if self.is_up:
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def copy(self) -> "Triangle":
        """Creates a copy of the Triangle object's state (neighbors are not copied)."""
        new_tri = Triangle(self.row, self.col, self.is_up, self.is_death)
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color
        return new_tri

    def __repr__(self) -> str:
        state = "D" if self.is_death else ("O" if self.is_occupied else ".")
        orient = "^" if self.is_up else "v"
        return f"T({self.row},{self.col} {orient}{state})"

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.row == other.row and self.col == other.col


File: alphatriangle\environment\grid\__init__.py
# File: alphatriangle/environment/grid/__init__.py
"""
Grid submodule handling the triangular grid structure, data, and logic.
"""

# Removed: from .triangle import Triangle
from . import logic
from .grid_data import GridData

# DO NOT import grid_features here. It has been moved up one level
# to alphatriangle/environment/grid_features.py to break circular dependencies.

__all__ = [
    "GridData",
    "logic",
]


File: alphatriangle\environment\logic\actions.py
import logging
from typing import TYPE_CHECKING

from ..core.action_codec import encode_action
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ...utils.types import ActionType
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def get_valid_actions(state: "GameState") -> list["ActionType"]:
    """
    Calculates and returns a list of all valid encoded action indices
    for the current game state.
    """
    valid_actions: list[ActionType] = []
    for shape_idx, shape in enumerate(state.shapes):
        if shape is None:
            continue

        for r in range(state.env_config.ROWS):
            for c in range(state.env_config.COLS):
                if GridLogic.can_place(state.grid_data, shape, r, c):
                    action_index = encode_action(shape_idx, r, c, state.env_config)
                    valid_actions.append(action_index)

    return valid_actions


File: alphatriangle\environment\logic\README.md
# File: alphatriangle/environment/logic/README.md
# Environment Logic Submodule (`alphatriangle.environment.logic`)

## Purpose and Architecture

This submodule contains higher-level game logic that operates on the `GameState` and its components (`GridData`, `Shape`). It bridges the gap between basic actions/rules and the overall game flow.

-   **`actions.py`:**
    -   `get_valid_actions`: Determines all possible valid moves (shape placements) from the current `GameState` by iterating through available shapes and grid positions, checking placement validity using [`GridLogic.can_place`](../grid/logic.py). Returns a list of encoded `ActionType` integers.
-   **`step.py`:**
    -   `execute_placement`: Performs the core logic when a shape is placed. It updates the `GridData` (occupancy and color), checks for and clears completed lines using [`GridLogic.check_and_clear_lines`](../grid/logic.py), calculates the reward for the step using `calculate_reward`, updates the game score and step counters, and **triggers a batch refill of shape slots using [`ShapeLogic.refill_shape_slots`](../shapes/logic.py) only if all slots become empty after the placement.**
    -   `calculate_reward`: Calculates the reward based on the number of triangles placed, triangles cleared, and whether the game ended.

## Exposed Interfaces

-   **Functions:**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float`
    -   `calculate_reward(placed_count: int, unique_coords_cleared: Set[Tuple[int, int]], is_game_over: bool, config: EnvConfig) -> float`

## Dependencies

-   **[`alphatriangle.environment.core`](../core/README.md)**:
    -   `GameState`: The primary object operated upon.
    -   `ActionCodec`: Used by `get_valid_actions`.
-   **[`alphatriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used for placement checks and line clearing.
-   **[`alphatriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used for shape refilling.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used for reward calculation and action encoding.
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   `Shape`, `Triangle`, `COLOR_TO_ID_MAP`, `NO_COLOR_ID`.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the logic for determining valid actions, executing placements (including reward calculation and shape refilling), or modifying dependencies.

File: alphatriangle\environment\logic\step.py
# File: alphatriangle/environment/logic/step.py
import logging
import random
from typing import TYPE_CHECKING

# Correct import path for constants
from ...structs.constants import COLOR_TO_ID_MAP, NO_COLOR_ID
from .. import shapes as ShapeLogic

# Import the logic submodule correctly
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ...config import EnvConfig
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def calculate_reward(
    placed_count: int,
    unique_coords_cleared: set[tuple[int, int]],
    is_game_over: bool,
    config: "EnvConfig",
) -> float:
    """
    Calculates the step reward based on the new specification (v3).

    Args:
        placed_count: Number of triangles successfully placed.
        unique_coords_cleared: Set of unique (r, c) coordinates cleared this step.
        is_game_over: Boolean indicating if the game ended *after* this step.
        config: Environment configuration containing reward constants.

    Returns:
        The calculated step reward.
    """
    reward = 0.0

    # 1. Placement Reward
    reward += placed_count * config.REWARD_PER_PLACED_TRIANGLE

    # 2. Line Clear Reward
    reward += len(unique_coords_cleared) * config.REWARD_PER_CLEARED_TRIANGLE

    # 3. Survival Reward OR Game Over Penalty
    if is_game_over:
        reward += config.PENALTY_GAME_OVER
    else:
        reward += config.REWARD_PER_STEP_ALIVE

    logger.debug(
        f"Calculated Reward: Placement({placed_count * config.REWARD_PER_PLACED_TRIANGLE:.3f}) "
        f"+ LineClear({len(unique_coords_cleared) * config.REWARD_PER_CLEARED_TRIANGLE:.3f}) "
        f"+ {'GameOver' if is_game_over else 'Survival'}({config.PENALTY_GAME_OVER if is_game_over else config.REWARD_PER_STEP_ALIVE:.3f}) "
        f"= {reward:.3f}"
    )
    return reward


def execute_placement(
    game_state: "GameState", shape_idx: int, r: int, c: int, rng: random.Random
) -> float:
    """
    Places a shape, clears lines, updates game state (NumPy arrays), and calculates reward.
    Handles batch refilling of shapes.

    Args:
        game_state: The current game state (will be modified).
        shape_idx: Index of the shape to place.
        r: Target row for placement.
        c: Target column for placement.
        rng: Random number generator for shape refilling.

    Returns:
        The reward obtained for this step.
    """
    shape = game_state.shapes[shape_idx]
    if not shape:
        logger.error(f"Attempted to place an empty shape slot: {shape_idx}")
        return 0.0

    # Use the NumPy-based can_place from GridLogic
    if not GridLogic.can_place(game_state.grid_data, shape, r, c):
        logger.error(f"Invalid placement attempted: Shape {shape_idx} at ({r},{c})")
        # It's possible this check fails even if valid_actions included it,
        # especially if the state changed unexpectedly (e.g., in multi-threaded envs, though not the case here).
        # Returning 0 reward is reasonable.
        return 0.0

    # --- Place the shape ---
    placed_coords: set[tuple[int, int]] = set()
    placed_count = 0
    # Get color ID from the shape's color
    color_id = COLOR_TO_ID_MAP.get(shape.color, NO_COLOR_ID)
    if color_id == NO_COLOR_ID:
        logger.warning(f"Shape color {shape.color} not found in COLOR_TO_ID_MAP!")
        # Assign a default color ID? Or handle as error? Let's use 0 for now.
        color_id = 0

    for dr, dc, _ in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        # Check validity using GridData method (which checks bounds)
        if game_state.grid_data.valid(tri_r, tri_c):
            # Check death and occupancy using NumPy arrays
            if (
                not game_state.grid_data._death_np[tri_r, tri_c]
                and not game_state.grid_data._occupied_np[tri_r, tri_c]
            ):
                # Update NumPy arrays
                game_state.grid_data._occupied_np[tri_r, tri_c] = True
                game_state.grid_data._color_id_np[tri_r, tri_c] = color_id
                placed_coords.add((tri_r, tri_c))
                placed_count += 1
            else:
                # This case should ideally not be reached if can_place passed. Log if it does.
                logger.error(
                    f"Placement conflict at ({tri_r},{tri_c}) during execution, though can_place was true."
                )
        else:
            # This case should ideally not be reached if can_place passed. Log if it does.
            logger.error(
                f"Invalid coordinates ({tri_r},{tri_c}) encountered during placement execution."
            )

    game_state.shapes[shape_idx] = None  # Remove shape from slot
    game_state.pieces_placed_this_episode += 1

    # --- Check and clear lines ---
    # Use check_and_clear_lines from GridLogic
    lines_cleared_count, unique_coords_cleared, _ = GridLogic.check_and_clear_lines(
        game_state.grid_data, placed_coords
    )
    game_state.triangles_cleared_this_episode += len(unique_coords_cleared)

    # --- Update Score (Optional tracking) ---
    game_state.game_score += placed_count + len(unique_coords_cleared) * 2

    # --- Refill shapes if all slots are empty ---
    if all(s is None for s in game_state.shapes):
        logger.debug("All shape slots empty, triggering batch refill.")
        ShapeLogic.refill_shape_slots(game_state, rng)

    # --- Check for game over AFTER placement and refill ---
    # Game is over if no valid moves remain for the *new* state
    if not game_state.valid_actions():
        game_state.game_over = True
        logger.info(
            f"Game over detected after placing shape {shape_idx} and potential refill."
        )

    # --- Calculate Reward based on the outcome of this step ---
    step_reward = calculate_reward(
        placed_count=placed_count,
        unique_coords_cleared=unique_coords_cleared,  # Pass the set of cleared coords
        is_game_over=game_state.game_over,
        config=game_state.env_config,
    )

    return step_reward


File: alphatriangle\environment\logic\__init__.py


File: alphatriangle\environment\shapes\logic.py
# File: alphatriangle/environment/shapes/logic.py
import logging
import random
from typing import TYPE_CHECKING

from ...structs import SHAPE_COLORS, Shape
from .templates import PREDEFINED_SHAPE_TEMPLATES

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def generate_random_shape(rng: random.Random) -> Shape:
    """Generates a random shape from predefined templates and colors."""
    template = rng.choice(PREDEFINED_SHAPE_TEMPLATES)
    color = rng.choice(SHAPE_COLORS)
    return Shape(template, color)


def refill_shape_slots(game_state: "GameState", rng: random.Random) -> None:
    """
    Refills ALL empty shape slots in the GameState, but ONLY if ALL slots are currently empty.
    This implements batch refilling.
    """
    # --- CHANGED: Check if ALL slots are None ---
    if all(shape is None for shape in game_state.shapes):
        logger.debug("All shape slots are empty. Refilling all slots.")
        for i in range(game_state.env_config.NUM_SHAPE_SLOTS):
            game_state.shapes[i] = generate_random_shape(rng)
            logger.debug(f"Refilled slot {i} with {game_state.shapes[i]}")
    else:
        logger.debug("Not all shape slots are empty. Skipping refill.")
    # --- END CHANGED ---


def get_neighbors(r: int, c: int, is_up: bool) -> list[tuple[int, int]]:
    """Gets potential neighbor coordinates for connectivity check."""
    if is_up:
        # Up triangle neighbors: (r, c-1), (r, c+1), (r+1, c)
        return [(r, c - 1), (r, c + 1), (r + 1, c)]
    else:
        # Down triangle neighbors: (r, c-1), (r, c+1), (r-1, c)
        return [(r, c - 1), (r, c + 1), (r - 1, c)]


def is_shape_connected(shape: Shape) -> bool:
    """Checks if all triangles in a shape are connected."""
    if not shape.triangles or len(shape.triangles) <= 1:
        return True

    coords_set = {(r, c) for r, c, _ in shape.triangles}
    start_node = shape.triangles[0][:2]  # (r, c) of the first triangle
    visited: set[tuple[int, int]] = set()
    queue = [start_node]
    visited.add(start_node)

    while queue:
        current_r, current_c = queue.pop(0)
        # Find the orientation of the current triangle in the shape list
        current_is_up = False
        for r, c, is_up in shape.triangles:
            if r == current_r and c == current_c:
                current_is_up = is_up
                break

        for nr, nc in get_neighbors(current_r, current_c, current_is_up):
            neighbor_coord = (nr, nc)
            if neighbor_coord in coords_set and neighbor_coord not in visited:
                visited.add(neighbor_coord)
                queue.append(neighbor_coord)

    return len(visited) == len(coords_set)


File: alphatriangle\environment\shapes\README.md
# File: alphatriangle/environment/shapes/README.md
# Environment Shapes Submodule (`alphatriangle.environment.shapes`)

## Purpose and Architecture

This submodule defines the logic for managing placeable shapes within the game environment.

-   **Shape Representation:** The `Shape` class (defined in [`alphatriangle.structs`](../../structs/README.md)) stores the geometry of a shape as a list of relative triangle coordinates (`(dr, dc, is_up)`) and its color.
-   **Shape Templates:** The [`templates.py`](templates.py) file contains the `PREDEFINED_SHAPE_TEMPLATES` list, which defines the geometry of all possible shapes used in the game. **This list should not be modified.**
-   **Shape Logic:** The [`logic.py`](logic.py) module (exposed as `ShapeLogic`) contains functions related to shapes:
    -   `generate_random_shape`: Creates a new `Shape` instance by randomly selecting a template from `PREDEFINED_SHAPE_TEMPLATES` and assigning a random color (using `SHAPE_COLORS` from [`alphatriangle.structs`](../../structs/README.md)).
    -   `refill_shape_slots`: **Refills ALL empty shape slots** in the `GameState`, but **only if ALL slots are currently empty**. This implements batch refilling.

## Exposed Interfaces

-   **Modules/Namespaces:**
    -   `logic` (often imported as `ShapeLogic`):
        -   `generate_random_shape(rng: random.Random) -> Shape`
        -   `refill_shape_slots(game_state: GameState, rng: random.Random)` **(Refills all slots only if all are empty)**
-   **Constants:**
    -   `PREDEFINED_SHAPE_TEMPLATES` (from `templates.py`): The list of shape geometries.

## Dependencies

-   **[`alphatriangle.environment.core`](../core/README.md)**:
    -   `GameState`: Used by `ShapeLogic.refill_shape_slots` to access and modify the list of available shapes.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Accessed via `GameState` (e.g., for `NUM_SHAPE_SLOTS`).
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   Uses `Shape`, `Triangle`, `SHAPE_COLORS`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the shape generation algorithm or the logic for managing shape slots in the game state (especially the batch refill mechanism). Accurate documentation is crucial for maintainability. **Do not modify `templates.py`.**

File: alphatriangle\environment\shapes\shape.py
class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(
        self, triangles: list[tuple[int, int, bool]], color: tuple[int, int, int]
    ):
        self.triangles: list[tuple[int, int, bool]] = triangles
        self.color: tuple[int, int, int] = color

    def bbox(self) -> tuple[int, int, int, int]:
        """Calculates bounding box (min_r, min_c, max_r, max_c) in relative coords."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))

    def copy(self) -> "Shape":
        """Creates a shallow copy (triangle list is copied, color is shared)."""
        new_shape = Shape.__new__(Shape)
        new_shape.triangles = list(self.triangles)
        new_shape.color = self.color
        return new_shape

    def __str__(self) -> str:
        return f"Shape(Color:{self.color}, Tris:{len(self.triangles)})"


File: alphatriangle\environment\shapes\templates.py
# ==============================================================================
# ==                    PREDEFINED SHAPE TEMPLATES                          ==
# ==                                                                        ==
# ==    DO NOT MODIFY THIS LIST MANUALLY unless you are absolutely sure!    ==
# == These shapes are fundamental to the game's design and balance.         ==
# == Modifying them can have unintended consequences on gameplay and agent  ==
# == training.                                                              ==
# ==============================================================================

# List of predefined shape templates. Each template is a list of relative triangle coordinates (dr, dc, is_up).
# Coordinates are relative to the shape's origin (typically the top-leftmost triangle).
# is_up = True for upward-pointing triangle, False for downward-pointing.
PREDEFINED_SHAPE_TEMPLATES: list[list[tuple[int, int, bool]]] = [
    [  # Shape 1
        (
            0,
            0,
            True,
        )
    ],
    [  # Shape 1
        (
            0,
            0,
            True,
        )
    ],
    [  # Shape 2
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 2
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 3
        (
            0,
            0,
            False,
        )
    ],
    [  # Shape 4
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
    ],
    [  # Shape 4
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
    ],
    [  # Shape 5
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
    ],
    [  # Shape 5
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
    ],
    [  # Shape 6
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
    ],
    [  # Shape 7
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            0,
            2,
            False,
        ),
    ],
    [  # Shape 8
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 9
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 10
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            0,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 11
        (
            0,
            0,
            True,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 12
        (
            0,
            0,
            True,
        ),
        (
            1,
            -2,
            False,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 13
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 14
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 15
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 16
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 17
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 18
        (
            0,
            0,
            True,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 19
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 20
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 21
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 22
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 23
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 24
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 25
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            0,
            2,
            False,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 26
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 27
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
    ],
]


File: alphatriangle\environment\shapes\__init__.py
"""
Shapes submodule handling shape generation and management.
"""

from .logic import (
    generate_random_shape,
    get_neighbors,
    is_shape_connected,
    refill_shape_slots,
)
from .templates import PREDEFINED_SHAPE_TEMPLATES

__all__ = [
    "generate_random_shape",
    "refill_shape_slots",
    "is_shape_connected",
    "get_neighbors",
    "PREDEFINED_SHAPE_TEMPLATES",
]


File: alphatriangle\features\extractor.py
# File: alphatriangle/features/extractor.py
import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from ..config import ModelConfig
from ..utils.types import StateType
from . import grid_features  # Keep this import

if TYPE_CHECKING:
    from ..environment import GameState


logger = logging.getLogger(__name__)


class GameStateFeatures:
    """Extracts features from GameState for NN input. Reads from GridData NumPy arrays."""

    def __init__(self, game_state: "GameState", model_config: ModelConfig):
        self.gs = game_state
        self.env_config = game_state.env_config
        self.model_config = model_config
        # Get direct references to NumPy arrays for efficiency
        self.occupied_np = game_state.grid_data._occupied_np
        self.death_np = game_state.grid_data._death_np
        # self.color_id_np = game_state.grid_data._color_id_np # Not used in current features

    def _get_grid_state(self) -> np.ndarray:
        """
        Returns grid state as a single channel numpy array based on NumPy arrays.
        Values: 1.0 (occupied playable), 0.0 (empty playable), -1.0 (death cell).
        Shape: (C, H, W) where C is GRID_INPUT_CHANNELS
        """
        rows, cols = self.env_config.ROWS, self.env_config.COLS
        # Initialize with 0.0 (empty playable)
        grid_state: np.ndarray = np.zeros(
            (self.model_config.GRID_INPUT_CHANNELS, rows, cols), dtype=np.float32
        )

        # Mark occupied playable cells as 1.0
        playable_occupied_mask = self.occupied_np & ~self.death_np
        grid_state[0, playable_occupied_mask] = 1.0

        # Mark death cells as -1.0
        grid_state[0, self.death_np] = -1.0

        # No need for the loop or isfinite check here if input arrays are guaranteed finite

        return grid_state

    def _get_shape_features(self) -> np.ndarray:
        """Extracts features for each shape slot. (No change needed here)"""
        num_slots = self.env_config.NUM_SHAPE_SLOTS

        FEATURES_PER_SHAPE_HERE = 7
        shape_feature_matrix = np.zeros(
            (num_slots, FEATURES_PER_SHAPE_HERE), dtype=np.float32
        )

        for i, shape in enumerate(self.gs.shapes):
            if shape and shape.triangles:
                n_tris = len(shape.triangles)
                ups = sum(1 for _, _, is_up in shape.triangles if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = shape.bbox()
                height = max_r - min_r + 1
                width_eff = (max_c - min_c + 1) * 0.75 + 0.25 if n_tris > 0 else 0

                # Populate features
                shape_feature_matrix[i, 0] = np.clip(n_tris / 5.0, 0, 1)
                shape_feature_matrix[i, 1] = ups / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 2] = downs / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 3] = np.clip(
                    height / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 4] = np.clip(
                    width_eff / self.env_config.COLS, 0, 1
                )
                shape_feature_matrix[i, 5] = np.clip(
                    ((min_r + max_r) / 2.0) / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 6] = np.clip(
                    ((min_c + max_c) / 2.0) / self.env_config.COLS, 0, 1
                )
        # Flatten the matrix to get a 1D array
        return shape_feature_matrix.flatten()

    def _get_shape_availability(self) -> np.ndarray:
        """Returns a binary vector indicating which shape slots are filled. (No change needed)"""
        return np.array([1.0 if s else 0.0 for s in self.gs.shapes], dtype=np.float32)

    def _get_explicit_features(self) -> np.ndarray:
        """
        Extracts scalar features like score, heights, holes, etc.
        Uses GridData NumPy arrays directly.
        """
        EXPLICIT_FEATURES_DIM_HERE = 6
        features = np.zeros(EXPLICIT_FEATURES_DIM_HERE, dtype=np.float32)
        # Use the direct references stored in self
        occupied = self.occupied_np
        death = self.death_np
        rows, cols = self.env_config.ROWS, self.env_config.COLS

        # Pass NumPy arrays directly to grid_features functions
        heights = grid_features.get_column_heights(occupied, death, rows, cols)
        holes = grid_features.count_holes(occupied, death, heights, rows, cols)
        bump = grid_features.get_bumpiness(heights)
        total_playable_cells = np.sum(~death)

        # Populate features
        features[0] = np.clip(self.gs.game_score / 100.0, -5.0, 5.0)
        features[1] = np.mean(heights) / rows if rows > 0 else 0
        features[2] = np.max(heights) / rows if rows > 0 else 0
        features[3] = holes / total_playable_cells if total_playable_cells > 0 else 0
        features[4] = (bump / (cols - 1)) / rows if cols > 1 and rows > 0 else 0
        features[5] = np.clip(self.gs.pieces_placed_this_episode / 100.0, 0, 1)

        return cast(
            "np.ndarray", np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        )

    def get_combined_other_features(self) -> np.ndarray:
        """Combines all non-grid features into a single flat vector."""
        shape_feats = self._get_shape_features()
        avail_feats = self._get_shape_availability()
        explicit_feats = self._get_explicit_features()
        combined = np.concatenate([shape_feats, avail_feats, explicit_feats])

        expected_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if combined.shape[0] != expected_dim:
            # Log error instead of raising ValueError immediately during feature extraction
            logger.error(
                f"Combined other_features dimension mismatch! Extracted {combined.shape[0]}, but ModelConfig expects {expected_dim}. Padding/truncating."
            )
            # Pad or truncate to match expected dimension
            if combined.shape[0] < expected_dim:
                padding = np.zeros(
                    expected_dim - combined.shape[0], dtype=combined.dtype
                )
                combined = np.concatenate([combined, padding])
            else:
                combined = combined[:expected_dim]

        if not np.all(np.isfinite(combined)):
            logger.error(
                f"Non-finite values detected in combined other_features! Min: {np.nanmin(combined)}, Max: {np.nanmax(combined)}, Mean: {np.nanmean(combined)}"
            )
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        return cast("np.ndarray", combined.astype(np.float32))


def extract_state_features(
    game_state: "GameState", model_config: ModelConfig
) -> StateType:
    """
    Extracts and returns the state dictionary {grid, other_features} for NN input.
    Requires ModelConfig to ensure dimensions match the network's expectations.
    Includes validation for non-finite values. Now reads from GridData NumPy arrays.
    """
    extractor = GameStateFeatures(game_state, model_config)
    state_dict: StateType = {
        "grid": extractor._get_grid_state(),
        "other_features": extractor.get_combined_other_features(),
    }
    grid_feat = state_dict["grid"]
    other_feat = state_dict["other_features"]
    logger.debug(
        f"Extracted Features (State {game_state.current_step}): Grid(shape={grid_feat.shape}, min={grid_feat.min():.2f}, max={grid_feat.max():.2f}, mean={grid_feat.mean():.2f}), Other(shape={other_feat.shape}, min={other_feat.min():.2f}, max={other_feat.max():.2f}, mean={other_feat.mean():.2f})"
    )
    return state_dict


File: alphatriangle\features\grid_features.py
import numpy as np
from numba import njit, prange


@njit(cache=True)
def get_column_heights(
    occupied: np.ndarray, death: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    """Calculates the height of each column (highest occupied non-death cell)."""
    heights = np.zeros(cols, dtype=np.int32)
    for c in prange(cols):
        max_r = -1
        for r in range(rows):
            if occupied[r, c] and not death[r, c]:
                max_r = r
        heights[c] = max_r + 1
    return heights


@njit(cache=True)
def count_holes(
    occupied: np.ndarray, death: np.ndarray, heights: np.ndarray, _rows: int, cols: int
) -> int:
    """Counts the number of empty, non-death cells below the column height."""
    holes = 0
    for c in prange(cols):
        col_height = heights[c]
        for r in range(col_height):
            if not occupied[r, c] and not death[r, c]:
                holes += 1
    return holes


@njit(cache=True)
def get_bumpiness(heights: np.ndarray) -> float:
    """Calculates the total absolute difference between adjacent column heights."""
    bumpiness = 0.0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


File: alphatriangle\features\README.md
# File: alphatriangle/features/README.md
# Feature Extraction Module (`alphatriangle.features`)

## Purpose and Architecture

This module is solely responsible for converting raw [`GameState`](../environment/core/game_state.py) objects from the [`alphatriangle.environment`](../environment/README.md) module into numerical representations (features) suitable for input into the neural network ([`alphatriangle.nn`](../nn/README.md)). It acts as a bridge between the game's internal state and the requirements of the machine learning model.

-   **Decoupling:** This module completely decouples feature engineering from the core game environment logic. The `environment` module focuses only on game rules and state transitions, while this module handles the transformation for the NN.
-   **Feature Engineering:**
    -   [`extractor.py`](extractor.py): Contains the `GameStateFeatures` class and the main `extract_state_features` function. This orchestrates the extraction process, calling helper functions to generate different feature types. It uses `Triangle` and `Shape` from [`alphatriangle.structs`](../structs/README.md).
    -   [`grid_features.py`](grid_features.py): Contains low-level, potentially performance-optimized (e.g., using Numba) functions for calculating specific scalar metrics derived from the grid state (like column heights, holes, bumpiness). **This module now operates directly on NumPy arrays passed from `GameStateFeatures`.**
-   **Output Format:** The `extract_state_features` function returns a `StateType` (a `TypedDict` defined in [`alphatriangle.utils.types`](../utils/types.py) containing `grid` and `other_features` numpy arrays), which is the standard input format expected by the `NeuralNetwork` interface.
-   **Configuration Dependency:** The extractor requires [`ModelConfig`](../config/model_config.py) to ensure the dimensions of the extracted features match the expectations of the neural network architecture.

## Exposed Interfaces

-   **Functions:**
    -   `extract_state_features(game_state: GameState, model_config: ModelConfig) -> StateType`: The main function to perform feature extraction.
    -   Low-level grid feature functions from `grid_features` (e.g., `get_column_heights`, `count_holes`, `get_bumpiness`).
-   **Classes:**
    -   `GameStateFeatures`: Class containing the feature extraction logic (primarily used internally by `extract_state_features`).

## Dependencies

-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: The input object for feature extraction.
    -   `GridData`: Accessed via `GameState` to get grid information (NumPy arrays).
-   **[`alphatriangle.config`](../config/README.md)**:
    -   `EnvConfig`: Accessed via `GameState` for environment dimensions.
    -   `ModelConfig`: Required by `extract_state_features` to ensure output dimensions match the NN input layer.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   `StateType`: The return type dictionary format.
-   **`numpy`**:
    -   Used extensively for creating and manipulating the numerical feature arrays.
-   **`numba`**:
    -   Used in `grid_features` for performance optimization.
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when changing the feature extraction logic, the set of extracted features, or the output format (`StateType`). Accurate documentation is crucial for maintainability.

File: alphatriangle\features\__init__.py
"""
Feature extraction module.
Converts raw GameState objects into numerical representations suitable for NN input.
"""

from . import grid_features
from .extractor import GameStateFeatures, extract_state_features

__all__ = [
    "extract_state_features",
    "GameStateFeatures",
    "grid_features",
]


File: alphatriangle\interaction\debug_mode_handler.py
# File: alphatriangle/interaction/debug_mode_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from ..environment import grid as env_grid

# Import constants from the structs package directly
from ..structs import DEBUG_COLOR_ID, NO_COLOR_ID
from ..visualization import core as vis_core

if TYPE_CHECKING:
    # Keep Triangle for type hinting if GridLogic still uses it temporarily
    from .input_handler import InputHandler

logger = logging.getLogger(__name__)


def handle_debug_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in debug mode (toggle triangle state using NumPy arrays)."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect:
        logger.error("Grid layout rectangle not available for debug click.")
        return

    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )
    if not grid_coords:
        return

    r, c = grid_coords
    if game_state.grid_data.valid(r, c):
        # Check death zone first
        if not game_state.grid_data._death_np[r, c]:
            # Toggle occupancy state in NumPy array
            current_occupied_state = game_state.grid_data._occupied_np[r, c]
            new_occupied_state = not current_occupied_state
            game_state.grid_data._occupied_np[r, c] = new_occupied_state

            # Update color ID based on new state
            new_color_id = DEBUG_COLOR_ID if new_occupied_state else NO_COLOR_ID
            game_state.grid_data._color_id_np[r, c] = new_color_id

            logger.debug(
                f": Toggled triangle ({r},{c}) -> {'Occupied' if new_occupied_state else 'Empty'}"
            )

            # Check for line clears if the cell became occupied
            if new_occupied_state:
                # Pass the coordinate tuple in a set
                lines_cleared, unique_tris_coords, _ = (
                    env_grid.logic.check_and_clear_lines(
                        game_state.grid_data, newly_occupied_coords={(r, c)}
                    )
                )
                if lines_cleared > 0:
                    logger.debug(
                        f"Cleared {lines_cleared} lines ({len(unique_tris_coords)} coords) after toggle."
                    )
        else:
            logger.info(f"Clicked on death cell ({r},{c}). No action.")


def update_debug_hover(handler: "InputHandler") -> None:
    """Updates the debug highlight position within the InputHandler."""
    handler.debug_highlight_coord = None  # Reset hover state

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect or not grid_rect.collidepoint(mouse_pos):
        return  # Not hovering over grid

    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    if grid_coords:
        r, c = grid_coords
        # Highlight only valid, non-death cells
        if game_state.grid_data.valid(r, c) and not game_state.grid_data.is_death(r, c):
            handler.debug_highlight_coord = grid_coords


File: alphatriangle\interaction\event_processor.py
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pygame

if TYPE_CHECKING:
    from ..visualization.core.visualizer import Visualizer

logger = logging.getLogger(__name__)


def process_pygame_events(
    visualizer: "Visualizer",
) -> Generator[pygame.event.Event, Any, bool]:
    """
    Processes basic Pygame events like QUIT, ESCAPE, VIDEORESIZE.
    Yields other events for mode-specific handlers.
    Returns False via StopIteration value if the application should quit, True otherwise.
    """
    should_quit = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            logger.info("Received QUIT event.")
            should_quit = True
            break
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            logger.info("Received ESCAPE key press.")
            should_quit = True
            break
        if event.type == pygame.VIDEORESIZE:
            try:
                w, h = max(320, event.w), max(240, event.h)
                visualizer.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                visualizer.layout_rects = None
                logger.info(f"Window resized to {w}x{h}")
            except pygame.error as e:
                logger.error(f"Error resizing window: {e}")
            yield event
        else:
            yield event
    return not should_quit


File: alphatriangle\interaction\input_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from .. import environment, visualization
from . import debug_mode_handler, event_processor, play_mode_handler

if TYPE_CHECKING:
    from ..structs import Shape


logger = logging.getLogger(__name__)


class InputHandler:
    """
    Handles user input, manages interaction state (selection, hover),
    and delegates actions to mode-specific handlers.
    """

    def __init__(
        self,
        game_state: environment.GameState,
        visualizer: visualization.Visualizer,
        mode: str,
        env_config: environment.EnvConfig,
    ):
        self.game_state = game_state
        self.visualizer = visualizer
        self.mode = mode
        self.env_config = env_config

        # Interaction state managed here
        self.selected_shape_idx: int = -1
        self.hover_grid_coord: tuple[int, int] | None = None
        self.hover_is_valid: bool = False
        self.hover_shape: Shape | None = None
        self.debug_highlight_coord: tuple[int, int] | None = None
        self.mouse_pos: tuple[int, int] = (0, 0)

    def handle_input(self) -> bool:
        """Processes Pygame events and updates state based on mode. Returns False to quit."""
        self.mouse_pos = pygame.mouse.get_pos()

        # Reset hover/highlight state each frame before processing events/updates
        self.hover_grid_coord = None
        self.hover_is_valid = False
        self.hover_shape = None
        self.debug_highlight_coord = None

        running = True
        event_generator = event_processor.process_pygame_events(self.visualizer)
        try:
            while True:
                event = next(event_generator)
                # Pass self to handlers so they can modify interaction state
                if self.mode == "play":
                    play_mode_handler.handle_play_click(event, self)
                elif self.mode == "debug":
                    debug_mode_handler.handle_debug_click(event, self)
        except StopIteration as e:
            running = e.value  # False if quit requested

        # Update hover state after processing events
        if running:
            if self.mode == "play":
                play_mode_handler.update_play_hover(self)
            elif self.mode == "debug":
                debug_mode_handler.update_debug_hover(self)

        return running

    def get_render_interaction_state(self) -> dict:
        """Returns interaction state needed by Visualizer.render"""
        return {
            "selected_shape_idx": self.selected_shape_idx,
            "hover_shape": self.hover_shape,
            "hover_grid_coord": self.hover_grid_coord,
            "hover_is_valid": self.hover_is_valid,
            "hover_screen_pos": self.mouse_pos,  # Pass current mouse pos
            "debug_highlight_coord": self.debug_highlight_coord,
        }


File: alphatriangle\interaction\play_mode_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from ..environment import core as env_core
from ..environment import grid as env_grid
from ..visualization import core as vis_core

if TYPE_CHECKING:
    from ..structs import Shape
    from .input_handler import InputHandler

logger = logging.getLogger(__name__)


def handle_play_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in play mode (select preview, place shape). Modifies handler state."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    if game_state.is_over():
        logger.info("Game is over, ignoring click.")
        return

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    # Get preview rects from visualizer cache
    preview_rects = visualizer.preview_rects

    # 1. Check for clicks on shape previews
    preview_idx = vis_core.coord_mapper.get_preview_index_from_screen(
        mouse_pos, preview_rects
    )
    if preview_idx is not None:
        if handler.selected_shape_idx == preview_idx:
            # Clicked selected shape again: deselect
            handler.selected_shape_idx = -1
            handler.hover_grid_coord = None  # Clear hover state on deselect
            handler.hover_shape = None
            logger.info("Deselected shape.")
        elif (
            0 <= preview_idx < len(game_state.shapes) and game_state.shapes[preview_idx]
        ):
            # Clicked a valid, available shape: select it
            handler.selected_shape_idx = preview_idx
            logger.info(f"Selected shape index: {preview_idx}")
            # Immediately update hover based on current mouse pos after selection
            update_play_hover(handler)  # Update hover state within handler
        else:
            # Clicked an empty or invalid slot
            logger.info(f"Clicked empty/invalid preview slot: {preview_idx}")
            # Deselect if clicking an empty slot while another is selected
            if handler.selected_shape_idx != -1:
                handler.selected_shape_idx = -1
                handler.hover_grid_coord = None
                handler.hover_shape = None
        return  # Handled preview click

    # 2. Check for clicks on the grid (if a shape is selected)
    selected_idx = handler.selected_shape_idx
    if selected_idx != -1 and grid_rect and grid_rect.collidepoint(mouse_pos):
        # A shape is selected, and the click is within the grid area.
        grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
            mouse_pos, grid_rect, game_state.env_config
        )
        # Use TYPE_CHECKING import for Shape type hint
        shape_to_place: Shape | None = game_state.shapes[selected_idx]

        # Check if the placement is valid *at the clicked location*
        if (
            grid_coords
            and shape_to_place
            and env_grid.logic.can_place(
                game_state.grid_data, shape_to_place, grid_coords[0], grid_coords[1]
            )
        ):
            # Valid placement click!
            r, c = grid_coords
            action = env_core.action_codec.encode_action(
                selected_idx, r, c, game_state.env_config
            )
            # Execute the step using the game state's method
            reward, done = game_state.step(action)  # Now returns (reward, done)
            logger.info(
                f"Placed shape {selected_idx} at {grid_coords}. R={reward:.1f}, Done={done}"
            )
            # Deselect shape after successful placement
            handler.selected_shape_idx = -1
            handler.hover_grid_coord = None  # Clear hover state
            handler.hover_shape = None
        else:
            # Clicked grid, shape selected, but not a valid placement spot for the click
            logger.info(f"Clicked grid at {grid_coords}, but placement invalid.")


def update_play_hover(handler: "InputHandler") -> None:
    """Updates the hover state within the InputHandler."""
    # Reset hover state first
    handler.hover_grid_coord = None
    handler.hover_is_valid = False
    handler.hover_shape = None

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    if game_state.is_over() or handler.selected_shape_idx == -1:
        return  # No hover if game over or no shape selected

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect or not grid_rect.collidepoint(mouse_pos):
        return  # Not hovering over grid

    shape_idx = handler.selected_shape_idx
    if not (0 <= shape_idx < len(game_state.shapes)):
        return
    shape: Shape | None = game_state.shapes[shape_idx]
    if not shape:
        return

    # Get grid coordinates under mouse
    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    if grid_coords:
        # Check if placement is valid at these coordinates
        is_valid = env_grid.logic.can_place(
            game_state.grid_data, shape, grid_coords[0], grid_coords[1]
        )
        # Update handler's hover state
        handler.hover_grid_coord = grid_coords
        handler.hover_is_valid = is_valid
        handler.hover_shape = shape  # Store the shape being hovered
    else:
        handler.hover_shape = shape  # Store shape for floating preview


File: alphatriangle\interaction\README.md
# File: alphatriangle/interaction/README.md
# Interaction Module (`alphatriangle.interaction`)

## Purpose and Architecture

This module handles user input (keyboard and mouse) for interactive modes of the application, such as "play" and "debug". It bridges the gap between raw Pygame events and actions within the game simulation ([`GameState`](../environment/core/game_state.py)).

-   **Event Processing:** [`event_processor.py`](event_processor.py) handles common Pygame events like quitting (QUIT, ESC) and window resizing. It acts as a generator, yielding other events for mode-specific processing.
-   **Input Handler:** The [`InputHandler`](input_handler.py) class is the main entry point.
    -   It receives Pygame events (via the `event_processor`).
    -   It **manages interaction-specific state** internally (e.g., `selected_shape_idx`, `hover_grid_coord`, `debug_highlight_coord`).
    -   It determines the current interaction mode ("play" or "debug") and delegates event handling and hover updates to specific handler functions ([`play_mode_handler`](play_mode_handler.py), [`debug_mode_handler`](debug_mode_handler.py)).
    -   It provides the necessary interaction state to the [`Visualizer`](../visualization/core/visualizer.py) for rendering feedback (hover previews, selection highlights).
-   **Mode-Specific Handlers:** `play_mode_handler.py` and `debug_mode_handler.py` contain the logic specific to each mode, operating on the `InputHandler`'s state and the `GameState`.
    -   `play`: Handles selecting shapes, checking placement validity, and triggering `GameState.step` on valid clicks. Updates hover state in the `InputHandler`.
    -   `debug`: Handles toggling the state of individual triangles directly on the `GameState.grid_data`. Updates hover state in the `InputHandler`.
-   **Decoupling:** It separates input handling logic from the core game simulation ([`environment`](../environment/README.md)) and rendering ([`visualization`](../visualization/README.md)), although it needs references to both to function. The `Visualizer` is now only responsible for drawing based on the state provided by the `GameState` and the `InputHandler`.

## Exposed Interfaces

-   **Classes:**
    -   `InputHandler`:
        -   `__init__(game_state: GameState, visualizer: Visualizer, mode: str, env_config: EnvConfig)`
        -   `handle_input() -> bool`: Processes events for one frame, returns `False` if quitting.
        -   `get_render_interaction_state() -> dict`: Returns interaction state needed by `Visualizer.render`.
-   **Functions:**
    -   `process_pygame_events(visualizer: Visualizer) -> Generator[pygame.event.Event, Any, bool]`: Processes common events, yields others.
    -   `handle_play_click(event: pygame.event.Event, handler: InputHandler)`: Handles clicks in play mode.
    -   `update_play_hover(handler: InputHandler)`: Updates hover state in play mode.
    -   `handle_debug_click(event: pygame.event.Event, handler: InputHandler)`: Handles clicks in debug mode.
    -   `update_debug_hover(handler: InputHandler)`: Updates hover state in debug mode.

## Dependencies

-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: Modifies the game state based on user actions (placing shapes, toggling debug cells).
    -   `EnvConfig`: Used for coordinate mapping and action encoding.
    -   `GridLogic`, `ActionCodec`: Used by mode-specific handlers.
-   **[`alphatriangle.visualization`](../visualization/README.md)**:
    -   `Visualizer`: Used to get layout information (`grid_rect`, `preview_rects`) and for coordinate mapping (`get_grid_coords_from_screen`, `get_preview_index_from_screen`). Also updated directly during resize events.
    -   `VisConfig`: Accessed via `Visualizer`.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   `Shape`, `Triangle`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **`pygame`**:
    -   Relies heavily on Pygame for event handling (`pygame.event`, `pygame.mouse`) and constants (`MOUSEBUTTONDOWN`, `KEYDOWN`, etc.).
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when adding new interaction modes, changing input handling logic, or modifying the interfaces between interaction, environment, and visualization. Accurate documentation is crucial for maintainability.

File: alphatriangle\interaction\__init__.py
from .debug_mode_handler import handle_debug_click, update_debug_hover
from .event_processor import process_pygame_events
from .input_handler import InputHandler
from .play_mode_handler import handle_play_click, update_play_hover

__all__ = [
    "InputHandler",
    "process_pygame_events",
    "handle_play_click",
    "update_play_hover",
    "handle_debug_click",
    "update_debug_hover",
]


File: alphatriangle\mcts\README.md
# File: alphatriangle/mcts/README.md
# Monte Carlo Tree Search Module (`alphatriangle.mcts`)

## Purpose and Architecture

This module implements the Monte Carlo Tree Search algorithm, a core component of the AlphaZero-style reinforcement learning agent. MCTS is used during self-play to explore the game tree and determine the next best move and generate training targets for the neural network.

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Node`: Represents a state in the search tree, storing visit counts, value estimates, prior probabilities, and child nodes. Holds a `GameState` object.
    -   `search`: Contains the main `run_mcts_simulations` function orchestrating the selection, expansion, and backpropagation phases. **This version uses batched neural network evaluation (`evaluate_batch`) for potentially improved performance.** It collects multiple leaf nodes before calling the network.
    -   `config`: Defines the `MCTSConfig` class holding hyperparameters like the number of simulations, PUCT coefficient, temperature settings, and Dirichlet noise parameters.
    -   `types`: Defines necessary type hints and protocols, notably `ActionPolicyValueEvaluator` which specifies the interface required for the neural network evaluator used by MCTS.
-   **Strategy Components ([`strategy/README.md`](strategy/README.md)):**
    -   `selection`: Implements the tree traversal logic (PUCT calculation, Dirichlet noise addition, leaf selection).
    -   `expansion`: Handles expanding leaf nodes **using pre-computed policy priors** obtained from batched network evaluation.
    -   `backpropagation`: Implements the process of updating node statistics back up the tree after a simulation.
    -   `policy`: Provides functions to select the final action based on visit counts (`select_action_based_on_visits`) and to generate the policy target vector for training (`get_policy_target`).

## Exposed Interfaces

-   **Core:**
    -   `Node`: The tree node class.
    -   `MCTSConfig`: Configuration class (defined in [`alphatriangle.config`](../config/README.md)).
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network_evaluator: ActionPolicyValueEvaluator)`: The main function to run MCTS (uses batched evaluation).
    -   `ActionPolicyValueEvaluator`: Protocol defining the NN evaluation interface.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary.
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Strategy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.

## Dependencies

-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: Represents the state within each `Node`. MCTS interacts heavily with `GameState` methods like `copy()`, `step()`, `is_over()`, `get_outcome()`, `valid_actions()`.
    -   `EnvConfig`: Accessed via `GameState`.
-   **[`alphatriangle.nn`](../nn/README.md)**:
    -   `NeuralNetwork`: An instance conforming to the `ActionPolicyValueEvaluator` protocol is required by `run_mcts_simulations` and `expansion` to evaluate states (specifically `evaluate_batch`).
-   **[`alphatriangle.config`](../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   `ActionType`, `PolicyValueOutput`: Used for actions and NN return types.
-   **`numpy`**:
    -   Used for Dirichlet noise generation and potentially in policy calculations.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `time`, `concurrent.futures`.

---

**Note:** Please keep this README updated when changing the MCTS algorithm phases (selection, expansion, backpropagation), the node structure, configuration options, or the interaction with the environment or neural network, especially regarding the batched evaluation. Accurate documentation is crucial for maintainability.

File: alphatriangle\mcts\__init__.py
"""
Monte Carlo Tree Search (MCTS) module.
Provides the core algorithm and components for game tree search.
"""

from alphatriangle.config import MCTSConfig

from .core.node import Node
from .core.search import (
    MCTSExecutionError,
    run_mcts_simulations,
)
from .core.types import ActionPolicyMapping, ActionPolicyValueEvaluator
from .strategy.policy import get_policy_target, select_action_based_on_visits

__all__ = [
    # Core
    "Node",
    "run_mcts_simulations",
    "MCTSConfig",
    "ActionPolicyValueEvaluator",
    "ActionPolicyMapping",
    "MCTSExecutionError",
    # Strategy
    "select_action_based_on_visits",
    "get_policy_target",
]


File: alphatriangle\mcts\core\node.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphatriangle.environment import GameState
    from alphatriangle.utils.types import ActionType

logger = logging.getLogger(__name__)


class Node:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        state: GameState,
        parent: Node | None = None,
        action_taken: ActionType | None = None,
        prior_probability: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children: dict[ActionType, Node] = {}

        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.prior_probability: float = prior_probability

    @property
    def is_expanded(self) -> bool:
        """Checks if the node has been expanded (i.e., children generated)."""
        return bool(self.children)

    @property
    def is_leaf(self) -> bool:
        """Checks if the node is a leaf (not expanded)."""
        return not self.is_expanded

    @property
    def value_estimate(self) -> float:
        """
        Calculates the Q-value (average action value) estimate for this node's state.
        This is the average value observed from simulations starting from this state.
        Refactored for clarity and safety.
        """
        if self.visit_count == 0:
            return 0.0

        visits = max(1, self.visit_count)
        q_value = self.total_action_value / visits

        return q_value

    def __repr__(self) -> str:
        return (
            f"Node(StateStep={self.state.current_step}, "
            f"FromAction={self.action_taken}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )


File: alphatriangle\mcts\core\README.md
# File: alphatriangle/mcts/core/README.md
# MCTS Core Submodule (`alphatriangle.mcts.core`)

## Purpose and Architecture

This submodule defines the fundamental building blocks and interfaces for the Monte Carlo Tree Search implementation.

-   **[`Node`](node.py):** The `Node` class is the cornerstone, representing a single state within the search tree. It stores the associated `GameState`, parent/child relationships, the action that led to it, and crucial MCTS statistics (visit count, total action value, prior probability). It provides properties like `value_estimate` (Q-value) and `is_expanded`.
-   **[`search`](search.py):** The `search.py` module contains the high-level `run_mcts_simulations` function. This function orchestrates the core MCTS loop for a given root node: repeatedly selecting leaves, batch-evaluating them using the network, expanding them, and backpropagating the results, using helper functions from the [`alphatriangle.mcts.strategy`](../strategy/README.md) submodule. **It uses a `ThreadPoolExecutor` for parallel traversals and batches network evaluations.**
-   **[`types`](types.py):** The `types.py` module defines essential type hints and protocols for the MCTS module. Most importantly, it defines the `ActionPolicyValueEvaluator` protocol, which specifies the `evaluate` and `evaluate_batch` methods that any neural network interface must implement to be usable by the MCTS expansion phase. It also defines `ActionPolicyMapping`.

## Exposed Interfaces

-   **Classes:**
    -   `Node`: Represents a node in the search tree.
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Functions:**
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network_evaluator: ActionPolicyValueEvaluator)`: Orchestrates the MCTS process using batched evaluation and parallel traversals.
-   **Protocols/Types:**
    -   `ActionPolicyValueEvaluator`: Defines the interface for the NN evaluator.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary (mapping action index to probability).

## Dependencies

-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`: Used within `Node` to represent the state. Methods like `is_over`, `get_outcome`, `valid_actions`, `copy`, `step` are used during the MCTS process (selection, expansion).
-   **[`alphatriangle.mcts.strategy`](../strategy/README.md)**:
    -   `selection`, `expansion`, `backpropagation`: The `run_mcts_simulations` function delegates the core algorithm phases to functions within this submodule.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`, `PolicyValueOutput`: Used in type hints and protocols.
-   **Standard Libraries:** `typing`, `math`, `logging`, `concurrent.futures`, `time`.
-   **`numpy`**: Used for validation checks.

---

**Note:** Please keep this README updated when modifying the `Node` structure, the `run_mcts_simulations` logic (especially parallelism and batching), or the `ActionPolicyValueEvaluator` interface definition. Accurate documentation is crucial for maintainability.

File: alphatriangle\mcts\core\search.py
# File: alphatriangle/mcts/core/search.py
import concurrent.futures
import logging
import time

import numpy as np

from ...config import MCTSConfig
from ..strategy import backpropagation, expansion, selection
from .node import Node
from .types import ActionPolicyValueEvaluator

logger = logging.getLogger(__name__)

# --- CHANGED: Default batch size, can be adjusted ---
MCTS_PARALLEL_TRAVERSALS = 16  # Number of traversals to run in parallel
# --- END CHANGED ---


class MCTSExecutionError(Exception):
    """Custom exception for errors during MCTS execution."""

    pass


def _run_single_traversal(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """Helper function to run a single MCTS traversal (selection phase)."""
    # This function is designed to be thread-safe as selection reads node stats
    # but doesn't modify them until backpropagation.
    try:
        leaf_node, selection_depth = selection.traverse_to_leaf(root_node, config)
        return leaf_node, selection_depth
    except Exception as e:
        # Log error within the thread/task for better context
        logger.error(
            f"[MCTS Traversal Task] Error during traversal: {e}", exc_info=True
        )
        # Re-raise or return an indicator? Re-raising is cleaner for future handling.
        raise MCTSExecutionError(f"Traversal failed: {e}") from e


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network_evaluator: ActionPolicyValueEvaluator,
) -> int:
    """
    Runs the specified number of MCTS simulations from the root node.
    Uses a ThreadPoolExecutor to run selection traversals in parallel.
    Neural network evaluations are batched. Backpropagation is sequential.

    Returns:
        The maximum tree depth reached during the simulations.
    """
    if root_node.state.is_over():
        logger.warning("[MCTS] MCTS started on a terminal state. No simulations run.")
        return 0

    max_depth_overall = 0
    sim_success_count = 0
    sim_error_count = 0
    eval_error_count = 0
    total_sims_run = 0

    # --- Initial Root Expansion (if needed) ---
    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node not expanded, performing initial evaluation...")
        try:
            action_policy, root_value = network_evaluator.evaluate(root_node.state)
            # Basic validation (can be enhanced)
            if not isinstance(action_policy, dict) or not isinstance(root_value, float):
                raise MCTSExecutionError("Initial evaluation returned invalid type.")
            if not np.all(np.isfinite(list(action_policy.values()))):
                raise MCTSExecutionError(
                    "Initial evaluation returned non-finite policy."
                )
            if not np.isfinite(root_value):
                raise MCTSExecutionError(
                    "Initial evaluation returned non-finite value."
                )

            expansion.expand_node_with_policy(root_node, action_policy)
            if root_node.is_expanded or root_node.state.is_over():
                depth_bp = backpropagation.backpropagate_value(root_node, root_value)
                max_depth_overall = max(max_depth_overall, depth_bp)
                selection.add_dirichlet_noise(
                    root_node, config
                )  # Apply noise after first expansion/backprop
            else:
                logger.warning("[MCTS] Initial root expansion failed.")
        except Exception as e:
            logger.error(
                f"[MCTS] Initial root evaluation/expansion failed: {e}", exc_info=True
            )
            raise MCTSExecutionError(
                f"Initial root evaluation/expansion failed: {e}"
            ) from e
    elif root_node.visit_count == 0:  # Apply noise if root is expanded but unvisited
        selection.add_dirichlet_noise(root_node, config)
    # --- End Initial Root Expansion ---

    logger.info(
        f"[MCTS] Starting MCTS loop for {config.num_simulations} simulations "
        f"(Parallel Traversals: {MCTS_PARALLEL_TRAVERSALS}). Root state step: {root_node.state.current_step}"
    )

    # Use ThreadPoolExecutor for parallel traversals
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MCTS_PARALLEL_TRAVERSALS
    ) as executor:
        pending_simulations = config.num_simulations
        processed_simulations = 0

        while pending_simulations > 0:
            num_to_launch = min(pending_simulations, MCTS_PARALLEL_TRAVERSALS)
            logger.debug(
                f"[MCTS Batch] Launching {num_to_launch} parallel traversals..."
            )

            # --- Submit Traversal Tasks ---
            futures_to_leaf: dict[concurrent.futures.Future, int] = {}
            for i in range(num_to_launch):
                future = executor.submit(_run_single_traversal, root_node, config)
                futures_to_leaf[future] = processed_simulations + i  # Store sim index

            leaves_to_evaluate: list[Node] = []
            paths_to_backprop: list[tuple[Node, float]] = []
            traversal_results: list[tuple[Node | None, int, Exception | None]] = []

            # --- Collect Traversal Results ---
            for future in concurrent.futures.as_completed(futures_to_leaf):
                sim_idx = futures_to_leaf[future]
                try:
                    leaf_node, selection_depth = future.result()
                    traversal_results.append((leaf_node, selection_depth, None))
                    logger.debug(
                        f"  [MCTS Traversal] Sim {sim_idx + 1} completed. Depth: {selection_depth}, Leaf: {leaf_node}"
                    )
                except Exception as exc:
                    sim_error_count += 1
                    traversal_results.append((None, 0, exc))
                    logger.error(f"  [MCTS Traversal] Sim {sim_idx + 1} failed: {exc}")

            # --- Process Traversal Results ---
            for leaf_node_optional, selection_depth, error in traversal_results:
                # --- CHANGED: Explicit check and assignment ---
                if error or leaf_node_optional is None:
                    continue  # Skip failed traversals

                # Now we know leaf_node_optional is not None, assign to typed variable
                valid_leaf_node: Node = leaf_node_optional
                # --- END CHANGED ---

                max_depth_overall = max(max_depth_overall, selection_depth)

                # --- Use valid_leaf_node ---
                if valid_leaf_node.state.is_over():
                    outcome = valid_leaf_node.state.get_outcome()
                    logger.debug(
                        f"    [Process] Sim result: TERMINAL leaf. Outcome: {outcome:.3f}. Adding to backprop."
                    )
                    paths_to_backprop.append((valid_leaf_node, outcome))
                elif not valid_leaf_node.is_expanded:
                    logger.debug(
                        "    [Process] Sim result: Leaf needs EVALUATION. Adding to batch."
                    )
                    leaves_to_evaluate.append(valid_leaf_node)
                else:  # Hit max depth or encountered selection error resulting in expanded node
                    logger.debug(
                        f"    [Process] Sim result: EXPANDED leaf (likely max depth). Value: {valid_leaf_node.value_estimate:.3f}. Adding to backprop."
                    )
                    paths_to_backprop.append(
                        (valid_leaf_node, valid_leaf_node.value_estimate)
                    )
                # --- END Use valid_leaf_node ---

            # --- Batch Evaluate Leaves ---
            evaluation_start_time = time.monotonic()
            if leaves_to_evaluate:
                logger.debug(
                    f"  [MCTS Eval] Evaluating batch of {len(leaves_to_evaluate)} leaves..."
                )
                try:
                    leaf_states = [node.state for node in leaves_to_evaluate]
                    batch_results = network_evaluator.evaluate_batch(leaf_states)

                    if batch_results is None or len(batch_results) != len(
                        leaves_to_evaluate
                    ):
                        raise MCTSExecutionError(
                            "Network evaluation returned invalid results."
                        )

                    for i, node in enumerate(leaves_to_evaluate):
                        action_policy, value = batch_results[i]
                        # Basic validation
                        if (
                            not isinstance(action_policy, dict)
                            or not isinstance(value, float)
                            or not np.isfinite(value)
                        ):
                            logger.error(
                                f"    [MCTS Eval] Invalid policy/value received for leaf {i}. Policy: {type(action_policy)}, Value: {value}. Using 0 value."
                            )
                            value = 0.0  # Use neutral value on error
                            action_policy = {}  # Use empty policy on error

                        if not node.is_expanded and not node.state.is_over():
                            expansion.expand_node_with_policy(node, action_policy)
                            logger.debug(
                                f"    [MCTS Eval/Expand] Expanded evaluated leaf node {i}: {node}"
                            )

                        paths_to_backprop.append(
                            (node, value)
                        )  # Add evaluated node for backprop

                except Exception as e:
                    eval_error_count += len(leaves_to_evaluate)
                    logger.error(
                        f"  [MCTS Eval] Error during batch evaluation/expansion: {e}",
                        exc_info=True,
                    )
                    # Skip backprop for these leaves if eval failed

            evaluation_duration = time.monotonic() - evaluation_start_time
            if leaves_to_evaluate:
                logger.debug(
                    f"  [MCTS Eval] Evaluation/Expansion phase finished. Duration: {evaluation_duration:.4f}s"
                )

            # --- Sequential Backpropagation ---
            backprop_start_time = time.monotonic()
            logger.debug(
                f"  [MCTS Backprop] Backpropagating {len(paths_to_backprop)} paths..."
            )
            for i, (leaf_node_bp, value_to_prop) in enumerate(paths_to_backprop):
                try:
                    depth_bp = backpropagation.backpropagate_value(
                        leaf_node_bp, value_to_prop
                    )
                    max_depth_overall = max(max_depth_overall, depth_bp)
                    sim_success_count += 1
                    logger.debug(
                        f"    [Backprop] Path {i}: Value={value_to_prop:.4f}, Depth={depth_bp}, Node={leaf_node_bp}"
                    )
                except Exception as bp_err:
                    logger.error(
                        f"    [Backprop] Error backpropagating path {i} (Value={value_to_prop:.4f}, Node={leaf_node_bp}): {bp_err}",
                        exc_info=True,
                    )
                    sim_error_count += 1  # Count backprop errors separately

            backprop_duration = time.monotonic() - backprop_start_time
            logger.debug(
                f"  [MCTS Backprop] Backpropagation phase finished. Duration: {backprop_duration:.4f}s"
            )

            # --- Update Loop Control ---
            processed_simulations += num_to_launch
            pending_simulations -= num_to_launch
            total_sims_run = (
                processed_simulations  # Track total sims attempted in this run
            )

            logger.debug(
                f"[MCTS Batch] Finished batch. Processed: {processed_simulations}/{config.num_simulations}. Pending: {pending_simulations}"
            )

    # --- Final Logging ---
    final_log_level = logging.INFO
    logger.log(
        final_log_level,
        f"[MCTS] MCTS loop finished. Target Sims: {config.num_simulations}. Attempted: {total_sims_run}. "
        f"Successful Backprops: {sim_success_count}. Traversal Errors: {sim_error_count}. Eval Errors: {eval_error_count}. "
        f"Root visits: {root_node.visit_count}. Max depth reached: {max_depth_overall}",
    )
    if root_node.children:
        child_visits_log = {a: c.visit_count for a, c in root_node.children.items()}
        logger.info(f"[MCTS] Root children visit counts: {child_visits_log}")
    elif not root_node.state.is_over():
        logger.warning("[MCTS] MCTS finished but root node still has no children.")

    # --- Error Check ---
    total_errors = sim_error_count + eval_error_count
    if total_errors > config.num_simulations * 0.01:  # Example threshold: 50% errors
        raise MCTSExecutionError(
            f"MCTS failed: High error rate ({total_errors} errors in {total_sims_run} simulations)."
        )
    elif total_errors > 0:
        logger.warning(f"[MCTS] Completed with {total_errors} errors.")

    return max_depth_overall


File: alphatriangle\mcts\core\types.py
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from ...utils.types import PolicyValueOutput

if TYPE_CHECKING:
    from ...environment import GameState
    from ...utils.types import ActionType

ActionPolicyMapping = Mapping["ActionType", float]


class ActionPolicyValueEvaluator(Protocol):
    """Defines the interface for evaluating a game state using a neural network."""

    def evaluate(self, state: "GameState") -> PolicyValueOutput:
        """
        Evaluates a single game state using the neural network.

        Args:
            state: The GameState object to evaluate.

        Returns:
            A tuple containing:
                - ActionPolicyMapping: A mapping from valid action indices
                    to their prior probabilities (output by the policy head).
                - float: The estimated value of the state (output by the value head).
        """
        ...

    def evaluate_batch(self, states: list["GameState"]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of game states using the neural network.
        (Optional but recommended for performance if MCTS supports batch evaluation).

        Args:
            states: A list of GameState objects to evaluate.

        Returns:
            A list of tuples, where each tuple corresponds to an input state and contains:
                - ActionPolicyMapping: Action probabilities for that state.
                - float: The estimated value of that state.
        """
        ...


File: alphatriangle\mcts\core\__init__.py


File: alphatriangle\mcts\strategy\backpropagation.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(leaf_node: "Node", value: float) -> int:
    """
    Propagates the simulation value back up the tree from the leaf node.
    Returns the depth of the backpropagation path (number of nodes updated).
    """
    current_node: Node | None = leaf_node
    path_str = []
    depth = 0
    logger.debug(
        f"[Backprop] Starting backprop from leaf (Action={leaf_node.action_taken}, StateStep={leaf_node.state.current_step}) with value={value:.4f}"
    )

    while current_node is not None:
        q_before = current_node.value_estimate
        total_val_before = current_node.total_action_value
        visits_before = current_node.visit_count

        current_node.visit_count += 1
        current_node.total_action_value += value

        q_after = current_node.value_estimate
        total_val_after = current_node.total_action_value
        visits_after = current_node.visit_count

        action_str = (
            f"Act={current_node.action_taken}"
            if current_node.action_taken is not None
            else "Root"
        )
        path_str.append(f"N({action_str},V={visits_after},Q={q_after:.3f})")

        logger.debug(
            f"  [Backprop] Depth {depth}: Node({action_str}), "
            f"Visits: {visits_before} -> {visits_after}, "
            f"AddedVal={value:.4f}, "
            f"TotalVal: {total_val_before:.3f} -> {total_val_after:.3f}, "
            f"Q: {q_before:.3f} -> {q_after:.3f}"
        )

        current_node = current_node.parent
        depth += 1

    logger.debug(f"[Backprop] Finished. Path: {' <- '.join(reversed(path_str))}")
    return depth


File: alphatriangle\mcts\strategy\expansion.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.types import ActionType

from ..core.node import Node
from ..core.types import (
    ActionPolicyMapping,
)

logger = logging.getLogger(__name__)


def expand_node_with_policy(node: Node, action_policy: ActionPolicyMapping):
    """
    Expands a node by creating children for valid actions using the
    pre-computed action policy priors from the network.
    Assumes the node is not terminal and not already expanded.
    Marks the node's state as game_over if no valid actions are found.
    """
    if node.is_expanded:
        logger.debug(f"[Expand] Attempted to expand an already expanded node: {node}")
        return
    if node.state.is_over():
        logger.warning(f"[Expand] Attempted to expand a terminal node: {node}")
        return

    logger.debug(f"[Expand] Expanding Node: {node}")

    # Use TYPE_CHECKING import for ActionType type hint
    valid_actions: list[ActionType] = node.state.valid_actions()
    logger.debug(
        f"[Expand] Found {len(valid_actions)} valid actions for state step {node.state.current_step}."
    )
    logger.debug(
        f"[Expand] Received action policy (first 5): {list(action_policy.items())[:5]}"
    )

    if not valid_actions:
        logger.warning(
            f"[Expand] Expanding node at step {node.state.current_step} with no valid actions but not terminal? Marking state as game over."
        )
        if hasattr(node.state, "game_over"):
            node.state.game_over = True
        elif hasattr(node.state, "_is_over"):
            node.state._is_over = True
        else:
            logger.error("[Expand] Cannot mark state as game over - attribute missing.")
        return

    children_created = 0
    for action in valid_actions:
        prior = action_policy.get(action, 0.0)
        if prior < 0.0:
            logger.warning(
                f"[Expand] Received negative prior ({prior}) for action {action}. Clamping to 0."
            )
            prior = 0.0
        elif prior == 0.0:
            logger.debug(
                f"[Expand] Valid action {action} received prior=0 from network."
            )

        next_state_copy = node.state.copy()
        try:
            # Correctly unpack the (reward, done) tuple returned by step
            _, done = next_state_copy.step(action)
        except Exception as e:
            logger.error(
                f"[Expand] Error stepping state for child node expansion (action {action}): {e}",
                exc_info=True,
            )
            continue  # Skip creating this child if stepping fails

        child = Node(
            state=next_state_copy,
            parent=node,
            action_taken=action,
            prior_probability=prior,
        )
        node.children[action] = child
        logger.debug(
            f"  [Expand] Created Child Node: Action={action}, Prior={prior:.4f}, StateStep={next_state_copy.current_step}, Done={done}"
        )
        children_created += 1

    logger.debug(f"[Expand] Expanded node {node} with {children_created} children.")


File: alphatriangle\mcts\strategy\policy.py
import logging
import random

import numpy as np

from ...utils.types import ActionType
from ..core.node import Node
from ..core.types import ActionPolicyMapping

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class PolicyGenerationError(Exception):
    """Custom exception for errors during policy generation or action selection."""

    pass


def select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType:
    """
    Selects an action from the root node based on visit counts and temperature.
    Raises PolicyGenerationError if selection is not possible.
    """
    if not root_node.children:
        raise PolicyGenerationError(
            f"Cannot select action: Root node (Step {root_node.state.current_step}) has no children."
        )

    actions = list(root_node.children.keys())
    visit_counts = np.array(
        [root_node.children[action].visit_count for action in actions],
        dtype=np.float64,
    )

    if len(actions) == 0:
        raise PolicyGenerationError(
            f"Cannot select action: No actions available in children for root node (Step {root_node.state.current_step})."
        )

    total_visits = np.sum(visit_counts)
    logger.debug(
        f"[PolicySelect] Selecting action for node step {root_node.state.current_step}. Total child visits: {total_visits}. Num children: {len(actions)}"
    )

    if total_visits == 0:
        logger.warning(
            f"[PolicySelect] Total visit count for children is zero at root node (Step {root_node.state.current_step}). MCTS might have failed. Selecting uniformly."
        )
        selected_action = random.choice(actions)
        logger.debug(
            f"[PolicySelect] Uniform random action selected: {selected_action}"
        )
        return selected_action

    if temperature == 0.0:
        max_visits = np.max(visit_counts)
        logger.debug(
            f"[PolicySelect] Greedy selection (temp=0). Max visits: {max_visits}"
        )
        best_action_indices = np.where(visit_counts == max_visits)[0]
        logger.debug(
            f"[PolicySelect] Greedy selection. Best action indices: {best_action_indices}"
        )
        # Use standard library random for tie-breaking
        chosen_index = random.choice(best_action_indices)
        selected_action = actions[chosen_index]
        logger.debug(f"[PolicySelect] Greedy action selected: {selected_action}")
        return selected_action

    else:
        logger.debug(f"[PolicySelect] Probabilistic selection: Temp={temperature:.4f}")
        logger.debug(f"  Visit Counts: {visit_counts}")
        log_visits = np.log(np.maximum(visit_counts, 1e-9))
        scaled_log_visits = log_visits / temperature
        scaled_log_visits -= np.max(scaled_log_visits)
        probabilities = np.exp(scaled_log_visits)
        sum_probs = np.sum(probabilities)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            raise PolicyGenerationError(
                f"Could not normalize visit probabilities (sum={sum_probs}). Visits: {visit_counts}"
            )
        else:
            probabilities /= sum_probs

        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PolicyGenerationError(
                f"Invalid probabilities generated after normalization: {probabilities}"
            )
        if abs(np.sum(probabilities) - 1.0) > 1e-5:
            logger.warning(
                f"[PolicySelect] Probabilities sum to {np.sum(probabilities):.6f} after normalization. Attempting re-normalization."
            )
            probabilities /= np.sum(probabilities)
            if abs(np.sum(probabilities) - 1.0) > 1e-5:
                raise PolicyGenerationError(
                    f"Probabilities still do not sum to 1 after re-normalization: {probabilities}, Sum: {np.sum(probabilities)}"
                )

        logger.debug(f"  Final Probabilities (normalized): {probabilities}")
        logger.debug(f"  Final Probabilities Sum: {np.sum(probabilities):.6f}")

        try:
            # Use NumPy's default_rng for weighted choice
            selected_action = rng.choice(actions, p=probabilities)
            logger.debug(
                f"[PolicySelect] Sampled action (temp={temperature:.2f}): {selected_action}"
            )
            # Ensure return type is ActionType (int)
            return int(selected_action)
        except ValueError as e:
            raise PolicyGenerationError(
                f"Error during np.random.choice: {e}. Probs: {probabilities}, Sum: {np.sum(probabilities)}"
            ) from e


def get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping:
    """
    Calculates the policy target distribution based on MCTS visit counts.
    Raises PolicyGenerationError if target cannot be generated.
    """
    action_dim = int(root_node.state.env_config.ACTION_DIM)  # type: ignore[call-overload]
    full_target = dict.fromkeys(range(action_dim), 0.0)

    if not root_node.children or root_node.visit_count == 0:
        logger.warning(
            f"[PolicyTarget] Cannot compute policy target: Root node (Step {root_node.state.current_step}) has no children or zero visits. Returning zero target."
        )
        return full_target

    child_visits = {
        action: child.visit_count for action, child in root_node.children.items()
    }
    actions = list(child_visits.keys())
    visits = np.array(list(child_visits.values()), dtype=np.float64)
    total_visits = np.sum(visits)

    if not actions:
        logger.warning(
            "[PolicyTarget] Cannot compute policy target: No actions found in children."
        )
        return full_target

    if temperature == 0.0:
        max_visits = np.max(visits)
        if max_visits == 0:
            logger.warning(
                "[PolicyTarget] Temperature is 0 but max visits is 0. Returning zero target."
            )
            return full_target

        best_actions = [actions[i] for i, v in enumerate(visits) if v == max_visits]
        prob = 1.0 / len(best_actions)
        for a in best_actions:
            if 0 <= a < action_dim:
                full_target[a] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Best action {a} is out of bounds ({action_dim}). Skipping."
                )

    else:
        visit_probs = visits ** (1.0 / temperature)
        sum_probs = np.sum(visit_probs)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            raise PolicyGenerationError(
                f"Could not normalize policy target probabilities (sum={sum_probs}). Visits: {visits}"
            )

        probabilities = visit_probs / sum_probs
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PolicyGenerationError(
                f"Invalid probabilities generated for policy target: {probabilities}"
            )
        if abs(np.sum(probabilities) - 1.0) > 1e-5:
            logger.warning(
                f"[PolicyTarget] Target probabilities sum to {np.sum(probabilities):.6f}. Re-normalizing."
            )
            probabilities /= np.sum(probabilities)
            if abs(np.sum(probabilities) - 1.0) > 1e-5:
                raise PolicyGenerationError(
                    f"Target probabilities still do not sum to 1 after re-normalization: {probabilities}, Sum: {np.sum(probabilities)}"
                )

        raw_policy = {action: probabilities[i] for i, action in enumerate(actions)}
        for action, prob in raw_policy.items():
            if 0 <= action < action_dim:
                full_target[action] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Action {action} from MCTS children is out of bounds ({action_dim}). Skipping."
                )

    final_sum = sum(full_target.values())
    if abs(final_sum - 1.0) > 1e-5 and total_visits > 0:
        logger.error(
            f"[PolicyTarget] Final policy target does not sum to 1 ({final_sum:.6f}). Target: {full_target}"
        )

    return full_target


File: alphatriangle\mcts\strategy\README.md
# File: alphatriangle/mcts/strategy/README.md
# MCTS Strategy Submodule (`alphatriangle.mcts.strategy`)

## Purpose and Architecture

This submodule implements the specific algorithms and heuristics used within the different phases of the Monte Carlo Tree Search, as orchestrated by [`alphatriangle.mcts.core.search.run_mcts_simulations`](../core/search.py).

-   **[`selection`](selection.py):** Contains the logic for traversing the tree from the root to a leaf node.
    -   `calculate_puct_score`: Implements the PUCT (Polynomial Upper Confidence Trees) formula, balancing exploitation (node value) and exploration (prior probability and visit counts).
    -   `add_dirichlet_noise`: Adds noise to the root node's prior probabilities to encourage exploration early in the search, as done in AlphaZero.
    -   `select_child_node`: Chooses the child with the highest PUCT score.
    -   `traverse_to_leaf`: Repeatedly applies `select_child_node` to navigate down the tree.
-   **[`expansion`](expansion.py):** Handles the expansion of a selected leaf node.
    -   `expand_node_with_policy`: Takes a node and a *pre-computed* policy dictionary (obtained from batched network evaluation) and creates child `Node` objects for all valid actions, initializing them with the corresponding prior probabilities.
-   **[`backpropagation`](backpropagation.py):** Implements the update step after a simulation.
    -   `backpropagate_value`: Traverses from the expanded leaf node back up to the root, incrementing the `visit_count` and adding the simulation's resulting `value` to the `total_action_value` of each node along the path.
-   **[`policy`](policy.py):** Provides functions related to action selection and policy target generation after MCTS has run.
    -   `select_action_based_on_visits`: Selects the final action to be played in the game based on the visit counts of the root's children, using a temperature parameter to control exploration vs. exploitation.
    -   `get_policy_target`: Generates the policy target vector (a probability distribution over actions) based on the visit counts, which is used as a training target for the neural network's policy head.

## Exposed Interfaces

-   **Selection:**
    -   `traverse_to_leaf(root_node: Node, config: MCTSConfig) -> Tuple[Node, int]`: Returns leaf node and depth.
    -   `add_dirichlet_noise(node: Node, config: MCTSConfig)`
    -   `select_child_node(node: Node, config: MCTSConfig) -> Node` (Primarily internal use)
    -   `calculate_puct_score(...) -> Tuple[float, float, float]` (Primarily internal use)
    -   `SelectionError`: Custom exception.
-   **Expansion:**
    -   `expand_node_with_policy(node: Node, action_policy: ActionPolicyMapping)`
-   **Backpropagation:**
    -   `backpropagate_value(leaf_node: Node, value: float) -> int`: Returns depth.
-   **Policy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.
    -   `PolicyGenerationError`: Custom exception.

## Dependencies

-   **[`alphatriangle.mcts.core`](../core/README.md)**:
    -   `Node`: The primary data structure operated upon.
    -   `ActionPolicyMapping`: Used in `expansion` and `policy`.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters (PUCT coeff, noise params, etc.).
-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`: Accessed via `Node.state` for methods like `is_over`, `get_outcome`, `valid_actions`, `step`.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`: Used for representing actions.
-   **`numpy`**:
    -   Used for Dirichlet noise and policy/selection calculations.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `random`.

---

**Note:** Please keep this README updated when modifying the algorithms within selection, expansion, backpropagation, or policy generation, or changing how they interact with the `Node` structure or `MCTSConfig`. Accurate documentation is crucial for maintainability.

File: alphatriangle\mcts\strategy\selection.py
import logging
import math

import numpy as np

from ...config import MCTSConfig
from ..core.node import Node

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class SelectionError(Exception):
    """Custom exception for errors during node selection."""

    pass


def calculate_puct_score(
    child_node: Node,
    parent_visit_count: int,
    config: MCTSConfig,
) -> tuple[float, float, float]:
    """Calculates the PUCT score and its components for a child node."""
    q_value = child_node.value_estimate
    prior = child_node.prior_probability
    child_visits = child_node.visit_count
    # Use parent_visit_count directly; sqrt comes later if needed (original AlphaGo used N(s), not sqrt(N(s)))
    # Let's use sqrt(parent_visit_count) for UCB1-like exploration bonus scaling
    parent_visits_sqrt = math.sqrt(max(1, parent_visit_count))

    exploration_term = (
        config.puct_coefficient * prior * (parent_visits_sqrt / (1 + child_visits))
    )
    score = q_value + exploration_term

    # Ensure score is finite, default to Q-value if exploration term explodes
    if not np.isfinite(score):
        logger.warning(
            f"Non-finite PUCT score calculated (Q={q_value}, P={prior}, ChildN={child_visits}, ParentN={parent_visit_count}, Exp={exploration_term}). Defaulting to Q-value."
        )
        score = q_value
        exploration_term = 0.0

    return score, q_value, exploration_term


def add_dirichlet_noise(node: Node, config: MCTSConfig):
    """Adds Dirichlet noise to the prior probabilities of the children of this node."""
    if (
        config.dirichlet_alpha <= 0.0
        or config.dirichlet_epsilon <= 0.0
        or not node.children
        or len(node.children) <= 1
    ):
        return

    actions = list(node.children.keys())
    # Use the module-level rng generator
    noise = rng.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon

    noisy_priors_sum = 0.0
    for i, action in enumerate(actions):
        child = node.children[action]
        original_prior = child.prior_probability
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]
        noisy_priors_sum += child.prior_probability
        logger.debug(
            f"  [Noise] Action {action}: OrigP={original_prior:.4f}, Noise={noise[i]:.4f} -> NewP={child.prior_probability:.4f}"
        )

    # Re-normalize priors after adding noise to ensure they sum to 1
    if abs(noisy_priors_sum - 1.0) > 1e-6:
        logger.debug(
            f"Re-normalizing priors after Dirichlet noise (Sum={noisy_priors_sum:.6f})"
        )
        for action in actions:
            if noisy_priors_sum > 1e-9:
                node.children[action].prior_probability /= noisy_priors_sum
            else:
                # Handle case where sum is zero - distribute equally? Or leave as 0?
                # Leaving as 0 might be safer if original priors were also 0.
                # Distributing equally might introduce unintended exploration.
                # Let's log a warning and leave them as potentially 0.
                logger.warning(
                    "Sum of priors after noise is near zero. Cannot normalize."
                )
                node.children[action].prior_probability = 0.0  # Or 1.0 / len(actions) ?

    logger.debug(
        f"[Noise] Added Dirichlet noise (alpha={config.dirichlet_alpha}, eps={eps}) to {len(actions)} root node priors."
    )


def select_child_node(node: Node, config: MCTSConfig) -> Node:
    """
    Selects the child node with the highest PUCT score. Assumes noise already added if root.
    Raises SelectionError if no valid child can be selected.
    Includes detailed logging of all child scores if DEBUG level is enabled.
    """
    if not node.children:
        raise SelectionError(f"Cannot select child from node {node} with no children.")

    best_score = -float("inf")
    best_child: Node | None = None
    child_scores_log = []

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"  [Select] Selecting child for Node (Visits={node.visit_count}, Children={len(node.children)}, StateStep={node.state.current_step}):"
        )

    # Use parent_visit_count from the node being considered for selection
    parent_visit_count = node.visit_count

    for action, child in node.children.items():
        # Pass the correct parent_visit_count for PUCT calculation
        score, q, exp_term = calculate_puct_score(child, parent_visit_count, config)

        if logger.isEnabledFor(logging.DEBUG):
            log_entry = (
                f"    Act={action}, Score={score:.4f} "
                f"(Q={q:.3f}, P={child.prior_probability:.4f}, N={child.visit_count}, Exp={exp_term:.4f})"
            )
            child_scores_log.append(log_entry)
            # Removed per-child log line here to reduce verbosity, summary below is sufficient

        if not np.isfinite(score):
            logger.warning(
                f"    [Select] Non-finite PUCT score ({score}) calculated for child action {action}. Skipping."
            )
            continue

        # Tie-breaking: add small random value? Or just take first max? Taking first max is simpler.
        if score > best_score:
            best_score = score
            best_child = child

    if logger.isEnabledFor(logging.DEBUG) and child_scores_log:
        try:

            def get_score_from_log(log_str):
                parts = log_str.split(",")
                for part in parts:
                    if "Score=" in part:
                        return float(part.split("=")[1].split(" ")[0])
                return -float("inf")

            child_scores_log.sort(key=get_score_from_log, reverse=True)
        except Exception as sort_err:
            logger.warning(f"Could not sort child score logs: {sort_err}")
        logger.debug("    [Select] All Child Scores Considered (Top 5):")
        for log_line in child_scores_log[:5]:  # Log only top 5 scores
            logger.debug(f"      {log_line}")

    if best_child is None:
        # Log available children details for debugging
        child_details = [
            f"Act={a}, N={c.visit_count}, P={c.prior_probability:.4f}, Q={c.value_estimate:.3f}"
            for a, c in node.children.items()
        ]
        logger.error(
            f"Could not select best child for node step {node.state.current_step}. Child details: {child_details}"
        )
        raise SelectionError(
            f"Could not select best child for node step {node.state.current_step}. Check scores and children."
        )

    logger.debug(
        f"  [Select] --> Selected Child: Action {best_child.action_taken}, Score {best_score:.4f}, Q-value {best_child.value_estimate:.3f}"
    )
    return best_child


def traverse_to_leaf(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """
    Traverses the tree from root to a leaf node using PUCT selection.
    A leaf is defined as a node that is not expanded OR is terminal.
    Stops also if the maximum search depth has been reached.
    Raises SelectionError if child selection fails during traversal.
    Returns the leaf node and the depth reached.
    """
    current_node = root_node
    depth = 0
    logger.debug(f"[Traverse] --- Start Traverse (Root Node: {root_node}) ---")
    stop_reason = "Unknown"

    while True:
        logger.debug(
            f"  [Traverse] Depth {depth}: Considering Node: {current_node} (Expanded={current_node.is_expanded}, Terminal={current_node.state.is_over()})"
        )

        if current_node.state.is_over():
            stop_reason = "Terminal State"
            logger.debug(  # Changed level from INFO to DEBUG
                f"  [Traverse] Depth {depth}: Node is TERMINAL. Stopping traverse."
            )
            break
        if not current_node.is_expanded:
            stop_reason = "Unexpanded Leaf"
            logger.debug(  # Changed level from INFO to DEBUG
                f"  [Traverse] Depth {depth}: Node is LEAF (not expanded). Stopping traverse."
            )
            break
        if config.max_search_depth is not None and depth >= config.max_search_depth:
            stop_reason = "Max Depth Reached"
            logger.debug(  # Changed level from INFO to DEBUG
                f"  [Traverse] Depth {depth}: Hit MAX DEPTH ({config.max_search_depth}). Stopping traverse."
            )
            break

        # Node is expanded, non-terminal, and below max depth - select child
        try:
            selected_child = select_child_node(current_node, config)
            logger.debug(
                f"  [Traverse] Depth {depth}: Selected child with action {selected_child.action_taken}"
            )
            current_node = selected_child
            depth += 1
        except SelectionError as e:
            stop_reason = f"Child Selection Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Error during child selection: {e}. Breaking traverse.",
                exc_info=False,  # Avoid full traceback for selection errors unless needed
            )
            # It's better to return the current node where selection failed than raise an exception
            # The MCTS search loop can then handle this (e.g., backpropagate current value)
            logger.warning(
                f"  [Traverse] Returning node {current_node} due to SelectionError."
            )
            break
        except Exception as e:
            stop_reason = f"Unexpected Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Unexpected error during child selection: {e}. Breaking traverse.",
                exc_info=True,
            )
            # Also return current node here instead of raising
            logger.warning(
                f"  [Traverse] Returning node {current_node} due to Unexpected Error."
            )
            break

    logger.debug(  # Changed level from INFO to DEBUG
        f"[Traverse] --- End Traverse: Reached Node at Depth {depth}. Reason: {stop_reason}. Final Node: {current_node} ---"
    )
    return current_node, depth


File: alphatriangle\mcts\strategy\__init__.py


File: alphatriangle\nn\model.py
# File: alphatriangle/nn/model.py
import math
from typing import cast

import torch
import torch.nn as nn

from ..config import EnvConfig, ModelConfig

# --- REMOVED: Incorrect self-import ---
# from .model import AlphaTriangleNet
# --- END REMOVED ---


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    use_batch_norm: bool,
    activation: type[nn.Module],
) -> nn.Sequential:
    """Creates a standard convolutional block."""
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_batch_norm,
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Standard Residual Block."""

    def __init__(
        self, channels: int, use_batch_norm: bool, activation: type[nn.Module]
    ):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1, use_batch_norm, activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    """Injects sinusoidal positional encoding. (Adapted from PyTorch tutorial)"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive for PositionalEncoding")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        # --- CHANGE: Simplified calculation based on tutorial ---
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model / 2]
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (if they exist)
        # Note: div_term is already the correct size for broadcasting with pe[:, 1::2]
        # because its length is ceil(d_model / 2). If d_model is odd,
        # the last element of div_term won't be used for the cos calculation anyway.
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Add the batch dimension (1) expected by register_buffer and forward pass
        # Shape becomes [max_len, 1, d_model]
        pe = pe.unsqueeze(1)
        # --- END CHANGE ---

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
                (Note: AlphaTriangleNet might pass [batch_size, embedding_dim, seq_len (H*W)])
                It needs to be permuted before applying positional encoding if that's the case.
                Here, we assume the input is already [seq_len, batch_size, embedding_dim].

        Returns:
            Tensor with added positional encoding.
        """
        pe_buffer = self.pe
        if not isinstance(pe_buffer, torch.Tensor):
            raise TypeError("PositionalEncoding buffer 'pe' is not a Tensor.")

        if x.shape[0] > pe_buffer.shape[0]:
            raise ValueError(
                f"Input sequence length {x.shape[0]} exceeds max_len {pe_buffer.shape[0]} of PositionalEncoding"
            )
        if x.shape[2] != pe_buffer.shape[2]:
            raise ValueError(
                f"Input embedding dimension {x.shape[2]} does not match PositionalEncoding dimension {pe_buffer.shape[2]}"
            )

        # Add positional encoding
        # Slicing pe_buffer[:x.size(0)] handles variable sequence lengths
        x = x + pe_buffer[: x.size(0)]
        return cast("torch.Tensor", self.dropout(x))


class AlphaTriangleNet(nn.Module):
    """
    Neural Network architecture for AlphaTriangle.
    Includes optional Transformer Encoder block after CNN body.
    Supports Distributional Value Head (C51).
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        # Cast ACTION_DIM to int
        self.action_dim = int(env_config.ACTION_DIM)  # type: ignore[call-overload]

        activation_cls: type[nn.Module] = getattr(nn, model_config.ACTIVATION_FUNCTION)

        # --- CNN Body ---
        conv_layers: list[nn.Module] = []
        in_channels = model_config.GRID_INPUT_CHANNELS
        for i, out_channels in enumerate(model_config.CONV_FILTERS):
            conv_layers.append(
                conv_block(
                    in_channels,
                    out_channels,
                    model_config.CONV_KERNEL_SIZES[i],
                    model_config.CONV_STRIDES[i],
                    model_config.CONV_PADDING[i],
                    model_config.USE_BATCH_NORM,
                    activation_cls,
                )
            )
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_layers)

        # --- Residual Body ---
        res_layers: list[nn.Module] = []
        if model_config.NUM_RESIDUAL_BLOCKS > 0:
            res_channels = model_config.RESIDUAL_BLOCK_FILTERS
            if in_channels != res_channels:
                # Add projection layer if channels don't match
                res_layers.append(
                    conv_block(
                        in_channels,
                        res_channels,
                        1,
                        1,
                        0,
                        model_config.USE_BATCH_NORM,
                        activation_cls,
                    )
                )
                in_channels = res_channels
            for _ in range(model_config.NUM_RESIDUAL_BLOCKS):
                res_layers.append(
                    ResidualBlock(
                        in_channels, model_config.USE_BATCH_NORM, activation_cls
                    )
                )
        self.res_body = nn.Sequential(*res_layers)
        self.cnn_output_channels = in_channels  # Channels after CNN/Res blocks

        # --- Transformer Body (Optional) ---
        self.transformer_body = None
        self.pos_encoder = None
        self.input_proj: nn.Module = nn.Identity()
        self.transformer_output_size = 0

        if model_config.USE_TRANSFORMER and model_config.TRANSFORMER_LAYERS > 0:
            transformer_input_dim = model_config.TRANSFORMER_DIM
            if self.cnn_output_channels != transformer_input_dim:
                self.input_proj = nn.Conv2d(
                    self.cnn_output_channels, transformer_input_dim, kernel_size=1
                )
            else:
                self.input_proj = nn.Identity()

            self.pos_encoder = PositionalEncoding(transformer_input_dim, dropout=0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_input_dim,
                nhead=model_config.TRANSFORMER_HEADS,
                dim_feedforward=model_config.TRANSFORMER_FC_DIM,
                activation=model_config.ACTIVATION_FUNCTION.lower(),
                batch_first=False,  # Expects (Seq, Batch, Dim)
                norm_first=True,
            )
            transformer_norm = nn.LayerNorm(transformer_input_dim)
            self.transformer_body = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config.TRANSFORMER_LAYERS,
                norm=transformer_norm,
            )

            # Calculate transformer output size using a dummy forward pass
            dummy_input_grid = torch.zeros(
                1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
            )
            with torch.no_grad():
                cnn_out = self.conv_body(dummy_input_grid)
                res_out = self.res_body(cnn_out)
                proj_out = self.input_proj(res_out)
                b, d, h, w = proj_out.shape
                # Size after flattening H*W dimensions
                self.transformer_output_size = h * w * d
        else:
            # Calculate flattened size after conv/res blocks if no transformer
            dummy_input_grid = torch.zeros(
                1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
            )
            with torch.no_grad():
                conv_output = self.conv_body(dummy_input_grid)
                res_output = self.res_body(conv_output)
                self.flattened_cnn_size = res_output.numel()

        # --- Shared Fully Connected Layers ---
        if model_config.USE_TRANSFORMER and model_config.TRANSFORMER_LAYERS > 0:
            combined_input_size = (
                self.transformer_output_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )
        else:
            combined_input_size = (
                self.flattened_cnn_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )

        shared_fc_layers: list[nn.Module] = []  # Explicitly type the list
        in_features = combined_input_size
        for hidden_dim in model_config.FC_DIMS_SHARED:
            shared_fc_layers.append(nn.Linear(in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                # Use BatchNorm1d for FC layers
                shared_fc_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_fc_layers.append(activation_cls())
            in_features = hidden_dim
        self.shared_fc = nn.Sequential(*shared_fc_layers)

        # --- Policy Head ---
        policy_head_layers: list[nn.Module] = []
        policy_in_features = in_features
        # Iterate through hidden dims if any
        for hidden_dim in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                policy_head_layers.append(nn.BatchNorm1d(hidden_dim))
            policy_head_layers.append(activation_cls())
            policy_in_features = hidden_dim
        # Final layer to output action dimension logits
        # Use self.action_dim which is already cast to int
        policy_head_layers.append(nn.Linear(policy_in_features, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)

        # --- Value Head (Distributional) --- CHANGED
        value_head_layers: list[nn.Module] = []
        value_in_features = in_features
        # Iterate through hidden dims if any
        for hidden_dim in model_config.VALUE_HEAD_DIMS:
            value_head_layers.append(nn.Linear(value_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                value_head_layers.append(nn.BatchNorm1d(hidden_dim))
            value_head_layers.append(activation_cls())
            value_in_features = hidden_dim
        # Final layer to output logits for each value atom
        value_head_layers.append(
            nn.Linear(value_in_features, model_config.NUM_VALUE_ATOMS)
        )
        # REMOVED: Tanh activation - we need logits for cross-entropy loss
        # value_head_layers.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_head_layers)
        # --- END CHANGED ---

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Returns: (policy_logits, value_distribution_logits)
        """
        conv_out = self.conv_body(grid_state)
        res_out = self.res_body(conv_out)

        # Optional Transformer Body
        if (
            self.model_config.USE_TRANSFORMER
            and self.transformer_body is not None
            and self.pos_encoder is not None
        ):
            proj_out = self.input_proj(res_out)  # Shape: (B, D, H, W)
            b, d, h, w = proj_out.shape
            # Reshape for transformer: (Seq, Batch, Dim) -> (H*W, B, D)
            transformer_input = proj_out.flatten(2).permute(2, 0, 1)
            # Add positional encoding
            transformer_input = self.pos_encoder(transformer_input)
            # Pass through transformer encoder
            transformer_output = self.transformer_body(
                transformer_input
            )  # Shape: (Seq, Batch, Dim)
            # Flatten transformer output: (Seq, Batch, Dim) -> (Batch, Seq*Dim)
            flattened_features = transformer_output.permute(1, 0, 2).flatten(1)
        else:
            # Flatten CNN output if no transformer
            flattened_features = res_out.view(res_out.size(0), -1)

        # Combine with other features
        combined_features = torch.cat([flattened_features, other_features], dim=1)

        # Shared FC Layers and Heads
        shared_out = self.shared_fc(combined_features)
        policy_logits = self.policy_head(shared_out)
        # --- CHANGED: Return value logits ---
        value_logits = self.value_head(shared_out)
        return policy_logits, value_logits
        # --- END CHANGED ---


File: alphatriangle\nn\network.py
# File: alphatriangle/nn/network.py
import logging
import sys  # Import sys for platform check
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..config import EnvConfig, ModelConfig, TrainConfig
from ..environment import GameState
from ..features import extract_state_features
from ..utils.types import ActionType, PolicyValueOutput, StateType
from .model import AlphaTriangleNet

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class NetworkEvaluationError(Exception):
    """Custom exception for errors during network evaluation."""

    pass


class NeuralNetwork:
    """
    Wrapper for the PyTorch model providing evaluation and state management.
    Handles distributional value head (C51) by calculating expected value for MCTS.
    Optionally compiles the model using torch.compile().
    """

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,
        train_config: TrainConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.env_config = env_config
        self.train_config = train_config
        self.device = device
        self.model = AlphaTriangleNet(model_config, env_config).to(device)
        self.action_dim = env_config.ACTION_DIM
        self.model.eval()

        self.num_atoms = model_config.NUM_VALUE_ATOMS
        self.v_min = model_config.VALUE_MIN
        self.v_max = model_config.VALUE_MAX
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_atoms, device=self.device
        )

        # --- ADDED: Check for Windows/MPS before attempting compile ---
        if self.train_config.COMPILE_MODEL:
            # --- ADDED: Skip compilation entirely on Windows due to Triton dependency ---
            if sys.platform == "win32":
                logger.warning(
                    "Model compilation requested but running on Windows. "
                    "Skipping torch.compile() as the default CUDA backend (Inductor) requires Triton, "
                    "which is not officially supported on Windows. Proceeding with eager execution."
                )
            # --- END ADDED ---
            elif self.device.type == "mps":
                logger.warning(
                    "Model compilation requested but device is 'mps'. "
                    "Skipping torch.compile() due to known compatibility issues with this backend. "
                    "Proceeding with eager execution."
                )
            elif hasattr(torch, "compile"):
                try:
                    logger.info(
                        f"Attempting to compile model with torch.compile() on device '{self.device}'..."
                    )
                    self.model = torch.compile(self.model)  # type: ignore
                    logger.info(
                        f"Model compiled successfully on device '{self.device}'."
                    )
                except Exception as e:
                    logger.warning(
                        f"torch.compile() failed on device '{self.device}': {e}. "
                        f"Proceeding without compilation (using eager mode). "
                        f"Compilation might not be supported for this model/backend combination.",
                        exc_info=False,
                    )
            else:
                logger.warning(
                    "torch.compile() requested but not available (requires PyTorch 2.0+). Proceeding without compilation."
                )
        else:
            logger.info(
                "Model compilation skipped (COMPILE_MODEL=False in TrainConfig)."
            )
        # --- END ADDED ---

    def _state_to_tensors(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from GameState and converts them to tensors."""
        state_dict: StateType = extract_state_features(state, self.model_config)
        grid_tensor = torch.from_numpy(state_dict["grid"]).unsqueeze(0).to(self.device)
        other_features_tensor = (
            torch.from_numpy(state_dict["other_features"]).unsqueeze(0).to(self.device)
        )
        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError(
                f"Non-finite values found in input grid_tensor for state {state}"
            )
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                f"Non-finite values found in input other_features_tensor for state {state}"
            )
        return grid_tensor, other_features_tensor

    def _batch_states_to_tensors(
        self, states: list[GameState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from a batch of GameStates and converts to batched tensors."""
        if not states:
            grid_shape = (
                0,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            )
            other_shape = (0, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
            return torch.empty(grid_shape, device=self.device), torch.empty(
                other_shape, device=self.device
            )

        batch_grid = []
        batch_other = []
        for state in states:
            state_dict: StateType = extract_state_features(state, self.model_config)
            batch_grid.append(state_dict["grid"])
            batch_other.append(state_dict["other_features"])

        grid_tensor = torch.from_numpy(np.stack(batch_grid)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(batch_other)).to(self.device)

        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values found in batched input grid_tensor"
            )
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values found in batched input other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    def _logits_to_expected_value(self, value_logits: torch.Tensor) -> torch.Tensor:
        """Calculates the expected value from the value distribution logits."""
        value_probs = F.softmax(value_logits, dim=1)
        # Expand support to match batch size for broadcasting
        support_expanded = self.support.expand_as(value_probs)
        expected_value = torch.sum(value_probs * support_expanded, dim=1, keepdim=True)
        return expected_value

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """
        Evaluates a single state.
        Returns policy mapping and EXPECTED value from the distribution.
        Raises NetworkEvaluationError on issues.
        """
        self.model.eval()
        try:
            grid_tensor, other_features_tensor = self._state_to_tensors(state)
            policy_logits, value_logits = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected for state {state}. Logits: {policy_logits}"
                )
            if not torch.all(torch.isfinite(value_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite value_logits detected for state {state}: {value_logits}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax for state {state}. Logits were: {policy_logits}"
                )

            policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
            policy_probs = np.maximum(policy_probs, 0)
            prob_sum = np.sum(policy_probs)
            if abs(prob_sum - 1.0) > 1e-5:
                logger.warning(
                    f"Evaluate: Policy probabilities sum to {prob_sum:.6f} (not 1.0) for state {state.current_step}. Re-normalizing."
                )
                if prob_sum <= 1e-9:
                    raise NetworkEvaluationError(
                        f"Policy probability sum is near zero ({prob_sum}) for state {state.current_step}. Cannot normalize."
                    )
                policy_probs /= prob_sum

            expected_value_tensor = self._logits_to_expected_value(value_logits)
            expected_value_scalar = expected_value_tensor.squeeze(
                0
            ).item()  # Squeeze batch and atom dim, get scalar

            action_policy: Mapping[ActionType, float] = {
                i: float(p) for i, p in enumerate(policy_probs)
            }

            num_non_zero = sum(1 for p in action_policy.values() if p > 1e-6)
            logger.debug(
                f"Evaluate Final Policy Dict (State {state.current_step}): {num_non_zero}/{self.action_dim} non-zero probs. Example: {list(action_policy.items())[:5]}"
            )

            return action_policy, expected_value_scalar

        except Exception as e:
            logger.error(
                f"Exception during single evaluation for state {state}: {e}",
                exc_info=True,
            )
            raise NetworkEvaluationError(
                f"Evaluation failed for state {state}: {e}"
            ) from e

    @torch.inference_mode()
    def evaluate_batch(self, states: list[GameState]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of states.
        Returns a list of (policy mapping, EXPECTED value).
        Raises NetworkEvaluationError on issues.
        """
        if not states:
            return []
        self.model.eval()
        logger.debug(f"Evaluating batch of {len(states)} states...")
        try:
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)
            policy_logits, value_logits = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected in batch evaluation. Logits shape: {policy_logits.shape}"
                )
            if not torch.all(torch.isfinite(value_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite value_logits detected in batch value output. Value shape: {value_logits.shape}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax in batch. Logits shape: {policy_logits.shape}"
                )

            policy_probs = policy_probs_tensor.cpu().numpy()
            expected_values_tensor = self._logits_to_expected_value(value_logits)
            expected_values = (
                expected_values_tensor.squeeze(1).cpu().numpy()
            )  # Squeeze the atom dim

            results: list[PolicyValueOutput] = []
            for batch_idx in range(len(states)):
                probs_i = np.maximum(policy_probs[batch_idx], 0)
                prob_sum_i = np.sum(probs_i)
                if abs(prob_sum_i - 1.0) > 1e-5:
                    logger.warning(
                        f"EvaluateBatch: Policy probabilities sum to {prob_sum_i:.6f} (not 1.0) for sample {batch_idx}. Re-normalizing."
                    )
                    if prob_sum_i <= 1e-9:
                        raise NetworkEvaluationError(
                            f"Policy probability sum is near zero ({prob_sum_i}) for batch sample {batch_idx}. Cannot normalize."
                        )
                    probs_i /= prob_sum_i

                policy_i: Mapping[ActionType, float] = {
                    i: float(p) for i, p in enumerate(probs_i)
                }
                value_i = float(expected_values[batch_idx])  # This is now a scalar
                results.append((policy_i, value_i))

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(f"Batch evaluation failed: {e}") from e

        logger.debug(f"  Batch evaluation finished. Returning {len(results)} results.")
        return results

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        # If model is compiled, access the original model for state_dict
        model_to_save = getattr(self.model, "_orig_mod", self.model)
        return {k: v.cpu() for k, v in model_to_save.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]):
        """Loads the model's state dictionary from the provided weights."""
        try:
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            # If model is compiled, load into the original model
            model_to_load = getattr(self.model, "_orig_mod", self.model)
            model_to_load.load_state_dict(weights_on_device)
            self.model.eval()  # Ensure the main (potentially compiled) model is in eval mode
            logger.debug("NN weights set successfully.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise


File: alphatriangle\nn\README.md
# File: alphatriangle/nn/README.md
# Neural Network Module (`alphatriangle.nn`)

## Purpose and Architecture

This module defines and manages the neural network used by the AlphaTriangle agent. It follows the AlphaZero paradigm, featuring a shared body and separate heads for policy and value prediction.

-   **Model Definition ([`model.py`](model.py)):**
    -   The `AlphaTriangleNet` class (inheriting from `torch.nn.Module`) defines the network architecture.
    -   It includes convolutional layers for processing the grid state, potentially residual blocks.
    -   **Optionally**, it can include a **Transformer Encoder block** after the CNN/ResNet body to apply self-attention over the spatial features before combining them with other input features. This is controlled by `ModelConfig.USE_TRANSFORMER`.
    -   The output from the CNN/Transformer body is combined with other extracted features (e.g., shape info) and passed through shared fully connected layers.
    -   It splits into two heads:
        -   **Policy Head:** Outputs logits representing the probability distribution over all possible actions.
        -   **Value Head:** Outputs logits representing a **distribution** over possible state values (C51 Distributional RL).
    -   The architecture is configurable via [`ModelConfig`](../config/model_config.py).
-   **Network Interface ([`network.py`](network.py)):**
    -   The `NeuralNetwork` class acts as a wrapper around the `AlphaTriangleNet` PyTorch model.
    -   It provides a clean interface for the rest of the system (MCTS, Trainer) to interact with the network, abstracting away PyTorch specifics.
    -   It **internally uses [`alphatriangle.features.extract_state_features`](../features/extractor.py)** to convert input `GameState` objects into tensors before feeding them to the underlying `AlphaTriangleNet` model.
    -   It handles the **distributional value head**, calculating the expected value from the predicted distribution for use by MCTS.
    -   It **optionally compiles** the underlying model using `torch.compile()` based on `TrainConfig.COMPILE_MODEL` for potential performance improvements.
    -   Key methods:
        -   `evaluate(state: GameState)`: Takes a `GameState`, extracts features, performs a forward pass, and returns the policy probabilities (as a dictionary) and the **expected scalar value estimate**. Conforms to the `ActionPolicyValueEvaluator` protocol required by MCTS.
        -   `evaluate_batch(states: List[GameState])`: Extracts features from a batch of `GameState` objects and performs batched evaluation for efficiency.
        -   `get_weights()`: Returns the model's state dictionary (on CPU).
        -   `set_weights(weights: Dict)`: Loads weights into the model (handles device placement).
    -   It handles device placement (`torch.device`).

## Exposed Interfaces

-   **Classes:**
    -   `AlphaTriangleNet(model_config: ModelConfig, env_config: EnvConfig)`: The PyTorch `nn.Module` defining the architecture.
    -   `NeuralNetwork(model_config: ModelConfig, env_config: EnvConfig, train_config: TrainConfig, device: torch.device)`: The wrapper class providing the primary interface.
        -   `evaluate(state: GameState) -> PolicyValueOutput`
        -   `evaluate_batch(states: List[GameState]) -> List[PolicyValueOutput]`
        -   `get_weights() -> Dict[str, torch.Tensor]`
        -   `set_weights(weights: Dict[str, torch.Tensor])`
        -   `model`: Public attribute to access the underlying `AlphaTriangleNet` instance.
        -   `device`: Public attribute indicating the `torch.device`.
        -   `model_config`: Public attribute.
        -   `num_atoms`, `v_min`, `v_max`, `delta_z`, `support`: Attributes related to the distributional value head.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**:
    -   `ModelConfig`: Defines the network architecture parameters (including expected feature dimensions and Transformer options).
    -   `EnvConfig`: Provides environment dimensions (grid size, action space size) needed by the model.
    -   `TrainConfig`: Used by `NeuralNetwork` init (e.g., for `COMPILE_MODEL`).
-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: Input type for `evaluate` and `evaluate_batch`.
-   **[`alphatriangle.features`](../features/README.md)**:
    -   `extract_state_features`: Used internally by `NeuralNetwork` to process `GameState` inputs.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   `ActionType`, `PolicyValueOutput`, `StateType`: Used in method signatures and return types.
-   **`torch`**:
    -   The core deep learning framework (`torch`, `torch.nn`, `torch.nn.functional`).
-   **`numpy`**:
    -   Used for converting state components to tensors.
-   **Standard Libraries:** `typing`, `os`, `logging`, `math`, `sys`.

---

**Note:** Please keep this README updated when changing the neural network architecture (`AlphaTriangleNet`, including Transformer usage or the distributional value head), the `NeuralNetwork` interface methods, or its interaction with configuration or other modules (especially `alphatriangle.features`). Accurate documentation is crucial for maintainability.

File: alphatriangle\nn\__init__.py
"""
Neural Network module for the AlphaTriangle agent.
Contains the model definition and a wrapper for inference and training interface.
"""

from .model import AlphaTriangleNet
from .network import NeuralNetwork

__all__ = [
    "AlphaTriangleNet",
    "NeuralNetwork",
]


File: alphatriangle\rl\README.md
# File: alphatriangle/rl/README.md
# Reinforcement Learning Module (`alphatriangle.rl`)

## Purpose and Architecture

This module contains core components related to the reinforcement learning algorithm itself, specifically the `Trainer` for network updates, the `ExperienceBuffer` for storing data, and the `SelfPlayWorker` actor for generating data. **The overall orchestration of the training process has been moved to the [`alphatriangle.training`](../training/README.md) module.**

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Trainer`: Responsible for performing the neural network update steps. It takes batches of experience from the buffer, calculates losses (policy cross-entropy, **distributional value cross-entropy**, optional entropy bonus), applies importance sampling weights if using PER, updates the network weights, and calculates TD errors for PER priority updates.
    -   `ExperienceBuffer`: A replay buffer storing `Experience` tuples (`(StateType, policy_target, n_step_return)`). Supports both uniform sampling and Prioritized Experience Replay (PER).
-   **Self-Play Components ([`self_play/README.md`](self_play/README.md)):**
    -   `worker`: Defines the `SelfPlayWorker` Ray actor. Each actor runs game episodes independently using MCTS and its local copy of the neural network. It collects experiences (including calculated n-step returns) and returns results via a `SelfPlayResult` object. It also logs stats and game state asynchronously.
-   **Types ([`types.py`](types.py)):**
    -   Defines Pydantic models like `SelfPlayResult` for structured data transfer between Ray actors and the training loop.

## Exposed Interfaces

-   **Core:**
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(per_sample: PERBatchSample) -> Optional[Tuple[Dict[str, float], np.ndarray]]`: Takes PER sample, returns loss info and TD errors.
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(experience: Experience)`
        -   `add_batch(experiences: List[Experience])`
        -   `sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[PERBatchSample]`: Samples batch, requires step for PER beta.
        -   `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray)`: Updates priorities for PER.
        -   `is_ready() -> bool`
        -   `__len__() -> int`
-   **Self-Play:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `run_episode() -> SelfPlayResult`
        -   `set_weights(weights: Dict)`
        -   `set_current_trainer_step(global_step: int)`
-   **Types:**
    -   `SelfPlayResult`: Pydantic model for self-play results.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `TrainConfig`, `EnvConfig`, `ModelConfig`, `MCTSConfig`.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.features`](../features/README.md)**: `extract_state_features`.
-   **[`alphatriangle.mcts`](../mcts/README.md)**: Core MCTS components.
-   **[`alphatriangle.environment`](../environment/README.md)**: `GameState`.
-   **[`alphatriangle.stats`](../stats/README.md)**: `StatsCollectorActor` (used indirectly via `alphatriangle.training`).
-   **[`alphatriangle.utils`](../utils/README.md)**: Types (`Experience`, `StateType`, `PERBatchSample`, `StepInfo`) and helpers (`SumTree`).
-   **[`alphatriangle.structs`](../structs/README.md)**: Implicitly used via `GameState`.
-   **`torch`**: Used by `Trainer` and `NeuralNetwork`.
-   **`ray`**: Used by `SelfPlayWorker`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `numpy`, `random`, `time`.

---

**Note:** Please keep this README updated when changing the responsibilities of the Trainer, Buffer, or SelfPlayWorker.

File: alphatriangle\rl\types.py
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Experience

logger = logging.getLogger(__name__)

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker."""

    model_config = arbitrary_types_config

    episode_experiences: list[Experience]
    final_score: float
    episode_steps: int

    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    avg_tree_depth: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_experience_structure(self) -> "SelfPlayResult":
        """Basic structural validation for experiences."""
        invalid_count = 0
        valid_experiences = []
        # Rename unused loop variable 'i' to '_i'
        for _i, exp in enumerate(self.episode_experiences):
            is_valid = False
            if isinstance(exp, tuple) and len(exp) == 3:
                state_type, policy_map, value = exp
                # Combine nested if statements
                if (
                    isinstance(state_type, dict)
                    and "grid" in state_type
                    and "other_features" in state_type
                    and isinstance(state_type["grid"], np.ndarray)
                    and isinstance(state_type["other_features"], np.ndarray)
                    and isinstance(policy_map, dict)
                    # Use isinstance with | for multiple types
                    and isinstance(value, float | int)
                    # Basic check for NaN/inf in features
                    and np.all(np.isfinite(state_type["grid"]))
                    and np.all(np.isfinite(state_type["other_features"]))
                ):
                    is_valid = True

            if is_valid:
                valid_experiences.append(exp)
            else:
                invalid_count += 1
                # Log only once per validation failure type if needed
                # logger.warning(f"SelfPlayResult validation: Invalid experience structure at index {i}: {type(exp)}")

        if invalid_count > 0:
            logger.warning(
                f"SelfPlayResult validation: Found {invalid_count} invalid experience structures. Keeping only valid ones."
            )
            # Note: Modifying self within validator is generally discouraged,
            # but here we filter invalid data before it propagates.
            # A cleaner approach might be a separate validation function called after creation.
            # However, for immediate use, this ensures the validated object has valid experiences.
            object.__setattr__(
                self, "episode_experiences", valid_experiences
            )  # Use object.__setattr__ to bypass Pydantic's immutability during validation

        return self


SelfPlayResult.model_rebuild(force=True)


File: alphatriangle\rl\__init__.py
"""
Reinforcement Learning (RL) module.
Contains the core components for training an agent using self-play and MCTS.
"""

from .core.buffer import ExperienceBuffer
from .core.trainer import Trainer
from .self_play.worker import SelfPlayWorker
from .types import SelfPlayResult

__all__ = [
    # Core components used by the training pipeline
    "Trainer",
    "ExperienceBuffer",
    # Self-play components
    "SelfPlayWorker",
    "SelfPlayResult",
]


File: alphatriangle\rl\core\buffer.py
import logging
import random
from collections import deque

import numpy as np

from ...config import TrainConfig
from ...utils.sumtree import SumTree
from ...utils.types import (
    Experience,
    ExperienceBatch,
    PERBatchSample,
)

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience Replay Buffer storing (StateType, PolicyTarget, Value).
    Supports both uniform sampling and Prioritized Experience Replay (PER)
    based on TrainConfig.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.capacity = config.BUFFER_CAPACITY
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN
        self.use_per = config.USE_PER

        if self.use_per:
            self.tree = SumTree(self.capacity)
            self.per_alpha = config.PER_ALPHA
            self.per_beta_initial = config.PER_BETA_INITIAL
            self.per_beta_final = config.PER_BETA_FINAL
            # Ensure anneal steps is at least 1 to avoid division by zero
            self.per_beta_anneal_steps = max(
                1, config.PER_BETA_ANNEAL_STEPS or config.MAX_TRAINING_STEPS or 1
            )
            self.per_epsilon = config.PER_EPSILON
            logger.info(
                f"Experience buffer initialized with PER (alpha={self.per_alpha}, beta_init={self.per_beta_initial}). Capacity: {self.capacity}"
            )
        else:
            self.buffer: deque[Experience] = deque(maxlen=self.capacity)
            logger.info(
                f"Experience buffer initialized with uniform sampling. Capacity: {self.capacity}"
            )

    def _get_priority(self, error: float) -> float:
        """Calculates priority from TD error."""
        # Ensure return type is float
        return float((np.abs(error) + self.per_epsilon) ** self.per_alpha)

    def add(self, experience: Experience):
        """Adds a single experience. Uses max priority if PER is enabled."""
        if self.use_per:
            max_p = self.tree.max_priority
            self.tree.add(max_p, experience)
        else:
            self.buffer.append(experience)

    def add_batch(self, experiences: list[Experience]):
        """Adds a batch of experiences. Uses max priority if PER is enabled."""
        if self.use_per:
            max_p = self.tree.max_priority
            for exp in experiences:
                self.tree.add(max_p, exp)
        else:
            self.buffer.extend(experiences)

    def _calculate_beta(self, current_step: int) -> float:
        """Linearly anneals beta from initial to final value."""
        fraction = min(1.0, current_step / self.per_beta_anneal_steps)
        beta = self.per_beta_initial + fraction * (
            self.per_beta_final - self.per_beta_initial
        )
        return beta

    def sample(
        self, batch_size: int, current_train_step: int | None = None
    ) -> PERBatchSample | None:
        """
        Samples a batch of experiences.
        Uses prioritized sampling if PER is enabled, otherwise uniform.
        Requires current_train_step if PER is enabled to calculate beta.
        """
        current_size = len(self)
        if current_size < batch_size or current_size < self.min_size_to_train:
            return None

        if self.use_per:
            if current_train_step is None:
                raise ValueError("current_train_step is required for PER sampling.")

            batch: ExperienceBatch = []
            idxs = np.empty((batch_size,), dtype=np.int32)
            is_weights = np.empty((batch_size,), dtype=np.float32)
            beta = self._calculate_beta(current_train_step)

            priority_segment = self.tree.total_priority / batch_size
            max_weight = 0.0

            for i in range(batch_size):
                a = priority_segment * i
                b = priority_segment * (i + 1)
                value = random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(value)

                if not isinstance(data, tuple):
                    logger.warning(
                        f"PER sampling encountered non-experience data at index {idx}. Resampling."
                    )
                    # Resample with a random value across the entire range
                    value = random.uniform(0, self.tree.total_priority)
                    idx, p, data = self.tree.get_leaf(value)
                    if not isinstance(data, tuple):
                        logger.error(f"PER resampling failed. Skipping sample {i}.")
                        # Fallback: sample a random valid index if possible
                        if self.tree.n_entries > 0:
                            rand_data_idx = random.randint(0, self.tree.n_entries - 1)
                            rand_tree_idx = rand_data_idx + self.capacity - 1
                            idx, p, data = self.tree.get_leaf(
                                self.tree.tree[rand_tree_idx]
                            )
                            if not isinstance(data, tuple):
                                continue  # Give up on this sample if fallback fails
                        else:
                            continue  # Cannot sample if tree is empty

                sampling_prob = p / self.tree.total_priority
                weight = (
                    (current_size * sampling_prob) ** (-beta)
                    if sampling_prob > 1e-9
                    else 0.0
                )
                is_weights[i] = weight
                max_weight = max(max_weight, weight)
                idxs[i] = idx
                batch.append(data)

            if max_weight > 1e-9:
                is_weights /= max_weight
            else:
                logger.warning(
                    "Max importance sampling weight is near zero. Weights might be invalid."
                )
                is_weights.fill(1.0)

            return {"batch": batch, "indices": idxs, "weights": is_weights}

        else:
            uniform_batch = random.sample(self.buffer, batch_size)
            dummy_indices = np.zeros(batch_size, dtype=np.int32)
            uniform_weights = np.ones(batch_size, dtype=np.float32)
            return {
                "batch": uniform_batch,
                "indices": dummy_indices,
                "weights": uniform_weights,
            }

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Updates the priorities of sampled experiences based on TD errors."""
        if not self.use_per:
            return

        if len(tree_indices) != len(td_errors):
            logger.error(
                f"Mismatch between tree_indices ({len(tree_indices)}) and td_errors ({len(td_errors)}) lengths."
            )
            return

        # Calculate priorities for each error
        priorities = np.array([self._get_priority(err) for err in td_errors])

        if not np.all(np.isfinite(priorities)):
            logger.warning("Non-finite priorities calculated. Clamping.")
            priorities = np.nan_to_num(
                priorities,
                nan=self.per_epsilon,
                posinf=self.tree.max_priority,
                neginf=self.per_epsilon,
            )
            priorities = np.maximum(priorities, self.per_epsilon)

        # Use strict=False for zip, although lengths should match after check above
        for idx, p in zip(tree_indices, priorities, strict=False):
            if not (0 <= idx < len(self.tree.tree)):
                logger.error(f"Invalid tree index {idx} provided for priority update.")
                continue
            self.tree.update(idx, p)

        # Update the overall max priority tracked by the tree
        if len(priorities) > 0:
            self.tree._max_priority = max(self.tree.max_priority, np.max(priorities))

    def __len__(self) -> int:
        """Returns the current number of experiences in the buffer."""
        return self.tree.n_entries if self.use_per else len(self.buffer)

    def is_ready(self) -> bool:
        """Checks if the buffer has enough samples to start training."""
        return len(self) >= self.min_size_to_train


File: alphatriangle\rl\core\README.md
# File: alphatriangle/rl/core/README.md
# RL Core Submodule (`alphatriangle.rl.core`)

## Purpose and Architecture

This submodule contains core classes directly involved in the reinforcement learning update process and data storage. **The orchestration logic previously found here (`TrainingOrchestrator`) has been moved to the [`alphatriangle.training`](../../training/README.md) module.**

-   **[`Trainer`](trainer.py):** This class encapsulates the logic for updating the neural network's weights.
    -   It holds the main `NeuralNetwork` interface, optimizer, and scheduler.
    -   Its `train_step` method takes a batch of experiences (potentially with PER indices and weights), performs forward/backward passes, calculates losses (policy cross-entropy, **distributional value cross-entropy**, optional entropy bonus), applies importance sampling weights if using PER, updates weights, and returns calculated TD errors for PER priority updates.
-   **[`ExperienceBuffer`](buffer.py):** This class implements a replay buffer storing `Experience` tuples (`(StateType, policy_target, n_step_return)`). It supports Prioritized Experience Replay (PER) via a SumTree, including prioritized sampling and priority updates, based on configuration.

## Exposed Interfaces

-   **Classes:**
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(per_sample: PERBatchSample) -> Optional[Tuple[Dict[str, float], np.ndarray]]`: Takes PER sample, returns loss info and TD errors.
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(experience: Experience)`
        -   `add_batch(experiences: List[Experience])`
        -   `sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[PERBatchSample]`: Samples batch, requires step for PER beta.
        -   `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray)`: Updates priorities for PER.
        -   `is_ready() -> bool`
        -   `__len__() -> int`

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**: `TrainConfig`, `EnvConfig`, `ModelConfig`.
-   **[`alphatriangle.nn`](../../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.utils`](../../utils/README.md)**: Types (`Experience`, `PERBatchSample`, `StateType`, etc.) and helpers (`SumTree`).
-   **`torch`**: Used heavily by `Trainer`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `numpy`, `random`.

---

**Note:** Please keep this README updated when changing the responsibilities or interfaces of the Trainer or Buffer.

File: alphatriangle\rl\core\trainer.py
# File: alphatriangle/rl/core/trainer.py
import logging
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from ...config import EnvConfig, TrainConfig
from ...nn import NeuralNetwork
from ...utils.types import (
    ExperienceBatch,
    PERBatchSample,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the neural network training process, including loss calculation
    and optimizer steps. Supports Distributional RL (C51) value loss.
    """

    def __init__(
        self,
        nn_interface: NeuralNetwork,
        train_config: TrainConfig,
        env_config: EnvConfig,
    ):
        self.nn = nn_interface
        self.model = nn_interface.model
        self.train_config = train_config
        self.env_config = env_config
        self.model_config = nn_interface.model_config
        self.device = nn_interface.device
        self.optimizer = self._create_optimizer()
        self.scheduler: _LRScheduler | None = self._create_scheduler(self.optimizer)

        # --- ADDED: Distributional Value Attributes (from NN interface) ---
        self.num_atoms = self.nn.num_atoms
        self.v_min = self.nn.v_min
        self.v_max = self.nn.v_max
        self.delta_z = self.nn.delta_z
        self.support = self.nn.support.to(self.device)  # Ensure support is on device
        # --- END ADDED ---

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer based on TrainConfig."""
        lr = self.train_config.LEARNING_RATE
        wd = self.train_config.WEIGHT_DECAY
        params = self.model.parameters()
        opt_type = self.train_config.OPTIMIZER_TYPE.lower()
        logger.info(f"Creating optimizer: {opt_type}, LR: {lr}, WD: {wd}")
        if opt_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}"
            )

    def _create_scheduler(self, optimizer: optim.Optimizer) -> _LRScheduler | None:
        """Creates the learning rate scheduler based on TrainConfig."""
        scheduler_type_config = self.train_config.LR_SCHEDULER_TYPE
        scheduler_type: str | None = None
        if scheduler_type_config:
            scheduler_type = scheduler_type_config.lower()

        if not scheduler_type or scheduler_type == "none":
            logger.info("No LR scheduler configured.")
            return None

        logger.info(f"Creating LR scheduler: {scheduler_type}")
        if scheduler_type == "steplr":
            step_size = getattr(self.train_config, "LR_SCHEDULER_STEP_SIZE", 100000)
            gamma = getattr(self.train_config, "LR_SCHEDULER_GAMMA", 0.1)
            logger.info(f"  StepLR params: step_size={step_size}, gamma={gamma}")
            # Cast return type
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            )
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            if t_max is None:
                logger.warning(
                    "LR_SCHEDULER_T_MAX is None for CosineAnnealingLR. Scheduler might not work as expected."
                )
                t_max = self.train_config.MAX_TRAINING_STEPS or 1_000_000
            logger.info(f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}")
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                ),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type_config}")

    def _prepare_batch(
        self, batch: ExperienceBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a batch of experiences into tensors.
        The 4th tensor is now the n-step return G (scalar).
        """
        batch_size = len(batch)
        grids = []
        other_features = []
        # --- Store n-step returns ---
        n_step_returns = []
        action_dim_int = int(self.env_config.ACTION_DIM)  # type: ignore[call-overload]
        policy_target_tensor = torch.zeros(
            (batch_size, action_dim_int),
            dtype=torch.float32,
            device=self.device,
        )

        # --- Unpack n_step_return ---
        for i, (state_features, policy_target_map, n_step_return) in enumerate(batch):
            grids.append(state_features["grid"])
            other_features.append(state_features["other_features"])
            n_step_returns.append(n_step_return)  # Store the scalar return G
            for action, prob in policy_target_map.items():
                if 0 <= action < action_dim_int:
                    policy_target_tensor[i, action] = prob
                else:
                    logger.warning(
                        f"Action {action} out of bounds in policy target map for sample {i}."
                    )

        grid_tensor = torch.from_numpy(np.stack(grids)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(other_features)).to(
            self.device
        )
        # --- Create tensor for n-step returns ---
        n_step_return_tensor = torch.tensor(
            n_step_returns, dtype=torch.float32, device=self.device
        )

        expected_other_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if batch_size > 0 and other_features_tensor.shape[1] != expected_other_dim:
            raise ValueError(
                f"Unexpected other_features tensor shape: {other_features_tensor.shape}, expected dim {expected_other_dim}"
            )

        # --- Return n_step_return_tensor ---
        return (
            grid_tensor,
            other_features_tensor,
            policy_target_tensor,
            n_step_return_tensor,
        )

    # --- REWRITTEN: Helper for calculating target distribution ---
    def _calculate_target_distribution(
        self, n_step_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Projects the n-step returns onto the fixed support atoms (z).
        Args:
            n_step_returns: Tensor of shape (batch_size,) containing scalar n-step returns (G).
        Returns:
            Tensor of shape (batch_size, num_atoms) representing the target distribution.
        """
        batch_size = n_step_returns.size(0)
        # Initialize target distribution tensor
        m = torch.zeros(
            (batch_size, self.num_atoms), dtype=torch.float32, device=self.device
        )

        # Clamp returns to the support range [V_min, V_max]
        target_returns = n_step_returns.clamp(self.v_min, self.v_max)

        # Calculate the fractional index b and lower/upper atom indices l, u
        b = (target_returns - self.v_min) / self.delta_z
        # --- CHANGED: Rename l to lower_idx ---
        lower_idx = b.floor().long()
        # --- END CHANGED ---
        u = b.ceil().long()

        # Handle cases where b is an integer (l == u)
        # Ensure indices stay within bounds [0, num_atoms - 1]
        # --- CHANGED: Use lower_idx ---
        lower_idx = torch.max(torch.tensor(0, device=self.device), lower_idx)
        # --- END CHANGED ---
        u = torch.min(torch.tensor(self.num_atoms - 1, device=self.device), u)
        # If l==u after clamping, it means the target hit an atom exactly.
        # We can assign full probability to that atom.
        # However, the logic below handles this implicitly.

        # Calculate probabilities for lower and upper atoms based on distance
        # --- CHANGED: Use lower_idx ---
        m_l = u.float() - b  # Weight for lower atom
        m_u = b - lower_idx.float()  # Weight for upper atom
        # --- END CHANGED ---

        # Distribute probability mass using direct indexing
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=self.device)

        # Add probabilities to the lower atoms
        # --- CHANGED: Use lower_idx ---
        m[batch_indices, lower_idx] += m_l
        # --- END CHANGED ---
        # Add probabilities to the upper atoms
        m[batch_indices, u] += m_u

        return m

    # --- END REWRITTEN ---

    def train_step(
        self, per_sample: PERBatchSample
    ) -> tuple[dict[str, float], np.ndarray] | None:
        """
        Performs a single training step on the given batch from PER buffer.
        Uses distributional cross-entropy loss for the value head.
        Returns loss info dictionary and TD errors for priority updates.
        """
        batch = per_sample["batch"]
        is_weights = per_sample["weights"]

        if not batch:
            logger.warning("train_step called with empty batch.")
            return None

        self.model.train()
        try:
            # --- Get n_step_return_t ---
            grid_t, other_t, policy_target_t, n_step_return_t = self._prepare_batch(
                batch
            )
            is_weights_t = torch.from_numpy(is_weights).to(self.device)
        except Exception as e:
            logger.error(f"Error preparing batch for training: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        # --- Get value_logits ---
        policy_logits, value_logits = self.model(grid_t, other_t)

        # --- Value Loss (Distributional Cross-Entropy) ---
        # Calculate target distribution
        target_distribution = self._calculate_target_distribution(n_step_return_t)
        # Calculate cross-entropy loss
        # F.cross_entropy expects logits (N, C) and targets (N,) with class indices
        # OR targets (N, C) with probabilities if soft labels are used.
        # We have target probabilities, so use KLDivLoss or manual cross-entropy.
        # Manual Cross-Entropy: - sum(target_prob * log_softmax(pred_logits))
        log_pred_dist = F.log_softmax(value_logits, dim=1)
        value_loss_elementwise = -torch.sum(target_distribution * log_pred_dist, dim=1)
        # Apply importance sampling weights
        value_loss = (value_loss_elementwise * is_weights_t).mean()

        # --- Policy Loss (Cross-Entropy) --- (No change needed here)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_target_t = torch.nan_to_num(policy_target_t, nan=0.0)
        policy_loss_elementwise = -torch.sum(policy_target_t * log_probs, dim=1)
        policy_loss = (policy_loss_elementwise * is_weights_t).mean()

        # --- Entropy Bonus --- (No change needed here)
        entropy_scalar: float = 0.0  # Initialize as float
        entropy_loss_term = torch.tensor(
            0.0, device=self.device
        )  # Initialize as tensor
        if self.train_config.ENTROPY_BONUS_WEIGHT > 0:
            policy_probs = F.softmax(policy_logits, dim=1)
            # Calculate entropy term: -Sum(p * log(p))
            entropy_term_elementwise: torch.Tensor = -torch.sum(
                policy_probs * torch.log(policy_probs + 1e-9), dim=1
            )
            # Calculate mean entropy across batch for logging
            entropy_scalar = float(
                entropy_term_elementwise.mean().item()
            )  # Cast result to float
            # Calculate the loss term (negative entropy bonus)
            entropy_loss_term = (
                -self.train_config.ENTROPY_BONUS_WEIGHT
                * entropy_term_elementwise.mean()
            )

        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + entropy_loss_term  # Use the calculated term
        )

        total_loss.backward()

        if (
            self.train_config.GRADIENT_CLIP_VALUE is not None
            and self.train_config.GRADIENT_CLIP_VALUE > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # --- TD Error Calculation for PER ---
        # Use the difference between the n-step return G and the expected value E[V(s)]
        with torch.no_grad():
            expected_value_pred = self.nn._logits_to_expected_value(value_logits)
        # Ensure n_step_return_t has shape (batch_size,)
        td_errors = (
            (n_step_return_t - expected_value_pred.squeeze(1)).detach().cpu().numpy()
        )

        loss_info = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_scalar,
            "mean_td_error": float(np.mean(np.abs(td_errors))),
        }

        return loss_info, td_errors

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer."""
        try:
            # Ensure return type is float
            return float(self.optimizer.param_groups[0]["lr"])
        except (IndexError, KeyError):
            logger.warning("Could not retrieve learning rate from optimizer.")
            return 0.0


File: alphatriangle\rl\core\visual_state_actor.py
# File: alphatriangle/rl/core/visual_state_actor.py
import logging
import time
from typing import Any

import ray

from ...environment import GameState

logger = logging.getLogger(__name__)


@ray.remote
class VisualStateActor:
    """A simple Ray actor to hold the latest game states from workers for visualization."""

    def __init__(self) -> None:
        self.worker_states: dict[int, GameState] = {}
        self.global_stats: dict[str, Any] = {}
        self.last_update_times: dict[int, float] = {}

    def update_state(self, worker_id: int, game_state: GameState):
        """Workers call this to update their latest state."""
        self.worker_states[worker_id] = game_state
        self.last_update_times[worker_id] = time.time()

    def update_global_stats(self, stats: dict[str, Any]):
        """Orchestrator calls this to update global stats."""
        # Ensure stats is a dictionary
        if isinstance(stats, dict):
            # Use update to merge instead of direct assignment
            self.global_stats.update(stats)
        else:
            # Handle error or log warning if stats is not a dict
            logger.error(
                f"VisualStateActor received non-dict type for global stats: {type(stats)}"
            )
            # Don't reset, just ignore the update
            # self.global_stats = {}

    def get_all_states(self) -> dict[int, Any]:
        """
        Called by the orchestrator to get states for the visual queue.
        Key -1 holds the global_stats dictionary.
        Other keys hold GameState objects.
        """
        # Use dict() constructor instead of comprehension for ruff C416
        # Cast worker_states to dict[int, Any] before combining
        combined_states: dict[int, Any] = dict(self.worker_states)
        combined_states[-1] = self.global_stats.copy()
        return combined_states

    def get_state(self, worker_id: int) -> GameState | None:
        """Get state for a specific worker (unused currently)."""
        return self.worker_states.get(worker_id)


File: alphatriangle\rl\core\__init__.py
"""
Core RL components: Trainer, Buffer.
The Orchestrator logic has been moved to the alphatriangle.training module.
"""

from .buffer import ExperienceBuffer
from .trainer import Trainer

__all__ = [
    "Trainer",
    "ExperienceBuffer",
]


File: alphatriangle\rl\self_play\README.md
# File: alphatriangle/rl/self_play/README.md
# RL Self-Play Submodule (`alphatriangle.rl.self_play`)

## Purpose and Architecture

This submodule focuses specifically on generating game episodes through self-play, driven by the current neural network and MCTS. It is designed to run in parallel using Ray actors managed by the [`alphatriangle.training.worker_manager`](../../training/worker_manager.py).

-   **[`worker.py`](worker.py):** Defines the `SelfPlayWorker` class, decorated with `@ray.remote`.
    -   Each `SelfPlayWorker` actor runs independently, typically on a separate CPU core.
    -   It initializes its own `GameState` environment and `NeuralNetwork` instance (usually on the CPU).
    -   It receives configuration objects (`EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`) during initialization.
    -   It has a `set_weights` method allowing the `TrainingLoop` to periodically update its local neural network with the latest trained weights from the central model. **It also has `set_current_trainer_step` to store the global step associated with the current weights, called by the `WorkerManager`.**
    -   Its main method, `run_episode`, simulates a complete game episode:
        -   Uses its local `NeuralNetwork` evaluator and `MCTSConfig` to run MCTS ([`alphatriangle.mcts.run_mcts_simulations`](../../mcts/core/search.py)), **reusing the search tree between moves**.
        -   Selects actions based on MCTS results ([`alphatriangle.mcts.strategy.policy.select_action_based_on_visits`](../../mcts/strategy/policy.py)).
        -   Generates policy targets ([`alphatriangle.mcts.strategy.policy.get_policy_target`](../../mcts/strategy/policy.py)).
        -   Stores `(StateType, policy_target, n_step_return)` tuples (using extracted features and calculated n-step returns).
        -   Steps its local game environment (`GameState.step`).
        -   Returns the collected `Experience` list, final score, episode length, and MCTS statistics via a `SelfPlayResult` object.
        -   **Asynchronously logs per-step statistics (score, reward, MCTS visits/depth) to the `StatsCollectorActor`, providing a `StepInfo` dictionary containing the `game_step_index` and the `current_trainer_step` (global step of its current network weights).**
        -   **Asynchronously reports its current `GameState` to the `StatsCollectorActor` for visualization.**

## Exposed Interfaces

-   **Classes:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `__init__(...)`
        -   `run_episode() -> SelfPlayResult`: Runs one episode and returns results.
        -   `set_weights(weights: Dict)`: Updates the actor's local network weights.
        -   `set_current_trainer_step(global_step: int)`: Updates the stored trainer step.
-   **Types:**
    -   `SelfPlayResult`: Pydantic model defined in [`alphatriangle.rl.types`](../types.py).

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **[`alphatriangle.nn`](../../nn/README.md)**:
    -   `NeuralNetwork`: Instantiated locally within the actor.
-   **[`alphatriangle.mcts`](../../mcts/README.md)**:
    -   Core MCTS functions and types. **MCTS uses batched evaluation.**
-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`, `EnvConfig`: Used to instantiate and step through the game simulation locally.
-   **[`alphatriangle.features`](../../features/README.md)**:
    -   `extract_state_features`: Used to generate `StateType` for experiences.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `types`: `Experience`, `ActionType`, `PolicyTargetMapping`, `StateType`, `StepInfo`.
    -   `helpers`: `get_device`, `set_random_seeds`.
-   **[`alphatriangle.rl.types`](../types.py)**:
    -   `SelfPlayResult`: Return type.
-   **[`alphatriangle.stats`](../../stats/README.md)**:
    -   `StatsCollectorActor`: Handle passed for logging.
-   **`numpy`**:
    -   Used by MCTS strategies.
-   **`ray`**:
    -   The `@ray.remote` decorator makes this a Ray actor.
-   **`torch`**:
    -   Used by the local `NeuralNetwork`.
-   **Standard Libraries:** `typing`, `logging`, `random`, `time`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the self-play episode generation logic, the data collected, the interaction with MCTS/environment, or the asynchronous logging behavior, especially regarding the inclusion of `current_trainer_step` in `StepInfo`. Accurate documentation is crucial for maintainability.

File: alphatriangle\rl\self_play\worker.py
# File: alphatriangle/rl/self_play/worker.py
import logging
import random
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import ray
import torch  # Import torch

from ...config import MCTSConfig, ModelConfig, TrainConfig
from ...environment import EnvConfig, GameState
from ...features import extract_state_features
from ...mcts import (
    MCTSExecutionError,
    Node,
    get_policy_target,
    run_mcts_simulations,
    select_action_based_on_visits,
)
from ...nn import NeuralNetwork
from ...utils import get_device, set_random_seeds

# --- REMOVED: Type imports moved below ---
# from ...utils.types import Experience, PolicyTargetMapping, StateType, StepInfo
# --- END REMOVED ---

if TYPE_CHECKING:
    from ...stats import StatsCollectorActor

    # --- ADDED: Type imports moved here ---
    from ...utils.types import Experience, PolicyTargetMapping, StateType, StepInfo

    # --- END ADDED ---


from ..types import SelfPlayResult

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running self-play episodes using MCTS and a NN.
    Implements MCTS tree reuse between steps.
    Stores extracted features (StateType) and the N-STEP RETURN in the experience buffer.
    Returns a SelfPlayResult Pydantic model including aggregated stats.
    Reports current state and step stats asynchronously using StepInfo including game_step and trainer_step.
    """

    def __init__(
        self,
        actor_id: int,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        stats_collector_actor: "StatsCollectorActor",
        initial_weights: dict | None = None,
        seed: int | None = None,
        worker_device_str: str = "cpu",
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.stats_collector_actor = stats_collector_actor
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str

        # --- N-Step Config ---
        self.n_step = self.train_config.N_STEP_RETURNS
        self.gamma = self.train_config.GAMMA

        # Store the global step of the current weights
        self.current_trainer_step = 0

        # Configure logging for the worker process
        worker_log_level = logging.INFO
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        global logger
        logger = logging.getLogger(__name__)

        mcts_log_level = logging.WARNING
        nn_log_level = logging.WARNING
        logging.getLogger("alphatriangle.mcts").setLevel(mcts_log_level)
        logging.getLogger("alphatriangle.nn").setLevel(nn_log_level)

        set_random_seeds(self.seed)

        self.device = get_device(self.worker_device_str)

        if self.device.type == "cuda":
            try:
                torch.cuda.set_device(self.device)
                logger.info(
                    f"Successfully set default CUDA device for worker {self.actor_id} to {self.device} (Index: {torch.cuda.current_device()})."
                )
                count = torch.cuda.device_count()
                if count != 1:
                    logger.warning(
                        f"Worker {self.actor_id} sees {count} CUDA devices, expected 1 after Ray assignment. This might indicate an issue."
                    )
                else:
                    logger.info(
                        f"Worker {self.actor_id} sees 1 CUDA device as expected."
                    )

            except Exception as cuda_set_err:
                logger.error(
                    f"Failed to set default CUDA device for worker {self.actor_id} to {self.device}: {cuda_set_err}. "
                    f"Compilation or CUDA operations might fail.",
                    exc_info=True,
                )

        self.nn_evaluator = NeuralNetwork(
            model_config=self.model_config,
            env_config=self.env_config,
            train_config=self.train_config,
            device=self.device,
        )

        if initial_weights:
            self.set_weights(initial_weights)
        else:
            self.nn_evaluator.model.eval()

        logger.debug(f"INIT: MCTS Config: {self.mcts_config.model_dump()}")
        logger.info(
            f"Worker initialized on device {self.device}. Seed: {self.seed}. LogLevel: {logging.getLevelName(logger.getEffectiveLevel())}"
        )
        logger.debug("Worker init complete.")

    def set_weights(self, weights: dict):
        """Updates the neural network weights."""
        try:
            # Removed attempt to get step from weights dict
            self.nn_evaluator.set_weights(weights)
            logger.debug("Weights updated.")
        except Exception as e:
            logger.error(f"Failed to set weights: {e}", exc_info=True)

    def set_current_trainer_step(self, global_step: int):
        """Sets the global step corresponding to the current network weights."""
        self.current_trainer_step = global_step
        logger.debug(f"Worker {self.actor_id} trainer step set to {global_step}")

    def _report_current_state(self, game_state: GameState):
        """Asynchronously sends the current game state to the collector."""
        if self.stats_collector_actor:
            try:
                state_copy = game_state.copy()
                self.stats_collector_actor.update_worker_game_state.remote(  # type: ignore
                    self.actor_id, state_copy
                )
                logger.debug(
                    f"Reported state step {state_copy.current_step} to collector."
                )
            except Exception as e:
                logger.error(f"Failed to report game state to collector: {e}")

    def _log_step_stats_async(
        self,
        game_state: GameState,
        mcts_visits: int,
        mcts_depth: int,
        step_reward: float,
    ):
        """
        Asynchronously logs per-step stats to the collector using StepInfo,
        including the current game_step_index and the stored current_trainer_step.
        """
        if self.stats_collector_actor:
            try:
                # Include current_trainer_step
                step_info: StepInfo = {
                    "game_step_index": game_state.current_step,
                    "global_step": self.current_trainer_step,  # Add trainer step context
                }
                step_stats: dict[str, tuple[float, StepInfo]] = {
                    "RL/Current_Score": (game_state.game_score, step_info),
                    "MCTS/Step_Visits": (float(mcts_visits), step_info),
                    "MCTS/Step_Depth": (float(mcts_depth), step_info),
                    "RL/Step_Reward": (step_reward, step_info),
                }
                logger.debug(f"Sending step stats to collector: {step_stats}")
                self.stats_collector_actor.log_batch.remote(step_stats)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to log step stats to collector: {e}")

    def run_episode(self) -> SelfPlayResult:
        """
        Runs a single episode of self-play using MCTS and the internal neural network.
        Implements MCTS tree reuse.
        Stores extracted features (StateType) and the N-STEP RETURN in the experience buffer.
        Returns a SelfPlayResult Pydantic model including aggregated stats.
        Reports current state and step stats asynchronously.
        """
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)

        if game.is_over():
            logger.error(
                f"Game is over immediately after reset with seed {episode_seed}. Returning empty result."
            )
            return SelfPlayResult(
                episode_experiences=[],
                final_score=0.0,
                episode_steps=0,
                total_simulations=0,
                avg_root_visits=0.0,
                avg_tree_depth=0.0,
            )

        n_step_state_policy_buffer: deque[tuple[StateType, PolicyTargetMapping]] = (
            deque(maxlen=self.n_step)
        )
        n_step_reward_buffer: deque[float] = deque(maxlen=self.n_step)
        episode_experiences: list[Experience] = []

        step_root_visits: list[int] = []
        step_tree_depths: list[int] = []
        step_simulations: list[int] = []

        logger.info(f"Starting episode with seed {episode_seed}")
        self._report_current_state(game)

        root_node: Node | None = Node(state=game.copy())

        while not game.is_over():
            step_start_time = time.monotonic()
            if root_node is None:
                logger.error(
                    "MCTS root node became None unexpectedly. Aborting episode."
                )
                break

            if root_node.state.is_over():
                logger.warning(
                    f"MCTS root node state (Step {root_node.state.current_step}) is already terminal before running simulations. Ending episode."
                )
                break

            logger.info(
                f"Step {game.current_step}: Running MCTS simulations ({self.mcts_config.num_simulations}) on state from step {root_node.state.current_step}..."
            )
            mcts_start_time = time.monotonic()
            mcts_max_depth = 0
            try:
                mcts_max_depth = run_mcts_simulations(
                    root_node, self.mcts_config, self.nn_evaluator
                )
            except MCTSExecutionError as mcts_err:
                logger.error(
                    f"Step {game.current_step}: MCTS simulation failed critically: {mcts_err}",
                    exc_info=False,
                )
                break
            except Exception as mcts_err:
                logger.error(
                    f"Step {game.current_step}: MCTS simulation failed unexpectedly: {mcts_err}",
                    exc_info=True,
                )
                break

            mcts_duration = time.monotonic() - mcts_start_time
            logger.info(
                f"Step {game.current_step}: MCTS finished ({mcts_duration:.3f}s). Max Depth: {mcts_max_depth}, Root Visits: {root_node.visit_count}"
            )

            # Log stats *before* taking the step
            self._log_step_stats_async(
                game, root_node.visit_count, mcts_max_depth, step_reward=0.0
            )

            action_selection_start_time = time.monotonic()
            temp = (
                self.mcts_config.temperature_initial
                if game.current_step < self.mcts_config.temperature_anneal_steps
                else self.mcts_config.temperature_final
            )
            try:
                policy_target = get_policy_target(root_node, temperature=1.0)
                action = select_action_based_on_visits(root_node, temperature=temp)
            except Exception as policy_err:
                logger.error(
                    f"Step {game.current_step}: MCTS policy/action selection failed: {policy_err}",
                    exc_info=True,
                )
                break

            action_selection_duration = time.monotonic() - action_selection_start_time

            logger.info(
                f"Step {game.current_step}: Selected Action {action} (Temp={temp:.3f}). Selection time: {action_selection_duration:.4f}s"
            )

            feature_start_time = time.monotonic()
            try:
                state_features: StateType = extract_state_features(
                    game, self.model_config
                )
            except Exception as e:
                logger.error(
                    f"Error extracting features at step {game.current_step}: {e}",
                    exc_info=True,
                )
                break

            feature_duration = time.monotonic() - feature_start_time
            logger.debug(
                f"Step {game.current_step}: Feature extraction time: {feature_duration:.4f}s"
            )

            n_step_state_policy_buffer.append((state_features, policy_target))

            step_simulations.append(self.mcts_config.num_simulations)
            step_root_visits.append(root_node.visit_count)
            step_tree_depths.append(mcts_max_depth)

            game_step_start_time = time.monotonic()
            step_reward = 0.0
            try:
                step_reward, done = game.step(action)
            except Exception as step_err:
                logger.error(
                    f"Error executing game step for action {action}: {step_err}",
                    exc_info=True,
                )
                break

            game_step_duration = time.monotonic() - game_step_start_time
            logger.info(
                f"Step {game.current_step}: Action {action} taken. Reward: {step_reward:.3f}, Done: {done}. Game step time: {game_step_duration:.4f}s"
            )

            n_step_reward_buffer.append(step_reward)

            if len(n_step_reward_buffer) == self.n_step:
                discounted_reward_sum = 0.0
                for i in range(self.n_step):
                    discounted_reward_sum += (self.gamma**i) * n_step_reward_buffer[i]

                bootstrap_value = 0.0
                if not done:
                    try:
                        _, bootstrap_value = self.nn_evaluator.evaluate(game)
                    except Exception as eval_err:
                        logger.error(
                            f"Error evaluating bootstrap state S_{game.current_step}: {eval_err}",
                            exc_info=True,
                        )
                        bootstrap_value = 0.0

                n_step_return = (
                    discounted_reward_sum + (self.gamma**self.n_step) * bootstrap_value
                )

                state_features_t_minus_n, policy_target_t_minus_n = (
                    n_step_state_policy_buffer[0]
                )

                episode_experiences.append(
                    (
                        state_features_t_minus_n,
                        policy_target_t_minus_n,
                        n_step_return,
                    )
                )

            self._report_current_state(game)
            # Log stats *after* taking the step
            self._log_step_stats_async(
                game,
                root_node.visit_count if root_node else 0,
                mcts_max_depth,
                step_reward=step_reward,
            )

            tree_reuse_start_time = time.monotonic()
            if not done:
                if root_node and action in root_node.children:  # Check root_node exists
                    root_node = root_node.children[action]
                    root_node.parent = None
                    logger.debug(
                        f"Reused MCTS subtree for action {action}. New root step: {root_node.state.current_step}"
                    )
                else:
                    logger.error(
                        f"Child node for selected action {action} not found in MCTS tree children: {list(root_node.children.keys()) if root_node else 'No Root'}. Resetting MCTS root to current game state."
                    )
                    root_node = Node(state=game.copy())
            else:
                root_node = None

            tree_reuse_duration = time.monotonic() - tree_reuse_start_time
            logger.debug(
                f"Step {game.current_step}: Tree reuse/reset time: {tree_reuse_duration:.4f}s"
            )

            step_duration = time.monotonic() - step_start_time
            logger.info(
                f"Step {game.current_step} total duration: {step_duration:.3f}s"
            )

            if done:
                break

        final_score = game.game_score
        logger.info(
            f"Episode finished. Final Score: {final_score:.2f}, Steps: {game.current_step}"
        )

        remaining_steps = len(n_step_reward_buffer)
        for k in range(remaining_steps):
            discounted_reward_sum = 0.0
            for i in range(remaining_steps - k):
                discounted_reward_sum += (self.gamma**i) * n_step_reward_buffer[k + i]

            n_step_return = discounted_reward_sum
            state_features_t, policy_target_t = n_step_state_policy_buffer[k]
            episode_experiences.append(
                (state_features_t, policy_target_t, n_step_return)
            )

        total_sims_episode = sum(step_simulations)
        avg_visits_episode = np.mean(step_root_visits) if step_root_visits else 0.0
        avg_depth_episode = np.mean(step_tree_depths) if step_tree_depths else 0.0

        if not episode_experiences:
            logger.warning(
                f"Episode finished with 0 experiences collected. Final score: {final_score}, Steps: {game.current_step}"
            )

        return SelfPlayResult(
            episode_experiences=episode_experiences,
            final_score=final_score,
            episode_steps=game.current_step,
            total_simulations=total_sims_episode,
            avg_root_visits=float(avg_visits_episode),
            avg_tree_depth=float(avg_depth_episode),
        )


File: alphatriangle\rl\self_play\__init__.py


File: alphatriangle\stats\collector.py
# File: alphatriangle/stats/collector.py
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, cast  # Added cast

import numpy as np
import ray

from ..utils.types import StatsCollectorData, StepInfo

if TYPE_CHECKING:
    from ..environment import GameState

logger = logging.getLogger(__name__)


@ray.remote
class StatsCollectorActor:
    """
    Ray actor for collecting time-series statistics and latest worker game states.
    Stores metrics as (StepInfo, value) tuples.
    """

    def __init__(self, max_history: int | None = 1000):
        self.max_history = max_history
        self._data: StatsCollectorData = {}
        # Store the latest GameState reported by each worker
        self._latest_worker_states: dict[int, GameState] = {}
        self._last_state_update_time: dict[int, float] = {}

        # Ensure logger is configured for the actor process
        log_level = logging.INFO
        # Check if runtime_context is available before using it
        actor_id_str = "UnknownActor"
        try:
            if ray.is_initialized():
                actor_id_str = ray.get_runtime_context().get_actor_id()
        except Exception:
            pass  # Ignore if context cannot be retrieved
        log_format = f"%(asctime)s [%(levelname)s] [StatsCollectorActor pid={actor_id_str}] %(name)s: %(message)s"
        logging.basicConfig(level=log_level, format=log_format, force=True)
        global logger  # Re-assign logger after config
        logger = logging.getLogger(__name__)

        logger.info(f"Initialized with max_history={max_history}.")

    # --- Metric Logging ---

    def log(self, metric_name: str, value: float, step_info: StepInfo):
        """Logs a single metric value with its associated step information."""
        logger.debug(
            f"Attempting to log metric='{metric_name}', value={value}, step_info={step_info}"
        )
        if not isinstance(metric_name, str):
            logger.error(f"Invalid metric_name type: {type(metric_name)}")
            return
        if not isinstance(step_info, dict):
            logger.error(f"Invalid step_info type: {type(step_info)}")
            return
        if not np.isfinite(value):
            logger.warning(
                f"Received non-finite value for metric '{metric_name}': {value}. Skipping log."
            )
            return

        try:
            if metric_name not in self._data:
                logger.debug(f"Creating new deque for metric: '{metric_name}'")
                self._data[metric_name] = deque(maxlen=self.max_history)

            # Ensure value is float for consistency
            value_float = float(value)
            # Store the StepInfo dict directly
            self._data[metric_name].append((step_info, value_float))
            logger.debug(
                f"Successfully logged metric='{metric_name}', value={value_float}, step_info={step_info}. Deque size: {len(self._data[metric_name])}"
            )
        except (ValueError, TypeError) as e:
            logger.error(
                f"Could not log metric '{metric_name}'. Invalid value conversion: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error logging metric '{metric_name}' (value={value}, step_info={step_info}): {e}",
                exc_info=True,
            )

    def log_batch(self, metrics: dict[str, tuple[float, StepInfo]]):
        """Logs a batch of metrics, each with its StepInfo."""
        received_keys = list(metrics.keys())
        logger.debug(
            f"Log batch received with {len(metrics)} metrics. Keys: {received_keys}"
        )
        for name, (value, step_info) in metrics.items():
            self.log(name, value, step_info)  # Delegate to single log method

    # --- Game State Handling (No change needed) ---

    def update_worker_game_state(self, worker_id: int, game_state: "GameState"):
        """Stores the latest game state received from a worker."""
        if not isinstance(worker_id, int):
            logger.error(f"Invalid worker_id type: {type(worker_id)}")
            return
        # Basic check if it looks like a GameState object (can add more checks if needed)
        if not hasattr(game_state, "grid_data") or not hasattr(game_state, "shapes"):
            logger.error(
                f"Invalid game_state object received from worker {worker_id}: type={type(game_state)}"
            )
            return
        # Store the received state (it should be a copy from the worker)
        self._latest_worker_states[worker_id] = game_state
        self._last_state_update_time[worker_id] = time.time()
        logger.debug(
            f"Updated game state for worker {worker_id} (Step: {game_state.current_step})"
        )

    def get_latest_worker_states(self) -> dict[int, "GameState"]:
        """Returns a shallow copy of the latest worker states dictionary."""
        logger.debug(
            f"get_latest_worker_states called. Returning states for workers: {list(self._latest_worker_states.keys())}"
        )
        return self._latest_worker_states.copy()

    # --- Data Retrieval & Management ---

    def get_data(self) -> StatsCollectorData:
        """Returns a copy of the collected statistics data."""
        logger.debug(f"get_data called. Returning {len(self._data)} metrics.")
        # Return copies of deques to prevent external modification
        return {k: dq.copy() for k, dq in self._data.items()}

    def get_metric_data(self, metric_name: str) -> deque[tuple[StepInfo, float]] | None:
        """Returns a copy of the data deque for a specific metric."""
        dq = self._data.get(metric_name)
        return dq.copy() if dq else None

    def clear(self):
        """Clears all collected statistics and worker states."""
        self._data = {}
        self._latest_worker_states = {}
        self._last_state_update_time = {}
        logger.info("Data and worker states cleared.")

    def get_state(self) -> dict[str, Any]:
        """Returns the internal state for saving."""
        # Convert deques to lists for serialization compatibility with cloudpickle/json
        # The items in the list are now (StepInfo, float) tuples
        serializable_metrics = {key: list(dq) for key, dq in self._data.items()}

        state = {
            "max_history": self.max_history,
            "_metrics_data_list": serializable_metrics,  # Use the list version
        }
        logger.info(
            f"get_state called. Returning state for {len(serializable_metrics)} metrics. Worker states NOT included."
        )
        return state

    def set_state(self, state: dict[str, Any]):
        """Restores the internal state from saved data."""
        self.max_history = state.get("max_history", self.max_history)
        loaded_metrics_list = state.get("_metrics_data_list", {})
        self._data = {}
        restored_metrics_count = 0
        for key, items_list in loaded_metrics_list.items():
            if isinstance(items_list, list) and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict)
                for item in items_list
            ):
                # Ensure items are (StepInfo, float)
                valid_items: list[tuple[StepInfo, float]] = []
                for item in items_list:
                    try:
                        # Basic check for StepInfo structure (can be enhanced)
                        if not isinstance(item[0], dict):
                            raise TypeError("StepInfo is not a dict")
                        # Ensure value is float
                        value = float(item[1])
                        # Cast the dict to StepInfo for type safety
                        step_info = cast("StepInfo", item[0])
                        valid_items.append((step_info, value))
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid item {item} in metric '{key}' during state restore: {e}"
                        )
                # Convert list back to deque with maxlen
                # Cast valid_items to the expected type for deque
                self._data[key] = deque(
                    cast("list[tuple[StepInfo, float]]", valid_items),
                    maxlen=self.max_history,
                )
                restored_metrics_count += 1
            else:
                logger.warning(
                    f"Skipping restore for metric '{key}'. Invalid data format: {type(items_list)}"
                )
        # Clear worker states on restore, as they are transient
        self._latest_worker_states = {}
        self._last_state_update_time = {}
        logger.info(
            f"State restored. Restored {restored_metrics_count} metrics. Max history: {self.max_history}. Worker states cleared."
        )


File: alphatriangle\stats\plotter.py
# File: alphatriangle/stats/plotter.py
import contextlib
import logging
import time
from collections import deque
from io import BytesIO
from typing import TYPE_CHECKING, Any

import matplotlib

if TYPE_CHECKING:
    import numpy as np

    # --- MOVED: Import vis_colors only for type checking ---

import pygame

# Use Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- MOVED: Import normalize_color_for_matplotlib here ---
from ..utils.helpers import normalize_color_for_matplotlib  # noqa: E402

# --- CHANGED: Import StepInfo ---
from ..utils.types import StatsCollectorData  # noqa: E402

# --- END CHANGED ---
from .plot_definitions import (  # noqa: E402
    WEIGHT_UPDATE_METRIC_KEY,  # Import key
    PlotDefinitions,
)
from .plot_rendering import render_subplot  # Import subplot rendering logic

logger = logging.getLogger(__name__)


class Plotter:
    """
    Handles creation and caching of the multi-plot Matplotlib surface.
    Uses PlotDefinitions for layout and plot_rendering for drawing subplots.
    """

    def __init__(self, plot_update_interval: float = 0.75):  # Increased interval
        self.plot_surface_cache: pygame.Surface | None = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = plot_update_interval
        self.colors = self._init_colors()
        self.plot_definitions = PlotDefinitions(self.colors)  # Instantiate definitions

        self.rolling_window_sizes: list[int] = [
            10,
            50,
            100,
            500,
            1000,
            5000,
        ]

        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None  # type: ignore # numpy is type-checked \only
        self.last_target_size: tuple[int, int] = (0, 0)
        self.last_data_hash: int | None = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> dict[str, tuple[float, float, float]]:
        """Initializes plot colors using hardcoded values to avoid vis import."""
        # This breaks the circular import. Ensure these match vis_colors.py
        colors_rgb = {
            "YELLOW": (230, 230, 40),
            "WHITE": (255, 255, 255),
            "LIGHT_GRAY": (180, 180, 180),
            "LIGHTG": (144, 238, 144),
            "RED": (220, 40, 40),
            "BLUE": (60, 60, 220),
            "GREEN": (40, 200, 40),
            "CYAN": (40, 200, 200),
            "PURPLE": (140, 40, 140),
            "BLACK": (0, 0, 0),
            "GRAY": (100, 100, 100),
            "ORANGE": (240, 150, 20),
            "HOTPINK": (255, 105, 180),
        }

        return {
            "RL/Current_Score": normalize_color_for_matplotlib(colors_rgb["YELLOW"]),
            "RL/Step_Reward": normalize_color_for_matplotlib(colors_rgb["WHITE"]),
            "MCTS/Step_Visits": normalize_color_for_matplotlib(
                colors_rgb["LIGHT_GRAY"]
            ),
            "MCTS/Step_Depth": normalize_color_for_matplotlib(colors_rgb["LIGHTG"]),
            "Loss/Total": normalize_color_for_matplotlib(colors_rgb["RED"]),
            "Loss/Value": normalize_color_for_matplotlib(colors_rgb["BLUE"]),
            "Loss/Policy": normalize_color_for_matplotlib(colors_rgb["GREEN"]),
            "LearningRate": normalize_color_for_matplotlib(colors_rgb["CYAN"]),
            "Buffer/Size": normalize_color_for_matplotlib(colors_rgb["PURPLE"]),
            WEIGHT_UPDATE_METRIC_KEY: normalize_color_for_matplotlib(
                colors_rgb["BLACK"]
            ),
            "placeholder": normalize_color_for_matplotlib(colors_rgb["GRAY"]),
            "Rate/Steps_Per_Sec": normalize_color_for_matplotlib(colors_rgb["ORANGE"]),
            "Rate/Episodes_Per_Sec": normalize_color_for_matplotlib(
                colors_rgb["HOTPINK"]
            ),
            "Rate/Simulations_Per_Sec": normalize_color_for_matplotlib(
                colors_rgb["LIGHTG"]
            ),
            "PER/Beta": normalize_color_for_matplotlib(colors_rgb["ORANGE"]),
            "Loss/Entropy": normalize_color_for_matplotlib(colors_rgb["PURPLE"]),
            "Loss/Mean_TD_Error": normalize_color_for_matplotlib(colors_rgb["RED"]),
            "Progress/Train_Step_Percent": normalize_color_for_matplotlib(
                colors_rgb["GREEN"]
            ),
            "Progress/Total_Simulations": normalize_color_for_matplotlib(
                colors_rgb["CYAN"]
            ),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes based on plot definitions."""
        logger.info(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Error closing previous figure: {e}")

        dpi = 96
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            nrows = self.plot_definitions.nrows
            ncols = self.plot_definitions.ncols
            self.fig, self.axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width_in, fig_height_in),
                dpi=dpi,
                sharex=False,
            )
            if self.axes is None:
                raise RuntimeError("Failed to create Matplotlib subplots.")

            self.fig.patch.set_facecolor((0.1, 0.1, 0.1))
            self.fig.subplots_adjust(
                hspace=0.40,
                wspace=0.08,
                left=0.03,
                right=0.99,
                bottom=0.05,
                top=0.98,
            )
            self.last_target_size = (target_width, target_height)
            logger.info(
                f"[Plotter] Matplotlib figure initialized ({nrows}x{ncols} grid)."
            )
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

    def _get_data_hash(self, plot_data: StatsCollectorData) -> int:
        """Generates a hash based on data lengths and recent values."""
        hash_val = 0
        sample_size = 5
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            hash_val ^= hash(key) ^ len(dq)
            if not dq:
                continue
            try:
                num_to_sample = min(len(dq), sample_size)
                for i in range(-1, -num_to_sample - 1, -1):
                    # Hash StepInfo dict and value
                    step_info, val = dq[i]
                    # Simple hash for dict: hash tuple of sorted items
                    step_info_hash = hash(tuple(sorted(step_info.items())))
                    hash_val ^= step_info_hash ^ hash(f"{val:.6f}")
            except IndexError:
                pass
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData) -> bool:
        """Updates the data on the existing Matplotlib axes using render_subplot."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            plot_defs = self.plot_definitions.get_definitions()
            num_plots = len(plot_defs)

            # Extract weight update steps (global_step values)
            weight_update_steps: list[int] = []
            if WEIGHT_UPDATE_METRIC_KEY in plot_data:
                dq = plot_data[WEIGHT_UPDATE_METRIC_KEY]
                if dq:
                    # Extract global_step from StepInfo
                    weight_update_steps = [
                        step_info["global_step"]
                        for step_info, _ in dq
                        if "global_step" in step_info
                    ]

            for i, plot_def in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                # Pass weight_update_steps
                render_subplot(
                    ax=ax,
                    plot_data=plot_data,
                    plot_def=plot_def,
                    colors=self.colors,
                    rolling_window_sizes=self.rolling_window_sizes,
                    weight_update_steps=weight_update_steps,  # Pass the list
                )

            for i in range(num_plots, len(axes_flat)):
                ax = axes_flat[i]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("gray")
                ax.spines["left"].set_color("gray")

            self._apply_final_axis_formatting(axes_flat)

            plot_update_duration = time.monotonic() - plot_update_start
            logger.debug(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass
            return False

    def _apply_final_axis_formatting(self, axes_flat: Any):
        """Hides x-axis labels for plots not in the bottom row."""
        if not hasattr(axes_flat, "__iter__"):
            logger.error("axes_flat is not iterable in _apply_final_axis_formatting")
            return

        nrows = self.plot_definitions.nrows
        ncols = self.plot_definitions.ncols
        num_plots = len(self.plot_definitions.get_definitions())

        for i, ax in enumerate(axes_flat):
            if i >= num_plots:
                continue

            if i < (nrows - 1) * ncols:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0)

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            self.fig.canvas.draw_idle()
            buf = BytesIO()
            self.fig.savefig(
                buf, format="png", transparent=False, facecolor=self.fig.get_facecolor()
            )
            buf.seek(0)
            plot_img_surface = pygame.image.load(buf, "png").convert_alpha()
            buf.close()

            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                plot_img_surface = pygame.transform.smoothscale(
                    plot_img_surface, (target_width, target_height)
                )
            render_duration = time.monotonic() - render_start
            logger.debug(
                f"[Plotter] Figure rendered to surface in {render_duration:.4f}s"
            )
            return plot_img_surface

        except Exception as e:
            logger.error(f"Error rendering Matplotlib figure: {e}", exc_info=True)
            return None

    def get_plot_surface(
        self, plot_data: StatsCollectorData, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(
            isinstance(dq, deque) and dq
            for key, dq in plot_data.items()
            if not key.startswith("Internal/")
        )
        target_size = (target_width, target_height)

        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed
        can_create_plot = target_width > 50 and target_height > 50

        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info("[Plotter] Target size too small, clearing cache/figure.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes, self.last_target_size = None, None, (0, 0)
            return None

        if not has_data:
            logger.debug("[Plotter] No plot data available, returning cache (if any).")
            return self.plot_surface_cache

        try:
            if needs_reinit:
                self._init_figure(target_width, target_height)
                needs_update = True

            if needs_update and self.fig:
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot update failed, returning stale cache if available."
                    )
            elif (
                self.plot_surface_cache is None
                and self.fig
                and self._update_plot_data(plot_data)
            ):
                self.plot_surface_cache = self._render_figure_to_surface(
                    target_width, target_height
                )
                self.last_plot_update_time = current_time
                self.last_data_hash = current_data_hash

        except Exception as e:
            logger.error(f"[Plotter] Error in get_plot_surface: {e}", exc_info=True)
            self.plot_surface_cache = None
            if self.fig:
                with contextlib.suppress(Exception):
                    plt.close(self.fig)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed."""
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                print(f"[Plotter] Error closing figure in destructor: {e}")


File: alphatriangle\stats\plot_definitions.py
# File: alphatriangle/stats/plot_definitions.py
from typing import Literal, NamedTuple

# Define type for x-axis data source
PlotXAxisType = Literal["index", "global_step", "buffer_size"]

# Define metric key constant for weight updates
WEIGHT_UPDATE_METRIC_KEY = "Internal/Weight_Update_Step"


class PlotDefinition(NamedTuple):
    """Configuration for a single subplot."""

    metric_key: str  # Key in the StatsCollectorData dictionary
    label: str  # Title displayed on the plot
    y_log_scale: bool  # Use logarithmic scale for y-axis
    x_axis_type: PlotXAxisType  # What the x-axis represents


class PlotDefinitions:
    """Holds the definitions for all plots in the dashboard."""

    def __init__(self, colors: dict[str, tuple[float, float, float]]):
        self.colors = colors  # Store colors if needed for default lookups
        self.nrows: int = 4
        self.ncols: int = 3
        # Key used to get weight update steps for vertical lines
        self.weight_update_key = WEIGHT_UPDATE_METRIC_KEY  # Use the constant

        # Define the layout and properties of each plot
        self._definitions: list[PlotDefinition] = [
            # Row 1
            # --- CHANGED: x_axis_type to "index" ---
            PlotDefinition("RL/Current_Score", "Score", False, "index"),
            PlotDefinition(
                "Rate/Episodes_Per_Sec", "Episodes/sec", False, "buffer_size"
            ),
            PlotDefinition("Loss/Total", "Total Loss", True, "global_step"),
            # Row 2
            PlotDefinition("RL/Step_Reward", "Step Reward", False, "index"),
            PlotDefinition(
                "Rate/Simulations_Per_Sec", "Sims/sec", False, "buffer_size"
            ),
            PlotDefinition("Loss/Policy", "Policy Loss", True, "global_step"),
            # Row 3
            PlotDefinition("MCTS/Step_Visits", "MCTS Visits", False, "index"),
            PlotDefinition("Buffer/Size", "Buffer Size", False, "buffer_size"),
            PlotDefinition("Loss/Value", "Value Loss", True, "global_step"),
            # Row 4
            PlotDefinition("MCTS/Step_Depth", "MCTS Depth", False, "index"),
            # --- END CHANGED ---
            PlotDefinition("Rate/Steps_Per_Sec", "Steps/sec", False, "global_step"),
            PlotDefinition("LearningRate", "Learn Rate", True, "global_step"),
        ]

        # Validate grid size
        if len(self._definitions) > self.nrows * self.ncols:
            raise ValueError(
                f"Number of plot definitions ({len(self._definitions)}) exceeds grid size ({self.nrows}x{self.ncols})"
            )

    def get_definitions(self) -> list[PlotDefinition]:
        """Returns the list of plot definitions."""
        return self._definitions


# Define PlotType for potential external use, though PlotDefinition is more specific
PlotType = PlotDefinition


File: alphatriangle\stats\plot_rendering.py
# File: alphatriangle/stats/plot_rendering.py
import logging
from collections import deque
from typing import TYPE_CHECKING  # Added cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..utils.types import StepInfo
from .plot_definitions import PlotDefinition
from .plot_utils import calculate_rolling_average, format_value

if TYPE_CHECKING:
    from .collector import StatsCollectorData

logger = logging.getLogger(__name__)


def _find_closest_index_for_global_step(
    target_global_step: int,
    step_info_list: list[StepInfo],
) -> int | None:
    """
    Finds the index in the step_info_list where the stored 'global_step'
    is closest to the target_global_step.
    Returns None if no suitable point is found or list is empty.
    """
    if not step_info_list:
        return None

    best_match_idx = None
    min_step_diff = float("inf")

    for i, step_info in enumerate(step_info_list):
        global_step_in_info = step_info.get("global_step")

        if global_step_in_info is not None:
            step_diff = abs(global_step_in_info - target_global_step)
            if step_diff < min_step_diff:
                min_step_diff = step_diff
                best_match_idx = i
            # Optimization: If we found an exact match, we can stop early
            # Also, if the steps start increasing again, we passed the best point
            if step_diff == 0 or (
                best_match_idx is not None and global_step_in_info > target_global_step
            ):
                break

    # Optional: Add a tolerance? If min_step_diff is too large, maybe don't return a match?
    # For now, return the index of the closest found value.
    return best_match_idx


def render_subplot(
    ax: plt.Axes,
    plot_data: "StatsCollectorData",
    plot_def: PlotDefinition,
    colors: dict[str, tuple[float, float, float]],
    rolling_window_sizes: list[int],
    weight_update_steps: list[int] | None = None,  # Global steps where updates happened
) -> bool:
    """
    Renders data for a single metric onto the given Matplotlib Axes object.
    Scatter point size/alpha decrease linearly as more data/longer averages are shown.
    Draws semi-transparent black, solid vertical lines for weight updates on all plots.
    Returns True if data was plotted, False otherwise.
    """
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    metric_key = plot_def.metric_key
    label = plot_def.label
    log_scale = plot_def.y_log_scale
    x_axis_type = plot_def.x_axis_type  # e.g., "global_step", "buffer_size", "index"

    x_data: list[int] = []
    y_data: list[float] = []
    x_label_text = "Index"  # Default label
    step_info_list: list[StepInfo] = []  # Store step info for mapping

    dq = plot_data.get(metric_key, deque())
    if dq:
        # Extract x-axis value and store StepInfo
        temp_x = []
        temp_y = []
        for i, (step_info, value) in enumerate(dq):
            x_val: int | None = None
            if x_axis_type == "global_step":
                x_val = step_info.get("global_step")
                x_label_text = "Train Step"
            elif x_axis_type == "buffer_size":
                x_val = step_info.get("buffer_size")
                x_label_text = "Buffer Size"
            else:  # index
                x_val = i  # Use the simple index 'i' as the x-value
                if (
                    "Score" in metric_key
                    or "Reward" in metric_key
                    or "MCTS" in metric_key
                ):
                    x_label_text = "Game Step Index"  # Label remains descriptive
                else:
                    x_label_text = "Data Point Index"

            if x_val is not None:
                temp_x.append(x_val)
                temp_y.append(value)
                step_info_list.append(
                    step_info
                )  # Keep StepInfo aligned with data points
            else:
                logger.warning(
                    f"Missing x-axis key '{x_axis_type}' in step_info for metric '{metric_key}'. Skipping point."
                )

        x_data = temp_x
        y_data = temp_y

    color_mpl = colors.get(metric_key, (0.5, 0.5, 0.5))
    placeholder_color_mpl = colors.get("placeholder", (0.5, 0.5, 0.5))

    data_plotted = False
    if not x_data or not y_data:
        ax.text(
            0.5,
            0.5,
            f"{label}\n(No Data)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=placeholder_color_mpl,
            fontsize=9,
        )
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        data_plotted = True

        # Determine best rolling average window
        num_points = len(y_data)
        best_window = 0
        for window in sorted(rolling_window_sizes, reverse=True):
            if num_points >= window:
                best_window = window
                break

        # Determine scatter size/alpha based on best_window
        # Initialize as float
        scatter_size: float = 0.0
        scatter_alpha: float = 0.0
        max_scatter_size = 10.0  # Use float
        min_scatter_size = 1.0  # Use float
        max_scatter_alpha = 0.3
        min_scatter_alpha = 0.03
        max_window_for_interp = float(max(rolling_window_sizes))

        if best_window == 0:
            scatter_size = max_scatter_size
            scatter_alpha = max_scatter_alpha
        elif best_window >= max_window_for_interp:
            scatter_size = min_scatter_size
            scatter_alpha = min_scatter_alpha
        else:
            interp_fraction = best_window / max_window_for_interp
            # Cast result of np.interp to float
            scatter_size = float(
                np.interp(interp_fraction, [0, 1], [max_scatter_size, min_scatter_size])
            )
            scatter_alpha = float(
                np.interp(
                    interp_fraction, [0, 1], [max_scatter_alpha, min_scatter_alpha]
                )
            )

        # Plot raw data with dynamic size/alpha
        ax.scatter(
            x_data,
            y_data,
            color=color_mpl,
            alpha=scatter_alpha,
            s=scatter_size,  # Pass float size
            label="_nolegend_",
            zorder=2,
        )

        # Plot best rolling average
        if best_window > 0:
            rolling_avg = calculate_rolling_average(y_data, best_window)
            if len(rolling_avg) == len(x_data):
                ax.plot(
                    x_data,
                    rolling_avg,
                    color=color_mpl,
                    alpha=0.9,
                    linewidth=1.5,
                    label=f"Avg {best_window}",
                    zorder=3,
                )
                ax.legend(
                    fontsize=6, loc="upper right", frameon=False, labelcolor="lightgray"
                )
            else:
                logger.warning(
                    f"Length mismatch for rolling avg ({len(rolling_avg)}) vs x_data ({len(x_data)}) for {label}. Skipping avg plot."
                )

        # Draw vertical lines by mapping global_step to current x-axis value
        if weight_update_steps and step_info_list:
            plotted_line_x_values: set[float] = set()  # Store plotted x-values as float
            for update_global_step in weight_update_steps:
                x_index_for_line = _find_closest_index_for_global_step(
                    update_global_step, step_info_list
                )

                if x_index_for_line is not None and x_index_for_line < len(x_data):
                    actual_x_value: int | float
                    if x_axis_type == "index":
                        actual_x_value = x_index_for_line  # int
                    else:
                        # Explicitly cast list access to int to satisfy MyPy
                        actual_x_value = int(x_data[x_index_for_line])  # int

                    # Cast to float for axvline and check if already plotted
                    actual_x_float = float(actual_x_value)
                    if actual_x_float not in plotted_line_x_values:
                        ax.axvline(
                            x=actual_x_float,  # Pass float
                            color="black",
                            linestyle="-",
                            linewidth=0.7,
                            alpha=0.5,
                            zorder=10,
                        )
                        plotted_line_x_values.add(actual_x_float)
                else:
                    logger.debug(
                        f"Could not map global_step {update_global_step} to an index for plot '{label}'"
                    )

        # Formatting
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.tick_params(axis="both", which="major", labelsize=7, colors="lightgray")
        ax.grid(
            True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4), zorder=1
        )  # Ensure grid is behind lines

        # Set y-axis scale
        if log_scale:
            ax.set_yscale("log")
            min_val = min((v for v in y_data if v > 0), default=1e-6)
            max_val = max(y_data, default=1.0)
            ylim_bottom = max(1e-9, min_val * 0.1)
            ylim_top = max_val * 10
            if ylim_bottom < ylim_top:
                ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
            else:
                ax.set_ylim(bottom=1e-9, top=1.0)
        else:
            ax.set_yscale("linear")
            min_val = min(y_data) if y_data else 0.0
            max_val = max(y_data) if y_data else 0.0
            val_range = max_val - min_val
            if abs(val_range) < 1e-6:
                center_val = (min_val + max_val) / 2.0
                buffer = max(abs(center_val * 0.1), 0.5)
                ylim_bottom, ylim_top = center_val - buffer, center_val + buffer
            else:
                buffer = val_range * 0.1
                ylim_bottom, ylim_top = min_val - buffer, max_val + buffer
            if all(v >= 0 for v in y_data) and ylim_bottom < 0:
                ylim_bottom = 0.0
            if ylim_bottom >= ylim_top:
                ylim_bottom, ylim_top = min_val - 0.5, max_val + 0.5
                if ylim_bottom >= ylim_top:
                    ylim_bottom, ylim_top = 0.0, max(1.0, max_val)
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

        # Format x-axis
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"
            )
        )
        ax.set_xlabel(x_label_text, fontsize=8, color="gray")

        # Format y-axis
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_value(y)))

        # Add info text (min/max/current)
        current_val_str = format_value(y_data[-1])
        min_val_overall = min(y_data)
        max_val_overall = max(y_data)
        min_str = format_value(min_val_overall)
        max_str = format_value(max_val_overall)
        info_text = f"Min:{min_str} | Max:{max_str} | Cur:{current_val_str}"
        ax.text(
            1.0,
            1.01,
            info_text,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=6,
            color="white",
        )

    # Common Axis Styling (applied regardless of data presence)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")

    return data_plotted


File: alphatriangle\stats\plot_utils.py
# File: alphatriangle/stats/plot_utils.py
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

logger = logging.getLogger(__name__)


def calculate_rolling_average(data: list[float], window: int) -> list[float]:
    """Calculates the rolling average with handling for edges."""
    if not data or window <= 0:
        return []
    if window > len(data):
        # If window is larger than data, return average of all data for all points
        avg = np.mean(data)
        # Cast to float explicitly
        return [float(avg)] * len(data)
    # Use convolution for efficient rolling average
    weights = np.ones(window) / window
    rolling_avg = np.convolve(data, weights, mode="valid")
    # Pad the beginning to match the original length
    padding = [float(np.mean(data[:i])) for i in range(1, window)]
    # Cast result to list of floats
    return padding + [float(x) for x in rolling_avg]


def calculate_trend_line(
    steps: list[int], values: list[float]
) -> tuple[list[int], list[float]]:
    """Calculates a simple linear trend line."""
    if len(steps) < 2:
        return [], []
    try:
        coeffs = np.polyfit(steps, values, 1)
        poly = np.poly1d(coeffs)
        trend_values = poly(steps)
        return steps, list(trend_values)
    except Exception as e:
        logger.warning(f"Could not calculate trend line: {e}")
        return [], []


def format_value(value: float) -> str:
    """Formats a float value for display on the plot."""
    if abs(value) < 1e-6:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}k"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def render_single_plot(
    ax: plt.Axes,
    steps: list[int],
    values: list[float],
    label: str,
    color: tuple[float, float, float],
    placeholder_color: tuple[float, float, float],
    rolling_window_sizes: list[int],
    show_placeholder: bool = False,
    placeholder_text: str | None = None,
    y_log_scale: bool = False,
):
    """
    Renders a single metric plot onto a Matplotlib Axes object.
    Plots raw data and the single best rolling average line.
    """
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    if show_placeholder or not steps or not values:
        text_to_display = placeholder_text if placeholder_text else "(No Data)"
        ax.text(
            0.5,
            0.5,
            text_to_display,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=placeholder_color,
            fontsize=9,
        )
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        return

    # Plot raw data (thin, semi-transparent)
    ax.plot(steps, values, color=color, alpha=0.3, linewidth=0.7, label="_nolegend_")

    # --- CHANGED: Plot only the single best rolling average ---
    num_points = len(steps)
    best_window = 0
    # Iterate through window sizes in descending order
    for window in sorted(rolling_window_sizes, reverse=True):
        if num_points >= window:
            best_window = window
            break  # Found the largest applicable window

    if best_window > 0:
        rolling_avg = calculate_rolling_average(values, best_window)
        ax.plot(
            steps,
            rolling_avg,
            color=color,
            alpha=0.9,  # Make it more prominent
            linewidth=1.5,
            label=f"Avg {best_window}",  # Label this single line
        )
        # Add legend only if rolling average was plotted
        ax.legend(fontsize=6, loc="upper right", frameon=False, labelcolor="lightgray")
    # --- END CHANGED ---

    # Formatting
    ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
    ax.tick_params(axis="both", which="major", labelsize=7, colors="lightgray")
    ax.grid(True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")

    # Set y-axis scale
    if y_log_scale:
        ax.set_yscale("log")
        # Ensure positive values for log scale, adjust limits if needed
        min_val = (
            min(v for v in values if v > 0) if any(v > 0 for v in values) else 1e-6
        )
        max_val = max(values) if values else 1.0
        # Add buffer for log scale
        ylim_bottom = max(1e-9, min_val * 0.1)
        ylim_top = max_val * 10
        # Prevent potential errors if limits are invalid
        if ylim_bottom < ylim_top:
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
        else:
            ax.set_ylim(bottom=1e-9, top=1.0)  # Fallback limits
    else:
        ax.set_yscale("linear")

    # Format x-axis (steps)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(
        # Remove unused 'p' argument
        FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}")
    )
    ax.set_xlabel("Step", fontsize=8, color="gray")

    # Format y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # Remove unused 'p' argument
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_value(y)))

    # Add current value text
    current_val_str = format_value(values[-1])
    ax.text(
        1.0,
        1.01,
        f"Cur: {current_val_str}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7,
        color="white",
    )


File: alphatriangle\stats\README.md
# File: alphatriangle/stats/README.md
# Statistics Module (`alphatriangle.stats`)

## Purpose and Architecture

This module provides utilities for collecting, storing, and visualizing time-series statistics generated during the reinforcement learning training process using Matplotlib rendered onto Pygame surfaces.

-   **[`collector.py`](collector.py):** Defines the `StatsCollectorActor` class, a **Ray actor**. This actor uses dictionaries of `deque`s to store metric values (like losses, rewards, learning rate) associated with **step context information** ([`StepInfo`](../utils/types.py) dictionary containing `global_step`, `buffer_size`, etc.). It provides **remote methods** (`log`, `log_batch`) for asynchronous logging from multiple sources and methods (`get_data`, `get_metric_data`) for fetching the stored data. It supports limiting the history size and includes `get_state` and `set_state` methods for checkpointing.
-   **[`plot_definitions.py`](plot_definitions.py):** Defines the structure and properties of each plot in the dashboard (`PlotDefinition`, `PlotDefinitions`), including which step information (`x_axis_type`) should be used for the x-axis. **Also defines the `WEIGHT_UPDATE_METRIC_KEY` constant.**
-   **[`plot_rendering.py`](plot_rendering.py):** Contains the `render_subplot` function, responsible for drawing a single metric onto a Matplotlib `Axes` object based on a `PlotDefinition`. **It now accepts a list of `global_step` values where weight updates occurred and draws semi-transparent black, solid vertical lines on all plots by mapping the `global_step` to the corresponding value on the plot's specific x-axis. The raw data scatter points are now rendered with dynamically adjusted size and opacity, starting larger and fading as more data accumulates.**
-   **[`plot_utils.py`](plot_utils.py):** Contains helper functions for Matplotlib plotting, including calculating rolling averages and formatting values.
-   **[`plotter.py`](plotter.py):** Defines the `Plotter` class which manages the overall Matplotlib figure and axes.
    -   It orchestrates the plotting of multiple metrics onto a grid within the figure using `render_subplot`.
    -   It handles rendering the Matplotlib figure to an in-memory buffer and then converting it to a `pygame.Surface`.
    -   It implements caching logic.
    -   **It now extracts the weight update steps (`global_step` values) from the collected data and passes them to `render_subplot`.**

## Exposed Interfaces

-   **Classes:**
    -   `StatsCollectorActor`: Ray actor for collecting stats.
        -   `log.remote(metric_name: str, value: float, step_info: StepInfo)`
        -   `log_batch.remote(metrics: Dict[str, Tuple[float, StepInfo]])`
        -   `get_data.remote() -> StatsCollectorData`
        -   `get_metric_data.remote(metric_name: str) -> Optional[Deque[Tuple[StepInfo, float]]]`
        -   (Other methods: `clear`, `get_state`, `set_state`)
    -   `Plotter`:
        -   `get_plot_surface(plot_data: StatsCollectorData, target_width: int, target_height: int) -> Optional[pygame.Surface]`
    -   `PlotDefinitions`: Holds the layout and properties of all plots.
    -   `PlotDefinition`: NamedTuple defining a single plot.
-   **Types:**
    -   `StatsCollectorData`: Type alias `Dict[str, Deque[Tuple[StepInfo, float]]]` representing the stored data.
    -   `StepInfo`: TypedDict holding step context.
    -   `PlotType`: Alias for `PlotDefinition`.
-   **Functions:**
    -   `render_subplot`: Renders a single subplot, including mapped weight update lines and dynamic scatter points.
-   **Modules:**
    -   `plot_utils`: Contains helper functions.
-   **Constants:**
    -   `WEIGHT_UPDATE_METRIC_KEY`: Key used for logging/retrieving weight update events.

## Dependencies

-   **[`alphatriangle.visualization`](../visualization/README.md)**: `colors` (used indirectly via `Plotter`).
-   **[`alphatriangle.utils`](../utils/README.md)**: `helpers`, `types` (including `StepInfo`).
-   **`pygame`**: Used by `plotter.py` to create the final surface.
-   **`matplotlib`**: Used by `plotter.py`, `plot_rendering.py`, and `plot_utils.py` for generating plots.
-   **`numpy`**: Used by `plot_utils.py` and `plot_rendering.py` for calculations.
-   **`ray`**: Used by `StatsCollectorActor`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `math`, `time`, `io`, `contextlib`.

## Integration

-   The `TrainingLoop` ([`alphatriangle.training.loop`](../training/loop.py)) instantiates `StatsCollectorActor` and calls its remote `log` or `log_batch` methods, **passing `StepInfo` dictionaries**. It logs the `WEIGHT_UPDATE_METRIC_KEY` when worker weights are updated.
-   The `SelfPlayWorker` ([`alphatriangle.rl.self_play.worker`](../rl/self_play/worker.py)) calls `log_batch` **passing `StepInfo` dictionaries containing `game_step_index` and `global_step` (of its current weights).**
-   The `DashboardRenderer` ([`alphatriangle.visualization.core.dashboard_renderer`](../visualization/core/dashboard_renderer.py)) holds a handle to the `StatsCollectorActor` and calls `get_data.remote()` periodically to fetch data for plotting.
-   The `DashboardRenderer` instantiates `Plotter` and calls `get_plot_surface` using the fetched stats data and the target plot area dimensions. It then blits the returned surface.
-   The `DataManager` ([`alphatriangle.data.data_manager`](../data/data_manager.py)) interacts with the `StatsCollectorActor` via `get_state.remote()` and `set_state.remote()` during checkpoint saving and loading.

---

**Note:** Please keep this README updated when changing the data collection methods (especially the `StepInfo` structure), the plotting functions, or the way statistics are managed and displayed. Accurate documentation is crucial for maintainability.

File: alphatriangle\stats\__init__.py
# File: alphatriangle/stats/__init__.py
"""
Statistics collection and plotting module.
"""

from alphatriangle.utils.types import StatsCollectorData

from . import plot_utils
from .collector import StatsCollectorActor
from .plot_definitions import PlotDefinitions, PlotType  # Import new definitions
from .plot_rendering import render_subplot  # Import new rendering function
from .plotter import Plotter

__all__ = [
    # Core Collector
    "StatsCollectorActor",
    "StatsCollectorData",
    # Plotting Orchestrator
    "Plotter",
    # Plotting Definitions & Rendering Logic
    "PlotDefinitions",
    "PlotType",
    "render_subplot",
    # Plotting Utilities
    "plot_utils",
]


File: alphatriangle\structs\constants.py
# File: alphatriangle/structs/constants.py

# Define standard colors used for shapes
# Ensure these colors are distinct and visually clear
# Also ensure BLACK (0,0,0) is NOT used here if it represents empty in color_np
SHAPE_COLORS: list[tuple[int, int, int]] = [
    (220, 40, 40),  # 0: Red
    (60, 60, 220),  # 1: Blue
    (40, 200, 40),  # 2: Green
    (230, 230, 40),  # 3: Yellow
    (240, 150, 20),  # 4: Orange
    (140, 40, 140),  # 5: Purple
    (40, 200, 200),  # 6: Cyan
    (200, 100, 180),  # 7: Pink (Example addition)
    (100, 180, 200),  # 8: Light Blue (Example addition)
]

# --- NumPy GridData Color Representation ---
# ID for empty cells in the _color_id_np array
NO_COLOR_ID: int = -1
# ID for debug-toggled cells
DEBUG_COLOR_ID: int = -2

# Mapping from Color ID (int >= 0) to RGB tuple.
# Index 0 corresponds to SHAPE_COLORS[0], etc.
# This list is used by visualization to get the RGB from the ID.
COLOR_ID_MAP: list[tuple[int, int, int]] = SHAPE_COLORS

# Reverse mapping for efficient lookup during placement (Color Tuple -> ID)
# Note: Ensure SHAPE_COLORS have unique tuples.
COLOR_TO_ID_MAP: dict[tuple[int, int, int], int] = {
    color: i for i, color in enumerate(COLOR_ID_MAP)
}

# Add special colors to the map if needed for rendering debug/other states
# These IDs won't be stored during normal shape placement.
# Example: If you want to render the debug color:
# DEBUG_RGB_COLOR = (255, 255, 0) # Example Yellow
# COLOR_ID_MAP.append(DEBUG_RGB_COLOR) # Append if needed elsewhere, but generally lookup handled separately.

# --- End NumPy GridData Color Representation ---


File: alphatriangle\structs\README.md
# File: alphatriangle/structs/README.md
# Core Structures Module (`alphatriangle.structs`)

## Purpose and Architecture

This module defines fundamental data structures and constants that are shared across multiple major components of the application (like [`environment`](../environment/README.md), [`visualization`](../visualization/README.md), [`features`](../features/README.md)). Its primary purpose is to break potential circular dependencies that arise when these components need to know about the same basic building blocks.

-   **[`triangle.py`](triangle.py):** Defines the `Triangle` class, representing a single cell on the game grid.
-   **[`shape.py`](shape.py):** Defines the `Shape` class, representing a placeable piece composed of triangles.
-   **[`constants.py`](constants.py):** Defines shared constants, such as the list of possible `SHAPE_COLORS` and color IDs used in `GridData`.

By placing these core definitions in a low-level module with minimal dependencies, other modules can import them without creating import cycles.

## Exposed Interfaces

-   **Classes:**
    -   `Triangle`: Represents a grid cell.
    -   `Shape`: Represents a placeable piece.
-   **Constants:**
    -   `SHAPE_COLORS`: A list of RGB tuples for shape generation.
    -   `NO_COLOR_ID`: Integer ID for empty cells in `GridData`.
    -   `DEBUG_COLOR_ID`: Integer ID for debug-toggled cells in `GridData`.
    -   `COLOR_ID_MAP`: List mapping color ID index to RGB tuple.
    -   `COLOR_TO_ID_MAP`: Dictionary mapping RGB tuple to color ID index.

## Dependencies

This module has minimal dependencies, primarily relying on standard Python libraries (`typing`). It should **not** import from higher-level modules like `environment`, `visualization`, `nn`, `rl`, etc.

---

**Note:** This module should only contain widely shared, fundamental data structures and constants. More complex logic or structures specific to a particular domain (like game rules or rendering details) should remain in their respective modules.
```

**22. File:** `alphatriangle/training/README.md`
**Explanation:** Review content and add relative links.

```markdown
# File: alphatriangle/training/README.md
# Training Module (`alphatriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **[`setup.py`](setup.py):** Contains `setup_training_components` which initializes Ray, detects resources, adjusts worker counts, loads configurations, and creates the core components bundle (`TrainingComponents`).
-   **[`components.py`](components.py):** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **[`loop.py`](loop.py):** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, **passing the current `global_step` to the workers**, and logging a special event (`Internal/Weight_Update_Step`) with the `global_step` to the `StatsCollectorActor` when updates occur.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **[`worker_manager.py`](worker_manager.py):** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. **It now passes the `global_step` to workers when updating weights.**
-   **[`loop_helpers.py`](loop_helpers.py):** Contains helper functions used by `TrainingLoop` for tasks like logging rates, updating the visual queue, validating experiences, and logging results. **It constructs the `StepInfo` dictionary containing relevant step counters (`global_step`, `buffer_size`) for logging.** It also includes logic to log the weight update event.
-   **[`runners.py`](runners.py):** Re-exports the main entry point functions (`run_training_headless_mode`, `run_training_visual_mode`) from their respective modules.
-   **[`headless_runner.py`](headless_runner.py) / [`visual_runner.py`](visual_runner.py):** Contain the top-level logic for running training in either headless or visual mode. They handle argument parsing (via CLI), setup logging, call `setup_training_components`, load initial state, run the `TrainingLoop` (potentially in a separate thread for visual mode), handle visualization setup (visual mode), and manage overall cleanup (MLflow, Ray shutdown).
-   **[`logging_utils.py`](logging_utils.py):** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`headless_runner`/`visual_runner`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingLoop`: Contains the core async loop logic.
    -   `TrainingComponents`: Dataclass holding initialized components.
    -   `WorkerManager`: Manages worker actors.
    -   `LoopHelpers`: Provides helper functions for the loop.
-   **Functions (from `runners.py`):**
    -   `run_training_headless_mode(...) -> int`
    -   `run_training_visual_mode(...) -> int`
-   **Functions (from `setup.py`):**
    -   `setup_training_components(...) -> Tuple[Optional[TrainingComponents], bool]`
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: All configuration classes.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.rl`](../rl/README.md)**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **[`alphatriangle.data`](../data/README.md)**: `DataManager`, `LoadedTrainingState`.
-   **[`alphatriangle.stats`](../stats/README.md)**: `StatsCollectorActor`, `PlotDefinitions`.
-   **[`alphatriangle.environment`](../environment/README.md)**: `GameState`.
-   **[`alphatriangle.utils`](../utils/README.md)**: Helper functions and types (including `StepInfo`).
-   **[`alphatriangle.visualization`](../visualization/README.md)**: `ProgressBar`, `DashboardRenderer`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`, `sys`, `traceback`, `pathlib`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the logging of step context information (`StepInfo`) and worker weight updates.

File: alphatriangle\structs\shape.py
from __future__ import annotations


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(
        self, triangles: list[tuple[int, int, bool]], color: tuple[int, int, int]
    ):
        self.triangles: list[tuple[int, int, bool]] = sorted(triangles)
        self.color: tuple[int, int, int] = color

    def bbox(self) -> tuple[int, int, int, int]:
        """Calculates bounding box (min_r, min_c, max_r, max_c) in relative coords."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))

    def copy(self) -> Shape:
        """Creates a shallow copy (triangle list is copied, color is shared)."""
        new_shape = Shape.__new__(Shape)
        new_shape.triangles = list(self.triangles)
        new_shape.color = self.color
        return new_shape

    def __str__(self) -> str:
        return f"Shape(Color:{self.color}, Tris:{len(self.triangles)})"

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on triangles and color."""
        if not isinstance(other, Shape):
            return NotImplemented
        return self.triangles == other.triangles and self.color == other.color

    def __hash__(self) -> int:
        """Allows shapes to be used in sets/dicts if needed."""
        return hash((tuple(self.triangles), self.color))


File: alphatriangle\structs\triangle.py
from __future__ import annotations


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up
        self.is_death = is_death
        self.is_occupied = is_death
        self.color: tuple[int, int, int] | None = None

        self.neighbor_left: Triangle | None = None
        self.neighbor_right: Triangle | None = None
        self.neighbor_vert: Triangle | None = None

    def get_points(
        self, ox: float, oy: float, cw: float, ch: float
    ) -> list[tuple[float, float]]:
        """Calculates vertex points for drawing, relative to origin (ox, oy)."""
        x = ox + self.col * (cw * 0.75)
        y = oy + self.row * ch
        if self.is_up:
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def copy(self) -> Triangle:
        """Creates a copy of the Triangle object's state (neighbors are not copied)."""
        new_tri = Triangle(self.row, self.col, self.is_up, self.is_death)
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color
        return new_tri

    def __repr__(self) -> str:
        state = "D" if self.is_death else ("O" if self.is_occupied else ".")
        orient = "^" if self.is_up else "v"
        return f"T({self.row},{self.col} {orient}{state})"

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.row == other.row and self.col == other.col


File: alphatriangle\structs\__init__.py
# File: alphatriangle/structs/__init__.py
"""
Module for core data structures used across different parts of the application,
like environment, visualization, and features. Helps avoid circular dependencies.
"""

# Correctly export constants from the constants submodule
from .constants import (
    COLOR_ID_MAP,
    COLOR_TO_ID_MAP,
    DEBUG_COLOR_ID,
    NO_COLOR_ID,
    SHAPE_COLORS,
)
from .shape import Shape
from .triangle import Triangle

__all__ = [
    "Triangle",
    "Shape",
    # Exported Constants
    "SHAPE_COLORS",
    "NO_COLOR_ID",
    "DEBUG_COLOR_ID",
    "COLOR_ID_MAP",
    "COLOR_TO_ID_MAP",
]


File: alphatriangle\training\components.py
# File: alphatriangle/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

# --- ADDED: Import ActorHandle ---
import ray

# --- END ADDED ---

if TYPE_CHECKING:
    # Keep ray import here as well for consistency if needed elsewhere
    # import ray

    from alphatriangle.config import (
        EnvConfig,
        MCTSConfig,
        ModelConfig,
        PersistenceConfig,
        TrainConfig,
    )
    from alphatriangle.data import DataManager
    from alphatriangle.nn import NeuralNetwork
    from alphatriangle.rl import ExperienceBuffer, Trainer

    # --- REMOVED: Import StatsCollectorActor class ---
    # from alphatriangle.stats import StatsCollectorActor
    # --- END REMOVED ---


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    data_manager: "DataManager"
    # --- CORRECTED: Use ActorHandle type hint ---
    stats_collector_actor: ray.actor.ActorHandle | None
    # --- END CORRECTED ---
    train_config: "TrainConfig"
    env_config: "EnvConfig"
    model_config: "ModelConfig"
    mcts_config: "MCTSConfig"
    persist_config: "PersistenceConfig"


File: alphatriangle\training\headless_runner.py
# File: alphatriangle/training/headless_runner.py
import logging
import sys
import traceback
from collections import deque
from pathlib import Path

import mlflow
import ray
import torch

from ..config import APP_NAME, PersistenceConfig, TrainConfig
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import (
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop  # Import TrainingLoop here
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    """Sets up MLflow tracking and starts a run."""
    try:
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
        logger.info(f"Set MLflow experiment to: {APP_NAME}")

        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
            return True
        else:
            logger.error("MLflow run failed to start.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        return False


def _load_and_apply_initial_state(components: TrainingComponents) -> TrainingLoop:
    """Loads initial state using DataManager and applies it to components, returning an initialized TrainingLoop."""
    loaded_state = components.data_manager.load_initial_state()
    training_loop = TrainingLoop(components)  # Instantiate loop first

    if loaded_state.checkpoint_data:
        cp_data = loaded_state.checkpoint_data
        logger.info(
            f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
        )

        if cp_data.model_state_dict:
            components.nn.set_weights(cp_data.model_state_dict)
        if cp_data.optimizer_state_dict:
            try:
                components.trainer.optimizer.load_state_dict(
                    cp_data.optimizer_state_dict
                )
                for state in components.trainer.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(components.nn.device)
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        # --- CHANGED: Removed isinstance check, rely on ActorHandle type hint ---
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            # --- END CHANGED ---
            try:
                # MyPy should now know this is an ActorHandle
                set_state_ref = components.stats_collector_actor.set_state.remote(
                    cp_data.stats_collector_state
                )
                ray.get(set_state_ref, timeout=5.0)
                logger.info("StatsCollectorActor state restored.")
            except Exception as e:
                logger.error(
                    f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                )

        training_loop.set_initial_state(
            cp_data.global_step,
            cp_data.episodes_played,
            cp_data.total_simulations_run,
        )
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    if loaded_state.buffer_data:
        if components.train_config.USE_PER:
            logger.info("Rebuilding PER SumTree from loaded buffer data...")
            if not hasattr(components.buffer, "tree") or components.buffer.tree is None:
                components.buffer.tree = SumTree(components.buffer.capacity)
            else:
                components.buffer.tree = SumTree(components.buffer.capacity)
            max_p = 1.0
            for exp in loaded_state.buffer_data.buffer_list:
                components.buffer.tree.add(max_p, exp)
            logger.info(f"PER buffer loaded. Size: {len(components.buffer)}")
        else:
            components.buffer.buffer = deque(
                loaded_state.buffer_data.buffer_list,
                maxlen=components.buffer.capacity,
            )
            logger.info(f"Uniform buffer loaded. Size: {len(components.buffer)}")
        if training_loop.buffer_fill_progress:
            training_loop.buffer_fill_progress.set_current_steps(len(components.buffer))
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    return training_loop


def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    if not training_loop:
        logger.warning("Cannot save final state: TrainingLoop not available.")
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        # Pass the actor handle directly
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_final=True,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


def run_training_headless_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline in headless mode."""
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    log_file_path = None
    file_handler = None
    ray_initialized_by_setup = False
    mlflow_run_active = False

    try:
        # --- Setup File Logging ---
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "headless"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
        )

        # --- Setup Components (includes Ray init) ---
        components, ray_initialized_by_setup = setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize MLflow ---
        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)  # Log configs after run starts
            # Log parameter counts after MLflow run starts
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        # --- Load State & Initialize Loop ---
        training_loop = _load_and_apply_initial_state(components)

        # --- Run Training Loop ---
        training_loop.initialize_workers()
        training_loop.run()

        # --- Determine Exit Code ---
        if training_loop.training_complete:
            exit_code = 0
        elif training_loop.training_exception:
            exit_code = 1  # Failed
        else:
            exit_code = 1  # Interrupted or other issue

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during headless training setup or execution: {e}"
        )
        traceback.print_exc()
        # Attempt to log failure status if MLflow run was started
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        # --- Cleanup ---
        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            # Save final state
            _save_final_state(training_loop)
            # Cleanup loop actors
            training_loop.cleanup_actors()
            # Determine final status
            if training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(training_loop.training_exception)
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"

        # End MLflow run
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        # Shutdown Ray ONLY if initialized by the setup function in this process
        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by headless runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        # Close file handler
        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Headless training finished with exit code {exit_code}.")
    return exit_code


File: alphatriangle\training\logging_utils.py
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np

if TYPE_CHECKING:
    from alphatriangle.config import PersistenceConfig

    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class Tee:
    """Helper class to redirect stdout/stderr to both console and a file."""

    def __init__(self, stream1, stream2, main_stream_for_fileno):
        self.stream1 = stream1
        self.stream2 = stream2
        self._main_stream_for_fileno = main_stream_for_fileno

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

    def fileno(self):
        return self._main_stream_for_fileno.fileno()

    def isatty(self):
        return self._main_stream_for_fileno.isatty()


def get_root_logger() -> logging.Logger:
    """Gets the root logger instance."""
    return logging.getLogger()


def setup_file_logging(
    persist_config: "PersistenceConfig", run_name: str, mode_suffix: str
) -> str:
    """Sets up file logging for the current run."""
    run_base_dir = Path(persist_config.get_run_base_dir(run_name))
    log_dir = run_base_dir / persist_config.LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{run_name}_{mode_suffix}.log"

    file_handler = logging.FileHandler(log_file_path, mode="w")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    root_logger = get_root_logger()
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
        logger.info(f"Added file handler logging to: {log_file_path}")
    else:
        logger.warning("File handler already exists for root logger.")

    return str(log_file_path)


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs configuration parameters to MLflow."""
    if not mlflow.active_run():
        logger.warning("No active MLflow run found. Cannot log configs.")
        return

    logger.info("Logging configuration parameters to MLflow...")
    try:
        mlflow.log_params(components.env_config.model_dump())
        mlflow.log_params(components.model_config.model_dump())
        mlflow.log_params(components.train_config.model_dump())
        mlflow.log_params(components.mcts_config.model_dump())
        mlflow.log_params(components.persist_config.model_dump())
        logger.info("Configuration parameters logged to MLflow.")
    except Exception as e:
        logger.error(f"Failed to log parameters to MLflow: {e}", exc_info=True)


def log_metrics_to_mlflow(metrics: dict[str, Any], step: int):
    """Logs metrics to MLflow."""
    if not mlflow.active_run():
        logger.warning("No active MLflow run found. Cannot log metrics.")
        return

    try:
        # Filter only numeric, finite metrics
        numeric_metrics = {}
        for k, v in metrics.items():
            # Use isinstance with | for multiple types
            if isinstance(v, int | float | np.number) and np.isfinite(v):
                numeric_metrics[k] = float(v)
            else:
                logger.debug(
                    f"Skipping non-numeric or non-finite metric for MLflow: {k}={v} (type: {type(v)})"
                )
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)
            logger.debug(
                f"Logged {len(numeric_metrics)} metrics to MLflow at step {step}."
            )
        else:
            logger.debug(f"No valid numeric metrics to log at step {step}.")
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {e}", exc_info=True)


File: alphatriangle\training\loop.py
# File: alphatriangle/training/loop.py
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

# --- MOVED: numpy import ---
# import numpy as np
# --- END MOVED ---
from ..rl import SelfPlayResult

# --- MOVED: ProgressBar import ---
# from ..visualization.ui import ProgressBar
# --- END MOVED ---
# --- MOVED: TrainingComponents import ---
# from .components import TrainingComponents
# --- END MOVED ---
from .loop_helpers import LoopHelpers
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    # --- ADDED: Imports under TYPE_CHECKING ---
    import numpy as np

    from ..utils.types import PERBatchSample
    from ..visualization.ui import ProgressBar
    from .components import TrainingComponents

    # --- END ADDED ---


logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Manages the core asynchronous training loop logic: coordinating worker tasks,
    processing results, triggering training steps, and updating visual queue.
    Receives initialized components via TrainingComponents. Runs indefinitely
    until stop_requested is set. Uses WorkerManager and LoopHelpers.
    """

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None = None,
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.train_config = components.train_config

        # Core components
        self.buffer = components.buffer
        self.trainer = components.trainer

        # State variables
        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.worker_weight_updates_count = 0  # Counter for worker updates
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.training_exception: Exception | None = None

        # Progress Bars (initialized later)
        self.train_step_progress: ProgressBar | None = None
        self.buffer_fill_progress: ProgressBar | None = None

        # Instantiate helpers
        self.worker_manager = WorkerManager(components)
        self.loop_helpers = LoopHelpers(
            components,
            self.visual_state_queue,
            self._get_loop_state,  # Pass method to get current state
        )

        logger.info("TrainingLoop initialized.")

    def _get_loop_state(self) -> dict[str, Any]:
        """Provides current loop state to helpers."""
        return {
            "global_step": self.global_step,
            "episodes_played": self.episodes_played,
            "total_simulations_run": self.total_simulations_run,
            "worker_weight_updates": self.worker_weight_updates_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "num_active_workers": self.worker_manager.get_num_active_workers(),
            "num_pending_tasks": self.worker_manager.get_num_pending_tasks(),
            "train_progress": self.train_step_progress,
            "buffer_progress": self.buffer_fill_progress,
            "start_time": self.start_time,
            "num_workers": self.train_config.NUM_SELF_PLAY_WORKERS,
        }

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        # Estimate initial weight updates based on loaded step and frequency
        self.worker_weight_updates_count = (
            global_step // self.train_config.WORKER_UPDATE_FREQ_STEPS
        )
        self.train_step_progress, self.buffer_fill_progress = (
            self.loop_helpers.initialize_progress_bars(
                global_step, len(self.buffer), self.start_time
            )
        )
        self.loop_helpers.reset_rate_counters(
            global_step, episodes_played, total_simulations
        )
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}, WeightUpdates={self.worker_weight_updates_count}"
        )

    def initialize_workers(self):
        """Initializes self-play workers using WorkerManager."""
        self.worker_manager.initialize_workers()

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _process_self_play_result(self, result: SelfPlayResult, worker_id: int):
        """Processes a validated result from a worker."""
        logger.debug(
            f"Processing result from worker {worker_id} (Ep Steps: {result.episode_steps}, Score: {result.final_score:.2f})"
        )

        valid_experiences, invalid_count = self.loop_helpers.validate_experiences(
            result.episode_experiences
        )
        if invalid_count > 0:
            logger.warning(
                f"Worker {worker_id}: {invalid_count} invalid experiences were filtered out before adding to buffer."
            )

        if valid_experiences:
            try:
                self.buffer.add_batch(valid_experiences)
                logger.debug(
                    f"Added {len(valid_experiences)} experiences from worker {worker_id} to buffer (Buffer size: {len(self.buffer)})."
                )
            except Exception as e:
                logger.error(
                    f"Error adding batch to buffer from worker {worker_id}: {e}",
                    exc_info=True,
                )
                return  # Don't update counters if add failed

            if self.buffer_fill_progress:
                self.buffer_fill_progress.set_current_steps(len(self.buffer))
            self.episodes_played += 1
            self.total_simulations_run += result.total_simulations
        else:
            logger.error(
                f"Worker {worker_id}: Self-play episode produced NO valid experiences (Steps: {result.episode_steps}, Score: {result.final_score:.2f}). This prevents buffer filling and training."
            )

    def _run_training_step(self) -> bool:
        """Runs one training step."""
        if not self.buffer.is_ready():
            return False
        per_sample: PERBatchSample | None = self.buffer.sample(
            self.train_config.BATCH_SIZE, current_train_step=self.global_step
        )
        if not per_sample:
            return False

        train_result: tuple[dict[str, float], np.ndarray] | None = (
            self.trainer.train_step(per_sample)
        )
        if train_result:
            loss_info, td_errors = train_result
            self.global_step += 1
            if self.train_step_progress:
                self.train_step_progress.set_current_steps(self.global_step)
            if self.train_config.USE_PER:
                self.buffer.update_priorities(per_sample["indices"], td_errors)
            self.loop_helpers.log_training_results_async(
                loss_info, self.global_step, self.total_simulations_run
            )

            # Check if it's time to update worker networks
            if self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS == 0:
                try:
                    # --- CHANGED: Pass global_step ---
                    self.worker_manager.update_worker_networks(self.global_step)
                    # --- END CHANGED ---
                    self.worker_weight_updates_count += 1  # Increment counter
                    # Log the update event using the helper
                    self.loop_helpers.log_weight_update_event(self.global_step)
                except Exception as update_err:
                    logger.error(
                        f"Failed to update worker networks at step {self.global_step}: {update_err}"
                    )

            if self.global_step % 50 == 0:
                logger.info(
                    f"Step {self.global_step}: P Loss={loss_info['policy_loss']:.4f}, V Loss={loss_info['value_loss']:.4f}, Ent={loss_info['entropy']:.4f}, TD Err={loss_info['mean_td_error']:.4f}"
                )
            return True
        else:
            logger.warning(f"Training step {self.global_step + 1} failed.")
            return False

    def run(self):
        """Main training loop."""
        max_steps_info = (
            f"Target steps: {self.train_config.MAX_TRAINING_STEPS}"
            if self.train_config.MAX_TRAINING_STEPS is not None
            else "Running indefinitely (no MAX_TRAINING_STEPS)"
        )
        logger.info(f"Starting TrainingLoop run... {max_steps_info}")
        self.start_time = time.time()

        try:
            # Initial task submission
            self.worker_manager.submit_initial_tasks()

            while not self.stop_requested.is_set():
                # Check if max steps reached
                if (
                    self.train_config.MAX_TRAINING_STEPS is not None
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached MAX_TRAINING_STEPS ({self.train_config.MAX_TRAINING_STEPS}). Stopping loop."
                    )
                    self.training_complete = True
                    self.request_stop()
                    break

                # Training Step
                if self.buffer.is_ready():
                    _ = self._run_training_step()  # Call training step
                else:
                    time.sleep(0.01)

                if self.stop_requested.is_set():
                    break

                # Handle Completed Worker Tasks
                wait_timeout = 0.1 if self.buffer.is_ready() else 0.5
                completed_tasks = self.worker_manager.get_completed_tasks(wait_timeout)

                for worker_id, result_or_error in completed_tasks:
                    if isinstance(result_or_error, SelfPlayResult):
                        try:
                            self._process_self_play_result(result_or_error, worker_id)
                        except Exception as proc_err:
                            logger.error(
                                f"Error processing result from worker {worker_id}: {proc_err}",
                                exc_info=True,
                            )
                    elif isinstance(result_or_error, Exception):
                        logger.error(
                            f"Worker {worker_id} task failed with exception: {result_or_error}"
                        )
                    else:
                        logger.error(
                            f"Received unexpected item from completed tasks for worker {worker_id}: {type(result_or_error)}"
                        )

                    self.worker_manager.submit_task(worker_id)

                if self.stop_requested.is_set():
                    break

                # Periodic Tasks (using LoopHelpers)
                self.loop_helpers.update_visual_queue()
                self.loop_helpers.log_progress_eta()
                self.loop_helpers.calculate_and_log_rates()

                if not completed_tasks and not self.buffer.is_ready():
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received in TrainingLoop. Stopping.")
            self.request_stop()
        except Exception as e:
            logger.critical(f"Unhandled exception in TrainingLoop: {e}", exc_info=True)
            self.training_exception = e
            self.request_stop()
        finally:
            if (
                self.training_exception
                or self.stop_requested.is_set()
                and not self.training_complete
            ):
                self.training_complete = False
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Cleans up worker actors using WorkerManager."""
        self.worker_manager.cleanup_actors()


File: alphatriangle\training\loop_helpers.py
# File: alphatriangle/training/loop_helpers.py
import logging
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import ray

from ..environment import GameState
from ..stats.plot_definitions import WEIGHT_UPDATE_METRIC_KEY
from ..utils import format_eta
from ..utils.types import Experience, StatsCollectorData, StepInfo
from ..visualization.core import colors
from ..visualization.ui import ProgressBar

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

VISUAL_UPDATE_INTERVAL = 0.2
STATS_FETCH_INTERVAL = 0.5
VIS_STATE_FETCH_TIMEOUT = 0.1
RATE_CALCULATION_INTERVAL = 5.0  # Check rates every 5 seconds


class LoopHelpers:
    """Provides helper functions for the TrainingLoop."""

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None,
        get_loop_state_func: Callable[[], dict[str, Any]],
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.get_loop_state = get_loop_state_func  # Function to get current loop state

        self.stats_collector_actor = components.stats_collector_actor
        self.train_config = components.train_config
        self.trainer = components.trainer  # Needed for LR

        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = 0
        self.last_rate_calc_episodes = 0
        self.last_rate_calc_sims = 0

    def reset_rate_counters(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Resets counters used for rate calculation."""
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = global_step
        self.last_rate_calc_episodes = episodes_played
        self.last_rate_calc_sims = total_simulations

    def initialize_progress_bars(
        self, global_step: int, buffer_size: int, start_time: float
    ) -> tuple[ProgressBar, ProgressBar]:
        """Initializes and returns progress bars."""
        train_total_steps = self.train_config.MAX_TRAINING_STEPS
        train_total_steps_for_bar = (
            train_total_steps if train_total_steps is not None else 1
        )

        train_step_progress = ProgressBar(
            "Training Steps",
            total_steps=train_total_steps_for_bar,
            start_time=start_time,
            initial_steps=global_step,
            initial_color=colors.GREEN,
        )
        buffer_fill_progress = ProgressBar(
            "Buffer Fill",
            self.train_config.BUFFER_CAPACITY,
            start_time=start_time,
            initial_steps=buffer_size,
            initial_color=colors.ORANGE,
        )
        return train_step_progress, buffer_fill_progress

    def _fetch_latest_stats(self):
        """Fetches the latest stats data from the actor."""
        current_time = time.time()
        if current_time - self.last_stats_fetch_time < STATS_FETCH_INTERVAL:
            return
        self.last_stats_fetch_time = current_time
        if self.stats_collector_actor:
            try:
                data_ref = self.stats_collector_actor.get_data.remote()  # type: ignore
                self.latest_stats_data = ray.get(data_ref, timeout=1.0)
            except Exception as e:
                logger.warning(f"Failed to fetch latest stats: {e}")

    def calculate_and_log_rates(self):
        """
        Calculates and logs steps/sec, episodes/sec, sims/sec, and buffer size.
        Ensures buffer-related rates are logged against buffer size.
        Logs metrics with StepInfo containing global_step and buffer_size.
        """
        current_time = time.time()
        time_delta = current_time - self.last_rate_calc_time
        if time_delta < RATE_CALCULATION_INTERVAL:
            return

        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        episodes_played = loop_state["episodes_played"]
        total_simulations = loop_state["total_simulations_run"]
        current_buffer_size = int(loop_state["buffer_size"])  # Use int for step info

        steps_delta = global_step - self.last_rate_calc_step
        episodes_delta = episodes_played - self.last_rate_calc_episodes
        sims_delta = total_simulations - self.last_rate_calc_sims

        steps_per_sec = steps_delta / time_delta if time_delta > 0 else 0.0
        episodes_per_sec = episodes_delta / time_delta if time_delta > 0 else 0.0
        sims_per_sec = sims_delta / time_delta if time_delta > 0 else 0.0

        if self.stats_collector_actor:
            step_info_buffer: StepInfo = {
                "global_step": global_step,
                "buffer_size": current_buffer_size,
            }
            step_info_global: StepInfo = {"global_step": global_step}

            rate_stats: dict[str, tuple[float, StepInfo]] = {
                "Rate/Episodes_Per_Sec": (episodes_per_sec, step_info_buffer),
                "Rate/Simulations_Per_Sec": (sims_per_sec, step_info_buffer),
                "Buffer/Size": (float(current_buffer_size), step_info_buffer),
            }
            log_msg_steps = "Steps/s=N/A"
            if steps_delta > 0:
                rate_stats["Rate/Steps_Per_Sec"] = (steps_per_sec, step_info_global)
                log_msg_steps = f"Steps/s={steps_per_sec:.2f}"

            try:
                self.stats_collector_actor.log_batch.remote(rate_stats)  # type: ignore
                logger.debug(
                    f"Logged rates/buffer at step {global_step} / buffer {current_buffer_size}: "
                    f"{log_msg_steps}, Eps/s={episodes_per_sec:.2f}, Sims/s={sims_per_sec:.1f}, "
                    f"Buffer={current_buffer_size}"
                )
            except Exception as e:
                logger.error(f"Failed to log rate/buffer stats to collector: {e}")

        self.reset_rate_counters(global_step, episodes_played, total_simulations)

    def log_progress_eta(self):
        """Logs progress and ETA."""
        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        train_progress = loop_state["train_progress"]

        if global_step == 0 or global_step % 100 != 0 or not train_progress:
            return

        elapsed_time = time.time() - loop_state["start_time"]
        steps_since_load = global_step - train_progress.initial_steps
        steps_per_sec = 0.0
        self._fetch_latest_stats()  # Fetch stats to get latest rate

        rate_dq = self.latest_stats_data.get("Rate/Steps_Per_Sec")
        if rate_dq:
            # Get the value from the last tuple (step_info, value)
            steps_per_sec = rate_dq[-1][1]
        elif elapsed_time > 1 and steps_since_load > 0:
            # Fallback calculation if rate not in stats yet
            steps_per_sec = steps_since_load / elapsed_time

        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = f"{target_steps:,}" if target_steps else "Infinite"
        progress_str = f"Step {global_step:,}/{target_steps_str}"
        eta_str = format_eta(train_progress.get_eta_seconds())

        buffer_fill_perc = (
            (loop_state["buffer_size"] / loop_state["buffer_capacity"]) * 100
            if loop_state["buffer_capacity"] > 0
            else 0.0
        )
        total_sims = loop_state["total_simulations_run"]
        total_sims_str = (
            f"{total_sims / 1e6:.2f}M"
            if total_sims >= 1e6
            else (f"{total_sims / 1e3:.1f}k" if total_sims >= 1000 else str(total_sims))
        )
        num_pending_tasks = loop_state["num_pending_tasks"]
        logger.info(
            f"Progress: {progress_str}, Episodes: {loop_state['episodes_played']:,}, Total Sims: {total_sims_str}, "
            f"Buffer: {loop_state['buffer_size']:,}/{loop_state['buffer_capacity']:,} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending_tasks}, Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def update_visual_queue(self):
        """Fetches latest states/stats and puts them onto the visual queue."""
        if not self.visual_state_queue or not self.stats_collector_actor:
            return
        current_time = time.time()
        if current_time - self.last_visual_update_time < VISUAL_UPDATE_INTERVAL:
            return
        self.last_visual_update_time = current_time

        latest_worker_states: dict[int, GameState] = {}
        try:
            states_ref = self.stats_collector_actor.get_latest_worker_states.remote()  # type: ignore
            latest_worker_states = ray.get(states_ref, timeout=VIS_STATE_FETCH_TIMEOUT)
            if not isinstance(latest_worker_states, dict):
                logger.warning(
                    f"StatsCollectorActor returned invalid type for states: {type(latest_worker_states)}"
                )
                latest_worker_states = {}
        except Exception as e:
            logger.warning(
                f"Failed to fetch latest worker states for visualization: {e}"
            )
            latest_worker_states = {}

        self._fetch_latest_stats()  # Fetch latest metric data

        visual_data: dict[int, Any] = {}
        for worker_id, state in latest_worker_states.items():
            if isinstance(state, GameState):
                visual_data[worker_id] = state
            else:
                logger.warning(
                    f"Received invalid state type for worker {worker_id} from collector: {type(state)}"
                )

        visual_data[-1] = {
            **self.get_loop_state(),
            "stats_data": self.latest_stats_data,
        }

        if not visual_data or len(visual_data) == 1:
            logger.debug(
                "No worker states available from collector to send to visual queue."
            )
            return

        worker_keys = [k for k in visual_data if k != -1]
        logger.debug(
            f"Putting visual data on queue. Worker IDs with states: {worker_keys}"
        )

        try:
            while self.visual_state_queue.qsize() > 2:
                try:
                    self.visual_state_queue.get_nowait()
                except queue.Empty:
                    break
            self.visual_state_queue.put_nowait(visual_data)
        except queue.Full:
            logger.warning("Visual state queue full, dropping state dictionary.")
        except Exception as qe:
            logger.error(f"Error putting state dict in visual queue: {qe}")

    def validate_experiences(
        self, experiences: list[Experience]
    ) -> tuple[list[Experience], int]:
        """Validates the structure and content of experiences."""
        valid_experiences = []
        invalid_count = 0
        for i, exp in enumerate(experiences):
            is_valid = False
            try:
                if isinstance(exp, tuple) and len(exp) == 3:
                    state_type, policy_map, value = exp
                    if (
                        isinstance(state_type, dict)
                        and "grid" in state_type
                        and "other_features" in state_type
                        and isinstance(state_type["grid"], np.ndarray)
                        and isinstance(state_type["other_features"], np.ndarray)
                        and isinstance(policy_map, dict)
                        and isinstance(value, float | int)
                    ):
                        if np.all(np.isfinite(state_type["grid"])) and np.all(
                            np.isfinite(state_type["other_features"])
                        ):
                            is_valid = True
                        else:
                            logger.warning(
                                f"Experience {i} contains non-finite features."
                            )
                    else:
                        logger.warning(
                            f"Experience {i} has incorrect types: state={type(state_type)}, policy={type(policy_map)}, value={type(value)}"
                        )
                else:
                    logger.warning(
                        f"Experience {i} is not a tuple of length 3: type={type(exp)}, len={len(exp) if isinstance(exp, tuple) else 'N/A'}"
                    )
            except Exception as e:
                logger.error(
                    f"Unexpected error validating experience {i}: {e}", exc_info=True
                )
                is_valid = False
            if is_valid:
                valid_experiences.append(exp)
            else:
                invalid_count += 1
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid experiences.")
        return valid_experiences, invalid_count

    def log_training_results_async(
        self, loss_info: dict[str, float], global_step: int, total_simulations: int
    ) -> None:
        """
        Logs training results asynchronously to the StatsCollectorActor.
        Logs metrics with StepInfo containing global_step.
        """
        current_lr = self.trainer.get_current_lr()
        loop_state = self.get_loop_state()
        train_progress = loop_state.get("train_progress")
        buffer = self.components.buffer

        train_step_perc = (
            (train_progress.get_progress_fraction() * 100) if train_progress else 0.0
        )
        per_beta = (
            buffer._calculate_beta(global_step) if self.train_config.USE_PER else None
        )

        if self.stats_collector_actor:
            step_info: StepInfo = {"global_step": global_step}
            stats_batch: dict[str, tuple[float, StepInfo]] = {
                "Loss/Total": (loss_info["total_loss"], step_info),
                "Loss/Policy": (loss_info["policy_loss"], step_info),
                "Loss/Value": (loss_info["value_loss"], step_info),
                "Loss/Entropy": (loss_info["entropy"], step_info),
                "Loss/Mean_TD_Error": (loss_info["mean_td_error"], step_info),
                "LearningRate": (current_lr, step_info),
                "Progress/Train_Step_Percent": (train_step_perc, step_info),
                "Progress/Total_Simulations": (float(total_simulations), step_info),
            }
            if per_beta is not None:
                stats_batch["PER/Beta"] = (per_beta, step_info)
            try:
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
                logger.debug(
                    f"Logged training batch to StatsCollectorActor for Step {global_step}."
                )
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

    def log_weight_update_event(self, global_step: int) -> None:
        """Logs the event of a worker weight update with StepInfo."""
        if self.stats_collector_actor:
            try:
                # Log with value 1.0 at the current global step
                step_info: StepInfo = {"global_step": global_step}
                update_metric = {WEIGHT_UPDATE_METRIC_KEY: (1.0, step_info)}
                self.stats_collector_actor.log_batch.remote(update_metric)  # type: ignore
                logger.info(f"Logged worker weight update event at step {global_step}.")
            except Exception as e:
                logger.error(f"Failed to log weight update event: {e}")


File: alphatriangle\training\README.md
# File: alphatriangle/training/README.md
# Training Module (`alphatriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **`setup.py`:** Contains `setup_training_components` which initializes Ray, detects resources, adjusts worker counts, loads configurations, and creates the core components bundle (`TrainingComponents`).
-   **`components.py`:** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **`loop.py`:** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, **passing the current `global_step` to the workers**, and logging a special event (`Internal/Weight_Update_Step`) with the `global_step` to the `StatsCollectorActor` when updates occur.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **`worker_manager.py`:** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. **It now passes the `global_step` to workers when updating weights.**
-   **`loop_helpers.py`:** Contains helper functions used by `TrainingLoop` for tasks like logging rates, updating the visual queue, validating experiences, and logging results. **It constructs the `StepInfo` dictionary containing relevant step counters (`global_step`, `buffer_size`) for logging.** It also includes logic to log the weight update event.
-   **`runners.py`:** Re-exports the main entry point functions (`run_training_headless_mode`, `run_training_visual_mode`) from their respective modules.
-   **`headless_runner.py` / `visual_runner.py`:** Contain the top-level logic for running training in either headless or visual mode. They handle argument parsing (via CLI), setup logging, call `setup_training_components`, load initial state, run the `TrainingLoop` (potentially in a separate thread for visual mode), handle visualization setup (visual mode), and manage overall cleanup (MLflow, Ray shutdown).
-   **`logging_utils.py`:** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`headless_runner`/`visual_runner`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingLoop`: Contains the core async loop logic.
    -   `TrainingComponents`: Dataclass holding initialized components.
    -   `WorkerManager`: Manages worker actors.
    -   `LoopHelpers`: Provides helper functions for the loop.
-   **Functions (from `runners.py`):**
    -   `run_training_headless_mode(...) -> int`
    -   `run_training_visual_mode(...) -> int`
-   **Functions (from `setup.py`):**
    -   `setup_training_components(...) -> Tuple[Optional[TrainingComponents], bool]`
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **`alphatriangle.config`**: All configuration classes.
-   **`alphatriangle.nn`**: `NeuralNetwork`.
-   **`alphatriangle.rl`**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **`alphatriangle.data`**: `DataManager`, `LoadedTrainingState`.
-   **`alphatriangle.stats`**: `StatsCollectorActor`, `PlotDefinitions`.
-   **`alphatriangle.environment`**: `GameState`.
-   **`alphatriangle.utils`**: Helper functions and types (including `StepInfo`).
-   **`alphatriangle.visualization`**: `ProgressBar`, `DashboardRenderer`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the logging of step context information (`StepInfo`) and worker weight updates.

File: alphatriangle\training\runners.py
# File: alphatriangle/training/runners.py
"""
Entry points for running training modes.
Imports functions from specific runner modules.
"""

from .headless_runner import run_training_headless_mode
from .visual_runner import run_training_visual_mode

__all__ = [
    "run_training_headless_mode",
    "run_training_visual_mode",
]


File: alphatriangle\training\setup.py
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


File: alphatriangle\training\visual_runner.py
# File: alphatriangle/training/visual_runner.py
import logging
import queue
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import mlflow
import pygame
import ray
import torch

from .. import config, environment, visualization
from ..config import APP_NAME, PersistenceConfig, TrainConfig
from ..utils.sumtree import SumTree
from .components import TrainingComponents
from .logging_utils import (
    Tee,
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop  # Import TrainingLoop here
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)

# Queue for loop to send combined state dict {worker_id: state, -1: global_stats}
visual_state_queue: queue.Queue[dict[int, Any] | None] = queue.Queue(maxsize=5)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    """Sets up MLflow tracking and starts a run."""
    try:
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
        logger.info(f"Set MLflow experiment to: {APP_NAME}")

        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
            return True
        else:
            logger.error("MLflow run failed to start.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        return False


def _load_and_apply_initial_state(components: TrainingComponents) -> TrainingLoop:
    """Loads initial state using DataManager and applies it to components, returning an initialized TrainingLoop."""
    loaded_state = components.data_manager.load_initial_state()
    # Pass visual queue to TrainingLoop constructor
    training_loop = TrainingLoop(components, visual_state_queue=visual_state_queue)

    if loaded_state.checkpoint_data:
        cp_data = loaded_state.checkpoint_data
        logger.info(
            f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
        )

        if cp_data.model_state_dict:
            components.nn.set_weights(cp_data.model_state_dict)
        if cp_data.optimizer_state_dict:
            try:
                components.trainer.optimizer.load_state_dict(
                    cp_data.optimizer_state_dict
                )
                for state in components.trainer.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(components.nn.device)
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        # --- CHANGED: Removed isinstance check, rely on ActorHandle type hint ---
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            # --- END CHANGED ---
            try:
                # MyPy should now know this is an ActorHandle
                set_state_ref = components.stats_collector_actor.set_state.remote(
                    cp_data.stats_collector_state
                )
                ray.get(set_state_ref, timeout=5.0)
                logger.info("StatsCollectorActor state restored.")
            except Exception as e:
                logger.error(
                    f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                )

        training_loop.set_initial_state(
            cp_data.global_step,
            cp_data.episodes_played,
            cp_data.total_simulations_run,
        )
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    if loaded_state.buffer_data:
        if components.train_config.USE_PER:
            logger.info("Rebuilding PER SumTree from loaded buffer data...")
            if not hasattr(components.buffer, "tree") or components.buffer.tree is None:
                components.buffer.tree = SumTree(components.buffer.capacity)
            else:
                components.buffer.tree = SumTree(components.buffer.capacity)
            max_p = 1.0
            for exp in loaded_state.buffer_data.buffer_list:
                components.buffer.tree.add(max_p, exp)
            logger.info(f"PER buffer loaded. Size: {len(components.buffer)}")
        else:
            components.buffer.buffer = deque(
                loaded_state.buffer_data.buffer_list,
                maxlen=components.buffer.capacity,
            )
            logger.info(f"Uniform buffer loaded. Size: {len(components.buffer)}")
        if training_loop.buffer_fill_progress:
            training_loop.buffer_fill_progress.set_current_steps(len(components.buffer))
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    return training_loop


def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    if not training_loop:
        logger.warning("Cannot save final state: TrainingLoop not available.")
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        # Pass the actor handle directly
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_final=True,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


def _training_loop_thread_func(training_loop: TrainingLoop):
    """Function to run the training loop in a separate thread."""
    logger = logging.getLogger(__name__)  # Get logger within thread
    try:
        logger.info("Training loop thread started.")
        training_loop.initialize_workers()
        training_loop.run()
        logger.info("Training loop thread finished.")
    except Exception as e:
        logger.critical(f"Error in training loop thread: {e}", exc_info=True)
        training_loop.training_exception = e
    finally:
        # Signal the main visualization loop to exit
        try:
            while not visual_state_queue.empty():
                try:
                    visual_state_queue.get_nowait()
                except queue.Empty:
                    break
            visual_state_queue.put(None, timeout=1.0)
        except queue.Full:
            logger.error("Visual queue still full during shutdown.")
        except Exception as e_q:
            logger.error(f"Error putting None signal into visual queue: {e_q}")


def run_training_visual_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline in visual mode."""
    main_thread_exception = None
    train_thread = None
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    file_handler = None
    tee_stdout = None
    tee_stderr = None
    ray_initialized_by_setup = False
    mlflow_run_active = False
    total_params: int | None = None
    trainable_params: int | None = None

    try:
        # --- Setup File Logging & Redirection ---
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "visual"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to: {log_file_path}"
        )
        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )

        if file_handler and hasattr(file_handler, "stream") and file_handler.stream:
            tee_stdout = Tee(
                original_stdout,
                file_handler.stream,
                main_stream_for_fileno=original_stdout,
            )
            tee_stderr = Tee(
                original_stderr,
                file_handler.stream,
                main_stream_for_fileno=original_stderr,
            )
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            print("--- Stdout/Stderr redirected to console and log file ---")
            logger.info("Stdout/Stderr redirected to console and log file.")
        else:
            logger.error(
                "Could not redirect stdout/stderr: File handler stream not available."
            )

        # --- Setup Components (includes Ray init) ---
        components, ray_initialized_by_setup = setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        # --- Initialize MLflow ---
        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)  # Log configs after run starts
            # Log parameter counts after MLflow run starts
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        # --- Load State & Initialize Loop ---
        training_loop = _load_and_apply_initial_state(components)

        # --- Start Training Thread ---
        train_thread = threading.Thread(
            target=_training_loop_thread_func, args=(training_loop,), daemon=True
        )
        train_thread.start()
        logger.info("Training loop thread launched.")

        # --- Initialize Visualization ---
        vis_config = config.VisConfig()
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode(
            (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"{config.APP_NAME} - Training Visual Mode ({components.train_config.RUN_NAME})"
        )
        clock = pygame.time.Clock()
        fonts = visualization.load_fonts()
        # Pass the actor handle directly
        dashboard_renderer = visualization.DashboardRenderer(
            screen,
            vis_config,
            components.env_config,
            fonts,
            components.stats_collector_actor,
            components.model_config,
            total_params=total_params,  # Pass param counts
            trainable_params=trainable_params,
        )

        current_worker_states: dict[int, environment.GameState] = {}
        current_global_stats: dict[str, Any] = {}
        has_received_data = False

        # --- Visualization Loop (Main Thread) ---
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.VIDEORESIZE:
                    try:
                        w, h = max(640, event.w), max(480, event.h)
                        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        dashboard_renderer.screen = screen
                        dashboard_renderer.layout_rects = None
                    except pygame.error as e:
                        logger.error(f"Error resizing window: {e}")

            # Process Visual Queue
            try:
                visual_data = visual_state_queue.get(timeout=0.05)
                if visual_data is None:
                    if train_thread and not train_thread.is_alive():
                        running = False
                        logger.info("Received exit signal from training thread.")
                elif isinstance(visual_data, dict):
                    has_received_data = True
                    global_stats_update = visual_data.pop(-1, {})
                    if isinstance(global_stats_update, dict):
                        if not isinstance(current_global_stats, dict):
                            current_global_stats = {}
                        current_global_stats.update(global_stats_update)
                    else:
                        logger.warning(
                            f"Received non-dict global stats update: {type(global_stats_update)}"
                        )

                    current_worker_states = {
                        k: v
                        for k, v in visual_data.items()
                        if isinstance(k, int)
                        and k >= 0
                        and isinstance(v, environment.GameState)
                    }
                    remaining_items = {
                        k: v
                        for k, v in visual_data.items()
                        if k != -1 and k not in current_worker_states
                    }
                    if remaining_items:
                        logger.warning(
                            f"Unexpected items remaining in visual_data after processing: {remaining_items.keys()}"
                        )
                else:
                    logger.warning(
                        f"Received unexpected item from visual queue: {type(visual_data)}"
                    )
            except queue.Empty:
                pass
            except Exception as q_get_err:
                logger.error(f"Error getting from visual queue: {q_get_err}")
                time.sleep(0.1)

            # Rendering Logic
            screen.fill(visualization.colors.DARK_GRAY)
            if has_received_data:
                try:
                    dashboard_renderer.render(
                        current_worker_states, current_global_stats
                    )
                except Exception as render_err:
                    logger.error(f"Error during rendering: {render_err}", exc_info=True)
                    err_font = fonts.get("help")
                    if err_font:
                        err_surf = err_font.render(
                            f"Render Error: {render_err}",
                            True,
                            visualization.colors.RED,
                        )
                        screen.blit(err_surf, (10, screen.get_height() // 2))
            else:
                help_font = fonts.get("help")
                if help_font:
                    wait_surf = help_font.render(
                        "Waiting for first data from training...",
                        True,
                        visualization.colors.LIGHT_GRAY,
                    )
                    wait_rect = wait_surf.get_rect(
                        center=(screen.get_width() // 2, screen.get_height() // 2)
                    )
                    screen.blit(wait_surf, wait_rect)

            pygame.display.flip()

            # Check Training Thread Status
            if train_thread and not train_thread.is_alive() and running:
                logger.warning("Training loop thread terminated unexpectedly.")
                if training_loop and training_loop.training_exception:
                    logger.error(
                        f"Training thread terminated due to exception: {training_loop.training_exception}"
                    )
                    main_thread_exception = training_loop.training_exception
                running = False

            clock.tick(vis_config.FPS)

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred in visual training script (main thread): {e}"
        )
        traceback.print_exc()
        main_thread_exception = e
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "VIS_FAILED")
                mlflow.log_param("error_message", f"MainThread: {str(e)}")
            except Exception as mlf_err:
                logger.error(f"Failed to log main thread error to MLflow: {mlf_err}")

    finally:
        # Restore stdout/stderr
        if tee_stdout:
            sys.stdout = original_stdout
        if tee_stderr:
            sys.stderr = original_stderr
        print("--- Restored stdout/stderr ---")

        logger.info("Initiating shutdown sequence...")
        if training_loop and not training_loop.stop_requested.is_set():
            training_loop.request_stop()

        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training loop thread to join...")
            train_thread.join(timeout=15.0)
            if train_thread.is_alive():
                logger.error("Training loop thread did not exit gracefully.")

        # --- Cleanup ---
        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            # Save final state
            _save_final_state(training_loop)
            # Cleanup loop actors
            training_loop.cleanup_actors()
            # Determine final status
            if main_thread_exception or training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(
                    main_thread_exception or training_loop.training_exception
                )
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"

        # End MLflow run
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        pygame.quit()
        logger.info("Pygame quit.")

        # Shutdown Ray ONLY if initialized by the setup function in this process
        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by visual runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        # Close file handler
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger = get_root_logger()
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Visual training finished with exit code {exit_code}.")
    return exit_code


File: alphatriangle\training\worker_manager.py
# File: alphatriangle/training/worker_manager.py
import logging
from typing import TYPE_CHECKING

import ray
from pydantic import ValidationError

from ..rl import SelfPlayResult, SelfPlayWorker

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the pool of SelfPlayWorker actors, task submission, and results."""

    def __init__(self, components: "TrainingComponents"):
        self.components = components
        self.train_config = components.train_config
        self.nn = components.nn
        self.stats_collector_actor = components.stats_collector_actor

        self.workers: list[ray.actor.ActorHandle | None] = []
        self.worker_tasks: dict[ray.ObjectRef, int] = {}
        self.active_worker_indices: set[int] = set()

    def initialize_workers(self):
        """Creates the pool of SelfPlayWorker Ray actors."""
        logger.info(
            f"Initializing {self.train_config.NUM_SELF_PLAY_WORKERS} self-play workers..."
        )
        initial_weights = self.nn.get_weights()
        weights_ref = ray.put(initial_weights)
        self.workers = [None] * self.train_config.NUM_SELF_PLAY_WORKERS

        for i in range(self.train_config.NUM_SELF_PLAY_WORKERS):
            try:
                worker = SelfPlayWorker.options(num_cpus=1).remote(
                    actor_id=i,
                    env_config=self.components.env_config,
                    mcts_config=self.components.mcts_config,
                    model_config=self.components.model_config,
                    train_config=self.train_config,
                    stats_collector_actor=self.stats_collector_actor,
                    initial_weights=weights_ref,
                    seed=self.train_config.RANDOM_SEED + i,
                    worker_device_str=self.train_config.WORKER_DEVICE,
                )
                self.workers[i] = worker
                self.active_worker_indices.add(i)
            except Exception as e:
                logger.error(f"Failed to initialize worker {i}: {e}", exc_info=True)

        logger.info(
            f"Initialized {len(self.active_worker_indices)} active self-play workers."
        )
        del weights_ref

    def submit_initial_tasks(self):
        """Submits the first task for each active worker."""
        logger.info("Submitting initial tasks to workers...")
        for worker_idx in self.active_worker_indices:
            self.submit_task(worker_idx)

    def submit_task(self, worker_idx: int):
        """Submits a new run_episode task to a specific worker."""
        if worker_idx not in self.active_worker_indices:
            logger.warning(f"Attempted to submit task to inactive worker {worker_idx}.")
            return
        worker = self.workers[worker_idx]
        if worker:
            try:
                task_ref = worker.run_episode.remote()
                self.worker_tasks[task_ref] = worker_idx
                logger.debug(f"Submitted task to worker {worker_idx}")
            except Exception as e:
                logger.error(
                    f"Failed to submit task to worker {worker_idx}: {e}", exc_info=True
                )
                self.active_worker_indices.discard(worker_idx)
                self.workers[worker_idx] = None
        else:
            logger.error(
                f"Worker {worker_idx} is None during task submission despite being in active set."
            )
            self.active_worker_indices.discard(worker_idx)

    def get_completed_tasks(
        self, timeout: float = 0.1
    ) -> list[tuple[int, SelfPlayResult | Exception]]:
        """
        Checks for completed worker tasks using ray.wait.
        Returns a list of tuples: (worker_id, result_or_exception).
        """
        completed_results: list[tuple[int, SelfPlayResult | Exception]] = []
        if not self.worker_tasks:
            return completed_results

        ready_refs, _ = ray.wait(
            list(self.worker_tasks.keys()), num_returns=1, timeout=timeout
        )

        if not ready_refs:
            return completed_results

        for ref in ready_refs:
            worker_idx = self.worker_tasks.pop(ref, -1)
            if worker_idx == -1 or worker_idx not in self.active_worker_indices:
                logger.warning(
                    f"Received result for unknown or inactive worker task: {ref}"
                )
                continue

            try:
                logger.debug(f"Attempting ray.get for worker {worker_idx} task {ref}")
                result_raw = ray.get(ref)
                logger.debug(f"ray.get succeeded for worker {worker_idx}")
                try:
                    result_validated = SelfPlayResult.model_validate(result_raw)
                    completed_results.append((worker_idx, result_validated))
                    logger.debug(
                        f"Pydantic validation passed for worker {worker_idx} result."
                    )
                except ValidationError as e_val:
                    error_msg = f"Pydantic validation failed for result from worker {worker_idx}: {e_val}"
                    logger.error(error_msg, exc_info=False)
                    logger.debug(f"Invalid data structure received: {result_raw}")
                    completed_results.append((worker_idx, ValueError(error_msg)))
                except Exception as e_other_val:
                    error_msg = f"Unexpected error during result validation for worker {worker_idx}: {e_other_val}"
                    logger.error(error_msg, exc_info=True)
                    completed_results.append((worker_idx, e_other_val))

            except ray.exceptions.RayActorError as e_actor:
                logger.error(
                    f"Worker {worker_idx} actor failed: {e_actor}", exc_info=True
                )
                completed_results.append((worker_idx, e_actor))
                self.workers[worker_idx] = None
                self.active_worker_indices.discard(worker_idx)
            except Exception as e_get:
                logger.error(
                    f"Error getting result from worker {worker_idx} task {ref}: {e_get}",
                    exc_info=True,
                )
                completed_results.append((worker_idx, e_get))

        return completed_results

    # --- CHANGED: Accept global_step ---
    def update_worker_networks(self, global_step: int):
        """Sends the latest network weights and current global_step to all active workers."""
        # --- END CHANGED ---
        active_workers = [
            w
            for i, w in enumerate(self.workers)
            if i in self.active_worker_indices and w is not None
        ]
        if not active_workers:
            return
        logger.debug(f"Updating worker networks for step {global_step}...")
        current_weights = self.nn.get_weights()
        weights_ref = ray.put(current_weights)
        # --- CHANGED: Create separate task lists ---
        set_weights_tasks = [
            worker.set_weights.remote(weights_ref) for worker in active_workers
        ]
        set_step_tasks = [
            worker.set_current_trainer_step.remote(global_step)
            for worker in active_workers
        ]
        # --- END CHANGED ---

        all_tasks = set_weights_tasks + set_step_tasks
        if not all_tasks:
            del weights_ref
            return
        try:
            # Wait for all tasks to complete
            ray.get(all_tasks, timeout=120.0)
            logger.debug(
                f"Worker networks updated for {len(active_workers)} workers to step {global_step}."
            )
            # Logging the update event is now handled in TrainingLoop after this call succeeds
        except ray.exceptions.RayActorError as e:
            logger.error(
                f"A worker actor failed during weight/step update: {e}", exc_info=True
            )
            # Consider attempting to identify and remove the failed worker
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout waiting for workers to update weights/step.")
        except Exception as e:
            logger.error(
                f"Unexpected error updating worker networks/step: {e}", exc_info=True
            )
        finally:
            del weights_ref  # Ensure ref is deleted

    def get_num_active_workers(self) -> int:
        """Returns the number of currently active workers."""
        return len(self.active_worker_indices)

    def get_num_pending_tasks(self) -> int:
        """Returns the number of tasks currently running."""
        return len(self.worker_tasks)

    def cleanup_actors(self):
        """Kills Ray actors associated with this manager."""
        logger.info("Cleaning up WorkerManager actors...")
        for task_ref in list(self.worker_tasks.keys()):
            try:
                ray.cancel(task_ref, force=True)
            except Exception as cancel_e:
                logger.warning(f"Error cancelling task {task_ref}: {cancel_e}")
        self.worker_tasks = {}

        for i, worker in enumerate(self.workers):
            if worker:
                try:
                    ray.kill(worker, no_restart=True)
                    logger.debug(f"Killed worker {i}.")
                except Exception as kill_e:
                    logger.warning(f"Error killing worker {i}: {kill_e}")
        self.workers = []
        self.active_worker_indices = set()
        logger.info("WorkerManager actors cleaned up.")


File: alphatriangle\training\__init__.py
# File: alphatriangle/training/__init__.py
"""
Training module containing the pipeline, loop, components, and utilities
for orchestrating the reinforcement learning training process.
"""

# Core components & classes
from .components import TrainingComponents

# Utilities
from .logging_utils import Tee, get_root_logger, setup_file_logging
from .loop import TrainingLoop
from .loop_helpers import LoopHelpers

# Re-export runner functions
from .runners import (
    run_training_headless_mode,
    run_training_visual_mode,
)
from .setup import setup_training_components

# from .pipeline import TrainingPipeline # REMOVED
from .worker_manager import WorkerManager

# Explicitly define __all__
__all__ = [
    # Core Components
    "TrainingComponents",
    "TrainingLoop",
    # "TrainingPipeline", # REMOVED
    # Helpers & Managers
    "WorkerManager",
    "LoopHelpers",
    "setup_training_components",
    # Runners (re-exported)
    "run_training_headless_mode",
    "run_training_visual_mode",
    # Logging Utilities
    "setup_file_logging",
    "get_root_logger",
    "Tee",
]


File: alphatriangle\utils\geometry.py
def is_point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    """
    Checks if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: Tuple (x, y) representing the point coordinates.
        polygon: List of tuples [(x1, y1), (x2, y2), ...] representing polygon vertices in order.

    Returns:
        True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        # Combine nested if statements
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            # Use ternary operator for xinters calculation
            xinters = ((y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x) if p1y != p2y else x

            # Check if point is on the segment boundary or crosses the ray
            if abs(p1x - p2x) < 1e-9:  # Vertical line segment
                if abs(x - p1x) < 1e-9:
                    return True  # Point is on the vertical segment
            elif abs(x - xinters) < 1e-9:  # Point is exactly on the intersection
                return True  # Point is on the boundary
            elif (
                p1x == p2x or x <= xinters
            ):  # Point is to the left or on a non-horizontal segment
                inside = not inside

        p1x, p1y = p2x, p2y

    # Check if the point is exactly one of the vertices
    for px, py in polygon:
        if abs(x - px) < 1e-9 and abs(y - py) < 1e-9:
            return True

    return inside


File: alphatriangle\utils\helpers.py
# File: alphatriangle/utils/helpers.py
import logging
import random
from typing import cast

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Gets the appropriate torch device based on preference and availability.
    Prioritizes MPS on Mac if 'auto' is selected.
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return torch.device("cuda")
    # --- CHANGED: Prioritize MPS if available and preferred/auto ---
    if (
        device_preference in ["auto", "mps"]
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()  # Check if MPS is built
    ):
        logger.info(f"Using MPS device (Preference: {device_preference}).")
        return torch.device("mps")
    # --- END CHANGED ---
    if device_preference == "cpu":
        logger.info("Using CPU device.")
        return torch.device("cpu")

    # Auto-detection fallback (after MPS check)
    if torch.cuda.is_available():
        logger.info("Auto-selected CUDA device.")
        return torch.device("cuda")
    # Check MPS again in fallback (should have been caught above if available)
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        logger.info("Auto-selected MPS device.")
        return torch.device("mps")

    logger.info("Auto-selected CPU device.")
    return torch.device("cpu")


def set_random_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    # Use NumPy's recommended way to seed the global RNG state if needed,
    # or preferably use a Generator instance. For simplicity here, we seed global.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional: Set deterministic algorithms for CuDNN
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    # --- ADDED: Seed MPS if available ---
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Use torch.mps.manual_seed if available (newer PyTorch versions)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)  # type: ignore
            else:
                # Fallback for older versions if needed, though less common
                # torch.manual_seed(seed) might cover MPS indirectly in some versions
                pass
        except Exception as e:
            logger.warning(f"Could not set MPS seed: {e}")
    # --- END ADDED ---
    logger.info(f"Set random seeds to {seed}")


def format_eta(seconds: float | None) -> str:
    """Formats seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    if seconds > 3600 * 24 * 30:
        return ">1 month"

    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def normalize_color_for_matplotlib(
    color_tuple_0_255: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        valid_color = tuple(max(0, min(255, c)) for c in color_tuple_0_255)
        return cast("tuple[float, float, float]", tuple(c / 255.0 for c in valid_color))
    logger.warning(
        f"Invalid color format for normalization: {color_tuple_0_255}, returning black."
    )
    return (0.0, 0.0, 0.0)


File: alphatriangle\utils\README.md
# File: alphatriangle/utils/README.md
# Utilities Module (`alphatriangle.utils`)

## Purpose and Architecture

This module provides common utility functions and type definitions used across various parts of the AlphaTriangle project. Its goal is to avoid code duplication and provide central definitions for shared concepts.

-   **Helper Functions ([`helpers.py`](helpers.py)):** Contains miscellaneous helper functions:
    -   `get_device`: Determines the appropriate PyTorch device (CPU, CUDA, MPS) based on availability and preference.
    -   `set_random_seeds`: Initializes random number generators for Python, NumPy, and PyTorch for reproducibility.
    -   `format_eta`: Converts a time duration (in seconds) into a human-readable string (HH:MM:SS).
    -   `normalize_color_for_matplotlib`: Converts RGB (0-255) to Matplotlib format (0.0-1.0).
-   **Type Definitions ([`types.py`](types.py)):** Defines common type aliases and `TypedDict`s used throughout the codebase, particularly for data structures passed between modules (like RL components, NN, and environment). This improves code readability and enables better static analysis. Examples include:
    -   `StateType`: A `TypedDict` defining the structure of the state representation passed to the NN and stored in the buffer (e.g., `{'grid': np.ndarray, 'other_features': np.ndarray}`).
    -   `ActionType`: An alias for `int`, representing encoded actions.
    -   `PolicyTargetMapping`: A mapping from `ActionType` to `float`, representing the policy target from MCTS.
    -   `Experience`: A tuple representing `(StateType, PolicyTargetMapping, float)` stored in the replay buffer (the float is the n-step return).
    -   `ExperienceBatch`: A list of `Experience` tuples.
    -   `PolicyValueOutput`: A tuple representing `(PolicyTargetMapping, float)` returned by the NN's `evaluate` method (the float is the expected value).
    -   `PERBatchSample`: A `TypedDict` defining the output of the PER buffer's sample method, including the batch, indices, and importance sampling weights.
    -   `StatsCollectorData`: Type alias for the data structure holding collected statistics (`Dict[str, Deque[Tuple[StepInfo, float]]]`).
    -   `StepInfo`: A `TypedDict` holding step context information (e.g., `global_step`, `buffer_size`).
-   **Geometry Utilities ([`geometry.py`](geometry.py)):** Contains geometric helper functions.
    -   `is_point_in_polygon`: Checks if a 2D point lies inside a given polygon.
-   **Data Structures ([`sumtree.py`](sumtree.py)):**
    -   `SumTree`: A simple SumTree implementation used for Prioritized Experience Replay.

## Exposed Interfaces

-   **Functions:**
    -   `get_device(device_preference: str = "auto") -> torch.device`
    -   `set_random_seeds(seed: int = 42)`
    -   `format_eta(seconds: Optional[float]) -> str`
    -   `normalize_color_for_matplotlib(color_tuple_0_255: tuple[int, int, int]) -> tuple[float, float, float]`
    -   `is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool`
-   **Classes:**
    -   `SumTree`: For PER.
-   **Types:**
    -   `StateType` (TypedDict)
    -   `ActionType` (TypeAlias for `int`)
    -   `PolicyTargetMapping` (TypeAlias for `Mapping[ActionType, float]`)
    -   `Experience` (TypeAlias for `Tuple[StateType, PolicyTargetMapping, float]`)
    -   `ExperienceBatch` (TypeAlias for `List[Experience]`)
    -   `PolicyValueOutput` (TypeAlias for `Tuple[Mapping[ActionType, float], float]`)
    -   `PERBatchSample` (TypedDict)
    -   `StatsCollectorData` (TypeAlias for `Dict[str, Deque[Tuple[StepInfo, float]]]`)
    -   `StepInfo` (TypedDict)

## Dependencies

-   **`torch`**:
    -   Used by `get_device` and `set_random_seeds`.
-   **`numpy`**:
    -   Used by `set_random_seeds` and potentially in type definitions (`np.ndarray`).
-   **Standard Libraries:** `typing`, `random`, `os`, `math`, `logging`, `collections.deque`.

---

**Note:** Please keep this README updated when adding or modifying utility functions or type definitions, especially those used as interfaces between different modules. Accurate documentation is crucial for maintainability.

File: alphatriangle\utils\sumtree.py
import numpy as np

from .types import Experience


class SumTree:
    """
    Simple SumTree implementation for efficient prioritized sampling.
    Stores priorities and allows sampling proportional to priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

        # Tree structure: Stores priorities. Size is 2*capacity - 1.
        # Leaves are indices capacity-1 to 2*capacity-2.
        self.tree = np.zeros(2 * capacity - 1)

        # Data storage: Stores the actual experiences. Size is capacity.
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0  # Points to the next available data slot
        self.n_entries = 0  # Current number of entries in the buffer
        self._max_priority = 1.0  # Track max priority for new entries

    def add(self, priority: float, data: Experience):
        """Adds an experience with a given priority."""
        # Calculate the tree index for the leaf corresponding to the data slot
        tree_idx = self.data_pointer + self.capacity - 1

        # Store the data
        self.data[self.data_pointer] = data

        # Update the tree with the new priority
        self.update(tree_idx, priority)

        # Move data pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Wrap around

        # Update entry count
        if self.n_entries < self.capacity:
            self.n_entries += 1

        # Update max priority seen
        self._max_priority = max(self._max_priority, priority)

    def update(self, tree_idx: int, priority: float):
        """Updates the priority of an experience at a given tree index."""
        # Calculate the change in priority
        change = priority - self.tree[tree_idx]

        # Update the leaf node
        self.tree[tree_idx] = priority

        # Propagate the change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # Move to parent index
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> tuple[int, float, Experience]:
        """Finds the leaf node corresponding to a given value (for sampling)."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If left child index is out of bounds, we've reached a leaf node
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # If the value is less than or equal to the left child's priority sum,
                # go down the left branch.
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                # Otherwise, subtract the left child's sum and go down the right branch.
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        # Calculate the corresponding data index in the self.data array
        data_idx = leaf_idx - self.capacity + 1

        # Return the tree index, the priority at that leaf, and the data
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Returns the total priority (root node value)."""
        # Ensure return type is float
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns the maximum priority seen so far."""
        # Return 1.0 if buffer is empty to avoid issues with initial adds
        return self._max_priority if self.n_entries > 0 else 1.0

    def __len__(self) -> int:
        return self.n_entries


File: alphatriangle\utils\types.py
# File: alphatriangle/utils/types.py
from collections import deque
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict


class StateType(TypedDict):
    grid: np.ndarray  # (C, H, W) float32
    other_features: np.ndarray  # (OtherFeatDim,) float32


# Action representation (integer index)
ActionType = int

# Policy target from MCTS (visit counts distribution)
# Mapping from action index to its probability (normalized visit count)
PolicyTargetMapping = Mapping[ActionType, float]


# --- ADDED: Step Information Dictionary ---
class StepInfo(TypedDict, total=False):
    """Dictionary to hold various step counters associated with a metric."""

    global_step: int
    buffer_size: int
    game_step_index: int  # Index within an episode or similar sequence
    # Add other relevant step types if needed


# --- END ADDED ---


# Experience tuple stored in buffer
# NOW stores the extracted StateType (features) instead of the raw GameState object.
# Kept as Tuple for performance in buffer operations.
# The third element (float) represents the calculated n-step return (G_t^n)
# starting from the state represented by the first element (StateType).
# This G_t^n is used by the Trainer to construct the target value distribution.
Experience = tuple[StateType, PolicyTargetMapping, float]


# Batch of experiences for training
ExperienceBatch = list[Experience]

# Output type from the neural network's evaluate method
# (Policy Mapping, Value Estimate)
# Kept as Tuple for performance.
# The second element (float) is the EXPECTED value calculated from the
# predicted value distribution (used for MCTS). The Trainer uses the raw logits.
PolicyValueOutput = tuple[Mapping[ActionType, float], float]


# Type alias for the data structure holding collected statistics
# --- CHANGED: Stores StepInfo dict instead of single step int ---
# Maps metric name to a deque of (step_info_dict, value) tuples
StatsCollectorData = dict[str, deque[tuple[StepInfo, float]]]
# --- END CHANGED ---

# --- Pydantic Models for Data Transfer ---
# SelfPlayResult moved to alphatriangle/rl/types.py to resolve circular import


# --- Prioritized Experience Replay Types ---
# TypedDict for the output of the PER buffer's sample method
class PERBatchSample(TypedDict):
    batch: ExperienceBatch
    indices: np.ndarray
    weights: np.ndarray


File: alphatriangle\utils\__init__.py
from .geometry import is_point_in_polygon
from .helpers import (
    format_eta,
    get_device,
    normalize_color_for_matplotlib,
    set_random_seeds,
)
from .sumtree import SumTree
from .types import (
    ActionType,
    Experience,
    ExperienceBatch,
    PERBatchSample,
    PolicyValueOutput,
    StateType,
    StatsCollectorData,
)

__all__ = [
    # helpers
    "get_device",
    "set_random_seeds",
    "format_eta",
    "normalize_color_for_matplotlib",
    # types
    "StateType",
    "ActionType",
    "Experience",
    "ExperienceBatch",
    "PolicyValueOutput",
    "StatsCollectorData",
    "PERBatchSample",
    # geometry
    "is_point_in_polygon",
    # structures
    "SumTree",
]


File: alphatriangle\visualization\README.md
# File: alphatriangle/visualization/README.md
# Visualization Module (`alphatriangle.visualization`)

## Purpose and Architecture

This module is responsible for rendering the game state visually using the Pygame library. It provides components for drawing the grid, shapes, previews, HUD elements, and statistics plots. **In training visualization mode, it now renders the states of multiple self-play workers in a grid layout alongside plots and progress bars (with specific information displayed on each bar).**

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Visualizer`: Orchestrates the rendering process for interactive modes ("play", "debug"). It manages the layout, calls drawing functions, and handles hover/selection states specific to visualization.
    -   `GameRenderer`: **Adapted renderer** for displaying **multiple** game states and statistics during training visualization (`run_training_visual.py`). It uses `layout.py` to divide the screen. It renders worker game states in one area and statistics plots/progress bars in another. It re-instantiates [`alphatriangle.stats.Plotter`](../stats/plotter.py).
    -   `DashboardRenderer`: Renderer specifically for the **training visualization mode**. It uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects (using `GameRenderer` instances) in the top grid and displays statistics plots (using `alphatriangle.stats.Plotter`) and progress bars in the bottom area. **The training progress bar shows model/parameter info, while the buffer progress bar shows global training stats (updates, episodes, sims, workers).** It takes a dictionary mapping worker IDs to `GameState` objects and a dictionary of global statistics.
    -   `layout`: Calculates the screen positions and sizes for different UI areas (worker grid, stats area, plots).
    -   `fonts`: Loads necessary font files.
    -   `colors`: Defines a centralized palette of RGB color tuples.
    -   `coord_mapper`: Provides functions to map screen coordinates to grid coordinates (`get_grid_coords_from_screen`) and preview indices (`get_preview_index_from_screen`).
-   **Drawing Components ([`drawing/README.md`](drawing/README.md)):**
    -   Contains specific functions for drawing different elements onto Pygame surfaces:
        -   `grid`: Draws the grid background and occupied/empty triangles.
        -   `shapes`: Draws individual shapes (used by previews).
        -   `previews`: Renders the shape preview area.
        -   `hud`: Renders text information like global training stats and help text at the bottom.
        -   `highlight`: Draws debug highlights.
-   **UI Components ([`ui/README.md`](ui/README.md)):**
    -   Contains reusable UI elements like `ProgressBar`.

## Exposed Interfaces

-   **Core Classes & Functions:**
    -   `Visualizer`: Main renderer for interactive modes.
    -   `GameRenderer`: Renderer for a single worker's game state.
    -   `DashboardRenderer`: Renderer for combined multi-game/stats training visualization.
    -   `calculate_interactive_layout`, `calculate_training_layout`: Calculates UI layout rectangles.
    -   `load_fonts`: Loads Pygame fonts.
    -   `colors`: Module containing color constants (e.g., `colors.WHITE`).
    -   `get_grid_coords_from_screen`: Maps screen to grid coordinates.
    -   `get_preview_index_from_screen`: Maps screen to preview index.
-   **Drawing Functions (primarily used internally by Visualizer/GameRenderer but exposed):**
    -   `draw_grid_background`, `draw_grid_triangles`, `draw_grid_indices`
    -   `draw_shape`
    -   `render_previews`, `draw_placement_preview`, `draw_floating_preview`
    -   `render_hud`
    -   `draw_debug_highlight`
-   **UI Components:**
    -   `ProgressBar`: Class for rendering progress bars.
-   **Config:**
    -   `VisConfig`: Configuration class (re-exported from [`alphatriangle.config`](../config/README.md)).

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**:
    -   `VisConfig`, `EnvConfig`, `ModelConfig`: Used extensively for layout, sizing, and coordinate mapping.
-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: The primary object whose state is visualized.
    -   `GridData`: Accessed via `GameState` or passed directly to drawing functions.
-   **[`alphatriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `COLOR_ID_MAP`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **[`alphatriangle.stats`](../stats/README.md)**:
    -   Uses `Plotter` within `DashboardRenderer`.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   Uses `geometry.is_point_in_polygon`, `helpers.format_eta`, `types.StatsCollectorData`.
-   **`pygame`**:
    -   The core library used for all drawing, surface manipulation, event handling (via `interaction`), and font rendering.
-   **`matplotlib`**:
    -   Used by `alphatriangle.stats.Plotter`.
-   **Standard Libraries:** `typing`, `logging`, `math`, `time`.

---

**Note:** Please keep this README updated when changing rendering logic, adding new visual elements, modifying layout calculations, or altering the interfaces exposed to other modules (like `interaction` or the main application scripts). Accurate documentation is crucial for maintainability.

File: alphatriangle\visualization\utils.py
import logging
from typing import cast

logger = logging.getLogger(__name__)


def normalize_color_for_matplotlib(
    color_tuple_0_255: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        # Ensure values are within 0-255 before dividing
        valid_color = tuple(max(0, min(255, c)) for c in color_tuple_0_255)
        # Cast the result to the expected precise tuple type
        return cast("tuple[float, float, float]", tuple(c / 255.0 for c in valid_color))
    logger.warning(
        f"Invalid color format for normalization: {color_tuple_0_255}, returning black."
    )
    return (0.0, 0.0, 0.0)


File: alphatriangle\visualization\__init__.py
"""
Visualization module for rendering the game state using Pygame.
"""

from ..config import VisConfig
from .core import colors
from .core.coord_mapper import (
    get_grid_coords_from_screen,
    get_preview_index_from_screen,
)
from .core.dashboard_renderer import DashboardRenderer
from .core.fonts import load_fonts
from .core.game_renderer import GameRenderer
from .core.layout import (
    calculate_interactive_layout,
    calculate_training_layout,
)
from .core.visualizer import Visualizer
from .drawing.grid import (
    draw_grid_background,
    draw_grid_indices,
    draw_grid_triangles,
)
from .drawing.highlight import draw_debug_highlight
from .drawing.hud import render_hud
from .drawing.previews import (
    draw_floating_preview,
    draw_placement_preview,
    render_previews,
)
from .drawing.shapes import draw_shape
from .ui.progress_bar import ProgressBar

__all__ = [
    # Core Renderers & Layout
    "Visualizer",
    "GameRenderer",
    "DashboardRenderer",
    "calculate_interactive_layout",
    "calculate_training_layout",
    "load_fonts",
    "colors",  # Export colors module
    "get_grid_coords_from_screen",
    "get_preview_index_from_screen",
    # Drawing Functions
    "draw_grid_background",
    "draw_grid_triangles",
    "draw_grid_indices",
    "draw_shape",
    "render_previews",
    "draw_placement_preview",
    "draw_floating_preview",
    "render_hud",
    "draw_debug_highlight",
    # UI Components
    "ProgressBar",
    # Config
    "VisConfig",
]


File: alphatriangle\visualization\core\colors.py
# File: alphatriangle/visualization/core/colors.py
"""Centralized color definitions (RGB tuples 0-255)."""

WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHT_GRAY: tuple[int, int, int] = (180, 180, 180)
GRAY: tuple[int, int, int] = (100, 100, 100)
DARK_GRAY: tuple[int, int, int] = (40, 40, 40)
RED: tuple[int, int, int] = (220, 40, 40)
DARK_RED: tuple[int, int, int] = (100, 10, 10)
BLUE: tuple[int, int, int] = (60, 60, 220)
YELLOW: tuple[int, int, int] = (230, 230, 40)
GREEN: tuple[int, int, int] = (40, 200, 40)
DARK_GREEN: tuple[int, int, int] = (10, 80, 10)
ORANGE: tuple[int, int, int] = (240, 150, 20)
PURPLE: tuple[int, int, int] = (140, 40, 140)
CYAN: tuple[int, int, int] = (40, 200, 200)
LIGHTG: tuple[int, int, int] = (144, 238, 144)
HOTPINK: tuple[int, int, int] = (255, 105, 180)  # Added for plots

GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]

# Game Specific Visuals
GRID_BG_DEFAULT: tuple[int, int, int] = (20, 20, 30)
GRID_BG_GAME_OVER: tuple[int, int, int] = DARK_RED
GRID_LINE_COLOR: tuple[int, int, int] = GRAY
TRIANGLE_EMPTY_COLOR: tuple[int, int, int] = (60, 60, 70)
PREVIEW_BG: tuple[int, int, int] = (30, 30, 40)
PREVIEW_BORDER: tuple[int, int, int] = GRAY
PREVIEW_SELECTED_BORDER: tuple[int, int, int] = BLUE
PLACEMENT_VALID_COLOR: tuple[int, int, int, int] = (*GREEN, 150)  # RGBA
PLACEMENT_INVALID_COLOR: tuple[int, int, int, int] = (*RED, 100)  # RGBA
DEBUG_TOGGLE_COLOR: tuple[int, int, int] = YELLOW

# --- ADDED: Colors for Progress Bar Cycling ---
PROGRESS_BAR_CYCLE_COLORS: list[tuple[int, int, int]] = [
    GREEN,
    BLUE,
    YELLOW,
    ORANGE,
    PURPLE,
    CYAN,
    HOTPINK,
    RED,  # Add red towards the end
]
# --- END ADDED ---


File: alphatriangle\visualization\core\coord_mapper.py
import pygame

from ...config import EnvConfig
from ...structs import Triangle
from ...utils.geometry import is_point_in_polygon


def _calculate_render_params(
    width: int, height: int, config: EnvConfig
) -> tuple[float, float, float, float]:
    """Calculates scale (cw, ch) and offset (ox, oy) for rendering the grid."""
    rows, cols = config.ROWS, config.COLS
    cols_eff = cols * 0.75 + 0.25 if cols > 0 else 1
    scale_w = width / cols_eff if cols_eff > 0 else 1
    scale_h = height / rows if rows > 0 else 1
    scale = max(1.0, min(scale_w, scale_h))
    cell_size = scale
    grid_w_px = cols_eff * cell_size
    grid_h_px = rows * cell_size
    offset_x = (width - grid_w_px) / 2
    offset_y = (height - grid_h_px) / 2
    return cell_size, cell_size, offset_x, offset_y


def get_grid_coords_from_screen(
    screen_pos: tuple[int, int], grid_area_rect: pygame.Rect, config: EnvConfig
) -> tuple[int, int] | None:
    """Maps screen coordinates (relative to screen) to grid row/column."""
    if not grid_area_rect or not grid_area_rect.collidepoint(screen_pos):
        return None

    local_x = screen_pos[0] - grid_area_rect.left
    local_y = screen_pos[1] - grid_area_rect.top
    cw, ch, ox, oy = _calculate_render_params(
        grid_area_rect.width, grid_area_rect.height, config
    )
    if cw <= 0 or ch <= 0:
        return None

    row = int((local_y - oy) / ch) if ch > 0 else -1
    approx_col_center_index = (local_x - ox - cw / 4) / (cw * 0.75) if cw > 0 else -1
    col = int(round(approx_col_center_index))

    for r_check in [row, row - 1, row + 1]:
        if not (0 <= r_check < config.ROWS):
            continue
        for c_check in [col, col - 1, col + 1]:
            if not (0 <= c_check < config.COLS):
                continue
            # Use corrected orientation check
            is_up = (r_check + c_check) % 2 != 0
            temp_tri = Triangle(r_check, c_check, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            if is_point_in_polygon((local_x, local_y), pts):
                return r_check, c_check

    if 0 <= row < config.ROWS and 0 <= col < config.COLS:
        return row, col
    return None


def get_preview_index_from_screen(
    screen_pos: tuple[int, int], preview_rects: dict[int, pygame.Rect]
) -> int | None:
    """Maps screen coordinates to a shape preview index."""
    if not preview_rects:
        return None
    for idx, rect in preview_rects.items():
        if rect and rect.collidepoint(screen_pos):
            return idx
    return None


File: alphatriangle\visualization\core\dashboard_renderer.py
# File: alphatriangle/visualization/core/dashboard_renderer.py
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import pygame
import ray  # Import ray

from ...environment import GameState
from ...stats.plotter import Plotter
from ..drawing import hud as hud_drawing
from ..ui import ProgressBar
from . import colors, layout
from .game_renderer import GameRenderer

if TYPE_CHECKING:
    from ...config import EnvConfig, ModelConfig, VisConfig
    from ...stats import StatsCollectorData


logger = logging.getLogger(__name__)


class DashboardRenderer:
    """
    Renders the training dashboard with minimal spacing.
    Displays worker states, plots, and progress bars with specific info lines.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
        stats_collector_actor: ray.actor.ActorHandle | None = None,
        model_config: Optional["ModelConfig"] = None,
        total_params: int | None = None,
        trainable_params: int | None = None,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.stats_collector_actor = stats_collector_actor
        self.model_config = model_config
        self.total_params = total_params
        self.trainable_params = trainable_params
        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.worker_sub_rects: dict[int, pygame.Rect] = {}
        self.last_worker_grid_size = (0, 0)
        self.last_num_workers = 0

        self.single_game_renderer = GameRenderer(vis_config, env_config, fonts)
        self.plotter = Plotter(plot_update_interval=0.2)

        self.progress_bar_height_per_bar = 25
        self.num_progress_bars = 2
        self.progress_bar_spacing = 2
        self.progress_bars_total_height = (
            (
                (self.progress_bar_height_per_bar * self.num_progress_bars)
                + (self.progress_bar_spacing * (self.num_progress_bars - 1))
            )
            if self.num_progress_bars > 0
            else 0
        )

        self._layout_calculated_for_size: tuple[int, int] = (0, 0)
        # Don't call ensure_layout here, wait for first render

    def ensure_layout(self):
        """Calculates or retrieves the main layout areas."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Pass the calculated total height needed for progress bars
            self.layout_rects = layout.calculate_training_layout(
                current_w,
                current_h,
                self.vis_config,
                progress_bars_total_height=self.progress_bars_total_height,
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated dashboard layout for size {current_size}: {self.layout_rects}"
            )
            self.last_worker_grid_size = (0, 0)
            self.worker_sub_rects = {}
        return self.layout_rects if self.layout_rects is not None else {}

    def _calculate_worker_sub_layout(
        self, worker_grid_area: pygame.Rect, worker_ids: list[int]
    ):
        """Calculates the grid layout within the worker_grid_area with NO padding."""
        area_w, area_h = worker_grid_area.size
        num_workers = len(worker_ids)

        if (
            area_w,
            area_h,
        ) == self.last_worker_grid_size and num_workers == self.last_num_workers:
            return

        logger.debug(
            f"Recalculating worker sub-layout for {num_workers} workers in area {area_w}x{area_h}"
        )
        self.last_worker_grid_size = (area_w, area_h)
        self.last_num_workers = num_workers
        self.worker_sub_rects = {}

        if area_h <= 10 or area_w <= 10 or num_workers <= 0:
            if num_workers > 0:
                logger.warning(
                    f"Worker grid area too small ({area_w}x{area_h}). Cannot calculate sub-layout."
                )
            return

        aspect_ratio = area_w / max(1, area_h)
        cols = math.ceil(math.sqrt(num_workers * aspect_ratio))
        rows = math.ceil(num_workers / cols)

        cols = max(1, cols)
        rows = max(1, rows)

        cell_w = max(1, area_w / cols)
        cell_h = max(1, area_h / rows)

        min_cell_w, min_cell_h = 60, 40
        if cell_w < min_cell_w or cell_h < min_cell_h:
            logger.warning(
                f"Worker grid cells potentially too small ({cell_w:.1f}x{cell_h:.1f})."
            )

        logger.info(
            f"Calculated worker sub-layout (no pad): {rows}x{cols} for {num_workers} workers. Cell: {cell_w:.1f}x{cell_h:.1f}"
        )

        sorted_worker_ids = sorted(worker_ids)
        for i, worker_id in enumerate(sorted_worker_ids):
            row = i // cols
            col = i % cols
            worker_area_x = int(worker_grid_area.left + col * cell_w)
            worker_area_y = int(worker_grid_area.top + row * cell_h)
            worker_w = int((col + 1) * cell_w) - int(col * cell_w)
            worker_h = int((row + 1) * cell_h) - int(row * cell_h)

            worker_rect = pygame.Rect(worker_area_x, worker_area_y, worker_w, worker_h)
            self.worker_sub_rects[worker_id] = worker_rect.clip(worker_grid_area)

    def render(
        self,
        worker_states: dict[int, GameState],
        global_stats: dict[str, Any] | None = None,
    ):
        """Renders the entire training dashboard."""
        self.screen.fill(colors.DARK_GRAY)
        layout_rects = self.ensure_layout()
        if not layout_rects:
            return

        worker_grid_area = layout_rects.get("worker_grid")
        plots_rect = layout_rects.get("plots")
        progress_bar_area_rect = layout_rects.get("progress_bar_area")

        worker_step_stats = (
            global_stats.get("worker_step_stats", {}) if global_stats else {}
        )

        # --- Render Worker Grid Area ---
        if (
            worker_grid_area
            and worker_grid_area.width > 0
            and worker_grid_area.height > 0
        ):
            worker_ids = list(worker_states.keys())
            if not worker_ids and global_stats and "num_workers" in global_stats:
                worker_ids = list(range(global_stats["num_workers"]))

            self._calculate_worker_sub_layout(worker_grid_area, worker_ids)

            for worker_id in self.worker_sub_rects:
                worker_area_rect = self.worker_sub_rects[worker_id]
                game_state = worker_states.get(worker_id)
                step_stats = worker_step_stats.get(worker_id)
                self.single_game_renderer.render_worker_state(
                    self.screen,
                    worker_area_rect,
                    worker_id,
                    game_state,
                    worker_step_stats=step_stats,
                )
                pygame.draw.rect(self.screen, colors.GRAY, worker_area_rect, 1)
        else:
            logger.warning("Worker grid area not available or too small.")

        # --- Render Plot Area ---
        if global_stats:
            plot_surface = None
            if plots_rect and plots_rect.width > 0 and plots_rect.height > 0:
                stats_data_for_plot: StatsCollectorData | None = global_stats.get(
                    "stats_data"
                )
                if stats_data_for_plot is not None:
                    has_any_metric_data = any(
                        isinstance(dq, deque) and dq
                        for dq in stats_data_for_plot.values()
                    )
                    if has_any_metric_data:
                        plot_surface = self.plotter.get_plot_surface(
                            stats_data_for_plot,
                            int(plots_rect.width),
                            int(plots_rect.height),
                        )
                    else:
                        logger.debug(
                            "Plot data received but all metric deques are empty."
                        )
                else:
                    logger.debug(
                        "No 'stats_data' key found in global_stats for plotting."
                    )

                if plot_surface:
                    self.screen.blit(plot_surface, plots_rect.topleft)
                else:
                    pygame.draw.rect(self.screen, colors.DARK_GRAY, plots_rect)
                    plot_font = self.fonts.get("help")
                    if plot_font:
                        wait_text = (
                            "Plot Area (Waiting for data...)"
                            if stats_data_for_plot is None
                            else "Plot Area (No data yet)"
                        )
                        wait_surf = plot_font.render(wait_text, True, colors.LIGHT_GRAY)
                        wait_rect = wait_surf.get_rect(center=plots_rect.center)
                        self.screen.blit(wait_surf, wait_rect)
                    pygame.draw.rect(self.screen, colors.GRAY, plots_rect, 1)

            # --- Render Progress Bars (in their dedicated area) ---
            if progress_bar_area_rect:
                current_y = (
                    progress_bar_area_rect.top
                )  # Start at the top of the PB area
                progress_bar_font = self.fonts.get("help")

                if progress_bar_font:
                    bar_width = progress_bar_area_rect.width  # Use the area width
                    bar_x = progress_bar_area_rect.left
                    bar_height = self.progress_bar_height_per_bar

                    # --- Generate Info Strings for Each Bar ---
                    train_bar_info_str = ""
                    buffer_bar_info_str = ""

                    # Info for Training Bar (Model/Params)
                    train_info_parts = []
                    if self.model_config:
                        model_str = f"CNN:{len(self.model_config.CONV_FILTERS)}L"
                        if self.model_config.NUM_RESIDUAL_BLOCKS > 0:
                            model_str += (
                                f"/Res:{self.model_config.NUM_RESIDUAL_BLOCKS}L"
                            )
                        if (
                            self.model_config.USE_TRANSFORMER
                            and self.model_config.TRANSFORMER_LAYERS > 0
                        ):
                            model_str += f"/TF:{self.model_config.TRANSFORMER_LAYERS}L"
                        train_info_parts.append(model_str)
                    if self.total_params is not None:
                        train_info_parts.append(
                            f"Params:{self.total_params / 1e6:.1f}M"
                        )
                    train_bar_info_str = " | ".join(train_info_parts)

                    # Info for Buffer Bar (Weight Updates, Episodes, Sims, Workers)
                    buffer_info_parts = []
                    # Use .get with default '?' for robustness
                    updates = global_stats.get("worker_weight_updates", "?")
                    episodes = global_stats.get("total_episodes", "?")
                    sims = global_stats.get("total_simulations_run", "?")  # Correct key
                    num_workers = global_stats.get("num_workers", "?")
                    pending_tasks = global_stats.get("pending_tasks", "?")

                    buffer_info_parts.append(f"Weight Updates:{updates}")
                    buffer_info_parts.append(f"Episodes:{episodes}")
                    if isinstance(sims, int | float):
                        sims_str = (
                            f"{sims / 1e6:.1f}M"
                            if sims >= 1e6
                            else (
                                f"{sims / 1e3:.0f}k" if sims >= 1000 else str(int(sims))
                            )
                        )
                        buffer_info_parts.append(f"Simulations:{sims_str}")
                    else:
                        buffer_info_parts.append(f"Simulations:{sims}")

                    # --- CHANGED: Robust worker status formatting ---
                    if isinstance(pending_tasks, int) and isinstance(num_workers, int):
                        buffer_info_parts.append(
                            f"Workers:{pending_tasks}/{num_workers}"
                        )
                    else:
                        buffer_info_parts.append(
                            f"Workers:{pending_tasks or '?'}/{num_workers or '?'}"
                        )
                    # --- END CHANGED ---

                    buffer_bar_info_str = " | ".join(buffer_info_parts)
                    # --- End Generate Info Strings ---

                    # Training Progress Bar (with model/param info)
                    train_progress = global_stats.get("train_progress")
                    if isinstance(train_progress, ProgressBar):
                        train_progress.render(
                            self.screen,
                            (bar_x, current_y),
                            int(bar_width),
                            bar_height,
                            progress_bar_font,
                            border_radius=3,
                            info_line=train_bar_info_str,  # Pass specific info
                        )
                        current_y += bar_height + self.progress_bar_spacing
                    else:
                        logger.debug(
                            "Train progress bar data not available or invalid type."
                        )

                    # Buffer Progress Bar (with global stats info)
                    buffer_progress = global_stats.get("buffer_progress")
                    if isinstance(buffer_progress, ProgressBar):
                        buffer_progress.render(
                            self.screen,
                            (bar_x, current_y),
                            int(bar_width),
                            bar_height,
                            progress_bar_font,
                            border_radius=3,
                            info_line=buffer_bar_info_str,  # Pass specific info
                        )
                    else:
                        logger.debug(
                            "Buffer progress bar data not available or invalid type."
                        )

        elif not global_stats:
            logger.debug("No global_stats provided to DashboardRenderer.")

        # --- Render HUD (Help Text Only) ---
        hud_drawing.render_hud(
            self.screen,
            mode="training_visual",
            fonts=self.fonts,
            display_stats=None,
        )


File: alphatriangle\visualization\core\fonts.py
import logging

import pygame

logger = logging.getLogger(__name__)

DEFAULT_FONT_NAME = None
FALLBACK_FONT_NAME = "arial,freesans"


def load_single_font(name: str | None, size: int) -> pygame.font.Font | None:
    """Loads a single font, handling potential errors."""
    try:
        font = pygame.font.SysFont(name, size)
        return font
    except Exception as e:
        logger.error(f"Error loading font '{name}' size {size}: {e}")
        if name != FALLBACK_FONT_NAME:
            logger.warning(f"Attempting fallback font: {FALLBACK_FONT_NAME}")
            try:
                font = pygame.font.SysFont(FALLBACK_FONT_NAME, size)
                logger.info(f"Loaded fallback font: {FALLBACK_FONT_NAME} size {size}")
                return font
            except Exception as e_fallback:
                logger.error(f"Fallback font failed: {e_fallback}")
                return None
        return None


def load_fonts(
    font_sizes: dict[str, int] | None = None,
) -> dict[str, pygame.font.Font | None]:
    """Loads standard game fonts."""
    if font_sizes is None:
        font_sizes = {
            "ui": 24,
            "score": 30,
            "help": 18,
            "title": 48,
        }

    fonts: dict[str, pygame.font.Font | None] = {}
    required_fonts = ["score", "help"]

    logger.info("Loading fonts...")
    for name, size in font_sizes.items():
        fonts[name] = load_single_font(DEFAULT_FONT_NAME, size)

    for name in required_fonts:
        if fonts.get(name) is None:
            logger.critical(
                f"Essential font '{name}' failed to load. Text rendering will be affected."
            )

    return fonts


File: alphatriangle\visualization\core\game_renderer.py
import logging
from typing import TYPE_CHECKING, Any

import pygame

from ...environment import GameState
from ..drawing import grid as grid_drawing
from ..drawing import previews as preview_drawing
from . import colors

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig

logger = logging.getLogger(__name__)


class GameRenderer:
    """
    Renders a single GameState (grid and previews) within a specified area.
    Used by DashboardRenderer for displaying worker states.
    Also displays latest per-step stats for the worker.
    """

    def __init__(
        self,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
    ):
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.preview_width_ratio = 0.2
        self.min_preview_width = 30
        self.max_preview_width = 60
        self.padding = 5

    def render_worker_state(
        self,
        target_surface: pygame.Surface,
        area_rect: pygame.Rect,
        worker_id: int,
        game_state: GameState | None,
        # Add worker_step_stats parameter
        worker_step_stats: dict[str, Any] | None = None,
    ):
        """
        Renders the game state of a single worker into the specified area_rect
        on the target_surface. Includes per-step stats display.
        """
        if not game_state:
            # Optionally draw a placeholder if state is None
            pygame.draw.rect(target_surface, colors.DARK_GRAY, area_rect)
            pygame.draw.rect(target_surface, colors.GRAY, area_rect, 1)
            id_font = self.fonts.get("help")
            if id_font:
                id_surf = id_font.render(
                    f"W{worker_id} (No State)", True, colors.LIGHT_GRAY
                )
                id_rect = id_surf.get_rect(center=area_rect.center)
                target_surface.blit(id_surf, id_rect)
            return

        # Calculate layout within the worker's area_rect
        preview_w = max(
            self.min_preview_width,
            min(area_rect.width * self.preview_width_ratio, self.max_preview_width),
        )
        grid_w = max(0, area_rect.width - preview_w - self.padding)
        grid_h = area_rect.height
        preview_h = area_rect.height

        grid_rect_local = pygame.Rect(0, 0, grid_w, grid_h)
        preview_rect_local = pygame.Rect(grid_w + self.padding, 0, preview_w, preview_h)

        # Create subsurfaces relative to the target_surface
        try:
            worker_surface = target_surface.subsurface(area_rect)
            worker_surface.fill(colors.DARK_GRAY)  # Background for the worker area
            pygame.draw.rect(
                target_surface, colors.GRAY, area_rect, 1
            )  # Border around worker area

            # Render Grid
            if grid_rect_local.width > 0 and grid_rect_local.height > 0:
                grid_surf = worker_surface.subsurface(grid_rect_local)
                bg_color = (
                    colors.GRID_BG_GAME_OVER
                    if game_state.is_over()
                    else colors.GRID_BG_DEFAULT
                )
                grid_drawing.draw_grid_background(grid_surf, bg_color)
                grid_drawing.draw_grid_triangles(
                    grid_surf, game_state.grid_data, self.env_config
                )

                # --- Render Worker Info Text ---
                id_font = self.fonts.get("help")
                if id_font:
                    line_y = 3
                    line_height = id_font.get_height() + 1

                    # Worker ID and Game Step
                    id_text = f"W{worker_id} (Step {game_state.current_step})"
                    id_surf = id_font.render(id_text, True, colors.LIGHT_GRAY)
                    grid_surf.blit(id_surf, (3, line_y))
                    line_y += line_height

                    # Current Score
                    score_text = f"Score: {game_state.game_score:.0f}"
                    score_surf = id_font.render(score_text, True, colors.YELLOW)
                    grid_surf.blit(score_surf, (3, line_y))
                    line_y += line_height

                    # MCTS Stats (if available)
                    if worker_step_stats:
                        visits = worker_step_stats.get("mcts_visits", "?")
                        depth = worker_step_stats.get("mcts_depth", "?")
                        mcts_text = f"MCTS: V={visits} D={depth}"
                        mcts_surf = id_font.render(mcts_text, True, colors.CYAN)
                        grid_surf.blit(mcts_surf, (3, line_y))
                        line_y += line_height

            # Render Previews
            if preview_rect_local.width > 0 and preview_rect_local.height > 0:
                preview_surf = worker_surface.subsurface(preview_rect_local)
                # Pass 0,0 as topleft because preview_surf is already positioned
                _ = preview_drawing.render_previews(
                    preview_surf,
                    game_state,
                    (0, 0),  # Relative to preview_surf
                    "training_visual",  # Mode context
                    self.env_config,
                    self.vis_config,
                    selected_shape_idx=-1,  # No selection in training view
                )

        except ValueError as e:
            # Handle cases where subsurface creation fails (e.g., zero dimensions)
            if "subsurface rectangle is invalid" not in str(e):
                logger.error(
                    f"Error creating subsurface for W{worker_id} in area {area_rect}: {e}"
                )
            # Draw error indicator if subsurface fails
            pygame.draw.rect(target_surface, colors.RED, area_rect, 2)
            id_font = self.fonts.get("help")
            if id_font:
                id_surf = id_font.render(f"W{worker_id} (Render Err)", True, colors.RED)
                id_rect = id_surf.get_rect(center=area_rect.center)
                target_surface.blit(id_surf, id_rect)


File: alphatriangle\visualization\core\layout.py
# File: alphatriangle/visualization/core/layout.py
# Changes:
# - Position progress_bar_area_rect precisely above the HUD.
# - Calculate plot_rect height to fill the gap between worker grid and progress bars.

import logging

import pygame

from ...config import VisConfig

logger = logging.getLogger(__name__)


def calculate_interactive_layout(
    screen_width: int, screen_height: int, vis_config: VisConfig
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for interactive modes (play/debug).
    Places grid on the left and preview on the right.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    hud_h = vis_config.HUD_HEIGHT
    preview_w = vis_config.PREVIEW_AREA_WIDTH

    available_h = max(0, sh - hud_h - 2 * pad)
    available_w = max(0, sw - 3 * pad)

    grid_w = max(0, available_w - preview_w)
    grid_h = available_h

    grid_rect = pygame.Rect(pad, pad, grid_w, grid_h)
    preview_rect = pygame.Rect(grid_rect.right + pad, pad, preview_w, grid_h)

    screen_rect = pygame.Rect(0, 0, sw, sh)
    grid_rect = grid_rect.clip(screen_rect)
    preview_rect = preview_rect.clip(screen_rect)

    logger.debug(
        f"Interactive Layout calculated: Grid={grid_rect}, Preview={preview_rect}"
    )

    return {
        "grid": grid_rect,
        "preview": preview_rect,
    }


def calculate_training_layout(
    screen_width: int,
    screen_height: int,
    vis_config: VisConfig,
    progress_bars_total_height: int,  # Height needed for progress bars
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for training visualization mode. MINIMAL SPACING.
    Worker grid top, progress bars bottom (above HUD), plots fill middle.
    """
    sw, sh = screen_width, screen_height
    pad = 2  # Minimal padding
    hud_h = vis_config.HUD_HEIGHT

    # --- Worker Grid Area (Top) ---
    # Calculate available height excluding HUD and minimal padding
    total_available_h_for_grid_plots_bars = max(0, sh - hud_h - 2 * pad)
    top_area_h = min(
        int(total_available_h_for_grid_plots_bars * 0.10), 80
    )  # 10% or 80px max
    top_area_w = sw - 2 * pad
    worker_grid_rect = pygame.Rect(pad, pad, top_area_w, top_area_h)

    # --- Progress Bar Area (Bottom, above HUD) ---
    # Position it precisely based on its required height
    pb_area_y = sh - hud_h - pad - progress_bars_total_height
    pb_area_w = sw - 2 * pad
    progress_bar_area_rect = pygame.Rect(
        pad, pb_area_y, pb_area_w, progress_bars_total_height
    )

    # --- Plot Area (Middle) ---
    # Calculate height to fill the gap precisely
    plot_area_y = worker_grid_rect.bottom + pad
    plot_area_w = sw - 2 * pad
    plot_area_h = max(
        0, progress_bar_area_rect.top - plot_area_y - pad
    )  # Fill space between worker grid and progress bars
    plot_rect = pygame.Rect(pad, plot_area_y, plot_area_w, plot_area_h)

    # Clip all rects to screen bounds
    screen_rect = pygame.Rect(0, 0, sw, sh)
    worker_grid_rect = worker_grid_rect.clip(screen_rect)
    plot_rect = plot_rect.clip(screen_rect)
    progress_bar_area_rect = progress_bar_area_rect.clip(screen_rect)

    logger.debug(
        f"Training Layout calculated (Compact V3): WorkerGrid={worker_grid_rect}, PlotRect={plot_rect}, ProgressBarArea={progress_bar_area_rect}"
    )

    return {
        "worker_grid": worker_grid_rect,
        "plots": plot_rect,
        "progress_bar_area": progress_bar_area_rect,  # Use this rect for drawing PBs
    }


calculate_layout = calculate_training_layout


File: alphatriangle\visualization\core\README.md
# File: alphatriangle/visualization/core/README.md
# Visualization Core Submodule (`alphatriangle.visualization.core`)

## Purpose and Architecture

This submodule contains the central classes and foundational elements for the visualization system. It orchestrates rendering, manages layout and coordinate systems, and defines core visual properties like colors and fonts.

-   **Render Orchestration:**
    -   [`Visualizer`](visualizer.py): The main class for rendering in **interactive modes** ("play", "debug"). It maintains the Pygame screen, calculates layout using `layout.py`, manages cached preview area rectangles, and calls appropriate drawing functions from [`alphatriangle.visualization.drawing`](../drawing/README.md). **It receives interaction state (hover position, selected index) via its `render` method to display visual feedback.**
    -   [`GameRenderer`](game_renderer.py): **Simplified renderer** responsible for drawing a **single** worker's `GameState` (grid and previews) within a specified sub-rectangle. Used by the `DashboardRenderer`.
    -   [`DashboardRenderer`](dashboard_renderer.py): Renderer specifically for the **training visualization mode**. It uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects (using `GameRenderer` instances) in the top grid and displays statistics plots (using [`alphatriangle.stats.Plotter`](../../stats/plotter.py)) and progress bars in the bottom area. **The training progress bar shows model/parameter info, while the buffer progress bar shows global training stats (worker weight updates, episodes, sims, worker status). Plots now include black, solid vertical lines (drawn on top) indicating the `global_step` when worker weights were updated, mapped to the appropriate position on each plot's x-axis.** It takes a dictionary mapping worker IDs to `GameState` objects and a dictionary of global statistics.
-   **Layout Management:**
    -   [`layout.py`](layout.py): Contains functions (`calculate_interactive_layout`, `calculate_training_layout`) to determine the size and position of the main UI areas based on the screen dimensions, mode, and `VisConfig`.
-   **Coordinate System:**
    -   [`coord_mapper.py`](coord_mapper.py): Provides essential mapping functions:
        -   `_calculate_render_params`: Internal helper to get scaling and offset for grid rendering.
        -   `get_grid_coords_from_screen`: Converts mouse/screen coordinates into logical grid (row, column) coordinates.
        -   `get_preview_index_from_screen`: Converts mouse/screen coordinates into the index of the shape preview slot being pointed at.
-   **Visual Properties:**
    -   [`colors.py`](colors.py): Defines a centralized palette of named color constants (RGB tuples).
    -   [`fonts.py`](fonts.py): Contains the `load_fonts` function to load and manage Pygame font objects.

## Exposed Interfaces

-   **Classes:**
    -   `Visualizer`: Renderer for interactive modes.
        -   `__init__(...)`
        -   `render(game_state: GameState, mode: str, **interaction_state)`: Renders based on game state and interaction hints.
        -   `ensure_layout() -> Dict[str, pygame.Rect]`
        -   `screen`: Public attribute (Pygame Surface).
        -   `preview_rects`: Public attribute (cached preview area rects).
    -   `GameRenderer`: Renderer for a single worker's game state.
        -   `__init__(...)`
        -   `render_worker_state(target_surface: pygame.Surface, area_rect: pygame.Rect, worker_id: int, game_state: Optional[GameState], worker_step_stats: Optional[Dict[str, Any]])`
    -   `DashboardRenderer`: Renderer for combined multi-game/stats training visualization.
        -   `__init__(...)`
        -   `render(worker_states: Dict[int, GameState], global_stats: Optional[Dict[str, Any]])`
        -   `screen`: Public attribute (Pygame Surface).
-   **Functions:**
    -   `calculate_interactive_layout(...) -> Dict[str, pygame.Rect]`
    -   `calculate_training_layout(...) -> Dict[str, pygame.Rect]`
    -   `load_fonts() -> Dict[str, Optional[pygame.font.Font]]`
    -   `get_grid_coords_from_screen(...) -> Optional[Tuple[int, int]]`
    -   `get_preview_index_from_screen(...) -> Optional[int]`
-   **Modules:**
    -   `colors`: Provides color constants (e.g., `colors.RED`).

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**: `VisConfig`, `EnvConfig`, `ModelConfig`.
-   **[`alphatriangle.environment`](../../environment/README.md)**: `GameState`, `GridData`.
-   **[`alphatriangle.stats`](../../stats/README.md)**: `Plotter`, `StatsCollectorActor`.
-   **[`alphatriangle.utils`](../../utils/README.md)**: `types`, `helpers`.
-   **[`alphatriangle.visualization.drawing`](../drawing/README.md)**: Drawing functions are called by renderers.
-   **[`alphatriangle.visualization.ui`](../ui/README.md)**: `ProgressBar` (used by `DashboardRenderer`).
-   **`pygame`**: Used for surfaces, rectangles, fonts, display management.
-   **`ray`**: Used by `DashboardRenderer` (for actor handle type hint).
-   **Standard Libraries:** `typing`, `logging`, `math`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the core rendering logic, layout calculations, coordinate mapping, or the interfaces of the renderers. Accurate documentation is crucial for maintainability.

File: alphatriangle\visualization\core\visualizer.py
import logging
from typing import TYPE_CHECKING

import pygame

from ...structs import Shape
from ..drawing import grid as grid_drawing
from ..drawing import highlight as highlight_drawing
from ..drawing import hud as hud_drawing
from ..drawing import previews as preview_drawing
from ..drawing.previews import (
    draw_floating_preview,
    draw_placement_preview,
)
from . import colors, layout

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig
    from ...environment.core.game_state import GameState

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Orchestrates rendering of a single game state for interactive modes.
    Receives interaction state (hover, selection) via render parameters.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.preview_rects: dict[int, pygame.Rect] = {}  # Cache preview rects
        self._layout_calculated_for_size: tuple[int, int] = (0, 0)
        self.ensure_layout()  # Initial layout calculation

    def ensure_layout(self) -> dict[str, pygame.Rect]:
        """Returns cached layout or calculates it if needed."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Use the interactive layout calculation
            self.layout_rects = layout.calculate_interactive_layout(
                current_w, current_h, self.vis_config
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated interactive layout for size {current_size}: {self.layout_rects}"
            )
            # Clear preview rect cache when layout changes
            self.preview_rects = {}

        return self.layout_rects if self.layout_rects is not None else {}

    def render(
        self,
        game_state: "GameState",
        mode: str,
        # Interaction state passed in:
        selected_shape_idx: int = -1,
        hover_shape: Shape | None = None,
        hover_grid_coord: tuple[int, int] | None = None,
        hover_is_valid: bool = False,
        hover_screen_pos: tuple[int, int] | None = None,
        debug_highlight_coord: tuple[int, int] | None = None,
    ):
        """
        Renders the entire game visualization for interactive modes.
        Uses interaction state passed as parameters for visual feedback.
        """
        self.screen.fill(colors.GRID_BG_DEFAULT)  # Clear screen
        layout_rects = self.ensure_layout()
        grid_rect = layout_rects.get("grid")
        preview_rect = layout_rects.get("preview")

        # Render Grid Area
        if grid_rect and grid_rect.width > 0 and grid_rect.height > 0:
            try:
                grid_surf = self.screen.subsurface(grid_rect)
                self._render_grid_area(
                    grid_surf,
                    game_state,
                    mode,
                    grid_rect,
                    hover_shape,
                    hover_grid_coord,
                    hover_is_valid,
                    hover_screen_pos,
                    debug_highlight_coord,
                )
            except ValueError as e:
                logger.error(f"Error creating grid subsurface ({grid_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, grid_rect, 1)

        # Render Preview Area
        if preview_rect and preview_rect.width > 0 and preview_rect.height > 0:
            try:
                preview_surf = self.screen.subsurface(preview_rect)
                # Pass selected_shape_idx for highlighting
                self._render_preview_area(
                    preview_surf, game_state, mode, preview_rect, selected_shape_idx
                )
            except ValueError as e:
                logger.error(f"Error creating preview subsurface ({preview_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, preview_rect, 1)

        # Render HUD
        # --- CORRECTED CALL ---
        hud_drawing.render_hud(
            surface=self.screen,
            mode=mode,
            fonts=self.fonts,
            display_stats=None,  # Pass None for display_stats in interactive modes
        )
        # --- END CORRECTION ---

    def _render_grid_area(
        self,
        grid_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        grid_rect: pygame.Rect,  # Pass grid_rect for hover calculations
        hover_shape: Shape | None,
        hover_grid_coord: tuple[int, int] | None,
        hover_is_valid: bool,
        hover_screen_pos: tuple[int, int] | None,
        debug_highlight_coord: tuple[int, int] | None,
    ):
        """Renders the main game grid and overlays onto the provided grid_surf."""
        # Background
        bg_color = (
            colors.GRID_BG_GAME_OVER if game_state.is_over() else colors.GRID_BG_DEFAULT
        )
        grid_drawing.draw_grid_background(grid_surf, bg_color)

        # Grid Triangles
        grid_drawing.draw_grid_triangles(
            grid_surf, game_state.grid_data, self.env_config
        )

        # Debug Indices
        if mode == "debug":
            grid_drawing.draw_grid_indices(
                grid_surf, game_state.grid_data, self.env_config, self.fonts
            )

        # Play Mode Hover Previews
        if mode == "play" and hover_shape:
            if hover_grid_coord:  # Snapped preview
                draw_placement_preview(
                    grid_surf,
                    hover_shape,
                    hover_grid_coord[0],
                    hover_grid_coord[1],
                    is_valid=hover_is_valid,  # Use validity passed in
                    config=self.env_config,
                )
            elif hover_screen_pos:  # Floating preview (relative to grid_surf)
                # Adjust screen pos to be relative to grid_surf
                local_hover_pos = (
                    hover_screen_pos[0] - grid_rect.left,
                    hover_screen_pos[1] - grid_rect.top,
                )
                if grid_surf.get_rect().collidepoint(local_hover_pos):
                    draw_floating_preview(
                        grid_surf,
                        hover_shape,
                        local_hover_pos,
                        self.env_config,
                    )

        # Debug Mode Highlight
        if mode == "debug" and debug_highlight_coord:
            r, c = debug_highlight_coord
            highlight_drawing.draw_debug_highlight(grid_surf, r, c, self.env_config)

        # --- ADDED: Display Score in Grid Area for Interactive Modes ---
        score_font = self.fonts.get("score")
        if score_font:
            score_text = f"Score: {game_state.game_score:.0f}"
            score_surf = score_font.render(score_text, True, colors.YELLOW)
            # Position score at top-left of grid area
            score_rect = score_surf.get_rect(topleft=(5, 5))
            grid_surf.blit(score_surf, score_rect)
        # --- END ADDED ---

    def _render_preview_area(
        self,
        preview_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        preview_rect: pygame.Rect,
        selected_shape_idx: int,  # Pass selected index
    ):
        """Renders the shape preview slots onto preview_surf and caches rects."""
        # Pass selected_shape_idx to render_previews for highlighting
        current_preview_rects = preview_drawing.render_previews(
            preview_surf,
            game_state,
            preview_rect.topleft,  # Pass absolute top-left
            mode,
            self.env_config,
            self.vis_config,
            selected_shape_idx=selected_shape_idx,  # Pass selection state
        )
        # Update cache only if it changed (or first time)
        if not self.preview_rects or self.preview_rects != current_preview_rects:
            self.preview_rects = current_preview_rects


File: alphatriangle\visualization\core\__init__.py


File: alphatriangle\visualization\drawing\grid.py
# File: alphatriangle/visualization/drawing/grid.py
from typing import TYPE_CHECKING

import pygame

# Import constants from the structs package directly
from ...structs import COLOR_ID_MAP, DEBUG_COLOR_ID, NO_COLOR_ID, Triangle
from ..core import colors, coord_mapper

if TYPE_CHECKING:
    from ...config import EnvConfig
    from ...environment import GridData


def draw_grid_background(surface: pygame.Surface, bg_color: tuple) -> None:
    """Fills the grid area surface with a background color."""
    surface.fill(bg_color)


def draw_grid_triangles(
    surface: pygame.Surface, grid_data: "GridData", config: "EnvConfig"
) -> None:
    """Draws all triangles (empty, occupied, death) on the grid surface using NumPy state."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Get direct references to NumPy arrays
    occupied_np = grid_data._occupied_np
    death_np = grid_data._death_np
    color_id_np = grid_data._color_id_np

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            is_death = death_np[r, c]
            is_occupied = occupied_np[r, c]
            color_id = color_id_np[r, c]
            is_up = (r + c) % 2 != 0  # Calculate orientation

            color: tuple[int, int, int] | None = None
            border_color = colors.GRID_LINE_COLOR
            border_width = 1

            if is_death:
                color = colors.DARK_GRAY
                border_color = colors.RED
            elif is_occupied:
                if color_id == DEBUG_COLOR_ID:
                    color = colors.DEBUG_TOGGLE_COLOR  # Special debug color
                elif color_id != NO_COLOR_ID and 0 <= color_id < len(COLOR_ID_MAP):
                    color = COLOR_ID_MAP[color_id]
                else:
                    # Fallback if occupied but no valid color ID (shouldn't happen)
                    color = colors.PURPLE  # Error color
            else:  # Empty playable cell
                color = colors.TRIANGLE_EMPTY_COLOR

            # Create temporary Triangle only for geometry calculation
            temp_tri = Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)

            if color:  # Should always be true unless error
                pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, border_color, pts, border_width)


def draw_grid_indices(
    surface: pygame.Surface,
    grid_data: "GridData",
    config: "EnvConfig",
    fonts: dict[str, pygame.font.Font | None],
) -> None:
    """Draws the index number inside each triangle, including death cells."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    font = fonts.get("help")
    if not font:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Get direct references to NumPy arrays
    occupied_np = grid_data._occupied_np
    death_np = grid_data._death_np
    color_id_np = grid_data._color_id_np

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            is_death = death_np[r, c]
            is_occupied = occupied_np[r, c]
            color_id = color_id_np[r, c]
            is_up = (r + c) % 2 != 0  # Calculate orientation

            # Create temporary Triangle only for geometry calculation
            temp_tri = Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            center_x = sum(p[0] for p in pts) / 3
            center_y = sum(p[1] for p in pts) / 3

            text_color = colors.WHITE  # Default

            if is_death:
                text_color = colors.LIGHT_GRAY
            elif is_occupied:
                bg_color: tuple[int, int, int] | None = None
                if color_id == DEBUG_COLOR_ID:
                    bg_color = colors.DEBUG_TOGGLE_COLOR
                elif color_id != NO_COLOR_ID and 0 <= color_id < len(COLOR_ID_MAP):
                    bg_color = COLOR_ID_MAP[color_id]

                if bg_color:
                    brightness = sum(bg_color) / 3
                    text_color = colors.WHITE if brightness < 128 else colors.BLACK
                else:  # Fallback if color missing
                    text_color = colors.RED
            else:  # Empty playable
                bg_color = colors.TRIANGLE_EMPTY_COLOR
                brightness = sum(bg_color) / 3
                text_color = colors.WHITE if brightness < 128 else colors.BLACK

            index = r * config.COLS + c
            text_surf = font.render(str(index), True, text_color)
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            surface.blit(text_surf, text_rect)


File: alphatriangle\visualization\drawing\highlight.py
from typing import TYPE_CHECKING

import pygame

from ...structs import Triangle
from ..core import colors, coord_mapper

if TYPE_CHECKING:
    from ...config import EnvConfig


def draw_debug_highlight(
    surface: pygame.Surface, r: int, c: int, config: "EnvConfig"
) -> None:
    """Highlights a specific triangle border for debugging."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    is_up = (r + c) % 2 != 0
    temp_tri = Triangle(r, c, is_up)
    pts = temp_tri.get_points(ox, oy, cw, ch)

    pygame.draw.polygon(surface, colors.DEBUG_TOGGLE_COLOR, pts, 3)


File: alphatriangle\visualization\drawing\hud.py
# File: alphatriangle/visualization/drawing/hud.py
from typing import Any

import pygame

from ..core import colors
from ..ui import ProgressBar


def render_hud(
    surface: pygame.Surface,
    mode: str,
    fonts: dict[str, pygame.font.Font | None],
    display_stats: dict[str, Any] | None = None,
) -> None:
    """
    Renders global information (like step count, worker status) at the bottom.
    Individual game scores are not shown here anymore.
    Help text is skipped in training_visual mode.
    """
    screen_w, screen_h = surface.get_size()
    help_font = fonts.get("help")
    stats_font = fonts.get("help")  # Use same font for stats line
    step_font = fonts.get("ui") or help_font  # Use UI font for step, fallback to help

    bottom_y = screen_h - 10  # Position from bottom

    stats_rect = None
    step_rect = None

    # --- Render Global Step as "Weight Updates" ---
    if step_font and display_stats:
        train_progress = display_stats.get("train_progress")
        global_step = (
            train_progress.current_steps
            if isinstance(train_progress, ProgressBar)  # Check type
            else display_stats.get("global_step", "?")
        )
        step_text = f"Weight Updates: {global_step}"
        step_surf = step_font.render(step_text, True, colors.YELLOW)
        step_rect = step_surf.get_rect(bottomleft=(15, bottom_y))
        surface.blit(step_surf, step_rect)

    # Render other global training stats if available, positioned after the step count
    if stats_font and display_stats and step_rect:
        stats_items = []
        episodes = display_stats.get("total_episodes", "?")
        sims = display_stats.get("total_simulations", "?")
        num_workers = display_stats.get("num_workers", "?")
        pending_tasks = display_stats.get("pending_tasks", "?")

        stats_items.append(f"Episodes: {episodes}")
        if isinstance(sims, int | float):
            sims_str = (
                f"{sims / 1e6:.2f}M"
                if sims >= 1e6
                else (f"{sims / 1e3:.1f}k" if sims >= 1000 else str(int(sims)))
            )
            stats_items.append(f"Sims: {sims_str}")
        else:
            stats_items.append(f"Sims: {sims}")

        stats_items.append(f"Workers: {pending_tasks}/{num_workers} busy")

        stats_text = " | ".join(stats_items)
        stats_surf = stats_font.render(stats_text, True, colors.CYAN)
        stats_rect = stats_surf.get_rect(bottomleft=(step_rect.right + 20, bottom_y))
        surface.blit(stats_surf, stats_rect)

    # --- CHANGED: Skip help text in training visual mode ---
    if help_font and mode != "training_visual":
        help_text = "[ESC] Quit"
        if mode == "play":
            help_text += " | [Click] Select/Place Shape"
        elif mode == "debug":
            help_text += " | [Click] Toggle Cell"
        # No need for training_visual case here anymore

        help_surf = help_font.render(help_text, True, colors.LIGHT_GRAY)
        help_rect = help_surf.get_rect(bottomright=(screen_w - 15, bottom_y))

        combined_left_width = (
            stats_rect.right if stats_rect else (step_rect.right if step_rect else 0)
        )
        if combined_left_width > help_rect.left - 20:
            help_rect.bottom = (
                stats_rect.top
                if stats_rect
                else (step_rect.top if step_rect else bottom_y)
            ) - 5
            help_rect.right = screen_w - 15

        surface.blit(help_surf, help_rect)
    # --- END CHANGED ---


File: alphatriangle\visualization\drawing\previews.py
import logging
from typing import TYPE_CHECKING

import pygame

from ...structs import Shape, Triangle
from ..core import colors, coord_mapper
from .shapes import draw_shape

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig
    from ...environment import GameState

logger = logging.getLogger(__name__)


def render_previews(
    surface: pygame.Surface,
    game_state: "GameState",
    area_topleft: tuple[int, int],
    _mode: str,
    env_config: "EnvConfig",
    vis_config: "VisConfig",
    selected_shape_idx: int = -1,
) -> dict[int, pygame.Rect]:
    """Renders shape previews in their area. Returns dict {index: screen_rect}."""
    surface.fill(colors.PREVIEW_BG)
    preview_rects_screen: dict[int, pygame.Rect] = {}
    num_slots = env_config.NUM_SHAPE_SLOTS
    pad = vis_config.PREVIEW_PADDING
    inner_pad = vis_config.PREVIEW_INNER_PADDING
    border = vis_config.PREVIEW_BORDER_WIDTH
    selected_border = vis_config.PREVIEW_SELECTED_BORDER_WIDTH

    if num_slots <= 0:
        return {}

    # Calculate dimensions for each slot
    total_pad_h = (num_slots + 1) * pad
    available_h = surface.get_height() - total_pad_h
    slot_h = available_h / num_slots if num_slots > 0 else 0
    slot_w = surface.get_width() - 2 * pad

    current_y = float(pad)  # Start y position as float

    for i in range(num_slots):
        # Calculate local rectangle for the slot within the preview surface
        slot_rect_local = pygame.Rect(pad, int(current_y), int(slot_w), int(slot_h))
        # Calculate screen rectangle by offsetting local rect
        slot_rect_screen = slot_rect_local.move(area_topleft)
        preview_rects_screen[i] = (
            slot_rect_screen  # Store screen rect for interaction mapping
        )

        shape: Shape | None = game_state.shapes[i]
        # Use the passed selected_shape_idx for highlighting
        is_selected = selected_shape_idx == i

        # Determine border style based on selection
        border_width = selected_border if is_selected else border
        border_color = (
            colors.PREVIEW_SELECTED_BORDER if is_selected else colors.PREVIEW_BORDER
        )
        # Draw the border rectangle onto the local preview surface
        pygame.draw.rect(surface, border_color, slot_rect_local, border_width)

        # Draw the shape if it exists
        if shape:
            # Calculate drawing area inside the border and padding
            draw_area_w = slot_w - 2 * (border_width + inner_pad)
            draw_area_h = slot_h - 2 * (border_width + inner_pad)

            if draw_area_w > 0 and draw_area_h > 0:
                # Calculate shape bounding box and required cell size
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_rows = max_r - min_r + 1
                # Effective width considering triangle geometry (0.75 factor)
                shape_cols_eff = (
                    (max_c - min_c + 1) * 0.75 + 0.25 if shape.triangles else 1
                )

                # Determine cell size based on available space and shape dimensions
                scale_w = (
                    draw_area_w / shape_cols_eff if shape_cols_eff > 0 else draw_area_w
                )
                scale_h = draw_area_h / shape_rows if shape_rows > 0 else draw_area_h
                cell_size = max(1.0, min(scale_w, scale_h))  # Use the smaller scale

                # Calculate centered top-left position for drawing the shape
                shape_render_w = shape_cols_eff * cell_size
                shape_render_h = shape_rows * cell_size
                draw_topleft_x = (
                    slot_rect_local.left
                    + border_width
                    + inner_pad
                    + (draw_area_w - shape_render_w) / 2
                )
                draw_topleft_y = (
                    slot_rect_local.top
                    + border_width
                    + inner_pad
                    + (draw_area_h - shape_render_h) / 2
                )

                # Draw the shape onto the local preview surface
                # Cast float coordinates to int for draw_shape
                # Use _is_selected to match the function signature
                draw_shape(
                    surface,
                    shape,
                    (int(draw_topleft_x), int(draw_topleft_y)),
                    cell_size,
                    _is_selected=is_selected,
                    origin_offset=(
                        -min_r,
                        -min_c,
                    ),  # Adjust drawing origin based on bbox
                )

        # Move to the next slot position
        current_y += slot_h + pad

    return preview_rects_screen


def draw_placement_preview(
    surface: pygame.Surface,
    shape: "Shape",
    r: int,
    c: int,
    is_valid: bool,
    config: "EnvConfig",
) -> None:
    """Draws a semi-transparent shape snapped to the grid."""
    if not shape or not shape.triangles:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Use valid/invalid colors (could be passed in or defined here)
    base_color = (
        colors.PLACEMENT_VALID_COLOR[:3]
        if is_valid
        else colors.PLACEMENT_INVALID_COLOR[:3]
    )
    alpha = (
        colors.PLACEMENT_VALID_COLOR[3]
        if is_valid
        else colors.PLACEMENT_INVALID_COLOR[3]
    )
    color = list(base_color) + [alpha]  # Combine RGB and Alpha

    # Use a temporary surface for transparency
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))  # Fully transparent background

    for dr, dc, is_up in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        # Create a temporary Triangle to get points easily
        temp_tri = Triangle(tri_r, tri_c, is_up)
        pts = temp_tri.get_points(ox, oy, cw, ch)
        pygame.draw.polygon(temp_surface, color, pts)

    # Blit the transparent preview onto the main grid surface
    surface.blit(temp_surface, (0, 0))


def draw_floating_preview(
    surface: pygame.Surface,
    shape: "Shape",
    screen_pos: tuple[int, int],  # Position relative to the surface being drawn on
    _config: "EnvConfig",  # Mark config as unused
) -> None:
    """Draws a semi-transparent shape floating at the screen position."""
    if not shape or not shape.triangles:
        return

    cell_size = 20.0  # Fixed size for floating preview? Or scale based on config?
    color = list(shape.color) + [100]  # Base color with fixed alpha

    # Use a temporary surface for transparency
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))

    # Center the shape around the screen_pos
    min_r, min_c, max_r, max_c = shape.bbox()
    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0

    for dr, dc, is_up in shape.triangles:
        # Calculate position relative to shape center and screen_pos
        pt_x = screen_pos[0] + (dc - center_c) * (cell_size * 0.75)
        pt_y = screen_pos[1] + (dr - center_r) * cell_size

        # Create a temporary Triangle at origin to get relative points
        temp_tri = Triangle(0, 0, is_up)
        # Get points relative to 0,0 and scale
        rel_pts = temp_tri.get_points(0, 0, cell_size, cell_size)
        # Translate points to the calculated screen position
        pts = [(px + pt_x, py + pt_y) for px, py in rel_pts]
        pygame.draw.polygon(temp_surface, color, pts)

    # Blit the transparent preview onto the target surface
    surface.blit(temp_surface, (0, 0))


File: alphatriangle\visualization\drawing\README.md
# File: alphatriangle/visualization/drawing/README.md
# Visualization Drawing Submodule (`alphatriangle.visualization.drawing`)

## Purpose and Architecture

This submodule contains specialized functions responsible for drawing specific visual elements of the game onto Pygame surfaces. These functions are typically called by the core renderers (`Visualizer`, `GameRenderer`) in [`alphatriangle.visualization.core`](../core/README.md). Separating drawing logic makes the core renderers cleaner and promotes reusability of drawing code.

-   **[`grid.py`](grid.py):** Functions for drawing the grid background (`draw_grid_background`), the individual triangles within it colored based on occupancy/emptiness (`draw_grid_triangles`), and optional indices (`draw_grid_indices`). Uses `Triangle` from [`alphatriangle.structs`](../../structs/README.md) and `GridData` from [`alphatriangle.environment`](../../environment/README.md).
-   **[`shapes.py`](shapes.py):** Contains `draw_shape`, a function to render a given `Shape` object at a specific location on a surface (used primarily for previews). Uses `Shape` and `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
-   **[`previews.py`](previews.py):** Handles rendering related to shape previews:
    -   `render_previews`: Draws the dedicated preview area, including borders and the shapes within their slots, handling selection highlights. Uses `Shape` from [`alphatriangle.structs`](../../structs/README.md).
    -   `draw_placement_preview`: Draws a semi-transparent version of a shape snapped to the grid, indicating a potential placement location (used in play mode hover). Uses `Shape` and `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
    -   `draw_floating_preview`: Draws a semi-transparent shape directly under the mouse cursor when hovering over the grid but not snapped (used in play mode hover). Uses `Shape` and `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
-   **[`hud.py`](hud.py):** `render_hud` draws Heads-Up Display elements like the game score, help text, and optional training statistics onto the main screen surface.
-   **[`highlight.py`](highlight.py):** `draw_debug_highlight` draws a distinct border around a specific triangle, used for visual feedback in debug mode. Uses `Triangle` from [`alphatriangle.structs`](../../structs/README.md).
-   **[`utils.py`](utils.py):** Contains general drawing utility functions (currently empty).

## Exposed Interfaces

-   **Grid Drawing:**
    -   `draw_grid_background(surface: pygame.Surface, bg_color: tuple)`
    -   `draw_grid_triangles(surface: pygame.Surface, grid_data: GridData, config: EnvConfig)`
    -   `draw_grid_indices(surface: pygame.Surface, grid_data: GridData, config: EnvConfig, fonts: Dict[str, Optional[pygame.font.Font]])`
-   **Shape Drawing:**
    -   `draw_shape(surface: pygame.Surface, shape: Shape, topleft: Tuple[int, int], cell_size: float, is_selected: bool = False, origin_offset: Tuple[int, int] = (0, 0))`
-   **Preview Drawing:**
    -   `render_previews(surface: pygame.Surface, game_state: GameState, area_topleft: Tuple[int, int], mode: str, env_config: EnvConfig, vis_config: VisConfig, selected_shape_idx: int = -1) -> Dict[int, pygame.Rect]`
    -   `draw_placement_preview(surface: pygame.Surface, shape: Shape, r: int, c: int, is_valid: bool, config: EnvConfig)`
    -   `draw_floating_preview(surface: pygame.Surface, shape: Shape, screen_pos: Tuple[int, int], config: EnvConfig)`
-   **HUD Drawing:**
    -   `render_hud(surface: pygame.Surface, mode: str, fonts: Dict[str, Optional[pygame.font.Font]], display_stats: Optional[Dict[str, Any]] = None)`
-   **Highlight Drawing:**
    -   `draw_debug_highlight(surface: pygame.Surface, r: int, c: int, config: EnvConfig)`
-   **Utility Functions:**
    -   (Currently empty or contains other drawing-specific utils)

## Dependencies

-   **[`alphatriangle.visualization.core`](../core/README.md)**:
    -   `colors`: Used extensively for drawing colors.
    -   `coord_mapper`: Used internally (e.g., by `draw_placement_preview`) or relies on its calculations passed in.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`, `VisConfig`: Provide dimensions, padding, etc., needed for drawing calculations.
-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`, `GridData`: Provide the data to be drawn.
-   **[`alphatriangle.structs`](../../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `COLOR_ID_MAP`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **[`alphatriangle.visualization.ui`](../ui/README.md)**:
    -   `ProgressBar` (used by `hud.py`).
-   **`pygame`**:
    -   The core library used for all drawing operations (`pygame.draw.polygon`, `surface.fill`, `surface.blit`, etc.) and font rendering.
-   **Standard Libraries:** `typing`, `logging`, `math`.

---

**Note:** Please keep this README updated when adding new drawing functions, modifying existing ones, or changing their dependencies on configuration or environment data structures. Accurate documentation is crucial for maintainability.

File: alphatriangle\visualization\drawing\shapes.py
import pygame

from ...structs import Shape, Triangle
from ..core import colors


def draw_shape(
    surface: pygame.Surface,
    shape: Shape,
    topleft: tuple[int, int],
    cell_size: float,
    _is_selected: bool = False,
    origin_offset: tuple[int, int] = (0, 0),
) -> None:
    """Draws a single shape onto a surface."""
    if not shape or not shape.triangles or cell_size <= 0:
        return

    shape_color = shape.color
    border_color = colors.GRAY

    cw = cell_size
    ch = cell_size

    for dr, dc, is_up in shape.triangles:
        adj_r, adj_c = dr + origin_offset[0], dc + origin_offset[1]

        tri_x = topleft[0] + adj_c * (cw * 0.75)
        tri_y = topleft[1] + adj_r * ch

        temp_tri = Triangle(0, 0, is_up)
        pts = [(px + tri_x, py + tri_y) for px, py in temp_tri.get_points(0, 0, cw, ch)]

        pygame.draw.polygon(surface, shape_color, pts)
        pygame.draw.polygon(surface, border_color, pts, 1)


File: alphatriangle\visualization\drawing\utils.py


File: alphatriangle\visualization\drawing\__init__.py


File: alphatriangle\visualization\ui\progress_bar.py
# File: alphatriangle/visualization/ui/progress_bar.py
# Changes:
# - Modify render text logic: If info_line is provided, prepend default progress info.
# - Cast return type of _get_pulsing_color to satisfy mypy.

import math
import random
import time
from typing import Any, cast  # Added cast

import pygame

from ...utils import format_eta
from ..core import colors


class ProgressBar:
    """
    A reusable progress bar component for visualization.
    Handles overflow by cycling colors and displaying actual progress count.
    Includes rounded corners and subtle pulsing effect.
    Can display a custom info line alongside default progress text.
    """

    def __init__(
        self,
        entity_title: str,
        total_steps: int,
        start_time: float | None = None,
        initial_steps: int = 0,
        initial_color: tuple[int, int, int] = colors.BLUE,
    ):
        self.entity_title = entity_title
        self._original_total_steps = max(
            1, total_steps if total_steps is not None else 1
        )
        self.initial_steps = max(0, initial_steps)
        self.current_steps = self.initial_steps
        self.start_time = start_time if start_time is not None else time.time()
        self._last_step_time = self.start_time
        self._step_times: list[float] = []
        self.extra_data: dict[str, Any] = {}
        self._current_bar_color = initial_color
        self._last_cycle = -1
        self._rng = random.Random()
        self._pulse_phase = random.uniform(0, 2 * math.pi)

    def add_steps(self, steps_added: int):
        """Adds steps to the progress bar's current count."""
        if steps_added <= 0:
            return
        self.current_steps += steps_added
        self._check_color_cycle()

    def set_current_steps(self, steps: int):
        """Directly sets the current step count."""
        self.current_steps = max(0, steps)
        self._check_color_cycle()

    def _check_color_cycle(self):
        """Updates the bar color if a new cycle is reached."""
        current_cycle = self.current_steps // self._original_total_steps
        if current_cycle > self._last_cycle:
            self._last_cycle = current_cycle
            if current_cycle > 0:
                available_colors = [
                    c
                    for c in colors.PROGRESS_BAR_CYCLE_COLORS
                    if c != self._current_bar_color
                ]
                if not available_colors:
                    available_colors = colors.PROGRESS_BAR_CYCLE_COLORS
                if available_colors:
                    self._current_bar_color = self._rng.choice(available_colors)

    def update_extra_data(self, data: dict[str, Any]):
        """Updates or adds key-value pairs to display."""
        self.extra_data.update(data)

    def reset_time(self):
        """Resets the start time to now, keeping current steps."""
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.initial_steps = self.current_steps

    def reset_all(self, new_total_steps: int | None = None):
        """Resets steps to 0 and start time to now. Optionally updates total steps."""
        self.current_steps = 0
        self.initial_steps = 0
        if new_total_steps is not None:
            self._original_total_steps = max(1, new_total_steps)
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.extra_data = {}
        self._last_cycle = -1
        self._current_bar_color = (
            colors.PROGRESS_BAR_CYCLE_COLORS[0]
            if colors.PROGRESS_BAR_CYCLE_COLORS
            else colors.BLUE
        )

    def get_progress_fraction(self) -> float:
        """Returns progress within the current cycle as a fraction (0.0 to 1.0)."""
        if self._original_total_steps <= 1:
            return 1.0
        if self.current_steps == 0:
            return 0.0
        progress_in_cycle = self.current_steps % self._original_total_steps
        if progress_in_cycle == 0 and self.current_steps > 0:
            return 1.0
        else:
            return progress_in_cycle / self._original_total_steps

    def get_elapsed_time(self) -> float:
        """Returns the time elapsed since the start time."""
        return time.time() - self.start_time

    def get_eta_seconds(self) -> float | None:
        """Calculates the estimated time remaining in seconds."""
        if (
            self._original_total_steps <= 1
            or self.current_steps >= self._original_total_steps
        ):
            return None
        steps_processed = self.current_steps - self.initial_steps
        if steps_processed <= 0:
            return None
        elapsed = self.get_elapsed_time()
        if elapsed < 1.0:
            return None
        speed = steps_processed / elapsed
        if speed < 1e-6:
            return None
        remaining_steps = self._original_total_steps - self.current_steps
        if remaining_steps <= 0:
            return 0.0
        eta = remaining_steps / speed
        return eta

    def _get_pulsing_color(self) -> tuple[int, int, int]:
        """Applies a subtle brightness pulse to the current bar color."""
        base_color = self._current_bar_color
        pulse_amplitude = 15
        brightness_offset = int(
            pulse_amplitude * math.sin(time.time() * 4 + self._pulse_phase)
        )
        # --- CHANGED: Explicitly cast to tuple[int, int, int] ---
        pulsed_color = cast(
            "tuple[int, int, int]",
            tuple(max(0, min(255, c + brightness_offset)) for c in base_color),
        )
        # --- END CHANGED ---
        return pulsed_color

    def render(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        width: int,
        height: int,
        font: pygame.font.Font,
        bg_color: tuple[int, int, int] = colors.DARK_GRAY,
        text_color: tuple[int, int, int] = colors.WHITE,
        border_width: int = 1,
        border_color: tuple[int, int, int] = colors.GRAY,
        border_radius: int = 3,
        info_line: str | None = None,  # Keep optional info_line
    ):
        """Draws the progress bar onto the given surface."""
        x, y = position
        progress_fraction = self.get_progress_fraction()

        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, bg_color, bg_rect, border_radius=border_radius)

        # Pulsing Bar Fill
        fill_width = int(width * progress_fraction)
        if fill_width > 0:
            fill_width = min(width, fill_width)
            fill_rect = pygame.Rect(x, y, fill_width, height)
            pulsing_bar_color = self._get_pulsing_color()
            pygame.draw.rect(
                surface, pulsing_bar_color, fill_rect, border_radius=border_radius
            )

        # Border
        if border_width > 0:
            pygame.draw.rect(
                surface,
                border_color,
                bg_rect,
                border_width,
                border_radius=border_radius,
            )

        # --- Text Rendering (Modified) ---
        line_height = font.get_height()
        if height >= line_height + 4:
            # Always generate default progress text parts
            elapsed_time_str = format_eta(self.get_elapsed_time())
            eta_seconds = self.get_eta_seconds()
            eta_str = format_eta(eta_seconds) if eta_seconds is not None else "--"
            processed_steps = self.current_steps
            expected_steps = self._original_total_steps
            processed_str = (
                f"{processed_steps / 1e6:.1f}M"
                if processed_steps >= 1e6
                else (
                    f"{processed_steps / 1e3:.0f}k"
                    if processed_steps >= 1000
                    else f"{processed_steps:,}"
                )
            )
            expected_str = (
                f"{expected_steps / 1e6:.1f}M"
                if expected_steps >= 1e6
                else (
                    f"{expected_steps / 1e3:.0f}k"
                    if expected_steps >= 1000
                    else f"{expected_steps:,}"
                )
            )
            progress_text = f"{processed_str}/{expected_str}"
            if self._original_total_steps <= 1:
                progress_text = f"{processed_str}"
            extra_text = ""
            if self.extra_data:
                extra_items = [f"{k}:{v}" for k, v in self.extra_data.items()]
                extra_text = " | " + " ".join(extra_items)

            # Construct the display text
            default_progress_info = f"{self.entity_title}: {progress_text} (T:{elapsed_time_str} ETA:{eta_str}){extra_text}"

            # --- CHANGED: Prepend default info if info_line is given ---
            if info_line is not None:
                display_text = (
                    f"{default_progress_info} || {info_line}"  # Combine with separator
                )
            else:
                display_text = default_progress_info  # Use only default
            # --- END CHANGED ---

            # Truncate if necessary
            max_text_width = width - 10
            if font.size(display_text)[0] > max_text_width:
                while (
                    font.size(display_text + "...")[0] > max_text_width
                    and len(display_text) > 20
                ):
                    display_text = display_text[:-1]
                display_text += "..."

            # Render and blit centered vertically
            text_surf = font.render(display_text, True, text_color)
            text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
            surface.blit(text_surf, text_rect)
        # --- End Text Rendering ---


File: alphatriangle\visualization\ui\__init__.py
"""
UI Components subpackage for visualization.
Contains reusable UI elements like progress bars, buttons, etc.
"""

from .progress_bar import ProgressBar

__all__ = [
    "ProgressBar",
]


File: tests\conftest.py
import random
from typing import cast

import numpy as np
import pytest
import torch
import torch.optim as optim

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, MCTSConfig, ModelConfig, TrainConfig
    from alphatriangle.nn import NeuralNetwork
    from alphatriangle.rl import ExperienceBuffer, Trainer
    from alphatriangle.utils.types import Experience, StateType
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, MCTSConfig, ModelConfig, TrainConfig
    from alphatriangle.nn import NeuralNetwork
    from alphatriangle.rl import ExperienceBuffer, Trainer
    from alphatriangle.utils.types import Experience, StateType


# Use default NumPy random number generator
rng = np.random.default_rng()


@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    """Provides a default, *valid* EnvConfig for tests (session-scoped)."""
    # Use a smaller, fully playable grid for easier testing of placement logic
    rows = 3
    cols = 3
    cols_per_row = [cols] * rows
    # Pydantic models with defaults can be instantiated without args
    # if all required fields have defaults or are computed.
    # Let's provide them explicitly for clarity in tests.
    return EnvConfig(
        ROWS=rows,
        COLS=cols,
        COLS_PER_ROW=cols_per_row,
        NUM_SHAPE_SLOTS=1,
        MIN_LINE_LENGTH=3,
    )


@pytest.fixture(scope="session")
def mock_model_config(mock_env_config: EnvConfig) -> ModelConfig:
    """Provides a default ModelConfig compatible with mock_env_config (session-scoped)."""
    # Simple CNN config for testing
    # Pydantic models with defaults can be instantiated without args
    # if all required fields have defaults or are computed.
    # Let's provide them explicitly for clarity in tests.
    action_dim_int = int(mock_env_config.ACTION_DIM)
    return ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4],
        CONV_KERNEL_SIZES=[3],
        CONV_STRIDES=[1],
        CONV_PADDING=[1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=4,
        USE_TRANSFORMER=False,
        TRANSFORMER_DIM=16,
        TRANSFORMER_HEADS=2,
        TRANSFORMER_LAYERS=0,
        TRANSFORMER_FC_DIM=32,
        FC_DIMS_SHARED=[8],
        POLICY_HEAD_DIMS=[action_dim_int],  # Use casted int
        VALUE_HEAD_DIMS=[1],
        OTHER_NN_INPUT_FEATURES_DIM=10,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=True,
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    """Provides a default TrainConfig for tests (session-scoped)."""
    # Pydantic models with defaults can be instantiated without args
    # if all required fields have defaults or are computed.
    return TrainConfig(
        BATCH_SIZE=4,
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        USE_PER=False,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.0,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=100,
        PER_EPSILON=1e-5,
        MAX_TRAINING_STEPS=200,
    )


@pytest.fixture(scope="session")
def mock_mcts_config() -> MCTSConfig:
    """Provides a default MCTSConfig for tests (session-scoped)."""
    # Pydantic models with defaults can be instantiated without args
    return MCTSConfig(
        num_simulations=10,
        puct_coefficient=1.5,
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_anneal_steps=5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_search_depth=10,
    )


# --- Fixtures Moved from tests/mcts/conftest.py ---


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    """Creates a mock StateType dictionary with correct shapes."""
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape, dtype=np.float32),
        "other_features": rng.random(other_shape, dtype=np.float32),
    }


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_experience(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> Experience:
    """Creates a mock Experience tuple."""
    # Cast ACTION_DIM to int
    action_dim = int(mock_env_config.ACTION_DIM)
    policy_target = (
        dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
        else {0: 1.0}
    )
    value_target = random.uniform(-1, 1)
    # Cast StateType to Any temporarily to satisfy Experience type hint if needed
    # (though StateType should match the TypedDict definition)
    return (mock_state_type, policy_target, value_target)


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance with a mock model for testing."""
    device = torch.device("cpu")  # Use CPU for testing
    nn_interface = NeuralNetwork(
        mock_model_config, mock_env_config, mock_train_config, device
    )
    # Optionally replace internal model with a simpler mock if needed,
    # but using the actual AlphaTriangleNet with simple config is often better.
    return nn_interface


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance."""
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture(scope="session")  # Make session-scoped if appropriate
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    """Provides the optimizer from the mock_trainer."""
    # --- CHANGE: Explicitly cast return type ---
    return cast("optim.Optimizer", mock_trainer.optimizer)
    # --- END CHANGE ---


@pytest.fixture  # Buffer should likely be function-scoped unless state doesn't matter
def mock_experience_buffer(mock_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an ExperienceBuffer instance."""
    return ExperienceBuffer(mock_train_config)


@pytest.fixture  # Buffer should likely be function-scoped unless state doesn't matter
def filled_mock_buffer(
    mock_experience_buffer: ExperienceBuffer, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a buffer filled with some mock experiences."""
    for _ in range(mock_experience_buffer.min_size_to_train + 5):
        # Create slightly different experiences
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy(),
            "other_features": mock_experience[0]["other_features"].copy(),
        }
        # Ensure grid is writeable before modifying (copy() already does this)
        state_copy["grid"] += (
            rng.standard_normal(state_copy["grid"].shape, dtype=np.float32) * 0.1
        )
        # Create the new experience tuple
        exp_copy: Experience = (state_copy, mock_experience[1], random.uniform(-1, 1))
        mock_experience_buffer.add(exp_copy)
    return mock_experience_buffer


File: tests\__init__.py


File: tests\environment\conftest.py
import pytest

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig
    from alphatriangle.environment import GameState
    from alphatriangle.structs import Shape
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig
    from alphatriangle.environment import GameState
    from alphatriangle.structs import Shape


# Use session-scoped config from top-level conftest
@pytest.fixture(scope="session")
def default_env_config() -> EnvConfig:
    """Provides the default EnvConfig used in the specification (session-scoped)."""
    # Pydantic models with defaults can be instantiated without args
    return EnvConfig()


@pytest.fixture
def game_state(default_env_config: EnvConfig) -> GameState:
    """Provides a fresh GameState instance for testing."""
    # Use a fixed seed for reproducibility within tests if needed
    return GameState(config=default_env_config, initial_seed=123)


@pytest.fixture
def game_state_with_fixed_shapes(default_env_config: EnvConfig) -> GameState:
    """
    Provides a game state with predictable initial shapes.
    Uses a modified EnvConfig with NUM_SHAPE_SLOTS=3 for this specific fixture.
    """
    # Create a specific config for this fixture
    config_3_slots = default_env_config.model_copy(update={"NUM_SHAPE_SLOTS": 3})
    gs = GameState(config=config_3_slots, initial_seed=456)

    # Override the random shapes with fixed ones for testing placement/refill
    fixed_shapes = [
        Shape([(0, 0, False)], (255, 0, 0)),  # Single down (matches grid at 0,0)
        Shape([(0, 0, True)], (0, 255, 0)),  # Single up (matches grid at 0,1)
        Shape(
            [(0, 0, False), (0, 1, True)], (0, 0, 255)
        ),  # Domino (matches grid at 0,0 and 0,1)
    ]
    # This fixture now guarantees NUM_SHAPE_SLOTS is 3
    assert len(fixed_shapes) == gs.env_config.NUM_SHAPE_SLOTS

    for i in range(len(fixed_shapes)):
        gs.shapes[i] = fixed_shapes[i]
    return gs


@pytest.fixture
def simple_shape() -> Shape:
    """Provides a simple 3-triangle shape (Down, Up, Down)."""
    # Example: L-shape (Down at 0,0; Up at 1,0; Down at 1,1 relative)
    # Grid at (r,c) is Down if r+c is even, Up if odd.
    # (0,0) is Down. (1,0) is Up. (1,1) is Down. This shape matches grid orientation.
    triangles = [(0, 0, False), (1, 0, True), (1, 1, False)]
    color = (255, 0, 0)
    return Shape(triangles, color)


File: tests\environment\test_actions.py
# File: tests/environment/test_actions.py
import pytest

from alphatriangle.config import EnvConfig
from alphatriangle.environment.core.action_codec import decode_action
from alphatriangle.environment.core.game_state import GameState

# Import GridLogic correctly
from alphatriangle.environment.grid import logic as GridLogic
from alphatriangle.environment.logic import actions as ActionLogic
from alphatriangle.structs import Shape

# Fixtures are now implicitly injected from tests/environment/conftest.py


@pytest.fixture
def game_state_almost_full(default_env_config: EnvConfig) -> GameState:
    """
    Provides a game state where only a few placements are possible.
    Grid is filled completely, then specific spots are made empty.
    """
    # Use a fresh GameState to avoid side effects from other tests using the shared 'game_state' fixture
    gs = GameState(config=default_env_config, initial_seed=987)
    # Fill the entire playable grid first using NumPy arrays
    playable_mask = ~gs.grid_data._death_np
    gs.grid_data._occupied_np[playable_mask] = True

    # Explicitly make specific spots empty: (0,4) [Down] and (0,5) [Up]
    empty_spots = [(0, 4), (0, 5)]
    for r_empty, c_empty in empty_spots:
        if gs.grid_data.valid(r_empty, c_empty):
            gs.grid_data._occupied_np[r_empty, c_empty] = False
            # Reset color ID as well
            gs.grid_data._color_id_np[
                r_empty, c_empty
            ] = -1  # Assuming -1 is NO_COLOR_ID
    return gs


# --- Test Action Logic ---
def test_get_valid_actions_initial(game_state: GameState):
    """Test valid actions in an initial empty state."""
    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert isinstance(valid_actions, list)
    assert len(valid_actions) > 0  # Should be many valid actions initially

    # Check if decoded actions are valid placements
    for action_index in valid_actions[:10]:  # Check a sample
        shape_idx, r, c = decode_action(action_index, game_state.env_config)
        shape = game_state.shapes[shape_idx]
        assert shape is not None
        assert GridLogic.can_place(game_state.grid_data, shape, r, c)


def test_get_valid_actions_almost_full(game_state_almost_full: GameState):
    """Test valid actions in a nearly full state with only (0,4) and (0,5) free."""
    gs = game_state_almost_full
    # Ensure cells (0,4) and (0,5) are indeed empty using NumPy array
    assert not gs.grid_data._occupied_np[0, 4], "Cell (0,4) should be empty"
    assert not gs.grid_data._occupied_np[0, 5], "Cell (0,5) should be empty"
    # Check orientation (calculated) - Apply Ruff suggestion
    assert (0 + 4) % 2 == 0, "Cell (0,4) should be Down"  # Changed from not (... != 0)
    assert (0 + 5) % 2 != 0, "Cell (0,5) should be Up"

    # Single down triangle fits at (0,4) [which is Down]
    gs.shapes[0] = Shape([(0, 0, False)], (255, 0, 0))
    # Single up triangle fits at (0,5) [which is Up]
    gs.shapes[1] = Shape([(0, 0, True)], (0, 255, 0))
    # Make other slots empty or contain unfittable shapes
    if len(gs.shapes) > 2:
        gs.shapes[2] = Shape([(0, 0, False), (1, 0, False)], (0, 0, 255))  # Unfittable
    if len(gs.shapes) > 3:
        gs.shapes[3] = None

    valid_actions = ActionLogic.get_valid_actions(gs)

    # Expect fewer valid actions
    assert isinstance(valid_actions, list)
    # The setup should allow placing shape 0 at (0,4) and shape 1 at (0,5)
    assert len(valid_actions) == 2, (
        f"Expected 2 valid actions, found {len(valid_actions)}. Actions: {valid_actions}"
    )

    expected_placements = {(0, 0, 4), (1, 0, 5)}  # (shape_idx, r, c)
    found_placements = set()

    # Check if decoded actions are valid placements in the few remaining spots
    for action_index in valid_actions:
        shape_idx, r, c = decode_action(action_index, gs.env_config)
        shape = gs.shapes[shape_idx]
        assert shape is not None, f"Shape at index {shape_idx} is None"
        assert GridLogic.can_place(gs.grid_data, shape, r, c), (
            f"can_place returned False for action {action_index} -> shape_idx={shape_idx}, r={r}, c={c}"
        )
        # Check if placement is in the expected empty area
        is_expected_placement = (r == 0 and c == 4 and shape_idx == 0) or (
            r == 0 and c == 5 and shape_idx == 1
        )
        assert is_expected_placement, (
            f"Action {action_index} -> {(shape_idx, r, c)} is not one of the expected placements (0,0,4) or (1,0,5)"
        )
        found_placements.add((shape_idx, r, c))

    assert found_placements == expected_placements


def test_get_valid_actions_no_shapes(game_state: GameState):
    """Test valid actions when no shapes are available."""
    game_state.shapes = [None] * game_state.env_config.NUM_SHAPE_SLOTS
    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert valid_actions == []


def test_get_valid_actions_no_space(game_state: GameState):
    """Test valid actions when the grid is completely full (or no space for any shape)."""
    # Fill the entire playable grid using NumPy arrays
    playable_mask = ~game_state.grid_data._death_np
    game_state.grid_data._occupied_np[playable_mask] = True

    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert valid_actions == []


File: tests\environment\test_grid_logic.py
# File: tests/environment/test_grid_logic.py
import pytest

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig
    from alphatriangle.environment.grid import GridData
    from alphatriangle.environment.grid import logic as GridLogic
    from alphatriangle.structs import Shape
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig
    from alphatriangle.environment.grid import GridData
    from alphatriangle.environment.grid import logic as GridLogic
    from alphatriangle.structs import Shape

# Use shared fixtures implicitly via pytest injection
# from .conftest import game_state, simple_shape # Import fixtures if needed


@pytest.fixture
def grid_data(default_env_config: EnvConfig) -> GridData:
    """Provides a fresh GridData instance."""
    return GridData(config=default_env_config)


# --- Test can_place with NumPy GridData ---
def test_can_place_empty_grid(grid_data: GridData, simple_shape: Shape):
    """Test placement on an empty grid."""
    # Place at (2,2). Grid(2,2) is Down (2+2=4, even). Shape(0,0) is Down. OK.
    # Grid(3,2) is Up (3+2=5, odd). Shape(1,0) is Up. OK.
    # Grid(3,3) is Down (3+3=6, even). Shape(1,1) is Down. OK.
    assert GridLogic.can_place(grid_data, simple_shape, 2, 2)


def test_can_place_occupied(grid_data: GridData, simple_shape: Shape):
    """Test placement fails if any target cell is occupied."""
    # Occupy one cell where the shape would go
    target_r, target_c = 3, 2
    grid_data._occupied_np[target_r, target_c] = True
    assert not GridLogic.can_place(grid_data, simple_shape, 2, 2)


# Remove unused simple_shape argument
def test_can_place_death_zone(grid_data: GridData):
    """Test placement fails if any target cell is in a death zone."""
    # Find a death zone cell (e.g., top-left corner in default config)
    death_r, death_c = 0, 0
    assert grid_data._death_np[death_r, death_c]
    # Try placing a single triangle shape there
    single_down_shape = Shape([(0, 0, False)], (255, 0, 0))
    assert not GridLogic.can_place(grid_data, single_down_shape, death_r, death_c)


def test_can_place_orientation_mismatch(grid_data: GridData):
    """Test placement fails if triangle orientations don't match."""
    # Shape: Single UP triangle at its origin (0,0)
    shape_up = Shape([(0, 0, True)], (0, 255, 0))
    # Target grid cell: (0,4), which is DOWN in default config (0+4=4, even)
    target_r_down, target_c_down = 0, 4
    assert grid_data.valid(target_r_down, target_c_down) and not grid_data.is_death(
        target_r_down, target_c_down
    )
    assert not GridLogic.can_place(grid_data, shape_up, target_r_down, target_c_down)

    # Shape: Single DOWN triangle at its origin (0,0)
    shape_down = Shape([(0, 0, False)], (255, 0, 0))
    # Target grid cell: (0,3), which is UP in default config (0+3=3, odd)
    target_r_up, target_c_up = 0, 3
    assert grid_data.valid(target_r_up, target_c_up) and not grid_data.is_death(
        target_r_up, target_c_up
    )
    assert not GridLogic.can_place(grid_data, shape_down, target_r_up, target_c_up)

    # Test valid placement using playable coordinates
    assert GridLogic.can_place(grid_data, shape_down, 0, 4)  # Down on Down at (0,4)
    assert GridLogic.can_place(grid_data, shape_up, 0, 3)  # Up on Up at (0,3)


# --- Test check_and_clear_lines with NumPy GridData ---


def occupy_coords(grid_data: GridData, coords: set[tuple[int, int]]):
    """Helper to occupy specific coordinates."""
    for r, c in coords:
        if grid_data.valid(r, c) and not grid_data._death_np[r, c]:
            grid_data._occupied_np[r, c] = True


def test_check_and_clear_lines_no_clear(grid_data: GridData):
    """Test when newly occupied cells don't complete any lines."""
    newly_occupied = {(2, 2), (3, 2), (3, 3)}  # Coords from simple_shape placement
    occupy_coords(grid_data, newly_occupied)
    lines_cleared, unique_cleared, cleared_lines_set = GridLogic.check_and_clear_lines(
        grid_data, newly_occupied
    )
    assert lines_cleared == 0
    assert not unique_cleared
    assert not cleared_lines_set
    # Check grid state unchanged (except for initial occupation)
    assert grid_data._occupied_np[2, 2]
    assert grid_data._occupied_np[3, 2]
    assert grid_data._occupied_np[3, 3]


def test_check_and_clear_lines_single_line(grid_data: GridData):
    """Test clearing a single horizontal line."""
    # Find a valid horizontal line from the precomputed set
    # Example: Look for a line in row 1 (often has long lines)
    expected_line_coords = None
    for line_fs in grid_data.potential_lines:
        coords = list(line_fs)
        # Check if it's horizontal and in row 1
        if len(coords) >= grid_data.config.MIN_LINE_LENGTH and all(
            r == 1 for r, c in coords
        ):
            expected_line_coords = frozenset(coords)
            break

    assert expected_line_coords is not None, (
        "Could not find a suitable horizontal line in row 1 for testing"
    )
    # line_len = len(expected_line_coords) # Removed unused variable
    coords_list = list(expected_line_coords)

    # Occupy all but one cell in the line
    occupy_coords(grid_data, set(coords_list[:-1]))
    # Occupy the last cell
    last_coord = coords_list[-1]
    newly_occupied = {last_coord}
    occupy_coords(grid_data, newly_occupied)

    lines_cleared, unique_cleared, cleared_lines_set = GridLogic.check_and_clear_lines(
        grid_data, newly_occupied
    )

    assert lines_cleared == 1
    assert unique_cleared == set(expected_line_coords)  # Expect set of coords
    assert cleared_lines_set == {
        expected_line_coords
    }  # Expect set of frozensets of coords

    # Verify the line is now empty in the NumPy array
    for r, c in expected_line_coords:
        assert not grid_data._occupied_np[r, c]


File: tests\environment\test_shape_logic.py
# File: tests/environment/test_shape_logic.py
import random

import pytest

from alphatriangle.environment import GameState
from alphatriangle.environment.shapes import logic as ShapeLogic
from alphatriangle.structs import Shape

# Fixtures are now implicitly injected from tests/environment/conftest.py


@pytest.fixture
def fixed_rng() -> random.Random:
    """Provides a Random instance with a fixed seed."""
    return random.Random(12345)


def test_generate_random_shape(fixed_rng: random.Random):
    """Test generating a single random shape."""
    shape = ShapeLogic.generate_random_shape(fixed_rng)
    assert isinstance(shape, Shape)
    assert shape.triangles is not None
    assert shape.color is not None
    assert len(shape.triangles) > 0
    # Check connectivity (optional but good)
    assert ShapeLogic.is_shape_connected(shape)


def test_generate_multiple_shapes(fixed_rng: random.Random):
    """Test generating multiple shapes to ensure variety (or lack thereof with fixed seed)."""
    shape1 = ShapeLogic.generate_random_shape(fixed_rng)
    # Re-seed or use different rng instance if true randomness is needed per call
    # For this test, using the same fixed_rng will likely produce the same shape again
    shape2 = ShapeLogic.generate_random_shape(fixed_rng)
    # --- REMOVED INCORRECT ASSERTION ---
    # assert shape1 == shape2  # Expect same shape due to fixed seed - THIS IS INCORRECT
    # --- END REMOVED ---
    # Check that subsequent calls produce different results with the same RNG instance
    assert shape1 != shape2, (
        "Two consecutive calls with the same RNG produced the exact same shape (template and color), which is highly unlikely."
    )

    # Use a different seed for variation
    rng2 = random.Random(54321)
    shape3 = ShapeLogic.generate_random_shape(rng2)
    # Check that different RNGs produce different results (highly likely)
    assert shape1 != shape3 or shape1.color != shape3.color


def test_refill_shape_slots_empty(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when all slots are initially empty."""
    game_state.shapes = [None] * game_state.env_config.NUM_SHAPE_SLOTS
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)
    assert len(game_state.shapes) == game_state.env_config.NUM_SHAPE_SLOTS


def test_refill_shape_slots_partial(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when some slots are empty - SHOULD NOT REFILL."""
    num_slots = game_state.env_config.NUM_SHAPE_SLOTS
    if num_slots < 2:
        pytest.skip("Test requires at least 2 shape slots")

    # Start with full slots
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)

    # Empty one slot
    game_state.shapes[0] = None
    # Store original state (important: copy shapes if they are mutable)
    original_shapes = [s.copy() if s else None for s in game_state.shapes]

    # Attempt refill - it should do nothing
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)

    # Check that shapes remain unchanged
    assert game_state.shapes == original_shapes, "Refill happened unexpectedly"


def test_refill_shape_slots_full(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when all slots are already full - SHOULD NOT REFILL."""
    # Start with full slots
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)
    original_shapes = [s.copy() if s else None for s in game_state.shapes]

    # Attempt refill - should do nothing
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)

    # Check shapes are unchanged
    assert game_state.shapes == original_shapes, "Refill happened when slots were full"


def test_refill_shape_slots_batch_trigger(game_state: GameState) -> None:
    """Test that refill only happens when ALL slots are empty."""
    num_slots = game_state.env_config.NUM_SHAPE_SLOTS
    if num_slots < 2:
        pytest.skip("Test requires at least 2 shape slots")

    # Fill all slots initially
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    initial_shapes = [s.copy() if s else None for s in game_state.shapes]
    assert all(s is not None for s in initial_shapes)

    # Empty one slot - refill should NOT happen
    game_state.shapes[0] = None
    shapes_after_one_empty = [s.copy() if s else None for s in game_state.shapes]
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    assert game_state.shapes == shapes_after_one_empty, (
        "Refill happened when only one slot was empty"
    )

    # Empty all slots - refill SHOULD happen
    game_state.shapes = [None] * num_slots
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    assert all(s is not None for s in game_state.shapes), (
        "Refill did not happen when all slots were empty"
    )
    # Check that the shapes are different from the initial ones (probabilistically)
    assert game_state.shapes != initial_shapes, (
        "Shapes after refill are identical to initial shapes (unlikely)"
    )


File: tests\environment\test_step.py
# File: tests/environment/test_step.py
import random
from time import sleep

import pytest

# Import mocker fixture from pytest-mock
from pytest_mock import MockerFixture

from alphatriangle.config import EnvConfig
from alphatriangle.environment.core.game_state import GameState
from alphatriangle.environment.grid import (
    logic as GridLogic,
)
from alphatriangle.environment.grid.grid_data import GridData
from alphatriangle.environment.logic.step import calculate_reward, execute_placement
from alphatriangle.structs import Shape, Triangle

# Fixtures are now implicitly injected from tests/environment/conftest.py


def occupy_line(
    grid_data: GridData, line_indices: list[int], config: EnvConfig
) -> set[Triangle]:
    """Helper to occupy triangles for a given line index list."""
    # occupied_tris: set[Triangle] = set() # Removed unused variable
    for idx in line_indices:
        r, c = divmod(idx, config.COLS)
        # Combine nested if using 'and'
        if grid_data.valid(r, c) and not grid_data.is_death(r, c):
            grid_data._occupied_np[r, c] = True
            # Cannot easily return Triangle objects anymore
    # Return empty set as Triangle objects are not the primary state
    return set()


def occupy_coords(grid_data: GridData, coords: set[tuple[int, int]]):
    """Helper to occupy specific coordinates."""
    for r, c in coords:
        if grid_data.valid(r, c) and not grid_data.is_death(r, c):
            grid_data._occupied_np[r, c] = True


# --- New Reward Calculation Tests (v3) ---


def test_calculate_reward_v3_placement_only(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: only placement, game not over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = set()
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_single_line_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + line clear, game not over."""
    placed_count = len(simple_shape.triangles)
    # Simulate a cleared line of 9 unique coordinates
    unique_coords_cleared: set[tuple[int, int]] = {(0, i) for i in range(9)}
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_multi_line_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + multi-line clear (overlapping coords), game not over."""
    placed_count = len(simple_shape.triangles)
    # Simulate two lines sharing coordinate (0,0)
    line1_coords = {(0, i) for i in range(9)}
    line2_coords = {(i, 0) for i in range(5)}
    unique_coords_cleared = line1_coords.union(line2_coords)  # Union handles uniqueness
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_game_over(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement, no line clear, game IS over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = set()
    is_game_over = True
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + default_env_config.PENALTY_GAME_OVER
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_game_over_with_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + line clear, game IS over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = {(0, i) for i in range(9)}
    is_game_over = True
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.PENALTY_GAME_OVER
    )
    assert reward == pytest.approx(expected_reward)


# --- Test execute_placement with new reward ---


def test_execute_placement_simple_no_refill_v3(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape without clearing lines, verify reward and NO immediate refill."""
    gs = game_state_with_fixed_shapes  # Uses 3 slots, initially filled
    config = gs.env_config
    shape_idx = 0
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]
    shape_to_place = gs.shapes[shape_idx]
    assert shape_to_place is not None
    placed_count = len(shape_to_place.triangles)

    r, c = 2, 2
    assert GridLogic.can_place(gs.grid_data, shape_to_place, r, c)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + survival)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE + config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)
    # Score is still tracked separately
    assert gs.game_score == placed_count

    # Verify grid state using NumPy arrays
    for dr, dc, _ in shape_to_place.triangles:
        tri_r, tri_c = r + dr, c + dc
        assert gs.grid_data._occupied_np[tri_r, tri_c]
        # Cannot easily check color ID without map here, trust placement logic

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1
    assert gs.shapes[2] is original_shape_in_slot_2

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == 0
    assert not gs.is_over()


def test_execute_placement_clear_line_no_refill_v3(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape that clears a line, verify reward and NO immediate refill."""
    gs = game_state_with_fixed_shapes
    config = gs.env_config
    shape_idx = 0
    shape_single_down = gs.shapes[shape_idx]
    assert (
        shape_single_down is not None
        and len(shape_single_down.triangles) == 1
        and not shape_single_down.triangles[0][2]
    )
    placed_count = len(shape_single_down.triangles)
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]

    # Pre-occupy line using coordinates
    # Line indices [3..11] correspond to r=0, c=3 to c=11
    line_coords_to_occupy = {(0, i) for i in range(3, 12) if i != 4}
    occupy_coords(gs.grid_data, line_coords_to_occupy)
    cleared_line_coords = {(0, i) for i in range(3, 12)}  # Coords (0,3) to (0,11)

    r, c = 0, 4  # Placement position
    assert GridLogic.can_place(gs.grid_data, shape_single_down, r, c)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + line clear + survival)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE
        + len(cleared_line_coords) * config.REWARD_PER_CLEARED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)
    # Score is still tracked separately
    assert gs.game_score == placed_count + len(cleared_line_coords) * 2

    # Verify line is cleared using NumPy array
    for row, col in cleared_line_coords:
        assert not gs.grid_data._occupied_np[row, col]

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1
    assert gs.shapes[2] is original_shape_in_slot_2

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == len(cleared_line_coords)
    assert not gs.is_over()


def test_execute_placement_batch_refill_v3(game_state_with_fixed_shapes: GameState):
    """Test that placing the last shape triggers a refill and correct reward."""
    gs = game_state_with_fixed_shapes
    config = gs.env_config
    mock_rng = random.Random(123)

    # Place first shape
    shape_1_coords = (0, 4)
    assert gs.shapes[0] is not None
    placed_count_1 = len(gs.shapes[0].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[0], *shape_1_coords)
    reward1 = execute_placement(gs, 0, 0, 4, mock_rng)
    expected_reward1 = (
        placed_count_1 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward1 == pytest.approx(expected_reward1)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is not None
    assert gs.shapes[2] is not None

    # Place second shape
    shape_2_coords = (0, 3)
    assert gs.shapes[1] is not None
    placed_count_2 = len(gs.shapes[1].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[1], *shape_2_coords)
    reward2 = execute_placement(gs, 1, 0, 3, mock_rng)
    expected_reward2 = (
        placed_count_2 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward2 == pytest.approx(expected_reward2)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is None
    assert gs.shapes[2] is not None

    # Place third shape (triggers refill)
    shape_3_coords = (2, 2)
    assert gs.shapes[2] is not None
    placed_count_3 = len(gs.shapes[2].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[2], *shape_3_coords)
    reward3 = execute_placement(gs, 2, 2, 2, mock_rng)
    expected_reward3 = (
        placed_count_3 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )  # Game not over yet
    assert reward3 == pytest.approx(expected_reward3)
    sleep(0.01)  # Allow time for refill to happen (though it should be synchronous)

    # --- Verify REFILL happened ---
    assert all(s is not None for s in gs.shapes), "Not all slots were refilled"
    assert gs.shapes[0] != Shape([(0, 0, False)], (255, 0, 0))
    assert gs.shapes[1] != Shape([(0, 0, True)], (0, 255, 0))
    assert gs.shapes[2] != Shape([(0, 0, False), (0, 1, True)], (0, 0, 255))

    assert gs.pieces_placed_this_episode == 3
    assert not gs.is_over()


# Add mocker fixture to the test signature
def test_execute_placement_game_over_v3(game_state: GameState, mocker: MockerFixture):
    """Test reward when placement leads to game over, mocking line clears."""
    config = game_state.env_config
    # Fill grid almost completely using NumPy arrays
    playable_mask = ~game_state.grid_data._death_np
    game_state.grid_data._occupied_np[playable_mask] = True

    # Make one spot empty
    empty_r, empty_c = 0, 4
    if not game_state.grid_data.is_death(empty_r, empty_c):  # Ensure it's playable
        game_state.grid_data._occupied_np[empty_r, empty_c] = False

    # Provide a shape that fits the empty spot
    shape_to_place = Shape([(0, 0, False)], (255, 0, 0))  # Single down triangle
    placed_count = len(shape_to_place.triangles)

    # --- Modify setup to prevent refill ---
    unplaceable_shape = Shape([(0, 0, False), (1, 0, False), (2, 0, False)], (1, 1, 1))
    game_state.shapes = [None] * config.NUM_SHAPE_SLOTS
    game_state.shapes[0] = shape_to_place
    if config.NUM_SHAPE_SLOTS > 1:
        game_state.shapes[1] = unplaceable_shape
    # --- End modification ---

    assert GridLogic.can_place(game_state.grid_data, shape_to_place, empty_r, empty_c)
    mock_rng = random.Random(999)

    # --- Mock check_and_clear_lines ---
    # Patch the function within the logic module where execute_placement imports it from
    mock_clear = mocker.patch(
        "alphatriangle.environment.grid.logic.check_and_clear_lines",
        return_value=(0, set(), set()),  # Simulate no lines cleared
    )
    # --- End Mock ---

    # Execute placement - this should fill the last spot and trigger game over
    reward = execute_placement(game_state, 0, empty_r, empty_c, mock_rng)

    # Verify the mock was called (optional but good practice)
    mock_clear.assert_called_once()

    # Verify game is over
    assert game_state.is_over(), (
        "Game should be over after placing the final piece with no other valid moves"
    )

    # Verify reward (placement + game over penalty)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE + config.PENALTY_GAME_OVER
    )
    # Use a slightly larger tolerance if needed, but approx should work
    assert reward == pytest.approx(expected_reward)


File: tests\environment\__init__.py


File: tests\mcts\conftest.py
import random
from collections.abc import Mapping

import numpy as np
import pytest

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig
    from alphatriangle.mcts.core.node import Node
    from alphatriangle.utils.types import ActionType, PolicyValueOutput
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig
    from alphatriangle.mcts.core.node import Node
    from alphatriangle.utils.types import ActionType, PolicyValueOutput


# Use default NumPy random number generator
rng = np.random.default_rng()


# --- Mock GameState ---
class MockGameState:
    """A simplified mock GameState for testing MCTS logic."""

    def __init__(
        self,
        current_step: int = 0,
        is_terminal: bool = False,
        outcome: float = 0.0,
        valid_actions: list[ActionType] | None = None,
        env_config: EnvConfig | None = None,
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        # Use a default EnvConfig if none provided, needed for action dim
        self.env_config = env_config if env_config else EnvConfig()
        # Cast ACTION_DIM to int
        action_dim_int = int(self.env_config.ACTION_DIM)
        self._valid_actions = (
            valid_actions if valid_actions is not None else list(range(action_dim_int))
        )

    def is_over(self) -> bool:
        return self._is_over

    def get_outcome(self) -> float:
        if not self._is_over:
            raise ValueError("Cannot get outcome of non-terminal state.")
        return self._outcome

    def valid_actions(self) -> list[ActionType]:
        return self._valid_actions

    def copy(self) -> "MockGameState":
        # Simple copy for testing, doesn't need full state copy
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,
        )

    def step(self, action: ActionType) -> tuple[float, bool]:
        """
        Simulates taking a step. Returns (reward, done).
        Matches the real GameState.step signature.
        """
        if action not in self.valid_actions():
            raise ValueError(
                f"Invalid action {action} for mock state. Valid: {self.valid_actions()}"
            )
        self.current_step += 1
        # Make terminal condition slightly more complex for testing
        self._is_over = self.current_step >= 10 or len(self._valid_actions) == 0
        self._outcome = 1.0 if self._is_over else 0.0
        # Simulate removing the taken action from valid actions
        if action in self._valid_actions:
            self._valid_actions.remove(action)
        # Simulate removing another random action sometimes
        elif self._valid_actions and random.random() < 0.5:
            self._valid_actions.pop(random.randrange(len(self._valid_actions)))

        # Return dummy reward and the 'done' status
        return 0.0, self._is_over

    def __hash__(self):
        return hash(
            (self.current_step, self._is_over, tuple(sorted(self._valid_actions)))
        )

    def __eq__(self, other):
        if not isinstance(other, MockGameState):
            return NotImplemented
        return (
            self.current_step == other.current_step
            and self._is_over == other._is_over
            and sorted(self._valid_actions) == sorted(other._valid_actions)
            and self.env_config == other.env_config
        )


# --- Mock Network Evaluator ---
class MockNetworkEvaluator:
    """A mock network evaluator for testing MCTS."""

    def __init__(
        self,
        default_policy: Mapping[ActionType, float] | None = None,
        default_value: float = 0.5,
        action_dim: int = 9,
    ):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim  # Store as int
        self.evaluation_history: list[MockGameState] = []
        self.batch_evaluation_history: list[list[MockGameState]] = []

    def _get_policy(self, state: MockGameState) -> Mapping[ActionType, float]:
        if self._default_policy is not None:
            valid_actions = state.valid_actions()
            policy = {
                a: p for a, p in self._default_policy.items() if a in valid_actions
            }
            # Normalize if specified policy doesn't sum to 1 over valid actions
            policy_sum = sum(policy.values())
            if policy_sum > 1e-9 and abs(policy_sum - 1.0) > 1e-6:
                policy = {a: p / policy_sum for a, p in policy.items()}
            elif not policy and valid_actions:  # Handle empty policy for valid actions
                prob = 1.0 / len(valid_actions)
                policy = dict.fromkeys(valid_actions, prob)
            return policy

        # Default uniform policy
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}
        prob = 1.0 / len(valid_actions)
        return dict.fromkeys(valid_actions, prob)

    def evaluate(self, state: MockGameState) -> PolicyValueOutput:
        self.evaluation_history.append(state)
        # Cast ACTION_DIM to int
        self._action_dim = int(state.env_config.ACTION_DIM)
        policy = self._get_policy(state)
        # Create full policy map respecting action_dim
        full_policy = dict.fromkeys(range(self._action_dim), 0.0)
        full_policy.update(policy)
        return full_policy, self._default_value

    def evaluate_batch(self, states: list[MockGameState]) -> list[PolicyValueOutput]:
        self.batch_evaluation_history.append(states)
        results = []
        for state in states:
            # Use single evaluate logic for consistency
            results.append(self.evaluate(state))
        return results


# --- Pytest Fixtures ---
# Session-scoped fixtures moved to top-level tests/conftest.py


@pytest.fixture
def mock_evaluator(mock_env_config: EnvConfig) -> MockNetworkEvaluator:
    """Provides a MockNetworkEvaluator instance configured with the mock EnvConfig."""
    # Cast ACTION_DIM to int
    action_dim_int = int(mock_env_config.ACTION_DIM)
    return MockNetworkEvaluator(action_dim=action_dim_int)


@pytest.fixture
def root_node_mock_state(mock_env_config: EnvConfig) -> Node:
    """Provides a root Node with a MockGameState using the mock EnvConfig."""
    # Cast ACTION_DIM to int
    action_dim_int = int(mock_env_config.ACTION_DIM)
    state = MockGameState(
        valid_actions=list(range(action_dim_int)),
        env_config=mock_env_config,
    )
    # Cast MockGameState to Any temporarily to satisfy Node's type hint
    return Node(state=state)  # type: ignore [arg-type]


@pytest.fixture
def expanded_node_mock_state(
    root_node_mock_state: Node, mock_evaluator: MockNetworkEvaluator
) -> Node:
    """Provides an expanded root node with mock children using mock EnvConfig."""
    root = root_node_mock_state
    # Cast root.state back to MockGameState for the evaluator
    mock_state: MockGameState = root.state  # type: ignore [assignment]
    # Ensure evaluator action_dim is int
    # Cast ACTION_DIM to int
    mock_evaluator._action_dim = int(mock_state.env_config.ACTION_DIM)
    policy, value = mock_evaluator.evaluate(mock_state)
    # Ensure policy is not empty before expanding
    if not policy:
        policy = (
            dict.fromkeys(
                mock_state.valid_actions(), 1.0 / len(mock_state.valid_actions())
            )
            if mock_state.valid_actions()
            else {}
        )

    for action in mock_state.valid_actions():
        prior = policy.get(action, 0.0)
        child_state = MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=mock_state.env_config
        )
        child = Node(
            state=child_state,  # type: ignore [arg-type]
            parent=root,
            action_taken=action,
            prior_probability=prior,
        )
        root.children[action] = child
    root.visit_count = 1  # Simulate one visit to root after expansion
    root.total_action_value = value
    return root


@pytest.fixture
def deep_expanded_node_mock_state(
    expanded_node_mock_state: Node,
    mock_evaluator: MockNetworkEvaluator,
    mock_env_config: EnvConfig,
) -> Node:
    """
    Provides a root node expanded two levels deep, specifically configured
    to encourage traversal down the path leading to action 0, then action 1.
    """
    root = expanded_node_mock_state
    # Ensure evaluator has correct action dim (as int)
    # Cast ACTION_DIM to int
    mock_evaluator._action_dim = int(mock_env_config.ACTION_DIM)

    # Ensure children exist
    if 0 not in root.children or 1 not in root.children:
        pytest.skip("Actions 0 or 1 not available in expanded node children")

    # --- Configure Root Node to strongly prefer Action 0 ---
    root.visit_count = 100  # Give root significant visits
    child0 = root.children[0]
    # child1 = root.children[1] # Unused variable

    # Child 0: High visit count, good value, high prior (after potential noise)
    child0.visit_count = 80
    child0.total_action_value = 40  # Q = 0.5
    child0.prior_probability = 0.8

    # Other children: Low visits, low value, low prior
    for action, child in root.children.items():
        if action != 0:
            child.visit_count = 2
            child.total_action_value = 0  # Q = 0.0
            child.prior_probability = 0.01

    # --- Configure Child 0 to strongly prefer Action 1 ---
    # Ensure Child 0 has children (expand it manually)
    # Use evaluator to get a policy, then manually create children
    # Cast child0.state back to MockGameState for the evaluator
    mock_child0_state: MockGameState = child0.state  # type: ignore [assignment]
    policy_gc, value_gc = mock_evaluator.evaluate(mock_child0_state)
    if not policy_gc:  # Handle case where mock state has no valid actions
        policy_gc = (
            dict.fromkeys(
                mock_child0_state.valid_actions(),
                1.0 / len(mock_child0_state.valid_actions()),
            )
            if mock_child0_state.valid_actions()
            else {}
        )

    valid_gc_actions = mock_child0_state.valid_actions()
    if (
        1 not in valid_gc_actions and valid_gc_actions
    ):  # If action 1 not valid, pick first valid one
        preferred_gc_action = valid_gc_actions[0]
    elif not valid_gc_actions:
        pytest.skip("Child 0 has no valid actions to create grandchildren")
    else:
        preferred_gc_action = 1

    # Create grandchild nodes
    for action_gc in valid_gc_actions:
        prior_gc = policy_gc.get(action_gc, 0.0)
        grandchild_state = MockGameState(
            current_step=2, valid_actions=[0], env_config=mock_child0_state.env_config
        )
        grandchild = Node(
            state=grandchild_state,  # type: ignore [arg-type]
            parent=child0,
            action_taken=action_gc,
            prior_probability=prior_gc,
        )
        child0.children[action_gc] = grandchild

    # Now configure grandchild stats
    preferred_grandchild = child0.children.get(preferred_gc_action)
    if preferred_grandchild:
        # Preferred Grandchild: High visits, good value, high prior
        preferred_grandchild.visit_count = 60
        preferred_grandchild.total_action_value = 30  # Q = 0.5
        preferred_grandchild.prior_probability = 0.7

    # Other grandchildren: Low visits, low value, low prior
    for action_gc, grandchild in child0.children.items():
        if action_gc != preferred_gc_action:
            grandchild.visit_count = 1
            grandchild.total_action_value = 0  # Q = 0.0
            grandchild.prior_probability = 0.05

    return root


File: tests\mcts\fixtures.py
from collections.abc import Mapping

import pytest

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, MCTSConfig
    from alphatriangle.mcts.core.node import Node
    from alphatriangle.utils.types import ActionType, PolicyValueOutput
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, MCTSConfig
    from alphatriangle.mcts.core.node import Node
    from alphatriangle.utils.types import ActionType, PolicyValueOutput


# --- Mock GameState ---
class MockGameState:
    """A simplified mock GameState for testing MCTS logic."""

    def __init__(
        self,
        current_step: int = 0,
        is_terminal: bool = False,
        outcome: float = 0.0,
        valid_actions: list[ActionType] | None = None,
        env_config: EnvConfig | None = None,
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        # Use a default EnvConfig if none provided, needed for action dim
        # Pydantic models with defaults can be instantiated without args
        self.env_config = env_config if env_config else EnvConfig()
        # Cast ACTION_DIM to int
        action_dim_int = int(self.env_config.ACTION_DIM)
        self._valid_actions = (
            valid_actions if valid_actions is not None else list(range(action_dim_int))
        )

    def is_over(self) -> bool:
        return self._is_over

    def get_outcome(self) -> float:
        if not self._is_over:
            raise ValueError("Cannot get outcome of non-terminal state.")
        return self._outcome

    def valid_actions(self) -> list[ActionType]:
        return self._valid_actions

    def copy(self) -> "MockGameState":
        # Simple copy for testing, doesn't need full state copy
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,
        )

    def step(self, action: ActionType) -> tuple[float, bool]:
        # Mock step: advances step, returns dummy values, becomes terminal sometimes
        if action not in self._valid_actions:
            raise ValueError(f"Invalid action {action} for mock state.")
        self.current_step += 1
        # Simple logic: become terminal after 5 steps for testing
        self._is_over = self.current_step >= 5
        self._outcome = 1.0 if self._is_over else 0.0
        # Return dummy reward and done status
        return 0.0, self._is_over

    def __hash__(self):
        # Simple hash for testing purposes
        return hash((self.current_step, self._is_over, tuple(self._valid_actions)))

    def __eq__(self, other):
        if not isinstance(other, MockGameState):
            return NotImplemented
        return (
            self.current_step == other.current_step
            and self._is_over == other._is_over
            and self._valid_actions == other._valid_actions
        )


# --- Mock Network Evaluator ---
class MockNetworkEvaluator:
    """A mock network evaluator for testing MCTS."""

    def __init__(
        self,
        default_policy: Mapping[ActionType, float] | None = None,
        default_value: float = 0.5,
        action_dim: int = 3,  # Default action dim
    ):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim  # Already int
        self.evaluation_history: list[MockGameState] = []
        self.batch_evaluation_history: list[list[MockGameState]] = []

    def _get_policy(self, state: MockGameState) -> Mapping[ActionType, float]:
        if self._default_policy is not None:
            return self._default_policy
        # Default uniform policy over valid actions
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}
        prob = 1.0 / len(valid_actions)
        # Return policy only for valid actions
        return dict.fromkeys(valid_actions, prob)

    def evaluate(self, state: MockGameState) -> PolicyValueOutput:
        self.evaluation_history.append(state)
        policy = self._get_policy(state)
        # Ensure policy sums to 1 if not empty
        if policy:
            policy_sum = sum(policy.values())
            if abs(policy_sum - 1.0) > 1e-6:
                policy = {a: p / policy_sum for a, p in policy.items()}

        # Create full policy map for the action dimension
        full_policy = dict.fromkeys(range(self._action_dim), 0.0)
        full_policy.update(policy)

        return full_policy, self._default_value

    def evaluate_batch(self, states: list[MockGameState]) -> list[PolicyValueOutput]:
        self.batch_evaluation_history.append(states)
        results = []
        for state in states:
            results.append(self.evaluate(state))  # Reuse single evaluate logic
        return results


# --- Pytest Fixtures ---
@pytest.fixture
def mock_env_config() -> EnvConfig:
    """Provides a default EnvConfig for tests."""
    # Pydantic models with defaults can be instantiated without args
    return EnvConfig()


@pytest.fixture
def mock_mcts_config() -> MCTSConfig:
    """Provides a default MCTSConfig for tests."""
    # Pydantic models with defaults can be instantiated without args
    return MCTSConfig()


@pytest.fixture
def mock_evaluator(mock_env_config: EnvConfig) -> MockNetworkEvaluator:
    """Provides a MockNetworkEvaluator instance."""
    # Cast ACTION_DIM to int
    action_dim_int = int(mock_env_config.ACTION_DIM)
    return MockNetworkEvaluator(action_dim=action_dim_int)


@pytest.fixture
def root_node_mock_state(mock_env_config: EnvConfig) -> Node:
    """Provides a root Node with a MockGameState."""
    # Cast ACTION_DIM to int
    action_dim_int = int(mock_env_config.ACTION_DIM)
    state = MockGameState(
        valid_actions=list(range(action_dim_int)),
        env_config=mock_env_config,
    )
    # Cast MockGameState to Any temporarily to satisfy Node's type hint
    return Node(state=state)  # type: ignore [arg-type]


@pytest.fixture
def expanded_node_mock_state(
    root_node_mock_state: Node, mock_evaluator: MockNetworkEvaluator
) -> Node:
    """Provides an expanded root node with mock children."""
    root = root_node_mock_state
    # Cast root.state back to MockGameState for the evaluator
    mock_state: MockGameState = root.state  # type: ignore [assignment]
    policy, value = mock_evaluator.evaluate(mock_state)
    # Manually expand for testing setup
    for action in mock_state.valid_actions():
        prior = policy.get(action, 0.0)
        # Create mock child state (doesn't need to be accurate step)
        child_state = MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=mock_state.env_config
        )
        child = Node(
            state=child_state,  # type: ignore [arg-type]
            parent=root,
            action_taken=action,
            prior_probability=prior,
        )
        root.children[action] = child
    # Simulate one backpropagation
    root.visit_count = 1
    root.total_action_value = value
    return root


File: tests\mcts\test_expansion.py
from typing import Any

import pytest

from alphatriangle.mcts.core.node import Node

# Import necessary components and fixtures
from alphatriangle.mcts.strategy import expansion

# Import session-scoped fixtures implicitly via pytest injection
# from alphatriangle.config import MCTSConfig # REMOVED - Provided by top-level conftest
from .conftest import (  # Import from conftest (local fixtures)
    # mock_env_config, # REMOVED - Provided by top-level conftest
    MockGameState,
)


def test_expand_node_with_policy_basic(root_node_mock_state: Node):
    """Test basic node expansion with a valid policy."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock method
    mock_state: Any = node.state
    valid_actions = mock_state.valid_actions()
    # Simple policy: uniform over valid actions
    policy = {action: 1.0 / len(valid_actions) for action in valid_actions}

    assert not node.is_expanded
    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    for action in valid_actions:
        assert action in node.children
        child = node.children[action]
        assert child.parent is node
        assert child.action_taken == action
        assert child.prior_probability == pytest.approx(1.0 / len(valid_actions))
        assert (
            child.state.current_step == node.state.current_step + 1
        )  # Check state stepped
        assert not child.is_expanded
        assert child.visit_count == 0
        assert child.total_action_value == 0.0


def test_expand_node_with_policy_partial(root_node_mock_state: Node):
    """Test expansion when policy doesn't cover all valid actions (should assign 0 prior)."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock method
    mock_state: Any = node.state
    valid_actions = mock_state.valid_actions()  # e.g., [0, 1, ..., 8] for 3x3
    # Policy only covers action 0 and 1
    policy = {0: 0.6, 1: 0.4}

    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(
        valid_actions
    )  # Should still create nodes for all valid actions

    assert 0 in node.children
    assert node.children[0].prior_probability == pytest.approx(0.6)
    assert 1 in node.children
    assert node.children[1].prior_probability == pytest.approx(0.4)
    # Check an action not in the policy but valid
    if 2 in valid_actions:
        assert 2 in node.children
        assert node.children[2].prior_probability == 0.0  # Prior should default to 0


def test_expand_node_with_policy_empty_valid_actions(root_node_mock_state: Node):
    """Test expansion when the node's state has no valid actions (but isn't terminal yet)."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock attribute
    mock_state: Any = node.state
    mock_state._valid_actions = []  # No valid actions
    policy = {0: 1.0}  # Policy doesn't matter here

    expansion.expand_node_with_policy(node, policy)

    assert not node.is_expanded  # Should not expand
    assert not node.children
    # The function should log a warning in this case
    # The node's state should be marked as terminal by the expansion function
    assert node.state.is_over()


def test_expand_node_with_policy_already_expanded(root_node_mock_state: Node):
    """Test that expanding an already expanded node does nothing."""
    node = root_node_mock_state
    policy = {0: 1.0}
    # Manually add a child to simulate expansion
    # Pass the env_config from the root node's state
    node.children[0] = Node(
        state=MockGameState(current_step=1, env_config=node.state.env_config),  # type: ignore [arg-type]
        parent=node,
        action_taken=0,
    )

    assert node.is_expanded
    original_children = node.children.copy()

    expansion.expand_node_with_policy(node, policy)

    assert node.children == original_children  # Children should not change


def test_expand_node_with_policy_terminal_node(root_node_mock_state: Node):
    """Test that expanding a terminal node does nothing."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock attribute
    mock_state: Any = node.state
    mock_state._is_over = True  # Mark as terminal
    policy = {0: 1.0}

    assert not node.is_expanded
    expansion.expand_node_with_policy(node, policy)
    assert not node.is_expanded  # Should not expand


def test_expand_node_with_invalid_policy_content(root_node_mock_state: Node):
    """Test expansion handles policy with invalid content (e.g., negative priors)."""
    # Note: The main search loop should validate policy *before* calling expand.
    # This test checks if expand handles it defensively (it currently clamps).
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock method
    mock_state: Any = node.state
    valid_actions = mock_state.valid_actions()
    policy = {0: 1.5, 1: -0.5}  # Invalid priors

    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    assert node.children[0].prior_probability == pytest.approx(
        1.5
    )  # Currently doesn't clamp > 1
    assert node.children[1].prior_probability == 0.0  # Clamps negative to 0
    if 2 in valid_actions:
        assert node.children[2].prior_probability == 0.0


File: tests\mcts\test_selection.py
import math
from typing import Any

import pytest

# Import session-scoped fixtures implicitly via pytest injection
from alphatriangle.config import MCTSConfig  # Keep MCTSConfig type hint if needed
from alphatriangle.mcts.core.node import Node

# Import necessary components and fixtures
from alphatriangle.mcts.strategy import selection

from .conftest import (  # Import from conftest (local fixtures)
    EnvConfig,  # Keep EnvConfig type hint if needed
    MockGameState,
)


# --- Test PUCT Calculation ---
def test_puct_calculation_unvisited_child(
    mock_mcts_config: MCTSConfig, mock_env_config: EnvConfig
):
    """Test PUCT score for an unvisited child node."""
    parent = Node(state=MockGameState(env_config=mock_env_config))  # type: ignore [arg-type]
    parent.visit_count = 10
    child = Node(
        state=MockGameState(current_step=1, env_config=mock_env_config),  # type: ignore [arg-type]
        parent=parent,
        action_taken=0,
        prior_probability=0.5,
    )
    child.visit_count = 0
    child.total_action_value = 0.0

    score, q_value, exploration = selection.calculate_puct_score(
        child, parent.visit_count, mock_mcts_config
    )

    assert q_value == 0.0, "Q-value should be 0 for unvisited node"
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.5 * (math.sqrt(10) / (1 + 0))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


def test_puct_calculation_visited_child(
    mock_mcts_config: MCTSConfig, mock_env_config: EnvConfig
):
    """Test PUCT score for a visited child node."""
    parent = Node(state=MockGameState(env_config=mock_env_config))  # type: ignore [arg-type]
    parent.visit_count = 25
    child = Node(
        state=MockGameState(current_step=1, env_config=mock_env_config),  # type: ignore [arg-type]
        parent=parent,
        action_taken=1,
        prior_probability=0.2,
    )
    child.visit_count = 5
    child.total_action_value = 3.0

    score, q_value, exploration = selection.calculate_puct_score(
        child, parent.visit_count, mock_mcts_config
    )

    assert q_value == pytest.approx(3.0 / 5.0)
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.2 * (math.sqrt(25) / (1 + 5))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


def test_puct_calculation_zero_parent_visits(
    mock_mcts_config: MCTSConfig, mock_env_config: EnvConfig
):
    """Test PUCT score when parent visit count is zero (should use sqrt(1))."""
    parent = Node(state=MockGameState(env_config=mock_env_config))  # type: ignore [arg-type]
    parent.visit_count = 0
    child = Node(
        state=MockGameState(current_step=1, env_config=mock_env_config),  # type: ignore [arg-type]
        parent=parent,
        action_taken=0,
        prior_probability=0.6,
    )
    child.visit_count = 0
    child.total_action_value = 0.0

    # Calculate PUCT with parent_visit_count=0
    score, q_value, exploration = selection.calculate_puct_score(
        child, 0, mock_mcts_config
    )

    assert q_value == 0.0
    # The formula uses max(1, parent_visit_count) for the sqrt term numerator
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.6 * (math.sqrt(1) / (1 + 0))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


# --- Test Child Selection ---
def test_select_child_node_basic(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test basic child selection based on PUCT."""
    parent = expanded_node_mock_state
    parent.visit_count = 10

    # Ensure children 0, 1, 2 exist before accessing them
    if 0 not in parent.children or 1 not in parent.children or 2 not in parent.children:
        pytest.skip("Required children (0, 1, 2) not present in fixture")

    child0 = parent.children[0]
    child0.visit_count = 1
    child0.total_action_value = 0.8  # Q = 0.8
    child0.prior_probability = 0.1  # Lower prior, higher Q

    child1 = parent.children[1]
    child1.visit_count = 5
    child1.total_action_value = 0.5  # Low Q (0.1), higher visits
    child1.prior_probability = 0.6  # High prior

    child2 = parent.children[2]
    child2.visit_count = 3
    child2.total_action_value = 1.5  # Mid Q (0.5), mid visits
    child2.prior_probability = 0.3  # Mid prior

    # Calculate scores with C=1.5 (from config fixture now)
    # Score0 = 0.8/1 + 1.5 * 0.1 * (sqrt(10) / (1 + 1)) ~ 0.8 + 0.237 = 1.037
    # Score1 = 0.5/5 + 1.5 * 0.6 * (sqrt(10) / (1 + 5)) ~ 0.1 + 0.474 = 0.574
    # Score2 = 1.5/3 + 1.5 * 0.3 * (sqrt(10) / (1 + 3)) ~ 0.5 + 0.355 = 0.855
    # Child 0 should have the highest score

    selected_child = selection.select_child_node(parent, mock_mcts_config)
    assert selected_child is child0


def test_select_child_node_no_children(
    root_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test selection raises error if node has no children."""
    parent = root_node_mock_state
    assert not parent.children  # Ensure it has no children
    with pytest.raises(selection.SelectionError):
        selection.select_child_node(parent, mock_mcts_config)


def test_select_child_node_tie_breaking(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test that selection handles ties (implementation detail, usually selects first max)."""
    parent = expanded_node_mock_state
    parent.visit_count = 10

    # Ensure children 0, 1, 2 exist
    if 0 not in parent.children or 1 not in parent.children or 2 not in parent.children:
        pytest.skip("Required children (0, 1, 2) not present in fixture")

    child0 = parent.children[0]
    child0.visit_count = 1
    child0.total_action_value = 0.9  # Q = 0.9
    child0.prior_probability = 0.4

    child1 = parent.children[1]
    child1.visit_count = 1
    child1.total_action_value = 0.9  # Q = 0.9
    child1.prior_probability = 0.4

    child2 = parent.children[2]
    child2.visit_count = 5
    child2.total_action_value = 0.1  # Q = 0.02
    child2.prior_probability = 0.1

    # Score0 = 0.9 + C * 0.4 * (sqrt(10)/2)
    # Score1 = 0.9 + C * 0.4 * (sqrt(10)/2)
    # Score2 = 0.02 + C * 0.1 * (sqrt(10)/6)
    # Child 0 and 1 have equal highest score
    selected_child = selection.select_child_node(parent, mock_mcts_config)
    assert (
        selected_child is child0 or selected_child is child1
    )  # Should select one of the tied best


# --- Test Dirichlet Noise ---
def test_add_dirichlet_noise(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test that Dirichlet noise modifies prior probabilities correctly."""
    node = expanded_node_mock_state
    # Create a copy of the config to modify locally for this test
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.dirichlet_alpha = 0.5
    config_copy.dirichlet_epsilon = 0.25

    n_children = len(node.children)
    if n_children == 0:
        pytest.skip("Node has no children to add noise to.")
    original_priors = {a: c.prior_probability for a, c in node.children.items()}
    # original_sum = sum(original_priors.values()) # Unused variable

    # Use default_rng for modern NumPy random generation
    # rng = np.random.default_rng(42) # Removed unused variable
    selection.add_dirichlet_noise(node, config_copy)
    # Resetting global seed is less ideal, rely on instance if needed elsewhere

    new_priors = {a: c.prior_probability for a, c in node.children.items()}
    mixed_sum = sum(new_priors.values())

    assert len(new_priors) == n_children
    priors_changed = False
    for action, new_p in new_priors.items():
        assert action in original_priors
        assert 0.0 <= new_p <= 1.0  # Check bounds
        if abs(new_p - original_priors[action]) > 1e-9:
            priors_changed = True

    assert priors_changed, "Priors did not change after adding noise"
    assert mixed_sum == pytest.approx(1.0, abs=1e-6), (
        f"Noisy priors sum to {mixed_sum}, not 1.0"
    )


def test_add_dirichlet_noise_disabled(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test that noise is not added if alpha or epsilon is zero."""
    node = expanded_node_mock_state
    if not node.children:
        pytest.skip("Node has no children.")
    original_priors = {a: c.prior_probability for a, c in node.children.items()}

    # Create copies of the config to modify locally
    config_alpha_zero = mock_mcts_config.model_copy(deep=True)
    config_alpha_zero.dirichlet_alpha = 0.0
    config_alpha_zero.dirichlet_epsilon = 0.25

    config_eps_zero = mock_mcts_config.model_copy(deep=True)
    config_eps_zero.dirichlet_alpha = 0.5
    config_eps_zero.dirichlet_epsilon = 0.0

    selection.add_dirichlet_noise(node, config_alpha_zero)
    priors_after_alpha_zero = {a: c.prior_probability for a, c in node.children.items()}
    assert priors_after_alpha_zero == original_priors, (
        "Priors changed when alpha was zero"
    )

    # Reset priors before next test
    for a, p in original_priors.items():
        node.children[a].prior_probability = p

    selection.add_dirichlet_noise(node, config_eps_zero)
    priors_after_eps_zero = {a: c.prior_probability for a, c in node.children.items()}
    assert priors_after_eps_zero == original_priors, (
        "Priors changed when epsilon was zero"
    )


# --- Test Traversal ---
def test_traverse_to_leaf_unexpanded(
    root_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal stops immediately at an unexpanded root."""
    leaf, depth = selection.traverse_to_leaf(root_node_mock_state, mock_mcts_config)
    assert leaf is root_node_mock_state
    assert depth == 0


def test_traverse_to_leaf_expanded(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal selects a child from an expanded node and stops (depth 1)."""
    root = expanded_node_mock_state
    for child in root.children.values():
        assert not child.is_expanded, (
            f"Child {child} should not be expanded in this fixture setup"
        )

    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)

    assert leaf in root.children.values()
    assert depth == 1


def test_traverse_to_leaf_max_depth(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal stops at max depth."""
    root = expanded_node_mock_state
    # Create a copy of the config to modify locally
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 0

    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    assert leaf is root
    assert depth == 0

    # --- Test max depth = 1 ---
    config_copy.max_search_depth = 1
    # Ensure root has children
    if not root.children:
        pytest.skip("Root node has no children for max depth 1 test")

    # Manually expand one child to test if traversal stops *before* selecting grandchild
    child0 = next(iter(root.children.values()))
    child0.children[0] = Node(
        state=MockGameState(current_step=2, env_config=root.state.env_config),  # type: ignore [arg-type]
        parent=child0,
        action_taken=0,
    )

    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    assert leaf in root.children.values(), (
        "Leaf node should be a direct child of the root"
    )
    assert depth == 1, "Depth should be 1 when max_search_depth is 1"


def test_traverse_to_terminal_leaf(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal stops at a terminal node."""
    root = expanded_node_mock_state
    # Ensure child 1 exists
    if 1 not in root.children:
        pytest.skip("Child 1 not present in fixture")
    child1 = root.children[1]
    # Cast child1.state to Any temporarily to access mock attribute
    mock_child1_state: Any = child1.state
    mock_child1_state._is_over = True  # Mark child 1 as terminal

    # Make child 1 highly attractive to ensure it's selected
    root.visit_count = 10
    for action, child in root.children.items():
        if action == 1:
            child.visit_count = 5
            child.total_action_value = 4  # High Q = 0.8
            child.prior_probability = 0.8  # High P
        else:
            child.visit_count = 1
            child.total_action_value = 0  # Low Q = 0.0
            child.prior_probability = 0.1  # Low P

    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)

    assert leaf is child1, "Traversal should stop at the terminal child node"
    assert depth == 1, "Depth should be 1 as traversal stops at the terminal child"


# --- Added Test for Deeper Traversal ---
def test_traverse_to_leaf_deeper(
    deep_expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal goes deeper than 1 level using the specifically configured fixture."""
    root = deep_expanded_node_mock_state  # This fixture is configured to prefer path 0 -> 1 (or first valid)
    # Create a copy of the config to modify locally
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 10  # Ensure max depth doesn't interfere

    # --- Assert fixture setup is correct ---
    assert 0 in root.children, "Fixture should have child 0"
    child0 = root.children[0]
    assert child0.is_expanded, "Child 0 should be expanded in the fixture"
    assert child0.children, "Child 0 should have grandchildren"

    # Determine the action preferred by the fixture logic for child0
    # Cast child0.state to Any temporarily to access mock method
    mock_child0_state: Any = child0.state
    valid_gc_actions = mock_child0_state.valid_actions()
    if 1 in valid_gc_actions:
        preferred_gc_action = 1
    elif valid_gc_actions:
        preferred_gc_action = valid_gc_actions[0]
    else:
        pytest.fail("Fixture error: Child 0 has no valid actions for grandchildren")

    expected_grandchild = child0.children.get(preferred_gc_action)
    assert expected_grandchild is not None, (
        f"Expected grandchild for action {preferred_gc_action} not found"
    )

    # --- Run traversal ---
    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    # --- Assertions ---
    # It should select child0, then the expected grandchild (which is a leaf in the fixture setup)
    assert leaf is expected_grandchild, (
        f"Expected leaf {expected_grandchild}, but got {leaf}"
    )
    assert depth == 2, f"Expected depth 2, but got {depth}"


File: tests\mcts\__init__.py


File: tests\nn\test_model.py
# File: tests/nn/test_model.py
import pytest
import torch

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, ModelConfig
    from alphatriangle.nn import AlphaTriangleNet
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, ModelConfig
    from alphatriangle.nn import AlphaTriangleNet


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def model(model_config: ModelConfig, env_config: EnvConfig) -> AlphaTriangleNet:
    """Provides an instance of the AlphaTriangleNet model."""
    return AlphaTriangleNet(model_config, env_config)


def test_model_initialization(
    model: AlphaTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test if the model initializes without errors."""
    assert model is not None
    # Cast ACTION_DIM to int for comparison
    assert model.action_dim == int(env_config.ACTION_DIM)
    # Add more checks based on config if needed (e.g., transformer presence)
    assert model.model_config.USE_TRANSFORMER == model_config.USE_TRANSFORMER
    if model_config.USE_TRANSFORMER:
        assert model.transformer_body is not None
    else:
        assert model.transformer_body is None


def test_model_forward_pass(
    model: AlphaTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test the forward pass with dummy input tensors."""
    batch_size = 4
    device = torch.device("cpu")  # Test on CPU
    model.to(device)
    model.eval()  # Set to eval mode
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)

    # Create dummy input tensors
    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        # --- CHANGED: Expect value_logits ---
        policy_logits, value_logits = model(dummy_grid, dummy_other)
        # --- END CHANGED ---

    # Check output shapes
    assert policy_logits.shape == (
        batch_size,
        action_dim_int,
    ), f"Policy logits shape mismatch: {policy_logits.shape}"
    # --- CHANGED: Check value logits shape ---
    assert value_logits.shape == (
        batch_size,
        model_config.NUM_VALUE_ATOMS,
    ), f"Value logits shape mismatch: {value_logits.shape}"
    # --- END CHANGED ---

    # Check output types
    assert policy_logits.dtype == torch.float32
    # --- CHANGED: Check value logits type ---
    assert value_logits.dtype == torch.float32
    # --- END CHANGED ---

    # --- REMOVED: Value range check (output is logits) ---
    # assert torch.all(value >= -1.0) and torch.all(value <= 1.0), (
    #     f"Value out of range [-1, 1]: {value}"
    # )
    # --- END REMOVED ---


@pytest.mark.parametrize(
    "use_transformer", [False, True], ids=["CNN_Only", "CNN_Transformer"]
)
def test_model_forward_transformer_toggle(use_transformer: bool, env_config: EnvConfig):
    """Test forward pass with transformer enabled/disabled."""
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)
    # Create a specific model config for this test, providing all required fields
    # --- CHANGED: Use default distributional params from ModelConfig ---
    model_config_test = ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4, 8],  # Simple CNN
        CONV_KERNEL_SIZES=[3, 3],
        CONV_STRIDES=[1, 1],
        CONV_PADDING=[1, 1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=8,
        USE_TRANSFORMER=use_transformer,
        TRANSFORMER_DIM=16,
        TRANSFORMER_HEADS=2,
        TRANSFORMER_LAYERS=1,
        TRANSFORMER_FC_DIM=32,
        FC_DIMS_SHARED=[16],
        POLICY_HEAD_DIMS=[action_dim_int],  # Use casted int
        # VALUE_HEAD_DIMS=[1], # Use default from ModelConfig
        OTHER_NN_INPUT_FEATURES_DIM=10,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=True,
        # NUM_VALUE_ATOMS=51, # Use default
        # VALUE_MIN=-10.0, # Use default
        # VALUE_MAX=10.0, # Use default
    )
    # --- END CHANGED ---
    model = AlphaTriangleNet(model_config_test, env_config)
    batch_size = 2
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config_test.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config_test.OTHER_NN_INPUT_FEATURES_DIM)
    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        # --- CHANGED: Expect value_logits ---
        policy_logits, value_logits = model(dummy_grid, dummy_other)
        # --- END CHANGED ---

    assert policy_logits.shape == (batch_size, action_dim_int)
    # --- CHANGED: Check value logits shape ---
    assert value_logits.shape == (batch_size, model_config_test.NUM_VALUE_ATOMS)
    # --- END CHANGED ---


File: tests\nn\test_network.py
# File: tests/nn/test_network.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, ModelConfig, TrainConfig
    from alphatriangle.environment import GameState
    from alphatriangle.nn import AlphaTriangleNet, NeuralNetwork
    from alphatriangle.utils.types import StateType
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, ModelConfig, TrainConfig
    from alphatriangle.environment import GameState
    from alphatriangle.nn import AlphaTriangleNet, NeuralNetwork
    from alphatriangle.utils.types import StateType

# Use module-level rng from tests/conftest.py
from tests.conftest import rng


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    # Ensure feature dim matches mock_state_type
    mock_model_config.OTHER_NN_INPUT_FEATURES_DIM = 10
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    # --- CHANGED: Use the default COMPILE_MODEL=True for this test fixture ---
    # Ensure the test runs against the default behavior
    cfg = mock_train_config.model_copy(deep=True)
    cfg.COMPILE_MODEL = True  # Explicitly set to True for clarity in test setup
    return cfg
    # --- END CHANGED ---


@pytest.fixture
def device() -> torch.device:
    # Use CPU for consistency in tests, even though compile might happen
    return torch.device("cpu")


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing."""
    # --- CHANGED: Pass the modified train_config ---
    return NeuralNetwork(model_config, env_config, train_config, device)
    # --- END CHANGED ---


@pytest.fixture
def mock_game_state(env_config: EnvConfig) -> GameState:
    """Provides a real GameState object for testing NN interface."""
    # Use a real GameState instance
    return GameState(config=env_config, initial_seed=123)


# --- Fixture providing the batch of copied states ---
@pytest.fixture
def mock_game_state_batch(mock_game_state: GameState) -> list[GameState]:
    """Provides a list of copied GameState objects."""
    batch_size = 3
    # The .copy() call happens here, where mypy knows mock_game_state is GameState
    return [mock_game_state.copy() for _ in range(batch_size)]


# --- End fixture ---


@pytest.fixture
def mock_state_type_nn(model_config: ModelConfig, env_config: EnvConfig) -> StateType:
    """Creates a mock StateType dictionary compatible with the NN test config."""
    grid_shape = (
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape).astype(np.float32),
        "other_features": rng.random(other_shape).astype(np.float32),
    }


# --- Test Initialization ---
def test_nn_initialization(nn_interface: NeuralNetwork, device: torch.device):
    """Test if the NeuralNetwork wrapper initializes correctly."""
    assert nn_interface is not None
    assert nn_interface.device == device
    # --- CHANGED: Check underlying model type if compiled ---
    if hasattr(nn_interface.model, "_orig_mod"):
        # If compiled, check the original module's type
        assert isinstance(nn_interface.model._orig_mod, AlphaTriangleNet)
        # Check that the compiled model is in eval mode
        assert not nn_interface.model.training
    else:
        # If not compiled, check the model directly
        assert isinstance(nn_interface.model, AlphaTriangleNet)
        assert not nn_interface.model.training  # Should be in eval mode initially
    # --- END CHANGED ---


# --- Test Feature Extraction Integration (using mock) ---
@patch("alphatriangle.nn.network.extract_state_features")
def test_state_to_tensors(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
):
    """Test the internal _state_to_tensors method mocks feature extraction."""
    mock_extract.return_value = mock_state_type_nn
    grid_t, other_t = nn_interface._state_to_tensors(mock_game_state)

    mock_extract.assert_called_once_with(mock_game_state, nn_interface.model_config)
    assert isinstance(grid_t, torch.Tensor)
    assert isinstance(other_t, torch.Tensor)
    assert grid_t.device == nn_interface.device
    assert other_t.device == nn_interface.device
    assert grid_t.shape[0] == 1  # Batch dimension
    assert other_t.shape[0] == 1
    assert grid_t.shape[1] == nn_interface.model_config.GRID_INPUT_CHANNELS
    assert other_t.shape[1] == nn_interface.model_config.OTHER_NN_INPUT_FEATURES_DIM


@patch("alphatriangle.nn.network.extract_state_features")
def test_batch_states_to_tensors(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    # --- Use the fixture that provides the already copied batch ---
    mock_game_state_batch: list[GameState],
    mock_state_type_nn: StateType,
):
    """Test the internal _batch_states_to_tensors method."""
    # --- Use the fixture directly, no more .copy() needed here ---
    mock_states = mock_game_state_batch
    batch_size = len(mock_states)
    # --- End change ---
    # Make mock return slightly different arrays each time if needed
    # --- CHANGE: Add isinstance check before v.copy() ---
    mock_extract.side_effect = [
        {
            k: (v.copy() + i * 0.1 if isinstance(v, np.ndarray) else v)
            for k, v in mock_state_type_nn.items()
        }
        for i in range(batch_size)
    ]
    # --- END CHANGE ---

    grid_t, other_t = nn_interface._batch_states_to_tensors(mock_states)

    assert mock_extract.call_count == batch_size
    assert isinstance(grid_t, torch.Tensor)
    assert isinstance(other_t, torch.Tensor)
    assert grid_t.device == nn_interface.device
    assert other_t.device == nn_interface.device
    assert grid_t.shape[0] == batch_size
    assert other_t.shape[0] == batch_size
    assert grid_t.shape[1] == nn_interface.model_config.GRID_INPUT_CHANNELS
    assert other_t.shape[1] == nn_interface.model_config.OTHER_NN_INPUT_FEATURES_DIM


# --- Test Evaluation Methods ---
@patch("alphatriangle.nn.network.extract_state_features")
def test_evaluate_single(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    """Test the evaluate method for a single state."""
    mock_extract.return_value = mock_state_type_nn
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)

    policy_map, value = nn_interface.evaluate(mock_game_state)

    assert isinstance(policy_map, dict)
    assert isinstance(value, float)
    assert len(policy_map) == action_dim_int
    assert all(
        isinstance(k, int) and isinstance(v, float) for k, v in policy_map.items()
    )
    assert abs(sum(policy_map.values()) - 1.0) < 1e-5, (
        f"Policy probs sum to {sum(policy_map.values())}"
    )
    # --- REMOVED: Value range check ---
    # assert -1.0 <= value <= 1.0
    # --- END REMOVED ---


@patch("alphatriangle.nn.network.extract_state_features")
def test_evaluate_batch(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    # --- Use the fixture that provides the already copied batch ---
    mock_game_state_batch: list[GameState],
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    """Test the evaluate_batch method."""
    # --- Use the fixture directly, no more .copy() needed here ---
    mock_states = mock_game_state_batch
    batch_size = len(mock_states)
    # --- End change ---
    # --- CHANGE: Add isinstance check before v.copy() ---
    mock_extract.side_effect = [
        {
            k: (v.copy() + i * 0.1 if isinstance(v, np.ndarray) else v)
            for k, v in mock_state_type_nn.items()
        }
        for i in range(batch_size)
    ]
    # --- END CHANGE ---
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)

    results = nn_interface.evaluate_batch(mock_states)

    assert isinstance(results, list)
    assert len(results) == batch_size
    for policy_map, value in results:
        assert isinstance(policy_map, dict)
        assert isinstance(value, float)
        assert len(policy_map) == action_dim_int
        assert all(
            isinstance(k, int) and isinstance(v, float) for k, v in policy_map.items()
        )
        assert abs(sum(policy_map.values()) - 1.0) < 1e-5
        # --- REMOVED: Value range check ---
        # assert -1.0 <= value <= 1.0
        # --- END REMOVED ---


# --- Test Weight Management ---
def test_get_set_weights(nn_interface: NeuralNetwork):
    """Test getting and setting model weights."""
    initial_weights = nn_interface.get_weights()
    assert isinstance(initial_weights, dict)
    assert all(
        isinstance(k, str) and isinstance(v, torch.Tensor)
        for k, v in initial_weights.items()
    )
    # Check weights are on CPU
    assert all(v.device == torch.device("cpu") for v in initial_weights.values())

    # Modify only parameters (which should be floats)
    modified_weights = {}
    for k, v in initial_weights.items():
        if v.dtype.is_floating_point:
            modified_weights[k] = v + 0.1
        else:
            modified_weights[k] = v  # Keep non-float tensors (e.g., buffers) unchanged

    # Set modified weights
    nn_interface.set_weights(modified_weights)

    # Get weights again and compare parameters
    new_weights = nn_interface.get_weights()
    assert len(new_weights) == len(initial_weights)
    for key in initial_weights:
        assert key in new_weights
        # Compare on CPU
        if initial_weights[key].dtype.is_floating_point:
            assert torch.allclose(modified_weights[key], new_weights[key], atol=1e-6), (
                f"Weight mismatch for key {key}"
            )
        else:
            assert torch.equal(initial_weights[key], new_weights[key]), (
                f"Non-float tensor mismatch for key {key}"
            )

    # Test setting back original weights
    nn_interface.set_weights(initial_weights)
    final_weights = nn_interface.get_weights()
    for key in initial_weights:
        assert torch.equal(initial_weights[key], final_weights[key]), (
            f"Weight mismatch after setting back original for key {key}"
        )


File: tests\nn\__init__.py


File: tests\rl\test_buffer.py
from collections import deque

import numpy as np
import pytest

from alphatriangle.config import TrainConfig
from alphatriangle.rl import ExperienceBuffer
from alphatriangle.utils.sumtree import SumTree
from alphatriangle.utils.types import Experience, StateType

# Use module-level rng from tests/conftest.py
from tests.conftest import rng

# --- Fixtures ---


@pytest.fixture
def uniform_train_config() -> TrainConfig:
    """TrainConfig for uniform buffer."""
    return TrainConfig(
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        BATCH_SIZE=4,
        USE_PER=False,
        # Provide defaults for other required fields
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.0,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=100,
        PER_EPSILON=1e-5,
        MAX_TRAINING_STEPS=200,  # Set a finite value for tests
    )


@pytest.fixture
def per_train_config() -> TrainConfig:
    """TrainConfig for PER buffer."""
    return TrainConfig(
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        BATCH_SIZE=4,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=50,  # Short anneal for testing
        PER_EPSILON=1e-5,
        # Provide defaults for other required fields
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.0,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        MAX_TRAINING_STEPS=200,  # Set a finite value for tests
    )


@pytest.fixture
def uniform_buffer(uniform_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an empty uniform ExperienceBuffer."""
    return ExperienceBuffer(uniform_train_config)


@pytest.fixture
def per_buffer(per_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an empty PER ExperienceBuffer."""
    return ExperienceBuffer(per_train_config)


# Use shared mock_experience fixture implicitly from tests/conftest.py
# REMOVED: @pytest.fixture
# REMOVED: def experience(mock_experience: Experience) -> Experience:
# REMOVED:    return mock_experience


# --- Uniform Buffer Tests ---


def test_uniform_buffer_init(uniform_buffer: ExperienceBuffer):
    assert not uniform_buffer.use_per
    assert isinstance(uniform_buffer.buffer, deque)
    assert uniform_buffer.capacity == 100
    assert len(uniform_buffer) == 0
    assert not uniform_buffer.is_ready()


# Use mock_experience directly injected by pytest
def test_uniform_buffer_add(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    assert len(uniform_buffer) == 0
    uniform_buffer.add(mock_experience)
    assert len(uniform_buffer) == 1
    assert uniform_buffer.buffer[0] == mock_experience


# Use mock_experience directly injected by pytest
def test_uniform_buffer_add_batch(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    batch = [mock_experience] * 5
    uniform_buffer.add_batch(batch)
    assert len(uniform_buffer) == 5


# Use mock_experience directly injected by pytest
def test_uniform_buffer_capacity(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    for i in range(uniform_buffer.capacity + 10):
        # Create slightly different experiences
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        uniform_buffer.add(exp_copy)
    assert len(uniform_buffer) == uniform_buffer.capacity
    # Check if the first added element is gone
    first_added_val = mock_experience[2] + 0
    assert not any(exp[2] == first_added_val for exp in uniform_buffer.buffer)
    # Check if the last added element is present
    last_added_val = mock_experience[2] + uniform_buffer.capacity + 9
    assert any(exp[2] == last_added_val for exp in uniform_buffer.buffer)


# Use mock_experience directly injected by pytest
def test_uniform_buffer_is_ready(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    assert not uniform_buffer.is_ready()
    for _ in range(uniform_buffer.min_size_to_train):
        uniform_buffer.add(mock_experience)
    assert uniform_buffer.is_ready()


# Use mock_experience directly injected by pytest
def test_uniform_buffer_sample(
    uniform_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Fill buffer until ready
    for i in range(uniform_buffer.min_size_to_train):
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        uniform_buffer.add(exp_copy)

    sample = uniform_buffer.sample(uniform_buffer.config.BATCH_SIZE)
    assert sample is not None
    assert isinstance(sample, dict)
    assert "batch" in sample
    assert "indices" in sample
    assert "weights" in sample
    assert len(sample["batch"]) == uniform_buffer.config.BATCH_SIZE
    assert isinstance(sample["batch"][0], tuple)  # Check if it's an Experience tuple
    assert sample["indices"].shape == (uniform_buffer.config.BATCH_SIZE,)
    assert sample["weights"].shape == (uniform_buffer.config.BATCH_SIZE,)
    assert np.allclose(sample["weights"], 1.0)  # Uniform weights should be 1.0


def test_uniform_buffer_sample_not_ready(uniform_buffer: ExperienceBuffer):
    sample = uniform_buffer.sample(uniform_buffer.config.BATCH_SIZE)
    assert sample is None


def test_uniform_buffer_update_priorities(uniform_buffer: ExperienceBuffer):
    # Should be a no-op
    initial_len = len(uniform_buffer)
    uniform_buffer.update_priorities(np.array([0, 1]), np.array([0.5, 0.1]))
    assert len(uniform_buffer) == initial_len  # No change expected


# --- PER Buffer Tests ---


def test_per_buffer_init(per_buffer: ExperienceBuffer):
    assert per_buffer.use_per
    assert isinstance(per_buffer.tree, SumTree)
    assert per_buffer.capacity == 100
    assert len(per_buffer) == 0
    assert not per_buffer.is_ready()
    assert per_buffer.tree.max_priority == 1.0  # Initial max priority


# Use mock_experience directly injected by pytest
def test_per_buffer_add(per_buffer: ExperienceBuffer, mock_experience: Experience):
    assert len(per_buffer) == 0
    initial_max_p = per_buffer.tree.max_priority
    per_buffer.add(mock_experience)
    assert len(per_buffer) == 1
    # Check if added with initial max priority
    # Find the tree index corresponding to the added data
    # data_pointer points to the *next* available slot, so the last added is at data_pointer - 1
    data_idx = (
        per_buffer.tree.data_pointer - 1 + per_buffer.capacity
    ) % per_buffer.capacity
    tree_idx = data_idx + per_buffer.capacity - 1
    assert per_buffer.tree.tree[tree_idx] == initial_max_p
    assert per_buffer.tree.data[data_idx] == mock_experience


# Use mock_experience directly injected by pytest
def test_per_buffer_add_batch(
    per_buffer: ExperienceBuffer, mock_experience: Experience
):
    batch = [mock_experience] * 5
    per_buffer.add_batch(batch)
    assert len(per_buffer) == 5


# Use mock_experience directly injected by pytest
def test_per_buffer_capacity(per_buffer: ExperienceBuffer, mock_experience: Experience):
    for i in range(per_buffer.capacity + 10):
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)  # Adds with current max priority
    assert len(per_buffer) == per_buffer.capacity
    # Cannot easily check which element was overwritten without tracking indices


# Use mock_experience directly injected by pytest
def test_per_buffer_is_ready(per_buffer: ExperienceBuffer, mock_experience: Experience):
    assert not per_buffer.is_ready()
    for _ in range(per_buffer.min_size_to_train):
        per_buffer.add(mock_experience)
    assert per_buffer.is_ready()


# Use mock_experience directly injected by pytest
def test_per_buffer_sample(per_buffer: ExperienceBuffer, mock_experience: Experience):
    # Fill buffer until ready
    for i in range(per_buffer.min_size_to_train):
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)

    # Need current_step for beta calculation
    sample = per_buffer.sample(per_buffer.config.BATCH_SIZE, current_train_step=10)
    assert sample is not None
    assert isinstance(sample, dict)
    assert "batch" in sample
    assert "indices" in sample
    assert "weights" in sample
    assert len(sample["batch"]) == per_buffer.config.BATCH_SIZE
    assert isinstance(sample["batch"][0], tuple)
    assert sample["indices"].shape == (per_buffer.config.BATCH_SIZE,)
    assert sample["weights"].shape == (per_buffer.config.BATCH_SIZE,)
    assert np.all(sample["weights"] >= 0) and np.all(
        sample["weights"] <= 1.0
    )  # Weights are normalized


# Use mock_experience directly injected by pytest
def test_per_buffer_sample_requires_step(
    per_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Fill buffer
    for _ in range(per_buffer.min_size_to_train):
        per_buffer.add(mock_experience)
    with pytest.raises(ValueError, match="current_train_step is required"):
        per_buffer.sample(per_buffer.config.BATCH_SIZE)


# Use mock_experience directly injected by pytest
def test_per_buffer_update_priorities(
    per_buffer: ExperienceBuffer, mock_experience: Experience
):
    # Add some items
    num_items = per_buffer.min_size_to_train
    for i in range(num_items):
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], mock_experience[2] + i)
        per_buffer.add(exp_copy)

    # Sample to get indices
    sample = per_buffer.sample(per_buffer.config.BATCH_SIZE, current_train_step=1)
    assert sample is not None
    indices = sample["indices"]  # These are tree indices

    # Update with some errors
    td_errors = rng.random(per_buffer.config.BATCH_SIZE) * 0.5  # Example errors
    per_buffer.update_priorities(indices, td_errors)

    # --- Verification Adjustment ---
    # Instead of comparing the whole batch, compare based on unique indices.
    # Create a mapping from tree index to the *last* expected priority for that index.
    expected_priorities_map = {}
    calculated_priorities = np.array(
        [per_buffer._get_priority(err) for err in td_errors]
    )
    for tree_idx, expected_p in zip(indices, calculated_priorities, strict=True):
        expected_priorities_map[tree_idx] = expected_p  # Last write wins

    # Get the actual updated priorities from the tree for the unique indices involved
    # Remove list() call from sorted()
    unique_indices = sorted(expected_priorities_map.keys())
    actual_updated_priorities = [per_buffer.tree.tree[idx] for idx in unique_indices]
    expected_final_priorities = [expected_priorities_map[idx] for idx in unique_indices]

    # Check if priorities changed (at least one should have)
    # initial_priorities_unique = [
    #     per_buffer.tree.tree[idx] for idx in unique_indices
    # ]  # Get initial values for comparison *before* update (this needs adjustment - get before update)
    # Re-sample or store initial priorities before update for a proper check if needed.
    # For now, just check if the final values match the expected final values.

    # Increase tolerance for floating point comparison
    assert np.allclose(
        actual_updated_priorities, expected_final_priorities, rtol=1e-4, atol=1e-4
    ), (
        f"Mismatch between actual tree priorities {actual_updated_priorities} and expected {expected_final_priorities} for unique indices {unique_indices}"
    )


def test_per_buffer_beta_annealing(per_buffer: ExperienceBuffer):
    config = per_buffer.config
    assert per_buffer._calculate_beta(0) == config.PER_BETA_INITIAL
    # Ensure anneal steps is not None and > 0 before division
    anneal_steps = per_buffer.per_beta_anneal_steps
    assert anneal_steps is not None and anneal_steps > 0
    mid_step = anneal_steps // 2
    expected_mid_beta = config.PER_BETA_INITIAL + 0.5 * (
        config.PER_BETA_FINAL - config.PER_BETA_INITIAL
    )
    assert per_buffer._calculate_beta(mid_step) == pytest.approx(expected_mid_beta)
    assert per_buffer._calculate_beta(anneal_steps) == config.PER_BETA_FINAL
    assert per_buffer._calculate_beta(anneal_steps * 2) == config.PER_BETA_FINAL


File: tests\rl\test_trainer.py
# File: tests/rl/test_trainer.py
import numpy as np
import pytest
import torch

from alphatriangle.config import EnvConfig, ModelConfig, TrainConfig
from alphatriangle.nn import NeuralNetwork
from alphatriangle.rl import ExperienceBuffer, Trainer
from alphatriangle.utils.types import Experience, PERBatchSample, StateType

# --- Fixtures ---


@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    # Ensure feature dim matches mock_state_type
    mock_model_config.OTHER_NN_INPUT_FEATURES_DIM = 10
    return mock_model_config


@pytest.fixture
def train_config_uniform(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = False
    return cfg


@pytest.fixture
def train_config_per(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = True
    cfg.PER_BETA_ANNEAL_STEPS = 100  # Set anneal steps
    return cfg


@pytest.fixture
def nn_interface(
    mock_model_config: ModelConfig,
    env_config: EnvConfig,
    train_config_uniform: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing, configured for uniform buffer."""
    # Use train_config_uniform here, or make it parameterizable if needed
    device = torch.device("cpu")  # Use CPU for testing
    nn_interface_instance = NeuralNetwork(
        mock_model_config, env_config, train_config_uniform, device
    )
    # Ensure model is on CPU for testing consistency
    nn_interface_instance.model.to(device)
    nn_interface_instance.model.eval()  # Ensure it starts in eval mode if needed by tests
    return nn_interface_instance


@pytest.fixture
def trainer_uniform(
    nn_interface: NeuralNetwork,
    train_config_uniform: TrainConfig,
    env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance configured for uniform sampling."""
    return Trainer(nn_interface, train_config_uniform, env_config)


@pytest.fixture
def trainer_per(
    nn_interface: NeuralNetwork, train_config_per: TrainConfig, env_config: EnvConfig
) -> Trainer:
    """Provides a Trainer instance configured for PER."""
    # Need to re-create NN interface if its config depends on train_config
    # For now, assume nn_interface created with uniform config is okay for PER trainer too
    return Trainer(nn_interface, train_config_per, env_config)


# Use mock_experience implicitly from tests/conftest.py
@pytest.fixture
def buffer_uniform(
    train_config_uniform: TrainConfig, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a filled uniform buffer."""
    buffer = ExperienceBuffer(train_config_uniform)
    for i in range(buffer.min_size_to_train + 5):
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (
            state_copy,
            mock_experience[1],
            mock_experience[2] + i * 0.1,
        )
        buffer.add(exp_copy)
    return buffer


# Use mock_experience implicitly from tests/conftest.py
@pytest.fixture
def buffer_per(
    train_config_per: TrainConfig, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a filled PER buffer."""
    buffer = ExperienceBuffer(train_config_per)
    for i in range(buffer.min_size_to_train + 5):
        # Correctly copy StateType dict and its NumPy arrays
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i,
            "other_features": mock_experience[0]["other_features"].copy() + i,
        }
        exp_copy: Experience = (
            state_copy,
            mock_experience[1],
            mock_experience[2] + i * 0.1,
        )
        buffer.add(exp_copy)  # Adds with max priority
    return buffer


# --- Tests ---


def test_trainer_initialization(trainer_uniform: Trainer):
    assert trainer_uniform.nn is not None
    assert trainer_uniform.model is not None
    assert trainer_uniform.optimizer is not None
    # Scheduler might be None depending on config
    assert hasattr(trainer_uniform, "scheduler")


# Use mock_experience implicitly from tests/conftest.py
def test_prepare_batch(trainer_uniform: Trainer, mock_experience: Experience):
    """Test the internal _prepare_batch method."""
    batch_size = trainer_uniform.train_config.BATCH_SIZE
    batch = [mock_experience] * batch_size
    # --- CHANGED: Variable name for clarity ---
    grid_t, other_t, policy_target_t, n_step_return_t = trainer_uniform._prepare_batch(
        batch
    )
    # --- END CHANGED ---

    assert grid_t.shape == (
        batch_size,
        trainer_uniform.model_config.GRID_INPUT_CHANNELS,
        trainer_uniform.env_config.ROWS,
        trainer_uniform.env_config.COLS,
    )
    assert other_t.shape == (
        batch_size,
        trainer_uniform.model_config.OTHER_NN_INPUT_FEATURES_DIM,
    )
    assert policy_target_t.shape == (batch_size, trainer_uniform.env_config.ACTION_DIM)
    # --- CHANGED: Assert shape is (batch_size,) ---
    assert n_step_return_t.shape == (batch_size,)
    # --- END CHANGED ---

    assert grid_t.device == trainer_uniform.device
    assert other_t.device == trainer_uniform.device
    assert policy_target_t.device == trainer_uniform.device
    # --- CHANGED: Check n_step_return_t device ---
    assert n_step_return_t.device == trainer_uniform.device
    # --- END CHANGED ---


def test_train_step_uniform(trainer_uniform: Trainer, buffer_uniform: ExperienceBuffer):
    """Test a single training step with uniform sampling."""
    initial_params = [p.clone() for p in trainer_uniform.model.parameters()]
    sample = buffer_uniform.sample(trainer_uniform.train_config.BATCH_SIZE)
    assert sample is not None

    result = trainer_uniform.train_step(sample)

    assert result is not None
    loss_info, td_errors = result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert loss_info["total_loss"] > 0  # Loss should generally be positive

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer_uniform.train_config.BATCH_SIZE,)

    # Check if model parameters changed
    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer_uniform.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change after training step."


def test_train_step_per(trainer_per: Trainer, buffer_per: ExperienceBuffer):
    """Test a single training step with PER."""
    initial_params = [p.clone() for p in trainer_per.model.parameters()]
    # Need current_step for PER beta calculation
    sample = buffer_per.sample(
        trainer_per.train_config.BATCH_SIZE, current_train_step=10
    )
    assert sample is not None

    result = trainer_per.train_step(sample)

    assert result is not None
    loss_info, td_errors = result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert loss_info["total_loss"] > 0

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer_per.train_config.BATCH_SIZE,)
    assert np.all(np.isfinite(td_errors))  # TD errors should be finite

    # Check if model parameters changed
    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer_per.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change after training step."


def test_train_step_empty_batch(trainer_uniform: Trainer):
    """Test train_step with an empty batch."""
    empty_sample: PERBatchSample = {
        "batch": [],
        "indices": np.array([]),
        "weights": np.array([]),
    }
    result = trainer_uniform.train_step(empty_sample)
    assert result is None


def test_get_current_lr(trainer_uniform: Trainer):
    """Test retrieving the current learning rate."""
    lr = trainer_uniform.get_current_lr()
    assert isinstance(lr, float)
    assert (
        lr == trainer_uniform.train_config.LEARNING_RATE
    )  # Initially should be the base LR

    # Simulate scheduler step if scheduler exists
    if trainer_uniform.scheduler:
        trainer_uniform.scheduler.step()
        lr_after_step = trainer_uniform.get_current_lr()
        assert isinstance(lr_after_step, float)
        # Cannot assert exact value without knowing scheduler type/params easily
        # Just check it's still a float


File: tests\rl\__init__.py


File: tests\stats\test_collector.py
# File: tests/stats/test_collector.py
import logging
from collections import deque

import cloudpickle
import pytest
import ray

from alphatriangle.stats import StatsCollectorActor
from alphatriangle.utils.types import StepInfo  # Import StepInfo

# --- Fixtures ---


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    if not ray.is_initialized():
        ray.init(logging_level=logging.WARNING, num_cpus=1)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def stats_actor():
    """Provides a fresh StatsCollectorActor instance for each test."""
    actor = StatsCollectorActor.remote(max_history=5)
    # Ensure actor is initialized before returning
    ray.get(actor.clear.remote())  # Use a simple remote call to wait for init
    yield actor
    # Clean up the actor after the test
    ray.kill(actor, no_restart=True)


# --- Helper to create StepInfo ---
def create_step_info(step: int) -> StepInfo:
    """Creates a basic StepInfo dict for testing."""
    return {"global_step": step}


# --- Tests ---


def test_actor_initialization(stats_actor):
    """Test if the actor initializes correctly."""
    assert ray.get(stats_actor.get_data.remote()) == {}
    # Also check initial worker states
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_single_metric(stats_actor):
    """Test logging a single metric."""
    metric_name = "test_metric"
    value = 10.5
    step = 1
    # --- CHANGED: Pass StepInfo ---
    step_info = create_step_info(step)
    ray.get(stats_actor.log.remote(metric_name, value, step_info))
    # --- END CHANGED ---
    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert data[metric_name][0] == (step_info, value)
    # --- END CHANGED ---


def test_log_batch_metrics(stats_actor):
    """Test logging a batch of metrics."""
    # --- CHANGED: Pass StepInfo ---
    step_info_1 = create_step_info(1)
    step_info_2 = create_step_info(2)
    ray.get(stats_actor.log.remote("metric_a", 1.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_b", 2.5, step_info_1))
    ray.get(stats_actor.log.remote("metric_a", 1.1, step_info_2))
    # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert "metric_a" in data
    assert "metric_b" in data
    assert len(data["metric_a"]) == 2, (
        f"Expected 2 entries for metric_a, found {len(data['metric_a'])}"
    )
    assert len(data["metric_b"]) == 1
    # --- CHANGED: Check StepInfo in results ---
    assert data["metric_a"][0] == (step_info_1, 1.0)
    assert data["metric_a"][1] == (step_info_2, 1.1)
    assert data["metric_b"][0] == (step_info_1, 2.5)
    # --- END CHANGED ---


def test_max_history(stats_actor):
    """Test if the max_history constraint is enforced."""
    metric_name = "history_test"
    max_hist = 5  # Matches fixture
    for i in range(max_hist + 3):
        # --- CHANGED: Pass StepInfo ---
        step_info = create_step_info(i)
        ray.get(stats_actor.log.remote(metric_name, float(i), step_info))
        # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == max_hist
    # Check if the first elements were dropped
    # --- CHANGED: Check StepInfo in result ---
    expected_first_step_info = create_step_info(3)
    assert data[metric_name][0] == (expected_first_step_info, 3.0)
    expected_last_step_info = create_step_info(max_hist + 2)
    assert data[metric_name][-1] == (expected_last_step_info, float(max_hist + 2))
    # --- END CHANGED ---


def test_get_metric_data(stats_actor):
    """Test retrieving data for a specific metric."""
    # --- CHANGED: Pass StepInfo ---
    step_info_1 = create_step_info(1)
    step_info_2 = create_step_info(2)
    ray.get(stats_actor.log.remote("metric_1", 10.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_2", 20.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_1", 11.0, step_info_2))
    # --- END CHANGED ---

    metric1_data = ray.get(stats_actor.get_metric_data.remote("metric_1"))
    metric2_data = ray.get(stats_actor.get_metric_data.remote("metric_2"))
    metric3_data = ray.get(stats_actor.get_metric_data.remote("metric_3"))

    assert isinstance(metric1_data, deque)
    assert len(metric1_data) == 2
    # --- CHANGED: Check StepInfo in results ---
    assert list(metric1_data) == [(step_info_1, 10.0), (step_info_2, 11.0)]
    # --- END CHANGED ---

    assert isinstance(metric2_data, deque)
    assert len(metric2_data) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert list(metric2_data) == [(step_info_1, 20.0)]
    # --- END CHANGED ---

    assert metric3_data is None


def test_clear_data(stats_actor):
    """Test clearing the collected data."""
    # --- CHANGED: Pass StepInfo ---
    step_info = create_step_info(1)
    ray.get(stats_actor.log.remote("metric_1", 10.0, step_info))
    # --- END CHANGED ---
    assert len(ray.get(stats_actor.get_data.remote())) == 1
    ray.get(stats_actor.clear.remote())
    assert ray.get(stats_actor.get_data.remote()) == {}
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_non_finite(stats_actor):
    """Test that non-finite values are not logged."""
    metric_name = "non_finite_test"
    # --- CHANGED: Pass StepInfo ---
    ray.get(stats_actor.log.remote(metric_name, float("inf"), create_step_info(1)))
    ray.get(stats_actor.log.remote(metric_name, float("-inf"), create_step_info(2)))
    ray.get(stats_actor.log.remote(metric_name, float("nan"), create_step_info(3)))
    step_info_4 = create_step_info(4)
    ray.get(stats_actor.log.remote(metric_name, 10.0, step_info_4))
    # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert data[metric_name][0] == (step_info_4, 10.0)
    # --- END CHANGED ---


def test_get_set_state(stats_actor):
    """Test saving and restoring the actor's state."""
    # --- CHANGED: Pass StepInfo ---
    step_info_10 = create_step_info(10)
    step_info_11 = create_step_info(11)
    ray.get(stats_actor.log.remote("m1", 1.0, step_info_10))
    ray.get(stats_actor.log.remote("m2", 2.0, step_info_10))
    ray.get(stats_actor.log.remote("m1", 1.5, step_info_11))
    # --- END CHANGED ---

    state = ray.get(stats_actor.get_state.remote())

    # Verify state structure (basic check)
    assert isinstance(state, dict)
    assert "max_history" in state
    assert "_metrics_data_list" in state
    assert isinstance(state["_metrics_data_list"], dict)
    assert "m1" in state["_metrics_data_list"]
    assert isinstance(state["_metrics_data_list"]["m1"], list)
    # --- CHANGED: Check StepInfo in results ---
    assert state["_metrics_data_list"]["m1"] == [
        (step_info_10, 1.0),
        (step_info_11, 1.5),
    ], f"Actual m1 list: {state['_metrics_data_list']['m1']}"
    assert state["_metrics_data_list"]["m2"] == [(step_info_10, 2.0)], (
        f"Actual m2 list: {state['_metrics_data_list']['m2']}"
    )
    # --- END CHANGED ---

    # Use cloudpickle to simulate saving/loading
    pickled_state = cloudpickle.dumps(state)
    unpickled_state = cloudpickle.loads(pickled_state)

    # Create a new actor and restore state
    new_actor = StatsCollectorActor.remote(
        max_history=10
    )  # Different initial max_history
    ray.get(new_actor.set_state.remote(unpickled_state))

    # Verify restored state
    restored_data = ray.get(new_actor.get_data.remote())
    original_data = ray.get(
        stats_actor.get_data.remote()
    )  # Get original data again for comparison

    assert len(restored_data) == len(original_data)
    assert "m1" in restored_data
    assert "m2" in restored_data
    # Compare the deques after converting to lists
    assert list(restored_data["m1"]) == list(original_data["m1"])
    assert list(restored_data["m2"]) == list(original_data["m2"])

    # Check max_history was restored
    # Check behavior by adding more data
    # --- CHANGED: Pass StepInfo ---
    step_info_12 = create_step_info(12)
    step_info_13 = create_step_info(13)
    step_info_14 = create_step_info(14)
    step_info_15 = create_step_info(15)
    ray.get(new_actor.log.remote("m1", 2.0, step_info_12))
    ray.get(new_actor.log.remote("m1", 2.5, step_info_13))
    ray.get(new_actor.log.remote("m1", 3.0, step_info_14))
    ray.get(new_actor.log.remote("m1", 3.5, step_info_15))
    # --- END CHANGED ---

    restored_m1 = ray.get(new_actor.get_metric_data.remote("m1"))
    assert len(restored_m1) == 5  # Max history from loaded state
    # --- CHANGED: Check StepInfo in result ---
    assert restored_m1[0] == (step_info_11, 1.5)  # Check first element is correct
    # --- END CHANGED ---

    # Check that worker states were cleared on restore
    assert ray.get(new_actor.get_latest_worker_states.remote()) == {}

    ray.kill(new_actor, no_restart=True)


# --- Tests for Game State Handling ---
# Mock GameState class for testing state updates
class MockGameStateForStats:
    def __init__(self, step: int, score: float):
        self.current_step = step
        self.game_score = score
        # Add dummy attributes expected by the check in update_worker_game_state
        self.grid_data = True
        self.shapes = True


def test_update_and_get_worker_state(stats_actor):
    """Test updating and retrieving worker game states."""
    worker_id = 1
    state1 = MockGameStateForStats(step=10, score=5.0)
    state2 = MockGameStateForStats(step=11, score=6.0)

    # Initial state should be empty
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}

    # Update state for worker 1
    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state1))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 10
    assert latest_states[worker_id].game_score == 5.0

    # Update state again for worker 1
    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state2))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id].game_score == 6.0

    # Update state for worker 2
    worker_id_2 = 2
    state3 = MockGameStateForStats(step=5, score=2.0)
    ray.get(stats_actor.update_worker_game_state.remote(worker_id_2, state3))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert worker_id_2 in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id_2].current_step == 5


File: tests\stats\__init__.py



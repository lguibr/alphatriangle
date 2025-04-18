# AlphaTriangle Project

<!-- Badges -->
[![CI/CD Status](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml)
[![codecov](https://codecov.io/gh/lguibr/alphatriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/alphatriangle)
[![PyPI version](https://badge.fury.io/py/alphatriangle.svg)](https://badge.fury.io/py/alphatriangle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

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
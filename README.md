
[![CI/CD Status](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/alphatriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/alphatriangle) - [![PyPI version](https://badge.fury.io/py/alphatriangle.svg)](https://badge.fury.io/py/alphatriangle)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# AlphaTriangle

<img src="bitmap.png" alt="AlphaTriangle Logo" width="300"/>

## Overview

AlphaTriangle is a project implementing an artificial intelligence agent based on AlphaZero principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through self-play reinforcement learning, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch). **It uses the `trianglengin` library for core game logic.**

The project includes:

*   An implementation of the MCTS algorithm tailored for the game.
*   A deep neural network (policy and value heads) implemented in PyTorch, featuring convolutional layers and **optional Transformer Encoder layers**.
*   A reinforcement learning pipeline coordinating **parallel self-play (using Ray)**, data storage, and network training, managed by the `alphatriangle.training` module.
*   Experiment tracking using MLflow and TensorBoard.
*   Unit tests for RL components.
*   A command-line interface for running the **headless** training pipeline.

## Core Technologies

*   **Python 3.10+**
*   **trianglengin:** Core game engine (state, actions, rules, basic viz/interaction).
*   **PyTorch:** For the deep learning model (CNNs, **optional Transformers**, Distributional Value Head) and training, with CUDA/MPS support.
*   **NumPy:** For numerical operations, especially state representation (used by `trianglengin` and features).
*   **Ray:** For parallelizing self-play data generation and statistics collection across multiple CPU cores/processes.
*   **Numba:** (Optional, used in `features.grid_features`) For performance optimization of specific grid calculations.
*   **Cloudpickle:** For serializing the experience replay buffer and training checkpoints.
*   **MLflow:** For logging parameters, metrics, and artifacts (checkpoints, buffers) during training runs.
*   **TensorBoard:** For visualizing metrics during training.
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
│           ├── tensorboard/ # TensorBoard logs (within MLflow artifacts)
│           └── configs.json
├── alphatriangle/          # Source code for the AlphaZero agent package
│   ├── __init__.py
│   ├── cli.py              # CLI logic (train command - headless only)
│   ├── config/             # Pydantic configuration models (MCTS, Model, Train, Persistence)
│   │   └── README.md
│   ├── data/               # Data saving/loading logic (DataManager, Schemas)
│   │   └── README.md
│   ├── features/           # Feature extraction logic (operates on trianglengin.GameState)
│   │   └── README.md
│   ├── mcts/               # Monte Carlo Tree Search (operates on trianglengin.GameState)
│   │   └── README.md
│   ├── nn/                 # Neural network definition and wrapper
│   │   └── README.md
│   ├── rl/                 # RL components (Trainer, Buffer, Worker)
│   │   └── README.md
│   ├── stats/              # Statistics collection actor (StatsCollectorActor)
│   │   └── README.md
│   ├── training/           # Training orchestration (Loop, Setup, Runner)
│   │   └── README.md
│   └── utils/              # Shared utilities and types (specific to AlphaTriangle)
│       └── README.md
├── tests/                  # Unit tests (for alphatriangle components)
│   ├── conftest.py
│   ├── mcts/
│   ├── nn/
│   ├── rl/
│   ├── stats/
│   └── training/
├── .gitignore
├── .python-version
├── LICENSE                 # License file (MIT)
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration (depends on trianglengin)
├── README.md               # This file
└── requirements.txt        # List of dependencies (includes trianglengin)
```

## Key Modules (`alphatriangle`)

*   **`cli`:** Defines the command-line interface using Typer (**only `train` command, headless**). ([`alphatriangle/cli.py`](alphatriangle/cli.py))
*   **`config`:** Centralized Pydantic configuration classes (excluding `EnvConfig` and `VisConfig`). ([`alphatriangle/config/README.md`](alphatriangle/config/README.md))
*   **`features`:** Contains logic to convert `trianglengin.GameState` objects into numerical features (`StateType`). ([`alphatriangle/features/README.md`](alphatriangle/features/README.md))
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`AlphaTriangleNet`) and a wrapper class (`NeuralNetwork`). ([`alphatriangle/nn/README.md`](alphatriangle/nn/README.md))
*   **`mcts`:** Implements the Monte Carlo Tree Search algorithm (`Node`, `run_mcts_simulations`), operating on `trianglengin.GameState`. ([`alphatriangle/mcts/README.md`](alphatriangle/mcts/README.md))
*   **`rl`:** Contains RL components: `Trainer` (network updates), `ExperienceBuffer` (data storage, **supports PER**), and `SelfPlayWorker` (Ray actor for parallel self-play using `trianglengin.GameState`). ([`alphatriangle/rl/README.md`](alphatriangle/rl/README.md))
*   **`training`:** Orchestrates the **headless** training process using `TrainingLoop`, managing workers, data flow, logging, and checkpoints. Includes `runner.py` for the callable training function. ([`alphatriangle/training/README.md`](alphatriangle/training/README.md))
*   **`stats`:** Contains the `StatsCollectorActor` (Ray actor) for asynchronous statistics collection. ([`alphatriangle/stats/README.md`](alphatriangle/stats/README.md))
*   **`data`:** Manages saving and loading of training artifacts (`DataManager`) using Pydantic schemas and `cloudpickle`. ([`alphatriangle/data/README.md`](alphatriangle/data/README.md))
*   **`utils`:** Provides common helper functions and shared type definitions specific to the AlphaZero implementation. ([`alphatriangle/utils/README.md`](alphatriangle/utils/README.md))

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
3.  **Install the package (including `trianglengin`):**
    *   **For users:**
        ```bash
        # This will automatically install trianglengin from PyPI if available
        pip install alphatriangle
        # Or install directly from Git (installs trianglengin from PyPI)
        # pip install git+https://github.com/lguibr/alphatriangle.git
        ```
    *   **For developers (editable install):**
        *   First, ensure `trianglengin` is installed (ideally in editable mode from its own directory if developing both):
            ```bash
            # From the trianglengin directory:
            # pip install -e .
            ```
        *   Then, install `alphatriangle` in editable mode:
            ```bash
            # From the alphatriangle directory:
            pip install -e .
            # Install dev dependencies (optional, for running tests/linting)
            pip install -e .[dev] # Installs dev deps from pyproject.toml
            ```
    *Note: Ensure you have the correct PyTorch version installed for your system (CPU/CUDA/MPS). See [pytorch.org](https://pytorch.org/). Ray may have specific system requirements.*
4.  **(Optional) Add data directory to `.gitignore`:**
    Create or edit the `.gitignore` file in your project root and add the line:
    ```
    .alphatriangle_data/
    ```

## Running the Code (CLI)

Use the `alphatriangle` command for training:

*   **Show Help:**
    ```bash
    alphatriangle --help
    ```
*   **Run Training (Headless Only):**
    ```bash
    alphatriangle train [--seed 42] [--log-level INFO]
    ```
*   **Interactive Play/Debug (Use `trianglengin` CLI):**
    ```bash
    # Ensure trianglengin is installed
    trianglengin play [--seed 42] [--log-level INFO]
    trianglengin debug [--seed 42] [--log-level DEBUG]
    ```
*   **Monitoring Training (MLflow UI):**
    While training, or after runs have completed, open a separate terminal in the project root and run:
    ```bash
    mlflow ui --backend-store-uri file:./.alphatriangle_data/mlruns
    ```
    Then navigate to `http://localhost:5000` (or the specified port) in your browser.
*   **Monitoring Training (TensorBoard):**
    While training, or after runs have completed, open a separate terminal in the project root and run:
    ```bash
    tensorboard --logdir .alphatriangle_data/runs/<run_name>/tensorboard
    # Or point to the parent directory if MLflow logged it under artifacts:
    # tensorboard --logdir .alphatriangle_data/mlruns/<experiment_id>/<run_id>/artifacts/tensorboard
    ```
    Then navigate to `http://localhost:6006` (or the specified port) in your browser.
*   **Running Unit Tests (Development):**
    ```bash
    pytest tests/
    ```

## Configuration

All major parameters for the AlphaZero agent (MCTS, Model, Training, Persistence) are defined in the Pydantic classes within the `alphatriangle/config/` directory. Modify these files to experiment with different settings. Environment configuration (`EnvConfig`) and basic visualization config (`VisConfig`) are defined within the `trianglengin` library.

## Data Storage

All persistent data, including MLflow tracking data, TensorBoard logs, and run-specific artifacts, is stored within the `.alphatriangle_data/` directory in the project root, managed by the `DataManager` and MLflow.

## Maintainability

This project includes README files within each major `alphatriangle` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic.
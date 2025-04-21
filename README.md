
[![CI/CD Status](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/alphatriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/alphatriangle) - [![PyPI version](https://badge.fury.io/py/alphatriangle.svg)](https://badge.fury.io/py/alphatriangle)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# AlphaTriangle

<img src="bitmap.png" alt="AlphaTriangle Logo" width="300"/>

## Overview

AlphaTriangle is a project implementing an artificial intelligence agent based on AlphaZero principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through **headless self-play reinforcement learning**, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch).

**Key Features:**

*   **Core Game Logic:** Uses the [`trianglengin>=2.0.1`](https://github.com/lguibr/trianglengin) library for the triangle puzzle game rules and state management, featuring a high-performance C++ core.
*   **High-Performance MCTS:** Integrates the [`trimcts`](https://github.com/lguibr/trimcts) library, providing a **C++ implementation of MCTS** for efficient search, callable from Python. MCTS parameters are configurable via `alphatriangle/config/mcts_config.py` (defaulting to 64 simulations for faster testing/profiling, adjust as needed).
*   **Deep Learning Model:** Features a PyTorch neural network with policy and distributional value heads, convolutional layers, and **optional Transformer Encoder layers**.
*   **Parallel Self-Play:** Leverages **Ray** for distributed self-play data generation across multiple CPU cores.
*   **Experiment Tracking:** Uses **MLflow** and **TensorBoard** for logging parameters, metrics, and artifacts, enabling web-based monitoring.
*   **Headless Training:** Focuses on a command-line interface for running the training pipeline without visual output.
*   **Unit Tests:** Includes tests for RL components.

---

## 🎮 The Triangle Puzzle Game Guide 🧩

This project trains an agent to play the game defined by the `trianglengin` library. The rules are detailed in the [`trianglengin` README](https://github.com/lguibr/trianglengin/blob/main/README.md#the-ultimate-triangle-puzzle-guide-).

---

## Core Technologies

*   **Python 3.10+**
*   **trianglengin>=2.0.1:** Core game engine (state, actions, rules) with C++ optimizations.
*   **trimcts>=0.1.0:** High-performance C++ MCTS implementation with Python bindings.
*   **PyTorch:** For the deep learning model (CNNs, **optional Transformers**, Distributional Value Head) and training, with CUDA/MPS support.
*   **NumPy:** For numerical operations, especially state representation (used by `trianglengin` and features).
*   **Ray:** For parallelizing self-play data generation and statistics collection across multiple CPU cores/processes.
*   **Numba:** (Optional, used in `features.grid_features`) For performance optimization of specific grid calculations.
*   **Cloudpickle:** For serializing the experience replay buffer and training checkpoints.
*   **MLflow:** For logging parameters, metrics, and artifacts (checkpoints, buffers) during training runs. **Provides the primary web UI dashboard for experiment management.**
*   **TensorBoard:** For visualizing metrics during training (e.g., detailed loss curves). **Provides a secondary web UI dashboard, often with faster graph updates.**
*   **Pydantic:** For configuration management and data validation.
*   **Typer:** For the command-line interface.
*   **Pytest:** For running unit tests.

## Project Structure

```markdown
.
├── .github/workflows/      # GitHub Actions CI/CD
│   └── ci_cd.yml
├── .alphatriangle_data/    # Root directory for ALL persistent data (GITIGNORED)
│   ├── mlruns/             # MLflow internal tracking data & artifact store (for UI)
│   └── runs/               # Local artifacts per run (checkpoints, buffers, TB logs, configs)
│       └── <run_name>/
│           ├── checkpoints/ # Saved model weights & optimizer states
│           ├── buffers/     # Saved experience replay buffers
│           ├── logs/        # Plain text log files for the run
│           ├── tensorboard/ # TensorBoard log files (scalars, etc.)
│           └── configs.json # Copy of run configuration
├── alphatriangle/          # Source code for the AlphaZero agent package
│   ├── __init__.py
│   ├── cli.py              # CLI logic (train command - headless only)
│   ├── config/             # Pydantic configuration models (Model, Train, Persistence, MCTS)
│   │   └── README.md
│   ├── data/               # Data saving/loading logic (DataManager, Schemas)
│   │   └── README.md
│   ├── features/           # Feature extraction logic (operates on trianglengin.GameState)
│   │   └── README.md
│   ├── nn/                 # Neural network definition and wrapper (implements trimcts.AlphaZeroNetworkInterface)
│   │   └── README.md
│   ├── rl/                 # RL components (Trainer, Buffer, Worker using trimcts)
│   │   └── README.md
│   ├── stats/              # Statistics collection actor (StatsCollectorActor)
│   │   └── README.md
│   ├── training/           # Training orchestration (Loop, Setup, Runner)
│   │   └── README.md
│   └── utils/              # Shared utilities and types (specific to AlphaTriangle)
│       └── README.md
├── tests/                  # Unit tests (for alphatriangle components, excluding MCTS)
│   ├── conftest.py
│   ├── nn/
│   ├── rl/
│   ├── stats/
│   └── training/
├── .gitignore
├── .python-version
├── LICENSE                 # License file (MIT)
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration (depends on trianglengin, trimcts)
├── README.md               # This file
└── requirements.txt        # List of dependencies (includes trianglengin, trimcts)
```

## Key Modules (`alphatriangle`)

*   **`cli`:** Defines the command-line interface using Typer (**only `train` command, headless**). ([`alphatriangle/cli.py`](alphatriangle/cli.py))
*   **`config`:** Centralized Pydantic configuration classes (Model, Train, Persistence, **MCTS**). ([`alphatriangle/config/README.md`](alphatriangle/config/README.md))
*   **`features`:** Contains logic to convert `trianglengin.GameState` objects into numerical features (`StateType`). ([`alphatriangle/features/README.md`](alphatriangle/features/README.md))
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`AlphaTriangleNet`) and a wrapper class (`NeuralNetwork`). **The `NeuralNetwork` class implicitly conforms to the `trimcts.AlphaZeroNetworkInterface` protocol.** ([`alphatriangle/nn/README.md`](alphatriangle/nn/README.md))
*   **`rl`:** Contains RL components: `Trainer` (network updates), `ExperienceBuffer` (data storage, **supports PER**), and `SelfPlayWorker` (Ray actor for parallel self-play **using `trimcts.run_mcts`**). ([`alphatriangle/rl/README.md`](alphatriangle/rl/README.md))
*   **`training`:** Orchestrates the **headless** training process using `TrainingLoop`, managing workers, data flow, logging (to console, file, MLflow, TensorBoard), and checkpoints. Includes `runner.py` for the callable training function. ([`alphatriangle/training/README.md`](alphatriangle/training/README.md))
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
3.  **Install the package (including `trianglengin` and `trimcts`):**
    *   **For users:**
        ```bash
        # This will automatically install trianglengin and trimcts from PyPI if available
        pip install alphatriangle
        # Or install directly from Git (installs dependencies from PyPI)
        # pip install git+https://github.com/lguibr/alphatriangle.git
        ```
    *   **For developers (editable install):**
        *   First, ensure `trianglengin` and `trimcts` are installed (ideally in editable mode from their own directories if developing all three):
            ```bash
            # From the trianglengin directory (requires C++ build tools):
            # pip install -e .
            # From the trimcts directory (requires C++ build tools):
            # pip install -e .
            ```
        *   Then, install `alphatriangle` in editable mode:
            ```bash
            # From the alphatriangle directory:
            pip install -e .
            # Install dev dependencies (optional, for running tests/linting)
            pip install -e .[dev] # Installs dev deps from pyproject.toml
            ```
    *Note: Ensure you have the correct PyTorch version installed for your system (CPU/CUDA/MPS). See [pytorch.org](https://pytorch.org/). Ray may have specific system requirements. `trianglengin` and `trimcts` require a C++ compiler (like GCC, Clang, or MSVC) and CMake.*
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
    *Note: Interactive modes are part of the `trianglengin` library, not this `alphatriangle` package.*
    ```bash
    # Ensure trianglengin is installed
    trianglengin play [--seed 42] [--log-level INFO]
    trianglengin debug [--seed 42] [--log-level DEBUG]
    ```
*   **Monitoring Training (Web Dashboards):**
    This project uses **MLflow** and **TensorBoard** to provide web-based dashboards for monitoring. It's recommended to run both concurrently for the best experience:
    *   **MLflow UI (Experiment Overview & Artifacts):**
        Provides the main dashboard for comparing runs, viewing parameters, high-level metrics, and accessing saved artifacts (checkpoints, buffers). Updates occur as data is logged, but may require a browser refresh for the latest points.
        ```bash
        # Run from the project root directory
        mlflow ui --backend-store-uri file:./.alphatriangle_data/mlruns
        ```
        Access via `http://localhost:5000`.
    *   **TensorBoard (Near Real-Time Graphs):**
        Offers more frequently updated graphs of scalar metrics (losses, rates, etc.) during a run, making it ideal for closely monitoring training progress.
        ```bash
        # Run from the project root directory, pointing to the *specific run's* TB logs
        tensorboard --logdir .alphatriangle_data/runs/<your_run_name>/tensorboard
        # Replace <your_run_name> with the actual name (e.g., train_20240101_120000)
        # You can also point to the parent 'runs' directory to see all runs:
        # tensorboard --logdir .alphatriangle_data/runs
        ```
        Access via `http://localhost:6006`.
*   **Running Unit Tests (Development):**
    ```bash
    pytest tests/
    ```

## Configuration

All major parameters for the AlphaZero agent (Model, Training, Persistence, **MCTS**) are defined in the Pydantic classes within the `alphatriangle/config/` directory. Modify these files to experiment with different settings. Environment configuration (`EnvConfig`) is defined within the `trianglengin` library.

## Data Storage

All persistent data is stored within the `.alphatriangle_data/` directory in the project root.
*   **`.alphatriangle_data/mlruns/`**: Managed by **MLflow**. Contains MLflow's internal tracking data (parameters, metrics) and its own copy of logged artifacts. This is the source for the MLflow UI.
*   **`.alphatriangle_data/runs/`**: Managed by **DataManager**. Contains locally saved artifacts for each run (checkpoints, buffers, TensorBoard logs, configs) before/during logging to MLflow. This directory is used for auto-resuming and direct access to TensorBoard logs during a run.

## Maintainability

This project includes README files within each major `alphatriangle` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic.
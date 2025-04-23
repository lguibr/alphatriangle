
[![CI/CD Status](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/alphatriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/alphatriangle) - [![PyPI version](https://badge.fury.io/py/alphatriangle.svg)](https://badge.fury.io/py/alphatriangle)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# AlphaTriangle

<img src="bitmap.png" alt="AlphaTriangle Logo" width="300"/>

## Overview

AlphaTriangle is a project implementing an artificial intelligence agent based on AlphaZero principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through **headless self-play reinforcement learning**, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch).

**Key Features:**

*   **Core Game Logic:** Uses the [`trianglengin>=2.0.7`](https://github.com/lguibr/trianglengin) library for the triangle puzzle game rules and state management, featuring a high-performance C++ core.
*   **High-Performance MCTS:** Integrates the [`trimcts>=1.2.1`](https://github.com/lguibr/trimcts) library, providing a **C++ implementation of MCTS** for efficient search, callable from Python. MCTS parameters are configurable via `alphatriangle/config/mcts_config.py`.
*   **Deep Learning Model:** Features a PyTorch neural network with policy and distributional value heads, convolutional layers, and **optional Transformer Encoder layers**.
*   **Parallel Self-Play:** Leverages **Ray** for distributed self-play data generation across multiple CPU cores. **The number of workers automatically adjusts based on detected CPU cores (reserving some for stability), capped by the `NUM_SELF_PLAY_WORKERS` setting in `TrainConfig`.**
*   **Asynchronous Stats & Persistence (NEW):** Uses the [`trieye>=0.1.2`](https://github.com/lguibr/trieye) library, which provides a dedicated **Ray actor (`TrieyeActor`)** for:
    *   Asynchronous collection of raw metric events from workers and the training loop.
    *   Configurable processing and aggregation of metrics.
    *   Logging processed metrics to **MLflow** and **TensorBoard**.
    *   Saving/loading training state (checkpoints, buffers) to the filesystem and logging artifacts to MLflow.
    *   Handling auto-resumption from previous runs.
    *   All persistent data managed by `trieye` is stored within the `.trieye_data/<app_name>` directory (default: `.trieye_data/alphatriangle`).
*   **Headless Training:** Focuses on a command-line interface for running the training pipeline without visual output.
*   **Enhanced CLI:** Uses the **`rich`** library for improved visual feedback (colors, panels, emojis) in the terminal.
*   **Centralized Logging:** Uses Python's standard `logging` module configured centrally for consistent log formatting (including `▲` prefix) and level control across the project. Run logs are saved to `.trieye_data/<app_name>/runs/<run_name>/logs/`.
*   **Optional Profiling:** Supports profiling worker 0 using `cProfile` via a command-line flag. Profile data is saved to `.trieye_data/<app_name>/runs/<run_name>/profile_data/`.
*   **Unit Tests:** Includes tests for RL components.

---

## 🎮 The Triangle Puzzle Game Guide 🧩

This project trains an agent to play the game defined by the `trianglengin` library. The rules are detailed in the [`trianglengin` README](https://github.com/lguibr/trianglengin/blob/main/README.md#the-ultimate-triangle-puzzle-guide-).

---

## Core Technologies

*   **Python 3.10+**
*   **trianglengin>=2.0.7:** Core game engine (state, actions, rules) with C++ optimizations.
*   **trimcts>=1.2.1:** High-performance C++ MCTS implementation with Python bindings.
*   **trieye>=0.1.2:** Asynchronous statistics collection, processing, logging (MLflow/TensorBoard), and data persistence via a Ray actor.
*   **PyTorch:** For the deep learning model (CNNs, **optional Transformers**, Distributional Value Head) and training, with CUDA/MPS support.
*   **NumPy:** For numerical operations, especially state representation (used by `trianglengin` and features).
*   **Ray:** For parallelizing self-play data generation and hosting the `TrieyeActor`. **Dynamically scales worker count based on available cores.**
*   **Numba:** (Optional, used in `features.grid_features`) For performance optimization of specific grid calculations.
*   **Cloudpickle:** For serializing the experience replay buffer and training checkpoints (used by `trieye`).
*   **MLflow:** For logging parameters, metrics, and artifacts (checkpoints, buffers) during training runs. **Provides the primary web UI dashboard for experiment management. Data stored in `.trieye_data/<app_name>/mlruns/`.** Managed by `trieye`.
*   **TensorBoard:** For visualizing metrics during training (e.g., detailed loss curves). **Provides a secondary web UI dashboard. Data stored in `.trieye_data/<app_name>/runs/<run_name>/tensorboard/`.** Managed by `trieye`.
*   **Pydantic:** For configuration management and data validation (used by `alphatriangle` and `trieye`).
*   **Typer:** For the command-line interface.
*   **Rich:** For enhanced CLI output formatting and styling.
*   **Pytest:** For running unit tests.

## Project Structure

```markdown
.
├── .github/workflows/      # GitHub Actions CI/CD
│   └── ci_cd.yml
├── .trieye_data/           # Root directory for ALL persistent data (GITIGNORED) - Managed by Trieye
│   └── alphatriangle/      # Default app_name
│       ├── mlruns/         # MLflow internal tracking data & artifact store (for MLflow UI)
│       │   └── <experiment_id>/
│       │       └── <mlflow_run_id>/
│       │           ├── artifacts/ # MLflow's copy of logged artifacts (checkpoints, buffers, etc.)
│       │           ├── metrics/
│       │           ├── params/
│       │           └── tags/
│       └── runs/           # Local artifacts per run (source for TensorBoard UI & resume)
│           └── <run_name>/ # e.g., train_YYYYMMDD_HHMMSS
│               ├── checkpoints/ # Saved model weights & optimizer states (*.pkl)
│               ├── buffers/     # Saved experience replay buffers (*.pkl)
│               ├── logs/        # Plain text log files for the run (*.log) - App + Trieye logs
│               ├── tensorboard/ # TensorBoard log files (event files)
│               ├── profile_data/ # cProfile output files (*.prof) if profiling enabled
│               └── configs.json # Copy of run configuration
├── alphatriangle/          # Source code for the AlphaZero agent package
│   ├── __init__.py
│   ├── cli.py              # CLI logic (train, ml, tb, ray commands - headless only, uses Rich)
│   ├── config/             # Pydantic configuration models (Model, Train, MCTS) - Stats/Persistence now in Trieye
│   │   └── README.md
│   ├── features/           # Feature extraction logic (operates on trianglengin.GameState)
│   │   └── README.md
│   ├── logging_config.py   # Centralized logging setup function (for console)
│   ├── nn/                 # Neural network definition and wrapper (implements trimcts.AlphaZeroNetworkInterface)
│   │   └── README.md
│   ├── rl/                 # RL components (Trainer, Buffer, Worker using trimcts)
│   │   └── README.md
│   ├── training/           # Training orchestration (Loop, Setup, Runner, WorkerManager) - Interacts with TrieyeActor
│   │   └── README.md
│   └── utils/              # Shared utilities and types (specific to AlphaTriangle)
│       └── README.md
├── tests/                  # Unit tests (for alphatriangle components, excluding MCTS, Stats, Data)
│   ├── conftest.py
│   ├── nn/
│   ├── rl/
│   └── training/
├── .gitignore
├── .python-version
├── LICENSE                 # License file (MIT)
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration (depends on trianglengin, trimcts, trieye, rich)
├── README.md               # This file
└── requirements.txt        # List of dependencies (includes trianglengin, trimcts, trieye, rich)
```

## Key Modules (`alphatriangle`)

*   **`cli`:** Defines the command-line interface using Typer (**`train`**, **`ml`**, **`tb`**, **`ray`** commands - headless). Uses **`rich`** for styling. ([`alphatriangle/cli.py`](alphatriangle/cli.py))
*   **`config`:** Centralized Pydantic configuration classes (Model, Train, **MCTS**). Imports `EnvConfig` from `trianglengin`. **Uses `TrieyeConfig` from `trieye` for stats/persistence.** ([`alphatriangle/config/README.md`](alphatriangle/config/README.md))
*   **`features`:** Contains logic to convert `trianglengin.GameState` objects into numerical features (`StateType`). ([`alphatriangle/features/README.md`](alphatriangle/features/README.md))
*   **`logging_config`:** Defines the `setup_logging` function for centralized **console** logger configuration. ([`alphatriangle/logging_config.py`](alphatriangle/logging_config.py))
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`AlphaTriangleNet`) and a wrapper class (`NeuralNetwork`). **The `NeuralNetwork` class implicitly conforms to the `trimcts.AlphaZeroNetworkInterface` protocol.** ([`alphatriangle/nn/README.md`](alphatriangle/nn/README.md))
*   **`rl`:** Contains RL components: `Trainer` (network updates), `ExperienceBuffer` (data storage, **supports PER**), and `SelfPlayWorker` (Ray actor for parallel self-play **using `trimcts.run_mcts`**). **Workers now send `RawMetricEvent`s to the `TrieyeActor`.** ([`alphatriangle/rl/README.md`](alphatriangle/rl/README.md))
*   **`training`:** Orchestrates the **headless** training process using `TrainingLoop`, managing workers, data flow, and interaction with the **`TrieyeActor`** for logging, checkpointing, and state loading. Includes `runner.py` for the callable training function. ([`alphatriangle/training/README.md`](alphatriangle/training/README.md))
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
3.  **Install the package (including `trianglengin`, `trimcts`, `trieye`, and `rich`):**
    *   **For users:**
        ```bash
        # This will automatically install trianglengin, trimcts, trieye, and rich from PyPI if available
        pip install alphatriangle
        # Or install directly from Git (installs dependencies from PyPI)
        # pip install git+https://github.com/lguibr/alphatriangle.git
        ```
    *   **For developers (editable install):**
        *   First, ensure `trianglengin`, `trimcts`, and `trieye` are installed (ideally in editable mode from their own directories if developing all):
            ```bash
            # From the trianglengin directory (requires C++ build tools):
            # pip install -e .
            # From the trimcts directory (requires C++ build tools):
            # pip install -e .
            # From the trieye directory:
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
4.  **(Optional but Recommended) Add data directory to `.gitignore`:**
    Ensure the `.gitignore` file in your project root contains the line:
    ```
    .trieye_data/
    ```

## Running the Code (CLI)

Use the `alphatriangle` command for training and monitoring. The CLI uses `rich` for enhanced output.

*   **Show Help:**
    ```bash
    alphatriangle --help
    ```
*   **Run Training (Headless Only):**
    ```bash
    alphatriangle train [--seed 42] [--log-level INFO] [--profile] [--run-name my_custom_run]
    ```
    *   `--log-level`: Set console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO. Logs are saved to `.trieye_data/alphatriangle/runs/<run_name>/logs/`.
    *   `--seed`: Set the random seed for reproducibility. Default: 42.
    *   `--profile`: Enable cProfile for worker 0. Generates `.prof` files in `.trieye_data/alphatriangle/runs/<run_name>/profile_data/`.
    *   `--run-name`: Specify a custom name for the run. Default: `train_YYYYMMDD_HHMMSS`.
*   **Launch MLflow UI:**
    Launches the MLflow web interface, automatically pointing to the `.trieye_data/alphatriangle/mlruns` directory.
    ```bash
    alphatriangle ml [--host 127.0.0.1] [--port 5000]
    ```
    Access via `http://localhost:5000` (or the specified host/port).
*   **Launch TensorBoard UI:**
    Launches the TensorBoard web interface, automatically pointing to the `.trieye_data/alphatriangle/runs` directory (which contains the individual run subdirectories with `tensorboard` logs).
    ```bash
    alphatriangle tb [--host 127.0.0.1] [--port 6006]
    ```
    Access via `http://localhost:6006` (or the specified host/port).
*   **Launch Ray Dashboard UI:**
    Launches the Ray Dashboard web interface. **Note:** This typically requires a Ray cluster to be running (e.g., started by `alphatriangle train` or manually).
    ```bash
    alphatriangle ray [--host 127.0.0.1] [--port 8265]
    ```
    Access via `http://localhost:8265` (or the specified host/port).
*   **Interactive Play/Debug (Use `trianglengin` CLI):**
    *Note: Interactive modes are part of the `trianglengin` library, not this `alphatriangle` package.*
    ```bash
    # Ensure trianglengin is installed
    trianglengin play [--seed 42] [--log-level INFO]
    trianglengin debug [--seed 42] [--log-level DEBUG]
    ```
*   **Running Unit Tests (Development):**
    ```bash
    pytest tests/
    ```
*   **Analyzing Profile Data (if `--profile` was used):**
    Use the provided `analyze_profiles.py` script (requires `typer`).
    ```bash
    python analyze_profiles.py .trieye_data/alphatriangle/runs/<run_name>/profile_data/worker_0_ep_<ep_seed>.prof [-n <num_lines>]
    ```

## Configuration

*   **AlphaTriangle Specific:** Parameters for the Model (`ModelConfig`), Training (`TrainConfig`), and MCTS (`AlphaTriangleMCTSConfig`) are defined in the Pydantic classes within the `alphatriangle/config/` directory.
*   **Environment:** Environment configuration (`EnvConfig`) is defined within the `trianglengin` library.
*   **Stats & Persistence:** Statistics logging and data persistence are configured via `TrieyeConfig` (which includes `StatsConfig` and `PersistenceConfig`) from the `trieye` library. These are typically instantiated in `alphatriangle/training/runner.py` or `alphatriangle/cli.py`.

## Data Storage

All persistent data is now managed by the `trieye` library and stored within the `.trieye_data/<app_name>/` directory (default: `.trieye_data/alphatriangle/`) in the project root. This directory should be added to your `.gitignore`.

*   **`.trieye_data/<app_name>/mlruns/`**: Managed by **MLflow** (via `trieye`). Contains MLflow's internal tracking data (parameters, metrics) and its own copy of logged artifacts. This is the source for the MLflow UI (`alphatriangle ml`).
*   **`.trieye_data/<app_name>/runs/`**: Managed by **`trieye`**. Contains locally saved artifacts for each run (checkpoints, buffers, TensorBoard logs, configs, **profile data**) before/during logging to MLflow. This directory is used for auto-resuming and is the source for the TensorBoard UI (`alphatriangle tb`).
    *   **Replay Buffer Content:** The saved buffer file (`buffer.pkl`) contains `Experience` tuples: `(StateType, PolicyTargetMapping, n_step_return)`. The `StateType` includes:
        *   `grid`: Numerical features representing grid occupancy.
        *   `other_features`: Numerical features derived from game state and available shapes.
    *   **Visualization:** This stored data allows offline analysis and visualization of grid occupancy and the available shapes for each recorded step. It does **not** contain the full sequence of actions or raw `GameState` objects needed for a complete, interactive game replay.

## Maintainability

This project includes README files within each major `alphatriangle` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic, especially regarding the interaction with the `trieye` library.
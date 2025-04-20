
[![CI/CD Status](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/alphatriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/alphatriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/alphatriangle) - [![PyPI version](https://badge.fury.io/py/alphatriangle.svg)](https://badge.fury.io/py/alphatriangle)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# AlphaTriangle

<img src="bitmap.png" alt="AlphaTriangle Logo" width="300"/>

## Overview

AlphaTriangle is a project implementing an artificial intelligence agent based on AlphaZero principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through **headless self-play reinforcement learning**, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch). **It uses the `trianglengin` library for core game logic.**

The project includes:

*   An implementation of the MCTS algorithm tailored for the game.
*   A deep neural network (policy and value heads) implemented in PyTorch, featuring convolutional layers and **optional Transformer Encoder layers**.
*   A reinforcement learning pipeline coordinating **parallel self-play (using Ray)**, data storage, and network training, managed by the `alphatriangle.training` module.
*   Experiment tracking and visualization using **MLflow** and **TensorBoard**.
*   Unit tests for RL components.
*   A command-line interface for running the **headless** training pipeline.

---

## 🎮 The Triangle Puzzle Game Guide 🧩

This project trains an agent to play the game defined by the `trianglengin` library. Here's a detailed explanation of the game rules:

### 1. Introduction: Your Mission! 🎯

The goal is to place colorful shapes onto a special triangular grid. By filling up lines of triangles, you make them disappear and score points! Keep placing shapes and clearing lines for as long as possible to get the highest score before the grid fills up and you run out of moves.

### 2. The Playing Field: The Grid 🗺️

- **Triangle Cells:** The game board is a grid made of many small triangles. Some point UP (🔺) and some point DOWN (🔻). They alternate like a checkerboard pattern based on their row and column index (specifically, `(row + col) % 2 != 0` means UP).
- **Shape:** The grid itself is rectangular overall, but the playable area within it is typically shaped like a triangle or hexagon, wider in the middle and narrower at the top and bottom.
- **Playable Area:** You can only place shapes within the designated playable area.
- **Death Zones 💀:** Around the edges of the playable area (often at the start and end of rows), some triangles are marked as "Death Zones". You **cannot** place any part of a shape onto these triangles. They are off-limits! Think of them as the boundaries within the rectangular grid.

### 3. Your Tools: The Shapes 🟦🟥🟩

- **Shape Formation:** Each shape is a collection of connected small triangles (🔺 and 🔻). They come in different colors and arrangements. Some might be a single triangle, others might be long lines, L-shapes, or more complex patterns.
- **Relative Positions:** The triangles within a shape have fixed positions _relative to each other_. When you move the shape, all its triangles move together as one block.
- **Preview Area:** You will always have **three** shapes available to choose from at any time. These are shown in a special "preview area".

### 4. Making Your Move: Placing Shapes 🖱️➡️▦

This is the core action! Here's exactly how to place a shape:

- **Step 4a: Select a Shape:** Choose one of the three shapes available in the preview area.
- **Step 4b: Aim on the Grid:** Select a target coordinate `(row, col)` on the main grid. This coordinate represents the *anchor* point for placing the shape.
- **Step 4c: The Placement Rules (MUST Follow!)**
  - 📏 **Rule 1: Fit Inside Playable Area:** ALL triangles of your chosen shape must land within the playable grid area. No part of the shape can land in a Death Zone 💀.
  - 🧱 **Rule 2: No Overlap:** ALL triangles of your chosen shape must land on currently _empty_ spaces on the grid. You cannot place a shape on top of triangles that are already filled with color from previous shapes.
  - 📐 **Rule 3: Orientation Match!** This is crucial!
    - If a part of your shape is an UP triangle (🔺), it MUST land on an UP space (🔺) on the grid.
    - If a part of your shape is a DOWN triangle (🔻), it MUST land on a DOWN space (🔻) on the grid.
    - 🔺➡️🔺 (OK!)
    - 🔻➡️🔻 (OK!)
    - 🔺➡️🔻 (INVALID! ❌)
    - 🔻➡️🔺 (INVALID! ❌)
- **Step 4d: Confirm Placement:** If the chosen shape can be placed at the target coordinate according to ALL three rules, the placement is valid. The shape is now placed permanently on the grid! ✨

### 5. Scoring Points: How You Win! 🏆

You score points in two main ways:

- **Placing Triangles:** You get a small number of points for _every single small triangle_ that makes up the shape you just placed. (e.g., placing a 3-triangle shape might give you 3 \* tiny_score points).
- **Clearing Lines:** This is where the BIG points come from! You get a much larger number of points for _every single small triangle_ that disappears when you clear a line (or multiple lines at once!). See the next section for details!

### 6. Line Clearing Magic! ✨ (The Key to High Scores!)

This is the most exciting part! When you place a shape, the game immediately checks if you've completed any lines. This section explains how the game _finds_ and _clears_ these lines.

- **What Lines Can Be Cleared?** There are **three** types of lines the game looks for:

  - **Horizontal Lines ↔️:** A straight, unbroken line of filled triangles going across a single row.
  - **Diagonal Lines (Top-Left to Bottom-Right) ↘️:** An unbroken diagonal line of filled triangles stepping down and to the right.
  - **Diagonal Lines (Bottom-Left to Top-Right) ↗️:** An unbroken diagonal line of filled triangles stepping up and to the right.

- **How Lines are Found: Pre-calculation of Maximal Lines**

  - **The Idea:** Instead of checking every possible line combination all the time, the game pre-calculates all *maximal* continuous lines of playable triangles when it starts. A **maximal line** is the longest possible straight segment of *playable* triangles (not in a Death Zone) in one of the three directions (Horizontal, Diagonal ↘️, Diagonal ↗️).
  - **Tracing:** For every playable triangle on the grid, the game traces outwards in each of the three directions to find the full extent of the continuous playable line passing through that triangle in that direction.
  - **Storing Maximal Lines:** Only the complete maximal lines found are stored. For example, if tracing finds a playable sequence `A-B-C-D`, only the line `(A,B,C,D)` is stored, not the sub-segments like `(A,B,C)` or `(B,C,D)`. These maximal lines represent the *potential* lines that can be cleared.
  - **Coordinate Map:** The game also builds a map linking each playable triangle coordinate `(r, c)` to the set of maximal lines it belongs to. This allows for quick lookup.

- **Defining the Paths (Neighbor Logic):** How does the game know which triangle is "next" when tracing? It depends on the current triangle's orientation (🔺 or 🔻) and the direction being traced:

  - **Horizontal ↔️:**
    - Left Neighbor: `(r, c-1)` (Always in the same row)
    - Right Neighbor: `(r, c+1)` (Always in the same row)
  - **Diagonal ↘️ (TL-BR):**
    - If current is 🔺 (Up): Next is `(r+1, c)` (Down triangle directly below)
    - If current is 🔻 (Down): Next is `(r, c+1)` (Up triangle to the right)
  - **Diagonal ↗️ (BL-TR):**
    - If current is 🔻 (Down): Next is `(r-1, c)` (Up triangle directly above)
    - If current is 🔺 (Up): Next is `(r, c+1)` (Down triangle to the right)

- **Visualizing the Paths:**

  - **Horizontal ↔️:**
    ```
    ... [🔻][🔺][🔻][🔺][🔻][🔺] ...  (Moves left/right in the same row)
    ```
  - **Diagonal ↘️ (TL-BR):** (Connects via shared horizontal edges)
    ```
    ...[🔺]...
    ...[🔻][🔺] ...
    ...     [🔻][🔺] ...
    ...         [🔻] ...
    (Path alternates row/col increments depending on orientation)
    ```
  - **Diagonal ↗️ (BL-TR):** (Connects via shared horizontal edges)
    ```
    ...           [🔺]  ...
    ...      [🔺][🔻]   ...
    ... [🔺][🔻]        ...
    ... [🔻]            ...
    (Path alternates row/col increments depending on orientation)
    ```

- **The "Full Line" Rule:** After you place a piece, the game looks at the coordinates `(r, c)` of the triangles you just placed. Using the pre-calculated map, it finds all the *maximal* lines that contain _any_ of those coordinates. For each of those maximal lines (that have at least 2 triangles), it checks: "Is _every single triangle coordinate_ in this maximal line now occupied?" If yes, that line is complete! (Note: Single isolated triangles don't count as clearable lines).

- **The _Poof_! 💨:**
  - If placing your shape completes one or MORE maximal lines (of any type, length >= 2) simultaneously, all the triangles in ALL completed lines vanish instantly!
  - The spaces become empty again.
  - You score points for _every single triangle_ that vanished. Clearing multiple lines at once is the best way to rack up points! 🥳

### 7. Getting New Shapes: The Refill 🪄

- **The Trigger:** The game only gives you new shapes when a specific condition is met.
- **The Condition:** New shapes appear **only when all three of your preview slots become empty at the exact same time.**
- **How it Happens:** This usually occurs right after you place your _last_ available shape (the third one).
- **The Refill:** As soon as the third slot becomes empty, _BAM!_ 🪄 Three brand new, randomly generated shapes instantly appear in the preview slots.
- **Important:** If you place a shape and only one or two slots are empty, you **do not** get new shapes yet. You must use up all three before the refill happens.

### 8. The End of the Road: Game Over 😭

So, how does the game end?

- **The Condition:** The game is over when you **cannot legally place _any_ of the three shapes currently available in your preview slots anywhere on the grid.**
- **The Check:** After every move (placing a shape and any resulting line clears), and after any potential shape refill, the game checks: "Is there at least one valid spot on the grid for Shape 1? OR for Shape 2? OR for Shape 3?"
- **No More Moves:** If the answer is "NO" for all three shapes (meaning none of them can be placed anywhere according to the Placement Rules), then the game immediately ends.
- **Strategy:** This means you need to be careful! Don't fill up the grid in a way that leaves no room for the types of shapes you might get later. Always try to keep options open! 🤔

---

## Core Technologies

*   **Python 3.10+**
*   **trianglengin:** Core game engine (state, actions, rules).
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
*   **`config`:** Centralized Pydantic configuration classes (excluding `EnvConfig` and `DisplayConfig`). ([`alphatriangle/config/README.md`](alphatriangle/config/README.md))
*   **`features`:** Contains logic to convert `trianglengin.GameState` objects into numerical features (`StateType`). ([`alphatriangle/features/README.md`](alphatriangle/features/README.md))
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`AlphaTriangleNet`) and a wrapper class (`NeuralNetwork`). ([`alphatriangle/nn/README.md`](alphatriangle/nn/README.md))
*   **`mcts`:** Implements the Monte Carlo Tree Search algorithm (`Node`, `run_mcts_simulations`), operating on `trianglengin.GameState`. ([`alphatriangle/mcts/README.md`](alphatriangle/mcts/README.md))
*   **`rl`:** Contains RL components: `Trainer` (network updates), `ExperienceBuffer` (data storage, **supports PER**), and `SelfPlayWorker` (Ray actor for parallel self-play using `trianglengin.GameState`). ([`alphatriangle/rl/README.md`](alphatriangle/rl/README.md))
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

All major parameters for the AlphaZero agent (MCTS, Model, Training, Persistence) are defined in the Pydantic classes within the `alphatriangle/config/` directory. Modify these files to experiment with different settings. Environment configuration (`EnvConfig`) is defined within the `trianglengin` library.

## Data Storage

All persistent data is stored within the `.alphatriangle_data/` directory in the project root.
*   **`.alphatriangle_data/mlruns/`**: Managed by **MLflow**. Contains MLflow's internal tracking data (parameters, metrics) and its own copy of logged artifacts. This is the source for the MLflow UI.
*   **`.alphatriangle_data/runs/`**: Managed by **DataManager**. Contains locally saved artifacts for each run (checkpoints, buffers, TensorBoard logs, configs) before/during logging to MLflow. This directory is used for auto-resuming and direct access to TensorBoard logs during a run.

## Maintainability

This project includes README files within each major `alphatriangle` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic.
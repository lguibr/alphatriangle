import logging
import shutil
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

# Import alphatriangle specific configs and runner
from alphatriangle.config import (
    PersistenceConfig,
    TrainConfig,
)
from alphatriangle.logging_config import setup_logging  # Import centralized setup
from alphatriangle.training.runners import run_training

# Initialize Rich Console
console = Console()

# Use a standard string for the main help, rely on rich_markup_mode for styling
app_help_text = (
    "▲ AlphaTriangle CLI ▲\nAlphaZero training pipeline for a triangle puzzle game."
)

app = typer.Typer(
    name="alphatriangle",
    help=app_help_text,  # Use the standard string here
    add_completion=False,
    rich_markup_mode="markdown",  # Keep markdown enabled for help text formatting
    pretty_exceptions_show_locals=False,
)

# --- CLI Option Annotations ---
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

ProfileOption = Annotated[
    bool,
    typer.Option(
        "--profile",
        help="Enable cProfile for worker 0.",
        is_flag=True,
    ),
]

HostOption = Annotated[
    str, typer.Option(help="The network address to listen on (default: 127.0.0.1).")
]

PortOption = Annotated[int, typer.Option(help="The port to listen on.")]


# --- Helper Function ---
def _run_external_ui(
    command_name: str,
    command_args: list[str],
    ui_name: str,
    default_url: str,
):
    """Runs an external UI command and handles common errors."""
    logger = logging.getLogger(__name__)
    executable = shutil.which(command_name)
    if not executable:
        console.print(
            f"[bold red]Error:[/bold red] Could not find '{command_name}' executable in PATH."
        )
        console.print(
            f"Please ensure {ui_name} is installed and accessible in your environment."
        )
        raise typer.Exit(code=1)

    full_command = [executable] + command_args
    console.print(
        Panel(
            f"🚀 Launching [bold cyan]{ui_name}[/]...\n"
            f"   ❯ Command: [dim]{' '.join(full_command)}[/]\n"
            f"   ❯ Access URL (approx): [link={default_url}]{default_url}[/]",
            title="External UI",
            border_style="blue",
            expand=False,
        )
    )

    try:
        # Use Popen for non-blocking execution if needed, but run is simpler for now
        process = subprocess.run(full_command, check=False)
        if process.returncode != 0:
            console.print(
                f"[bold red]Error:[/bold red] {ui_name} command failed with exit code {process.returncode}"
            )
            raise typer.Exit(code=process.returncode)
    except FileNotFoundError as e:
        console.print(
            f"[bold red]Error:[/bold red] '{executable}' command not found. Is {ui_name} installed and in your PATH?"
        )
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:
        console.print(f"\n[yellow]🟡 {ui_name} interrupted by user.[/]")
        raise typer.Exit(code=0) from None
    except Exception as e:
        console.print(
            f"[bold red]❌ An unexpected error occurred launching {ui_name}:[/]"
        )
        logger.error(f"Unexpected error launching {ui_name}: {e}", exc_info=True)
        raise typer.Exit(code=1) from e


# --- CLI Commands ---
@app.command()
def train(
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
    profile: ProfileOption = False,
):
    """
    🚀 Run the AlphaTriangle training pipeline (headless).

    Initiates the self-play and learning process. Logs will be saved to the run directory.
    """
    # Setup logging using the centralized function (file logging handled by runner)
    setup_logging(log_level)
    logging.getLogger(__name__)  # Get logger after setup

    # Use alphatriangle configs here
    train_config_override = TrainConfig()
    persist_config_override = PersistenceConfig()
    train_config_override.RANDOM_SEED = seed
    train_config_override.PROFILE_WORKERS = profile  # Set profile config
    # Ensure run name is set for persistence config
    persist_config_override.RUN_NAME = train_config_override.RUN_NAME

    console.print(
        Panel(
            f"Starting Training Run: '[bold cyan]{train_config_override.RUN_NAME}[/]'\n"
            f"Seed: {seed}, Log Level: {log_level.upper()}, Profiling: {'✅ Enabled' if profile else '❌ Disabled'}",
            title="[bold green]Training Setup[/]",
            border_style="green",
            expand=False,
        )
    )

    # Call the single runner function directly, passing the profile flag
    exit_code = run_training(
        log_level_str=log_level,
        train_config_override=train_config_override,
        persist_config_override=persist_config_override,
        profile=profile,
    )

    if exit_code == 0:
        console.print(
            Panel(
                f"✅ Training run '[bold cyan]{train_config_override.RUN_NAME}[/]' completed successfully.",
                title="[bold green]Training Finished[/]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"❌ Training run '[bold cyan]{train_config_override.RUN_NAME}[/]' failed with exit code {exit_code}.",
                title="[bold red]Training Failed[/]",
                border_style="red",
            )
        )
    sys.exit(exit_code)


@app.command()
def ml(
    host: HostOption = "127.0.0.1",
    port: PortOption = 5000,
):
    """
    📊 Launch the MLflow UI for experiment tracking.

    Requires MLflow to be installed. Points to the `.alphatriangle_data/mlruns` directory.
    """
    setup_logging("INFO")  # Basic logging for this command
    persist_config = PersistenceConfig()
    # Use the computed property which resolves the path and creates the dir
    mlflow_uri = persist_config.MLFLOW_TRACKING_URI
    mlflow_path = persist_config.get_mlflow_abs_path()

    if not mlflow_path.exists() or not any(mlflow_path.iterdir()):
        console.print(
            f"[yellow]Warning:[/yellow] MLflow directory not found or empty at expected location: [dim]{mlflow_path}[/]"
        )
        console.print("[yellow]Attempting to launch MLflow UI anyway...[/]")
    else:
        console.print(f"Found MLflow data at: [dim]{mlflow_path}[/]")

    command_args = [
        "ui",
        "--backend-store-uri",
        mlflow_uri,
        "--host",
        host,
        "--port",
        str(port),
    ]
    _run_external_ui("mlflow", command_args, "MLflow UI", f"http://{host}:{port}")


@app.command()
def tb(
    host: HostOption = "127.0.0.1",
    port: PortOption = 6006,
):
    """
    📈 Launch TensorBoard UI pointing to the runs directory.

    Requires TensorBoard to be installed. Points to the `.alphatriangle_data/runs` directory.
    """
    setup_logging("INFO")  # Basic logging for this command
    persist_config = PersistenceConfig()
    # Point to the parent directory containing all individual run folders
    runs_root_dir = persist_config.get_runs_root_dir()

    if not runs_root_dir.exists() or not any(runs_root_dir.iterdir()):
        console.print(
            f"[yellow]Warning:[/yellow] TensorBoard 'runs' directory not found or empty at: [dim]{runs_root_dir}[/]"
        )
        console.print(
            "[yellow]Attempting to launch TensorBoard UI anyway (it might show no data)...[/]"
        )
    else:
        console.print(f"Found TensorBoard runs data at: [dim]{runs_root_dir}[/]")

    command_args = [
        "--logdir",
        str(runs_root_dir),  # Pass the absolute path string
        "--host",
        host,
        "--port",
        str(port),
    ]
    _run_external_ui(
        "tensorboard", command_args, "TensorBoard UI", f"http://{host}:{port}"
    )


@app.command()
def ray(
    host: HostOption = "127.0.0.1",
    port: PortOption = 8265,
):
    """
    ☀️ Launch the Ray Dashboard web UI.

    Requires Ray to be installed and potentially running (e.g., started by `alphatriangle train`).
    """
    setup_logging("INFO")  # Basic logging for this command
    console.print(
        "[yellow]Note:[/yellow] This command attempts to open the Ray Dashboard for an existing Ray cluster."
    )
    console.print(
        "If Ray is not running, this command might fail. Start Ray first (e.g., via `alphatriangle train`)."
    )

    command_args = [
        "dashboard",
        "--host",
        host,
        "--port",
        str(port),
    ]
    _run_external_ui("ray", command_args, "Ray Dashboard", f"http://{host}:{port}")


if __name__ == "__main__":
    app()

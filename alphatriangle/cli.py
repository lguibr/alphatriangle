# File: alphatriangle/cli.py
import logging
import shutil
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

# Import Trieye config
from trieye import PersistenceConfig, TrieyeConfig

# Import alphatriangle specific configs and runner
from alphatriangle.config import (
    APP_NAME,  # Use APP_NAME from config
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

RunNameOption = Annotated[
    str | None,
    typer.Option(
        "--run-name",
        help="Specify a custom name for the run (overrides default timestamp).",
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
            # Don't exit immediately, let the calling function handle it if needed
            # raise typer.Exit(code=process.returncode)
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
    run_name: RunNameOption = None,  # Add run_name option
):
    """
    🚀 Run the AlphaTriangle training pipeline (headless).

    Initiates the self-play and learning process. Uses Trieye for stats/persistence.
    Logs will be saved to the run directory within `.trieye_data/alphatriangle/runs/`.
    This command also initializes Ray and starts the Ray Dashboard. Check the logs for the dashboard URL.
    """
    # Setup logging using the centralized function (file logging handled by Trieye)
    setup_logging(log_level)
    logging.getLogger(__name__)  # Get logger after setup

    # Use alphatriangle TrainConfig
    train_config_override = TrainConfig()
    train_config_override.RANDOM_SEED = seed
    train_config_override.PROFILE_WORKERS = profile  # Set profile config

    # Create TrieyeConfig, overriding run_name if provided
    trieye_config_override = TrieyeConfig(app_name=APP_NAME)
    if run_name:
        trieye_config_override.run_name = run_name
        # Sync run_name to persistence config within TrieyeConfig
        trieye_config_override.persistence.RUN_NAME = run_name
    else:
        # Use the default factory-generated run_name from TrieyeConfig
        run_name = trieye_config_override.run_name

    console.print(
        Panel(
            f"Starting Training Run: '[bold cyan]{run_name}[/]'\n"
            f"Seed: {seed}, Log Level: {log_level.upper()}, Profiling: {'✅ Enabled' if profile else '❌ Disabled'}",
            title="[bold green]Training Setup[/]",
            border_style="green",
            expand=False,
        )
    )

    # Call the single runner function directly, passing configs
    exit_code = run_training(
        log_level_str=log_level,
        train_config_override=train_config_override,
        trieye_config_override=trieye_config_override,
        profile=profile,
    )

    if exit_code == 0:
        console.print(
            Panel(
                f"✅ Training run '[bold cyan]{run_name}[/]' completed successfully.",
                title="[bold green]Training Finished[/]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"❌ Training run '[bold cyan]{run_name}[/]' failed with exit code {exit_code}.",
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

    Requires MLflow to be installed. Points to the `.trieye_data/<app_name>/mlruns` directory.
    """
    setup_logging("INFO")  # Basic logging for this command
    # Use Trieye's PersistenceConfig to find the path
    persist_config = PersistenceConfig(APP_NAME=APP_NAME)
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
    try:
        _run_external_ui("mlflow", command_args, "MLflow UI", f"http://{host}:{port}")
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(
                f"[yellow]MLflow UI failed to start (Exit Code: {e.exit_code}). "
                f"Is port {port} already in use? Try specifying a different port with --port.[/]"
            )
        sys.exit(e.exit_code)


@app.command()
def tb(
    host: HostOption = "127.0.0.1",
    port: PortOption = 6006,
):
    """
    📈 Launch TensorBoard UI pointing to the runs directory.

    Requires TensorBoard to be installed. Points to the `.trieye_data/<app_name>/runs` directory.
    """
    setup_logging("INFO")  # Basic logging for this command
    # Use Trieye's PersistenceConfig to find the path
    persist_config = PersistenceConfig(APP_NAME=APP_NAME)
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
    try:
        _run_external_ui(
            "tensorboard", command_args, "TensorBoard UI", f"http://{host}:{port}"
        )
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(
                f"[yellow]TensorBoard UI failed to start (Exit Code: {e.exit_code}). "
                f"Is port {port} already in use? Try specifying a different port with --port.[/]"
            )
        sys.exit(e.exit_code)


@app.command()
def ray(
    host: HostOption = "127.0.0.1",  # Keep host/port options for reference
    port: PortOption = 8265,
):
    """
    ☀️ Provides instructions to view the Ray Dashboard.

    The dashboard is automatically started when you run `alphatriangle train`.
    Check the output logs of the `train` command for the correct URL.
    """
    setup_logging("INFO")  # Basic logging for this command
    console.print(
        Panel(
            f"💡 To view the Ray Dashboard:\n\n"
            f"1. The Ray Dashboard is started automatically when you run the `[bold]alphatriangle train[/]` command.\n"
            f"2. Check the console output or the log file for the `train` command (located in `.trieye_data/{APP_NAME}/runs/<run_name>/logs/`).\n"
            f"3. Look for a line similar to: '[bold cyan]Ray Dashboard running at: http://<address>:<port>[/]' \n"
            f"4. Open that specific URL in your web browser.\n\n"
            f"[dim]Note: The default URL is often http://{host}:{port}, but it might differ. "
            f"If you cannot access the URL, check firewall settings or if the port is blocked.[/]",
            title="[bold yellow]Ray Dashboard Instructions[/]",
            border_style="yellow",
            expand=False,
        )
    )


if __name__ == "__main__":
    app()

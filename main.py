import argparse
import optuna
from typing import Any, Dict
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from optimizer_agent import HardwareOptimizerAgent

console: Console = Console()

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="LM Studio Hardware Optimizer Agent"
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of optimization trials to run"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.json",
        help="Path to the dataset JSON file",
    )

    args: argparse.Namespace = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold blue]LM Studio Hardware Optimizer[/bold blue]\n"
            "Dynamic Load Benchmarking",
            border_style="blue",
        )
    )

    agent: HardwareOptimizerAgent = HardwareOptimizerAgent(dataset_path=args.dataset)

    table: Table = Table(title="Hardware Optimization Trials")
    table.add_column("Trial", style="cyan", justify="right")
    table.add_column("Context Length", style="magenta")
    table.add_column("GPU Ratio", style="yellow")
    table.add_column("TPS", style="green", justify="right")
    table.add_column("TTFT (s)", style="blue", justify="right")
    table.add_column("Status", justify="center")

    def live_callback(trial: optuna.Trial, stats: Dict[str, Any]) -> None:
        status_color: str = "[green]" if stats["status"] == "Success" else "[red]"
        table.add_row(
            str(trial.number),
            str(stats["context_length"]),
            f"{stats['gpu_ratio']:.2f}",
            f"{stats['tps']:.2f}",
            f"{stats['ttft']:.2f}",
            f"{status_color}{stats['status']}[/]"
        )

    console.print(
        f"[yellow]Starting {args.trials} trials. "
        "LM Studio model will be dynamically unloaded and loaded.[/yellow]"
    )

    try:
        with Live(table, refresh_per_second=4):
            best_trial: optuna.trial.FrozenTrial = agent.run_optimization(
                n_trials=args.trials, callback=live_callback
            )

        if best_trial.value == 0.0:
            console.print(Panel.fit(
                "[bold red]Optimization Failed[/bold red]\n\n"
                "All hardware configurations resulted in a load failure or "
                "OOM error.\n"
                "Please verify that LM Studio is running and the TARGET_MODEL "
                "in .env is exactly correct.",
                border_style="red"
            ))
        else:
            best_params = best_trial.params
            console.print(Panel.fit(
                f"[bold green]Hardware Optimization Complete![/bold green]\n\n"
                f"Best Hardware Configuration:\n"
                f"Context Length: {best_params['context_length']}\n"
                f"GPU Offload Ratio: {best_params['gpu_ratio']:.2f}\n\n"
                f"Saved to: [yellow]optimal_settings.json[/yellow]",
                border_style="green"
            ))
    except Exception as e:
        console.print(f"[bold red]Error during optimization:[/bold red] {e}")

if __name__ == "__main__":
    main()

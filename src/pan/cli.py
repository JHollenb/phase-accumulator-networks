"""CLI entry point — powered by Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pan.config import TrainConfig
from pan.constants import DEVICE

# Import experiment modules so they register themselves
import pan.experiments.compare     # noqa: F401
import pan.experiments.sweep       # noqa: F401
import pan.experiments.primes      # noqa: F401
import pan.experiments.tier3       # noqa: F401
import pan.experiments.sweeps      # noqa: F401
import pan.experiments as registry

app = typer.Typer(help="Phase Accumulator Network — sinusoidal phase arithmetic for grokking",
                  no_args_is_help=True, rich_markup_mode="rich")
console = Console()


@app.command()
def run(
    experiment: str = typer.Argument(help="Experiment name (see `pan list`)"),
    p: int = typer.Option(113, help="Prime for modular arithmetic"),
    k: int = typer.Option(5, "--k", help="Phase frequencies (PAN)"),
    steps: int = typer.Option(50_000, help="Training steps"),
    seed: int = typer.Option(42, help="Random seed"),
    weight_decay: float = typer.Option(0.01, "--wd", help="AdamW weight decay"),
    diversity_weight: float = typer.Option(0.01, "--dw", help="Diversity regularisation"),
    val_samples: Optional[int] = typer.Option(None, help="Subsample val set (None = full)"),
    no_compile: bool = typer.Option(False, help="Disable torch.compile"),
    no_early_stop: bool = typer.Option(False, help="Train past grokking"),
    log_every: int = typer.Option(200, help="Log interval"),
    output_dir: Path = typer.Option(".", "--out", help="Output directory"),
    save_model: bool = typer.Option(False, help="Save model weights"),
    dry_run: bool = typer.Option(False, help="Print config, skip training"),
):
    """Run a registered experiment."""
    cfg = TrainConfig(
        p=p, k_freqs=k, n_steps=steps, seed=seed,
        weight_decay=weight_decay, diversity_weight=diversity_weight,
        val_samples=val_samples, use_compile=not no_compile,
        early_stop=not no_early_stop, log_every=log_every,
        output_dir=output_dir, save_model=save_model, dry_run=dry_run,
    )

    console.print(f"\n[bold]Phase Accumulator Network[/]")
    console.print(f"  Device: {DEVICE}   Experiment: [cyan]{experiment}[/]")
    console.print(f"  P={cfg.p}  K={cfg.k_freqs}  steps={cfg.n_steps:,}  seed={cfg.seed}")
    console.print(f"  wd={cfg.weight_decay}  dw={cfg.diversity_weight}  "
                  f"compile={cfg.use_compile}  early_stop={cfg.early_stop}\n")

    fn = registry.get(experiment)
    return fn(cfg)


@app.command("list")
def list_experiments():
    """Show all registered experiments."""
    table = Table(title="Available Experiments")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name, help_text in registry.list_experiments().items():
        table.add_row(name, help_text)
    console.print(table)


if __name__ == "__main__":
    app()

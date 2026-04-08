"""CLI entry point for running a PAN vs Transformer comparison."""

from __future__ import annotations

from pathlib import Path

import typer
import wandb
from rich.console import Console

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.models import PAN, Transformer
from pan.training import define_wandb_metrics, train_loop

app = typer.Typer(
    help="Phase Accumulator Network — PAN vs Transformer comparison",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def compare(
    p: int = 113,
    k: int = 5,
    steps: int = 50_000,
    seed: int = 42,
    wd: float = 0.01,
    dw: float = 0.01,
    no_compile: bool = False,
    no_early_stop: bool = False,
    log_every: int = 200,
    out: Path = Path("."),
    dry_run: bool = False,
) -> None:
    """Head-to-head PAN vs Transformer."""
    cfg = TrainConfig(
        p=p,
        k_freqs=k,
        n_steps=steps,
        seed=seed,
        weight_decay=wd,
        diversity_weight=dw,
        use_compile=not no_compile,
        early_stop=not no_early_stop,
        log_every=log_every,
        output_dir=out,
        dry_run=dry_run,
    )

    console.print("\n[bold]Phase Accumulator Network[/]  ·  compare")
    console.print(
        f"  device={DEVICE}  p={cfg.p}  K={cfg.k_freqs}  "
        f"steps={cfg.n_steps:,}  seed={cfg.seed}"
    )

    tx, ty, vx, vy = make_modular_dataset(cfg.p, seed=cfg.seed)

    pan = PAN(cfg.p, k=cfg.k_freqs).to(DEVICE)
    tf = Transformer(cfg.p, cfg.d_model, cfg.n_heads, cfg.d_mlp).to(DEVICE)
    console.print(
        f"  PAN: {pan.count_parameters():,}  TF: {tf.count_parameters():,}  "
        f"ratio: {tf.count_parameters() / pan.count_parameters():.0f}×"
    )

    wandb.init(project="pan", name=f"compare-p{p}-k{k}-s{seed}", config=cfg.model_dump())
    define_wandb_metrics("PAN")
    define_wandb_metrics("TF")

    train_loop(pan, cfg, tx, ty, vx, vy, label="PAN")
    train_loop(tf, cfg.overlay(weight_decay=1.0), tx, ty, vx, vy, label="TF")
    wandb.finish()


if __name__ == "__main__":
    app()

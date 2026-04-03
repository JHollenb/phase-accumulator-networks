"""CLI — each command is a thin experiment definition."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import wandb
from rich.console import Console

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.models import PAN, Transformer, build
from pan.training import train
from pan.sweep import grid_search, print_results

app = typer.Typer(
    help="Phase Accumulator Network — sinusoidal phase arithmetic for grokking",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


# ── Shared options ──────────────────────────────────────────────────────────

def _cfg(
    p: int = 113, k: int = 5, steps: int = 50_000, seed: int = 42,
    wd: float = 0.01, dw: float = 0.01, val_samples: Optional[int] = None,
    no_compile: bool = False, no_early_stop: bool = False, log_every: int = 200,
    out: Path = Path("."), save_model: bool = False, dry_run: bool = False,
    record_checkpoints: bool = False,
) -> TrainConfig:
    return TrainConfig(
        p=p, k_freqs=k, n_steps=steps, seed=seed,
        weight_decay=wd, diversity_weight=dw, val_samples=val_samples,
        use_compile=not no_compile, early_stop=not no_early_stop,
        log_every=log_every, output_dir=out, save_model=save_model,
        dry_run=dry_run, record_checkpoints=record_checkpoints,
    )


def _banner(cfg: TrainConfig, name: str):
    console.print(f"\n[bold]Phase Accumulator Network[/]  ·  {name}")
    console.print(f"  device={DEVICE}  p={cfg.p}  K={cfg.k_freqs}  "
                  f"steps={cfg.n_steps:,}  seed={cfg.seed}")


# ── Commands ────────────────────────────────────────────────────────────────

@app.command()
def compare(
    p: int = 113, k: int = 5, steps: int = 50_000, seed: int = 42,
    wd: float = 0.01, dw: float = 0.01,
    no_compile: bool = False, no_early_stop: bool = False,
    log_every: int = 200, out: Path = Path("."), dry_run: bool = False,
):
    """Head-to-head PAN vs Transformer."""
    cfg = _cfg(p=p, k=k, steps=steps, seed=seed, wd=wd, dw=dw,
               no_compile=no_compile, no_early_stop=no_early_stop,
               log_every=log_every, out=out, dry_run=dry_run)
    _banner(cfg, "compare")
    tx, ty, vx, vy = make_modular_dataset(cfg.p, seed=cfg.seed)

    pan = PAN(cfg.p, k=cfg.k_freqs).to(DEVICE)
    tf = Transformer(cfg.p, cfg.d_model, cfg.n_heads, cfg.d_mlp).to(DEVICE)
    console.print(f"  PAN: {pan.count_parameters():,}  TF: {tf.count_parameters():,}  "
                  f"ratio: {tf.count_parameters() / pan.count_parameters():.0f}×")

    wandb.init(project="pan", name=f"compare-p{p}-k{k}-s{seed}", config=cfg.model_dump())
    # Define metrics for both models upfront — avoids duplicate panels from
    # calling define_metric mid-run after data has already been logged.
    from pan.training import define_wandb_metrics
    define_wandb_metrics("PAN")
    define_wandb_metrics("TF")
    train(pan, cfg, tx, ty, vx, vy, label="PAN")
    train(tf, cfg.overlay(weight_decay=1.0), tx, ty, vx, vy, label="TF")
    wandb.finish()


@app.command()
def tier3(
    p: int = 113, k: int = 9, steps: int = 100_000, seed: int = 42,
    wd: float = 0.01, dw: float = 0.01,
    log_every: int = 200, out: Path = Path("."), dry_run: bool = False,
):
    """Mechanistic equivalence with full frequency checkpoint logging."""
    cfg = _cfg(p=p, k=k, steps=steps, seed=seed, wd=wd, dw=dw,
               no_compile=True, no_early_stop=True,
               log_every=log_every, out=out, dry_run=dry_run,
               record_checkpoints=True)
    _banner(cfg, "tier3")

    pan = PAN(cfg.p, k=cfg.k_freqs).to(DEVICE)
    wandb.init(project="pan", name=f"tier3-p{p}-k{k}-s{seed}", config=cfg.model_dump())
    gs = train(pan, cfg, *make_modular_dataset(cfg.p, seed=cfg.seed), label="PAN-T3")

    if cfg.save_model and not cfg.dry_run:
        import torch
        path = cfg.output_dir / f"model_PAN-T3.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=pan.state_dict(), config=cfg.model_dump(), grok_step=gs), path)
        wandb.save(str(path))

    wandb.finish()


@app.command()
def sweep(
    p: int = 113, steps: int = 50_000, seed: int = 42,
    wd: float = 0.01, dw: float = 0.01,
    log_every: int = 200, dry_run: bool = False,
):
    """Find minimum K for reliable grokking (K=1..15 × 3 seeds)."""
    cfg = _cfg(p=p, steps=steps, seed=seed, wd=wd, dw=dw, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "K sweep")
    wandb.init(project="pan", group="k-sweep", name="parent", config=cfg.model_dump())
    results = grid_search(cfg, vary={"k_freqs": list(range(1, 16))})
    print_results(results, "K", "K Sweep — Minimum Reliable K")
    reliable = [k for k, r in results.items() if r["n_grokked"] >= 2]
    if reliable:
        console.print(f"  [green]Minimum reliable K: {min(reliable)}[/]")
    wandb.finish()


@app.command()
def primes(
    k: int = 9, steps: int = 50_000, seed: int = 42,
    wd: float = 0.01, dw: float = 0.01,
    log_every: int = 200, dry_run: bool = False,
):
    """Cross-prime generalisation (43, 67, 89, 113, 127)."""
    cfg = _cfg(k=k, steps=steps, seed=seed, wd=wd, dw=dw, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "primes")
    wandb.init(project="pan", group="primes", name="parent", config=cfg.model_dump())
    results = grid_search(cfg, vary={"p": [43, 67, 89, 113, 127]}, seeds=[seed])
    print_results(results, "P", "Cross-Prime Generalisation")
    wandb.finish()


@app.command()
def held_out(
    steps: int = 200_000, seed: int = 42,
    log_every: int = 200, dry_run: bool = False,
):
    """Held-out primes never used in development (59, 71, 97)."""
    cfg = _cfg(k=9, steps=steps, seed=seed, wd=0.01, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "held-out primes")
    wandb.init(project="pan", group="held-out", name="parent", config=cfg.model_dump())
    results = grid_search(cfg, vary={"p": [59, 71, 97]}, seeds=[seed],
                          fixed={"k_freqs": 9, "weight_decay": 0.01})
    n = sum(1 for r in results.values() if r["n_grokked"] > 0)
    console.print(f"  Held-out: [bold]{n}/3[/] grokked")
    wandb.finish()


@app.command()
def dw_sweep(
    p: int = 113, steps: int = 100_000, seed: int = 42,
    log_every: int = 200, dry_run: bool = False,
):
    """Diversity weight sweep (6 values × 5 seeds, K=9)."""
    cfg = _cfg(p=p, k=9, steps=steps, seed=seed, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "dw sweep")
    wandb.init(project="pan", group="dw-sweep", name="parent", config=cfg.model_dump())
    results = grid_search(cfg, vary={"diversity_weight": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]},
                          seeds=[42, 123, 456, 789, 999], fixed={"weight_decay": 0.01})
    print_results(results, "DW", "Diversity Weight Sweep")
    wandb.finish()


@app.command()
def wd_sweep(
    p: int = 113, steps: int = 100_000, seed: int = 42,
    log_every: int = 200, dry_run: bool = False,
):
    """Weight decay sweep (6 values × 3 seeds, K=9)."""
    cfg = _cfg(p=p, k=9, steps=steps, seed=seed, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "wd sweep")
    wandb.init(project="pan", group="wd-sweep", name="parent", config=cfg.model_dump())
    results = grid_search(cfg, vary={"weight_decay": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]})
    print_results(results, "WD", "Weight Decay Sweep")
    wandb.finish()


@app.command()
def k8(
    p: int = 113, steps: int = 200_000,
    log_every: int = 200, dry_run: bool = False,
):
    """K=8 anomaly investigation (10 seeds)."""
    cfg = _cfg(p=p, k=8, steps=steps, wd=0.01, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "K=8 anomaly")
    seeds = [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]
    wandb.init(project="pan", group="k8-anomaly", name="parent", config=cfg.model_dump())
    # Vary seed directly — no dummy outer loop needed
    results = grid_search(cfg, vary={"seed": seeds}, seeds=[0],  # 0 is overridden by vary
                          fixed={"k_freqs": 8, "weight_decay": 0.01})
    n = sum(1 for r in results.values() if r["n_grokked"] > 0)
    console.print(f"  K=8: [bold]{n}/{len(seeds)}[/] grokked"
                  + ("  → sampling noise" if n > 0 else ""))
    wandb.finish()


@app.command()
def tf_sweep(
    p: int = 113, steps: int = 100_000, seed: int = 42,
    log_every: int = 200, dry_run: bool = False,
):
    cfg = _cfg(p=p, k=k, steps=steps, seed=seed, wd=wd, dw=dw,
               no_compile=no_compile, no_early_stop=no_early_stop,
               log_every=log_every, out=out, dry_run=dry_run)
    _banner(cfg, "compare")
    tx, ty, vx, vy = make_modular_dataset(cfg.p, seed=cfg.seed)

    pan = PAN(cfg.p, k=cfg.k_freqs).to(DEVICE)
    console.print(f"  PAN: {pan.count_parameters():,}  TF: {tf.count_parameters():,}  "
                  f"ratio: {tf.count_parameters() / pan.count_parameters():.0f}×")

    # Define metrics for both models upfront — avoids duplicate panels from
    # calling define_metric mid-run after data has already been logged.
    train(pan, cfg, tx, ty, vx, vy, label="PAN")

@app.command()
def sweep_test(
    p: int = 113, steps: int = 50_000, seed: int = 42,
    wd: float = 0.01, dw: float = 0.01,
    log_every: int = 200, dry_run: bool = False,
):
    """Find minimum K for reliable grokking (K=1..15 × 3 seeds)."""
    cfg = _cfg(p=p, steps=steps, seed=seed, wd=wd, dw=dw, log_every=log_every, dry_run=dry_run)
    _banner(cfg, "K sweep")
    wandb.init(project="pan", group="k-sweep", name="parent", config=cfg.model_dump())
    results = grid_search(cfg, vary={"k_freqs": list(range(1, 16))})
    print_results(results, "K", "K Sweep — Minimum Reliable K")
    reliable = [k for k, r in results.items() if r["n_grokked"] >= 2]
    if reliable:
        console.print(f"  [green]Minimum reliable K: {min(reliable)}[/]")
    wandb.finish()


if __name__ == "__main__":
    app()

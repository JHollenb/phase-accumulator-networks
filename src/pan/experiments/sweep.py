"""Tier 2: Minimum K for reliable grokking (3 seeds per K)."""

import numpy as np
from rich.console import Console
from rich.table import Table

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments import experiment
from pan.models import PAN
from pan.training import train

console = Console()


@experiment("sweep", help="Find minimum K for reliable grokking")
def run(cfg: TrainConfig) -> dict:
    console.rule("[bold]K Sweep — minimum reliable K")
    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)
    seeds = [42, 123, 456]
    results = {}

    for k in range(1, 16):
        grok_steps = []
        for seed in seeds:
            pan = PAN(cfg.p, k_freqs=k).to(DEVICE)
            sub = cfg.overlay(k_freqs=k, seed=seed, use_compile=False)
            hist = train(pan, sub, train_x, train_y, val_x, val_y, label=f"K{k}-s{seed}")
            grok_steps.append(hist.grok_step)

        n = sum(1 for s in grok_steps if s is not None)
        mean = np.mean([s for s in grok_steps if s]) if n else None
        results[k] = dict(grok_steps=grok_steps, n_grokked=n, mean_step=mean,
                          params=PAN(cfg.p, k).count_parameters())

    table = Table(title="K Sweep Results")
    table.add_column("K", style="cyan"); table.add_column("Grokked"); table.add_column("Mean Step")
    table.add_column("Params")
    for k, r in results.items():
        ms = f"{r['mean_step']:.0f}" if r['mean_step'] else "—"
        table.add_row(str(k), f"{r['n_grokked']}/3", ms, f"{r['params']:,}")
    console.print(table)

    reliable = [k for k, r in results.items() if r["n_grokked"] >= 2]
    if reliable:
        console.print(f"  [green]Minimum reliable K: {min(reliable)}[/]")
    return results

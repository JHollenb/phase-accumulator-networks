"""Grid search over one parameter × N seeds — the core sweep primitive."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import wandb
from rich.console import Console
from rich.table import Table

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.models import build
from pan.models.pan import PAN
from pan.training import train

console = Console()


def grid_search(
    cfg: TrainConfig,
    *,
    vary: dict[str, list[Any]],
    seeds: list[int] = (42, 123, 456),
    arch: str = "pan",
    fixed: dict[str, Any] | None = None,
) -> dict:
    """
    Sweep over the cartesian product of `vary` values × `seeds`.

    Each run gets its own wandb run (grouped). Returns a results dict
    keyed by the vary values (or tuples if multiple vary keys).

    Example:
        grid_search(cfg, vary={"k_freqs": range(1,16)}, seeds=[42,123,456])
        grid_search(cfg, vary={"weight_decay": [0.001, 0.01, 0.1]}, arch="pan")
        grid_search(cfg, vary={"d_model": [8,16,32,64,128]}, arch="transformer")
    """
    fixed = fixed or {}
    vary_names = list(vary.keys())
    vary_values = list(vary.values())
    results = {}

    for combo in product(*vary_values):
        overrides = dict(zip(vary_names, combo))
        key = combo[0] if len(combo) == 1 else combo
        grok_steps = []

        for seed in seeds:
            sub = cfg.overlay(seed=seed, use_compile=False, **overrides, **fixed)

            wandb.init(
                project=wandb.run.project if wandb.run else "pan",
                group=wandb.run.group if wandb.run else "sweep",
                name=f"{'-'.join(f'{k}={v}' for k,v in overrides.items())}-s{seed}",
                config=sub.model_dump(),
                reinit=True,
            )

            model = build(arch, sub)
            tx, ty, vx, vy = make_modular_dataset(sub.p, seed=cfg.seed)
            label = f"{'-'.join(f'{v}' for v in combo)}-s{seed}"
            gs = train(model, sub, tx, ty, vx, vy, label=label)
            grok_steps.append(gs)

            # Log PAN-specific diagnostics
            if isinstance(model, PAN):
                wandb.summary["mode_collapsed"] = model.is_mode_collapsed()

            wandb.finish()

        n = sum(1 for s in grok_steps if s is not None)
        gs_valid = [s for s in grok_steps if s is not None]
        results[key] = dict(
            grok_steps=grok_steps,
            n_grokked=n,
            n_seeds=len(seeds),
            mean_step=float(np.mean(gs_valid)) if gs_valid else None,
        )

    return results


def print_results(results: dict, param_name: str, title: str):
    """Pretty-print grid search results as a Rich table."""
    table = Table(title=title)
    table.add_column(param_name, style="cyan")
    table.add_column("Grokked")
    table.add_column("Mean Step")
    for val, r in results.items():
        ms = f"{r['mean_step']:,.0f}" if r["mean_step"] else "—"
        table.add_row(str(val), f"{r['n_grokked']}/{r['n_seeds']}", ms)
    console.print(table)

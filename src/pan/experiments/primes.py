"""Tier 4: Cross-prime generalisation test."""

from rich.console import Console
from rich.table import Table

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments import experiment
from pan.models import PAN
from pan.training import train

console = Console()


@experiment("primes", help="Test generalisation across primes")
def run(cfg: TrainConfig) -> dict:
    primes = [43, 67, 89, 113, 127]
    console.rule(f"[bold]Cross-Prime — K={cfg.k_freqs}")
    results = {}

    for p in primes:
        train_x, train_y, val_x, val_y = make_modular_dataset(p, seed=cfg.seed)
        pan = PAN(p, k_freqs=cfg.k_freqs).to(DEVICE)
        sub = cfg.overlay(p=p, use_compile=False)
        hist = train(pan, sub, train_x, train_y, val_x, val_y, label=f"P{p}")
        results[p] = dict(
            grok_step=hist.grok_step,
            final_acc=hist.val_acc[-1] if hist.val_acc else 0.0,
            params=pan.count_parameters(),
        )

    table = Table(title="Cross-Prime Results")
    table.add_column("P", style="cyan"); table.add_column("Grok Step")
    table.add_column("Final Acc"); table.add_column("Params")
    for p, r in results.items():
        gs = f"{r['grok_step']:,}" if r["grok_step"] else "—"
        table.add_row(str(p), gs, f"{r['final_acc']:.3f}", f"{r['params']:,}")
    console.print(table)
    return results


@experiment("held_out_primes", help="Primes never used in development (59, 71, 97)")
def run_held_out(cfg: TrainConfig) -> dict:
    primes = [59, 71, 97]
    console.rule("[bold]Held-Out Primes — K=9")
    results = {}

    for p in primes:
        train_x, train_y, val_x, val_y = make_modular_dataset(p, seed=cfg.seed)
        pan = PAN(p, k_freqs=9).to(DEVICE)
        sub = cfg.overlay(p=p, k_freqs=9, weight_decay=0.01, use_compile=False)
        hist = train(pan, sub, train_x, train_y, val_x, val_y, label=f"HO-P{p}")
        results[p] = dict(grok_step=hist.grok_step,
                          final_acc=hist.val_acc[-1] if hist.val_acc else 0.0)

    n = sum(1 for r in results.values() if r["grok_step"] is not None)
    console.print(f"  Held-out: [bold]{n}/{len(primes)}[/] grokked")
    return results

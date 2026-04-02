"""Overnight sweep experiments: diversity weight, weight decay, K=8, TF sizing."""

import numpy as np
from rich.console import Console
from rich.table import Table

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments import experiment
from pan.models import PAN, Transformer
from pan.training import train
from pan.analysis import detect_mode_collapse

console = Console()


def _sweep(cfg, param_name, values, seeds, *, k_freqs=9, model_cls=PAN, **fixed):
    """Generic sweep: vary one parameter across seeds, return results dict."""
    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)
    results = {}

    for val in values:
        grok_steps, final_accs, collapses = [], [], []
        for seed in seeds:
            overrides = {param_name: val, "seed": seed, "use_compile": False,
                         "k_freqs": k_freqs, **fixed}
            sub = cfg.overlay(**overrides)

            if model_cls is PAN:
                model = PAN(cfg.p, k_freqs=overrides.get("k_freqs", k_freqs)).to(DEVICE)
            else:
                d = overrides.get("d_model", 128)
                nh = max(1, d // 16)
                model = Transformer(cfg.p, d_model=d, n_heads=nh, d_mlp=4 * d).to(DEVICE)

            hist = train(model, sub, train_x, train_y, val_x, val_y,
                         label=f"{param_name}={val}-s{seed}")
            grok_steps.append(hist.grok_step)
            final_accs.append(hist.val_acc[-1] if hist.val_acc else 0.0)
            if model_cls is PAN:
                collapses.append(detect_mode_collapse(model))

        n = sum(1 for s in grok_steps if s is not None)
        gs = [s for s in grok_steps if s is not None]
        results[val] = dict(
            grok_steps=grok_steps, n_grokked=n, n_seeds=len(seeds),
            mean_step=np.mean(gs) if gs else None, mean_acc=np.mean(final_accs),
            n_collapse=sum(collapses) if collapses else 0,
        )
    return results


def _print_sweep_table(results, param_name, title):
    table = Table(title=title)
    table.add_column(param_name, style="cyan")
    table.add_column("Grokked"); table.add_column("Mean Step"); table.add_column("Mean Acc")
    for val, r in results.items():
        ms = f"{r['mean_step']:,.0f}" if r["mean_step"] else "—"
        table.add_row(str(val), f"{r['n_grokked']}/{r['n_seeds']}", ms, f"{r['mean_acc']:.3f}")
    console.print(table)


@experiment("dw_sweep", help="Diversity weight reliability sweep (K=9, 5 seeds)")
def run_dw(cfg: TrainConfig) -> dict:
    console.rule("[bold]Diversity Weight Sweep")
    results = _sweep(cfg, "diversity_weight",
                     [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
                     [42, 123, 456, 789, 999], weight_decay=0.01)
    _print_sweep_table(results, "DW", "Diversity Weight Sweep")
    return results


@experiment("wd_sweep", help="Weight decay reliability sweep (K=9, 3 seeds)")
def run_wd(cfg: TrainConfig) -> dict:
    console.rule("[bold]Weight Decay Sweep")
    results = _sweep(cfg, "weight_decay",
                     [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                     [42, 123, 456])
    _print_sweep_table(results, "WD", "Weight Decay Sweep")
    return results


@experiment("k8_sweep", help="K=8 anomaly investigation (10 seeds)")
def run_k8(cfg: TrainConfig) -> dict:
    console.rule("[bold]K=8 Anomaly — 10 seeds")
    seeds = [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]
    results = _sweep(cfg, "seed", seeds, [0],  # dummy outer, seed is the param
                     k_freqs=8, weight_decay=0.01)
    # Reformat: the sweep varied seed, flatten results
    grok_steps = [results[s]["grok_steps"][0] for s in seeds]
    n = sum(1 for s in grok_steps if s is not None)
    console.print(f"  K=8: [bold]{n}/{len(seeds)}[/] grokked")
    if n > 0:
        console.print("  → Anomaly was sampling noise")
    return dict(grok_steps=grok_steps, n_grokked=n)


@experiment("tf_sweep", help="Minimum transformer size for mod-P (5 d_model × 3 seeds)")
def run_tf(cfg: TrainConfig) -> dict:
    console.rule("[bold]Transformer Size Sweep")
    results = _sweep(cfg, "d_model", [8, 16, 32, 64, 128],
                     [42, 123, 456], model_cls=Transformer, weight_decay=1.0)

    pan_params = PAN(cfg.p, cfg.k_freqs).count_parameters()
    table = Table(title="Transformer Size Sweep")
    table.add_column("d_model", style="cyan"); table.add_column("Params")
    table.add_column("Grokked"); table.add_column("Mean Step"); table.add_column("vs PAN")
    for d, r in results.items():
        nh = max(1, d // 16)
        params = Transformer(cfg.p, d_model=d, n_heads=nh, d_mlp=4 * d).count_parameters()
        ms = f"{r['mean_step']:,.0f}" if r["mean_step"] else "—"
        table.add_row(str(d), f"{params:,}", f"{r['n_grokked']}/3", ms,
                      f"{params / pan_params:.1f}×")
    console.print(table)
    return results

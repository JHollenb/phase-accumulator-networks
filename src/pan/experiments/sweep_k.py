from pathlib import Path

import pandas as pd

from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments.plots import plot_sweep_k
from pan.experiments.results import ExperimentResult
from pan.models import PhaseAccumulatorNetwork
from pan.training import TrainConfig, train
from pan.utils import ensure_dir, save_json


def run_sweep_k(cfg: TrainConfig) -> ExperimentResult:
    run_dir = ensure_dir(Path(cfg.output_dir) / "sweep_k")
    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)

    rows = []
    k_values = list(range(1, 16))
    seeds = [42, 123, 456]

    for k in k_values:
        for seed in seeds:
            scfg = TrainConfig(
                p=cfg.p,
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                log_every=cfg.log_every,
                seed=seed,
                k_freqs=k,
                diversity_weight=cfg.diversity_weight,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_mlp=cfg.d_mlp,
                val_samples=cfg.val_samples,
                use_compile=False,
                early_stop=cfg.early_stop,
                output_dir=cfg.output_dir,
                save_model=False,
                dry_run=cfg.dry_run,
            )

            pan = PhaseAccumulatorNetwork(cfg.p, k_freqs=k).to(DEVICE)
            hist = train(pan, scfg, train_x, train_y, val_x, val_y, label=f"PAN-K{k}-s{seed}")

            rows.append({
                "experiment": "sweep_k",
                "p": cfg.p,
                "k": k,
                "seed": seed,
                "n_steps": cfg.n_steps,
                "grok_step": hist.grok_step,
                "final_val_acc": hist.val_acc[-1] if hist.val_acc else None,
                "param_count": pan.count_parameters(),
                "success": int(hist.grok_step is not None),
            })

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby("k", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            n_grokked=("success", "sum"),
            mean_grok_step=("grok_step", "mean"),
            median_grok_step=("grok_step", "median"),
            param_count=("param_count", "first"),
        )
    )
    summary_df["grok_rate"] = summary_df["n_grokked"] / summary_df["n_runs"]

    raw_path = run_dir / "raw_runs.csv"
    summary_path = run_dir / "summary_by_k.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    save_json(run_dir / "config.json", cfg.to_dict())

    plot_path = run_dir / "sweep_k.png"
    plot_sweep_k(summary_df, str(plot_path))

    summary = {
        "experiment": "sweep_k",
        "p": cfg.p,
        "k_min": int(summary_df["k"].min()),
        "k_max": int(summary_df["k"].max()),
        "num_rows": int(len(raw_df)),
        "num_summary_rows": int(len(summary_df)),
    }
    save_json(run_dir / "summary.json", summary)

    return ExperimentResult(
        name="sweep_k",
        run_dir=run_dir,
        summary=summary,
        artifacts={
            "config": str(run_dir / "config.json"),
            "summary": str(run_dir / "summary.json"),
            "raw_runs": str(raw_path),
            "summary_by_k": str(summary_path),
            "plot": str(plot_path),
        },
    )

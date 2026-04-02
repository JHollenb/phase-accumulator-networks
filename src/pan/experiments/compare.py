"""Tier 1: Head-to-head PAN vs Transformer on mod-P addition."""

from pathlib import Path

from rich.console import Console

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments import experiment
from pan.models import PAN, Transformer
from pan.training import train
from pan.analysis import ablation_test, save_weights
from pan.plotting import plot_comparison, plot_frequencies

console = Console()


@experiment("compare", help="PAN vs Transformer head-to-head")
def run(cfg: TrainConfig) -> dict:
    console.rule(f"[bold]Compare — a + b mod {cfg.p}")
    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)

    pan = PAN(cfg.p, cfg.k_freqs).to(DEVICE)
    tf = Transformer(cfg.p, cfg.d_model, cfg.n_heads, cfg.d_mlp).to(DEVICE)

    console.print(f"  PAN: {pan.count_parameters():,}   TF: {tf.count_parameters():,}   "
                  f"ratio: {tf.count_parameters() / pan.count_parameters():.0f}×")

    pan_cfg = cfg.overlay(weight_decay=cfg.weight_decay)
    tf_cfg = cfg.overlay(weight_decay=1.0)  # Nanda's default

    hist_pan = train(pan, pan_cfg, train_x, train_y, val_x, val_y, label="PAN")
    hist_tf = train(tf, tf_cfg, train_x, train_y, val_x, val_y, label="TF")

    if not cfg.dry_run:
        ablation_test(pan, val_x, val_y)
        out = Path(cfg.output_dir)
        plot_comparison(hist_pan, hist_tf, pan.count_parameters(),
                        tf.count_parameters(), cfg.p, out / "pan_vs_transformer.png")
        plot_frequencies(pan, out / "pan_frequencies.png")

        if cfg.save_model:
            save_weights(pan, cfg, "PAN", hist_pan.grok_step, out)
            save_weights(tf, cfg, "TF", hist_tf.grok_step, out)

    return dict(pan_grok=hist_pan.grok_step, tf_grok=hist_tf.grok_step)

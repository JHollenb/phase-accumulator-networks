"""Tier 3: Mechanistic equivalence — full checkpoint logging."""

from pathlib import Path

from rich.console import Console

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments import experiment
from pan.models import PAN
from pan.training import train
from pan.analysis import ablation_test, save_history, save_weights
from pan.plotting import plot_tier3, plot_freq_lock, plot_training_curve

console = Console()


@experiment("tier3", help="Mechanistic equivalence with full checkpoint logging")
def run(cfg: TrainConfig) -> dict:
    console.rule(f"[bold]Tier 3 — mod-{cfg.p} K={cfg.k_freqs}")

    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)
    pan = PAN(cfg.p, cfg.k_freqs).to(DEVICE)
    console.print(f"  PAN params: {pan.count_parameters():,}")

    # Force deterministic: no compile (changes MPS accumulation), no early stop
    sub = cfg.overlay(use_compile=False, early_stop=False)
    hist = train(pan, sub, train_x, train_y, val_x, val_y,
                 label="PAN-T3", record_checkpoints=True)

    if not cfg.dry_run:
        ablation_test(pan, val_x, val_y)
        out = Path(cfg.output_dir)
        plot_tier3(hist, pan, cfg.p, out / "tier3_mechanistic.png")
        plot_freq_lock(hist, cfg.p, cfg.k_freqs, out / "tier3_freq_lock.png")
        plot_training_curve(hist, cfg.p, cfg.k_freqs, out / "tier3_training_curve.png")
        save_history(hist, sub, "PAN-T3", out)
        save_weights(pan, sub, "PAN-T3", hist.grok_step, out)

    return dict(grok_step=hist.grok_step, checkpoints=len(hist.freq_checkpoints))

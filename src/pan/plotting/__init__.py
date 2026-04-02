"""Matplotlib visualizations for PAN experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pan.config import TrainHistory
from pan.constants import TWO_PI, SIFP_QUANT_ERROR
from pan.models.pan import PAN

COLORS = {"pan": "#e63946", "tf": "#457b9d"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Comparison plot ─────────────────────────────────────────────────────────


def plot_comparison(
    hist_pan: TrainHistory,
    hist_tf: TrainHistory,
    pan_params: int,
    tf_params: int,
    p: int,
    path: Path,
):
    """Four-panel PAN-vs-Transformer summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PAN vs Transformer — a + b mod {p}", fontsize=14, fontweight="bold")

    # Val accuracy
    ax = axes[0, 0]
    ax.plot(hist_pan.steps, hist_pan.val_acc, color=COLORS["pan"], lw=2,
            label=f"PAN ({pan_params:,})")
    ax.plot(hist_tf.steps, hist_tf.val_acc, color=COLORS["tf"], lw=2, ls="--",
            label=f"Transformer ({tf_params:,})")
    ax.axhline(0.99, color="gray", ls=":", alpha=0.5, label="99 %")
    for h, c, ls in [(hist_pan, COLORS["pan"], "-"), (hist_tf, COLORS["tf"], "--")]:
        if h.grok_step:
            ax.axvline(h.grok_step, color=c, alpha=0.3, ls=ls)
    ax.set(title="Validation Accuracy", xlabel="Step", ylabel="Accuracy", ylim=(-0.05, 1.05))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Loss curves
    ax = axes[0, 1]
    for h, c, tag, ls in [(hist_pan, COLORS["pan"], "PAN", "-"),
                           (hist_tf, COLORS["tf"], "TF", "--")]:
        ax.plot(h.steps, h.train_loss, color=c, alpha=0.5, lw=1, ls=ls, label=f"{tag} train")
        ax.plot(h.steps, h.val_loss, color=c, lw=2, ls=ls, label=f"{tag} val")
    ax.set(title="Loss", xlabel="Step", ylabel="Cross-Entropy", yscale="log")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Parameter bar chart
    ax = axes[1, 0]
    bars = ax.bar(["PAN", "Transformer"], [pan_params, tf_params],
                  color=[COLORS["pan"], COLORS["tf"]], alpha=0.8, edgecolor="black", lw=0.8)
    ax.set(title="Parameters", ylabel="Count", yscale="log")
    for bar, n in zip(bars, [pan_params, tf_params]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{n:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if tf_params > 0 and pan_params > 0:
        ax.text(0.5, 0.5, f"{tf_params / pan_params:.0f}× fewer",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="darkred"))
    ax.grid(alpha=0.3, axis="y")

    # Grokking step comparison
    ax = axes[1, 1]
    gp = hist_pan.grok_step or (max(hist_pan.steps) + 1)
    gt = hist_tf.grok_step or (max(hist_tf.steps) + 1)
    bars = ax.bar(["PAN", "Transformer"], [gp, gt],
                  color=[COLORS["pan"], COLORS["tf"]], alpha=0.8, edgecolor="black", lw=0.8)
    ax.set(title="Steps to Grokking", ylabel="Steps")
    for bar, step, h in zip(bars, [gp, gt], [hist_pan, hist_tf]):
        lbl = f"{step:,}" if h.grok_step else "No grok"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                lbl, ha="center", va="bottom", fontsize=11, fontweight="bold")
    if hist_pan.grok_step and hist_tf.grok_step:
        ratio = gt / gp
        color = "darkgreen" if ratio > 1 else "darkred"
        msg = f"PAN {ratio:.1f}× faster" if ratio > 1 else f"PAN {1/ratio:.1f}× slower"
        ax.text(0.5, 0.7, msg, ha="center", va="center", transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor=color))
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, path)


# ── Frequency convergence ──────────────────────────────────────────────────


def plot_frequencies(model: PAN, path: Path):
    """Learned vs theoretical frequencies — two-panel scatter."""
    info = model.get_learned_frequencies()
    k = model.k_freqs
    x = np.arange(1, k + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Learned vs Theoretical Frequencies (P={model.p})", fontsize=13, fontweight="bold")

    for ax, key, title in [(axes[0], "learned_a", "Encoder A"),
                           (axes[1], "learned_b", "Encoder B")]:
        ax.scatter(x, info["theoretical"], s=100, marker="*", color="black", zorder=5,
                   label="Theory k×2π/P")
        ax.scatter(x, info[key], s=80, marker="o", color="#e63946", zorder=4, label="Learned")
        for i in range(k):
            ax.plot([x[i], x[i]], [info["theoretical"][i], info[key][i]],
                    color="gray", alpha=0.5, lw=1)
        ax.set(title=title, xlabel="k", ylabel="Frequency (rad/token)", xticks=x)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.text(0.97, 0.04, f"Max err: {info[f'error_{key[-1]}'].max():.5f}",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=9, color="gray")

    plt.tight_layout()
    _save(fig, path)


# ── Tier 3 mechanistic panels ──────────────────────────────────────────────


def plot_tier3(history: TrainHistory, model: PAN, p: int, path: Path):
    """Four-panel mechanistic analysis: freq trajectories, Fourier concentration, angular error."""
    if not history.freq_checkpoints:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Tier 3 Mechanistic — mod-{p} (K={model.k_freqs})", fontsize=13, fontweight="bold")

    ck = sorted(history.freq_checkpoints.keys())
    theory = np.array([(k + 1) * TWO_PI / p for k in range(model.k_freqs)])
    colors = plt.cm.tab10(np.linspace(0, 0.9, model.k_freqs))

    for ax, enc, title in [(axes[0, 0], "learned_a", "Encoder A"), (axes[0, 1], "learned_b", "Encoder B")]:
        for k in range(model.k_freqs):
            vals = [history.freq_checkpoints[s][enc][k] for s in ck]
            ax.plot(ck, vals, color=colors[k], lw=1.5, label=f"k={k+1}")
            ax.axhline(theory[k], color=colors[k], lw=0.8, ls="--", alpha=0.5)
        if history.grok_step:
            ax.axvline(history.grok_step, color="black", lw=1.5, ls=":", alpha=0.7,
                       label=f"grok @ {history.grok_step:,}")
        ax.set_title(f"{title} — freq trajectory", fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("freq (rad)")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)

    # Fourier concentration
    ax = axes[1, 0]
    if history.fourier_conc_steps:
        ax.plot(history.fourier_conc_steps, history.fourier_conc_values, color="#2196F3", lw=2)
        ax.fill_between(history.fourier_conc_steps, history.fourier_conc_values,
                        alpha=0.15, color="#2196F3")
        if history.grok_step:
            ax.axvline(history.grok_step, color="black", lw=1.5, ls=":", alpha=0.7)
        ax.set(title="Decoder Fourier Concentration", xlabel="step",
               ylabel="top-K energy / total", ylim=(0, 1))
        ax.grid(alpha=0.25)

    # Angular error at end
    ax = axes[1, 1]
    final = history.freq_checkpoints[ck[-1]]
    x_pos = np.arange(model.k_freqs)
    w = 0.35
    ax.bar(x_pos - w / 2, final["error_a"], w, label="Enc A", color="#4CAF50", alpha=0.8)
    ax.bar(x_pos + w / 2, final["error_b"], w, label="Enc B", color="#FF9800", alpha=0.8)
    ax.axhline(SIFP_QUANT_ERROR, color="red", lw=1.5, ls="--",
               label=f"SIFP-16 ({SIFP_QUANT_ERROR:.5f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"k={k+1}" for k in range(model.k_freqs)], fontsize=8)
    ax.set(title="Angular Error (final)", xlabel="k", ylabel="error (rad)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    _save(fig, path)


# ── Frequency lock-in timeline ─────────────────────────────────────────────


def plot_freq_lock(history: TrainHistory, p: int, k_freqs: int, path: Path,
                   lock_threshold: float = 0.05):
    """When does each frequency lock relative to grokking?"""
    if not history.freq_checkpoints:
        return

    ck = sorted(history.freq_checkpoints.keys())
    theory = np.array([(k + 1) * TWO_PI / p for k in range(k_freqs)])
    colors = plt.cm.tab10(np.linspace(0, 0.9, k_freqs))

    def _first_lock(errs, thr):
        for i in range(len(errs)):
            if all(e < thr for e in errs[i:i + 3]):
                return i
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"Freq lock-in — mod-{p} (K={k_freqs}, thr={lock_threshold:.2f})",
                 fontsize=12, fontweight="bold")

    for ax, enc, title in [(axes[0], "error_a", "Encoder A"), (axes[1], "error_b", "Encoder B")]:
        for k in range(k_freqs):
            errs = [history.freq_checkpoints[s][enc][k] for s in ck]
            ax.plot(ck, errs, color=colors[k], lw=1.5, alpha=0.85, label=f"k={k+1}")
            li = _first_lock(errs, lock_threshold)
            if li is not None:
                ax.plot(ck[li], errs[li], "o", color=colors[k], ms=7, zorder=5)

        ax.axhline(SIFP_QUANT_ERROR, color="red", lw=1, ls=":", alpha=0.7, label="SIFP-16")
        ax.axhline(lock_threshold, color="gray", lw=1, ls="--", alpha=0.6, label="threshold")
        if history.grok_step:
            ax.axvline(history.grok_step, color="black", lw=2, ls=":", alpha=0.8,
                       label=f"grok @ {history.grok_step:,}")
        ax.set(title=title, xlabel="step", ylabel="angular error (rad)", yscale="log")
        ax.set_ylim(1e-5, TWO_PI + 0.5)
        ax.legend(fontsize=7, loc="upper right", ncol=2); ax.grid(alpha=0.2, which="both")

    plt.tight_layout()
    _save(fig, path)


# ── Simple training curve ──────────────────────────────────────────────────


def plot_training_curve(history: TrainHistory, p: int, k_freqs: int, path: Path):
    """Single-panel training curve for standalone PAN runs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history.steps, history.val_acc, color="#4CAF50", lw=2, label="val accuracy")
    peak = max(history.train_loss) if history.train_loss else 1
    ax.plot(history.steps, [l / peak for l in history.train_loss],
            color="#2196F3", lw=1.5, ls="--", alpha=0.7, label="train loss (normalised)")
    if history.grok_step:
        ax.axvline(history.grok_step, color="black", lw=1.5, ls=":", alpha=0.6,
                   label=f"grokked @ {history.grok_step:,}")
    ax.set(xlabel="step", ylabel="value", ylim=(0, 1.05),
           title=f"PAN training — mod-{p} K={k_freqs}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, path)

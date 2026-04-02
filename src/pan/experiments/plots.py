import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_comparison(
    hist_pan,
    hist_tf,
    pan_params: int,
    tf_params: int,
    p: int,
    save_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Phase Accumulator Network vs Transformer\nModular Arithmetic: a + b mod {p}",
        fontsize=14,
        fontweight="bold",
    )

    colors = {"pan": "#e63946", "tf": "#457b9d"}

    ax = axes[0, 0]
    ax.plot(hist_pan.steps, hist_pan.val_acc, color=colors["pan"], linewidth=2,
            label=f"PAN ({pan_params:,} params)")
    ax.plot(hist_tf.steps, hist_tf.val_acc, color=colors["tf"], linewidth=2,
            linestyle="--", label=f"Transformer ({tf_params:,} params)")
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="Grokked (99%)")
    if hist_pan.grok_step is not None:
        ax.axvline(hist_pan.grok_step, color=colors["pan"], alpha=0.3)
    if hist_tf.grok_step is not None:
        ax.axvline(hist_tf.grok_step, color=colors["tf"], alpha=0.3, linestyle="--")
    ax.set(title="Validation Accuracy", xlabel="Step", ylabel="Accuracy", ylim=(-0.05, 1.05))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(hist_pan.steps, hist_pan.train_loss, color=colors["pan"], alpha=0.5, linewidth=1, label="PAN train")
    ax.plot(hist_pan.steps, hist_pan.val_loss, color=colors["pan"], linewidth=2, label="PAN val")
    ax.plot(hist_tf.steps, hist_tf.train_loss, color=colors["tf"], alpha=0.5, linewidth=1, linestyle="--", label="TF train")
    ax.plot(hist_tf.steps, hist_tf.val_loss, color=colors["tf"], linewidth=2, linestyle="--", label="TF val")
    ax.set(title="Loss Curves", xlabel="Step", ylabel="Cross-Entropy Loss", yscale="log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bars = ax.bar(["PAN", "Transformer"], [pan_params, tf_params],
                  color=[colors["pan"], colors["tf"]], alpha=0.8,
                  edgecolor="black", linewidth=0.8)
    ax.set(title="Parameter Count", ylabel="Parameters", yscale="log")
    for bar, count in zip(bars, [pan_params, tf_params]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{count:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    grok_pan = hist_pan.grok_step or (max(hist_pan.steps) + 1 if hist_pan.steps else 0)
    grok_tf = hist_tf.grok_step or (max(hist_tf.steps) + 1 if hist_tf.steps else 0)
    bars = ax.bar(["PAN", "Transformer"], [grok_pan, grok_tf],
                  color=[colors["pan"], colors["tf"]], alpha=0.8,
                  edgecolor="black", linewidth=0.8)
    ax.set(title="Steps to Grokking (lower = better)", ylabel="Steps")
    for bar, step, hist in zip(bars, [grok_pan, grok_tf], [hist_pan, hist_tf]):
        lbl = f"{step:,}" if hist.grok_step else "Did not grok"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                lbl, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_sweep_k(df, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(df["k"], df["grok_rate"], marker="o")
    ax.set_title("Grok Rate vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("Grok Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df["k"], df["mean_grok_step"], marker="o")
    ax.set_title("Mean Grok Step vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean Grok Step")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")

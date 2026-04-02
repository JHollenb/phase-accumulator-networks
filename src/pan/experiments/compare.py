from pathlib import Path
import torch

from pan.constants import DEVICE
from pan.data import make_modular_dataset
from pan.experiments.plots import plot_comparison
from pan.experiments.results import ExperimentResult
from pan.models import PhaseAccumulatorNetwork, TransformerBaseline
from pan.training import TrainConfig, train
from pan.utils import ensure_dir, save_json


def _save_model(path: Path, model, cfg: TrainConfig, label: str, grok_step):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    payload = {
        "state_dict": raw_model.state_dict(),
        "arch": label,
        "param_count": raw_model.count_parameters(),
        "grok_step": grok_step,
        "config": cfg.to_dict(),
    }
    torch.save(payload, path)


def run_compare(cfg: TrainConfig) -> ExperimentResult:
    run_dir = ensure_dir(Path(cfg.output_dir) / "compare")
    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)

    pan = PhaseAccumulatorNetwork(cfg.p, cfg.k_freqs).to(DEVICE)
    tf = TransformerBaseline(cfg.p, cfg.d_model, cfg.n_heads, cfg.d_mlp).to(DEVICE)

    print(f"PAN parameters:         {pan.count_parameters():,}")
    print(f"Transformer parameters: {tf.count_parameters():,}")

    hist_pan = train(pan, cfg, train_x, train_y, val_x, val_y, label="PAN")
    hist_tf = train(tf, cfg, train_x, train_y, val_x, val_y, label="TF")

    pan_params = pan.count_parameters()
    tf_params = tf.count_parameters()

    summary = {
        "experiment": "compare",
        "p": cfg.p,
        "k_freqs": cfg.k_freqs,
        "seed": cfg.seed,
        "pan_params": pan_params,
        "tf_params": tf_params,
        "param_ratio": tf_params / pan_params,
        "pan_grok_step": hist_pan.grok_step,
        "tf_grok_step": hist_tf.grok_step,
        "pan_final_val_acc": hist_pan.val_acc[-1] if hist_pan.val_acc else None,
        "tf_final_val_acc": hist_tf.val_acc[-1] if hist_tf.val_acc else None,
    }

    save_json(run_dir / "config.json", cfg.to_dict())
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "history_pan.json", hist_pan.to_dict())
    save_json(run_dir / "history_tf.json", hist_tf.to_dict())

    artifacts = {
        "config": str(run_dir / "config.json"),
        "summary": str(run_dir / "summary.json"),
        "history_pan": str(run_dir / "history_pan.json"),
        "history_tf": str(run_dir / "history_tf.json"),
    }

    plot_path = run_dir / "compare.png"
    plot_comparison(hist_pan, hist_tf, pan_params, tf_params, cfg.p, str(plot_path))
    artifacts["plot"] = str(plot_path)

    if cfg.save_model and not cfg.dry_run:
        pan_path = run_dir / "pan.pt"
        tf_path = run_dir / "tf.pt"
        _save_model(pan_path, pan, cfg, "PAN", hist_pan.grok_step)
        _save_model(tf_path, tf, cfg, "TransformerBaseline", hist_tf.grok_step)
        artifacts["pan_model"] = str(pan_path)
        artifacts["tf_model"] = str(tf_path)

    return ExperimentResult(
        name="compare",
        run_dir=run_dir,
        summary=summary,
        artifacts=artifacts,
    )

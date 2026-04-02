"""Save / load model weights and training history."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pan.config import TrainConfig, TrainHistory
from pan.constants import DEVICE


def _to_json_safe(v):
    """Recursively convert numpy types to plain Python for JSON."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _to_json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json_safe(i) for i in v]
    return v


def save_history(hist: TrainHistory, cfg: TrainConfig, label: str, save_dir: Path) -> Path:
    """Serialise full TrainHistory to JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        meta=dict(label=label, timestamp=datetime.datetime.now().isoformat(), device=DEVICE),
        config=cfg.model_dump(),
        training_curve=dict(
            steps=hist.steps, train_loss=hist.train_loss,
            val_loss=hist.val_loss, val_acc=hist.val_acc, grok_step=hist.grok_step,
        ),
        freq_checkpoints=_to_json_safe(hist.freq_checkpoints),
        fourier_concentration=dict(steps=hist.fourier_conc_steps, values=hist.fourier_conc_values),
    )
    path = save_dir / f"checkpoints_{label}.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def save_weights(model: nn.Module, cfg: TrainConfig, label: str,
                 grok_step: Optional[int], save_dir: Path) -> Path:
    """Save state_dict + metadata."""
    from pan.models.pan import PAN

    raw = getattr(model, "_orig_mod", model)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"model_{label}.pt"
    torch.save(dict(
        state_dict=raw.state_dict(),
        arch="PAN" if isinstance(raw, PAN) else "Transformer",
        param_count=raw.count_parameters(),
        grok_step=grok_step,
        config=cfg.model_dump(),
    ), path)
    return path

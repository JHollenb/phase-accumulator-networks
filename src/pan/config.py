"""Training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Immutable, serializable training configuration."""

    model_config = {"frozen": True}

    # Task
    p: int = 113
    seed: int = 42

    # Architecture
    k_freqs: int = 5
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512

    # Optimization
    n_steps: int = 50_000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    diversity_weight: float = 0.01

    # Runtime
    val_samples: Optional[int] = None
    use_compile: bool = True
    early_stop: bool = True
    log_every: int = 200
    output_dir: Path = Field(default_factory=lambda: Path("."))
    save_model: bool = False
    dry_run: bool = False
    record_checkpoints: bool = False

    def overlay(self, **kw) -> "TrainConfig":
        return self.model_copy(update=kw)

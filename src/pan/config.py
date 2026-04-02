"""Configuration and training history data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def overlay(self, **overrides) -> "TrainConfig":
        """Return a new config with selected fields replaced."""
        return self.model_copy(update=overrides)


@dataclass
class TrainHistory:
    """Mutable accumulator for metrics recorded during training."""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    grok_step: Optional[int] = None

    # Tier 3 mechanistic checkpoints
    freq_checkpoints: dict = field(default_factory=dict)
    fourier_conc_steps: list[int] = field(default_factory=list)
    fourier_conc_values: list[float] = field(default_factory=list)

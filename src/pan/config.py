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
    log_console: bool = True  # print progress lines during training

    def overlay(self, **kw) -> "TrainConfig":
        return self.model_copy(update=kw)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (Path → str for JSON safety)."""
        d = self.model_dump()
        d["output_dir"] = str(d["output_dir"])
        return d

    def to_str(self) -> str:
        """Human-readable multi-line summary of non-default fields."""
        defaults = TrainConfig()
        lines = []
        for key, val in self.to_dict().items():
            default_val = defaults.to_dict().get(key)
            marker = "" if val == default_val else " ←"
            lines.append(f"  {key:>20}: {val}{marker}")
        return "\n".join(lines)

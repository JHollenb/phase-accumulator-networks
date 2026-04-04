"""Training configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from dataclasses import dataclass

from pan.constants import DEVICE


@dataclass(frozen=True)
class RunConfig:
    project: str = "pan"
    group: str = "default"
    label: str = "pan"
    arch: str = "pan"


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # task
    p: int = 113
    seed: int = 42

    # architecture
    k_freqs: int = 13
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    device: str = DEVICE

    # optimization
    n_steps: int = 100_000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    diversity_weight: float = 0.01

    # runtime
    val_samples: Optional[int] = None
    use_compile: bool = True
    early_stop: bool = True
    log_every: int = 200
    output_dir: Path = Field(default_factory=lambda: Path("."))
    save_model: bool = False
    dry_run: bool = False
    record_checkpoints: bool = False
    log_console: bool = True

    def overlay(self, **kw) -> "TrainConfig":
        return self.model_copy(update=kw)

    def wandb_payload(self) -> dict:
        d = self.model_dump()
        d["output_dir"] = str(d["output_dir"])
        return d

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

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class TrainConfig:
    p:            int   = 113
    n_steps:      int   = 50_000
    batch_size:   int   = 256
    lr:           float = 1e-3
    weight_decay: float = 0.01
    log_every:    int   = 200
    seed:         int   = 42
    # PAN-specific
    k_freqs:        int   = 5
    diversity_weight: float = 0.01   # off-diagonal Gram penalty; 0 = disabled
    # Transformer-specific
    d_model:      int   = 128
    n_heads:      int   = 4
    d_mlp:        int   = 512
    # Performance
    val_samples:  Optional[int] = None   # None = full val set; 1024 = fast
    use_compile:  bool  = True
    early_stop:   bool  = True           # stop once grokked (saves time on sweep)
    output_dir:   str   = '.'            # where to write PNGs
    save_model:   bool  = False           # write state_dict .pt after training
    dry_run:      bool  = False           # print config and exit without training

    def to_dict(self) -> dict:
        return asdict(self)

    def output_path(self) -> Path:
        return Path(self.output_dir)

@dataclass
class TrainHistory:
    steps:      list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss:   list = field(default_factory=list)
    val_acc:    list = field(default_factory=list)
    grok_step:  Optional[int] = None
    # Tier 3 checkpoint fields — only populated when record_checkpoints=True
    # Each entry is keyed by step and holds the full freq_info dict
    freq_checkpoints:    dict = field(default_factory=dict)
    fourier_conc_steps:  list = field(default_factory=list)
    fourier_conc_values: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "grok_step": self.grok_step,
        }


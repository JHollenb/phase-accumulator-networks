"""Model construction from config."""

from pan.config import TrainConfig
from pan.models.base import ModularModel
from pan.models.pan import PAN
from pan.models.transformer import Transformer


def build(arch: str, cfg: TrainConfig) -> ModularModel:
    """Construct a model by name. Handles n_heads derivation for transformer."""
    if arch == "pan":
        return PAN(cfg.p, k=cfg.k_freqs).to(cfg.device)
    if arch == "transformer":
        nh = max(1, cfg.d_model // 16) if cfg.n_heads is None else cfg.n_heads
        return Transformer(cfg.p, cfg.d_model, nh, cfg.d_mlp).to(cfg.device)
    raise ValueError(f"Unknown arch: {arch}")

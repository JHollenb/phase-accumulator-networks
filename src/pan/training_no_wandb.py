"""Training helpers that do not depend on Weights & Biases.

This module mirrors the core PAN training loop but keeps logging local.
When enabled, per-step metrics are accumulated and returned as a pandas
DataFrame for downstream analysis.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from pan.config import RunConfig, TrainConfig
from pan.data import make_modular_dataset
from pan.models import build
from pan.models.base import ModularModel
from pan.models.pan import PAN

console = Console()


@dataclass(frozen=True)
class LocalTrainingResult:
    """Return value for no-wandb training runs."""

    grok_step: Optional[int]
    best_val_acc: float
    best_val_loss: float
    final_val_acc: Optional[float]
    final_val_loss: Optional[float]
    elapsed_sec: float
    metrics: list[dict[str, Any]]
    metrics_df: Any | None = None


def _maybe_compile(model: nn.Module, use: bool) -> nn.Module:
    if not use or not hasattr(torch, "compile"):
        return model
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
        return torch.compile(model, backend="aot_eager")
    except Exception as exc:
        warnings.warn(f"torch.compile failed ({exc}); falling back to eager mode.")
        return model


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _fourier_concentration(W: torch.Tensor, top_k: int = 10) -> float:
    energy = torch.fft.fft(W.float(), dim=0).abs() ** 2
    total = energy.sum().item()
    if total <= 1e-10:
        return 0.0
    top = energy.reshape(-1).topk(min(top_k, energy.numel())).values.sum().item()
    return top / total


def _mixing_weight_entropy(mix_weight: torch.Tensor) -> tuple[float, float]:
    probs = torch.softmax(mix_weight.detach().float().abs(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return float(entropy.mean().item()), float(entropy.min().item())


def _plateau_depth(val_acc_history: list[float], threshold: float = 0.90) -> int:
    return sum(1 for v in val_acc_history if threshold <= v < 0.99)


def _to_native_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _build_metrics_df(rows: list[dict[str, Any]]):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pandas is required for log_to_dataframe=True. Install with `pip install pandas`."
        ) from exc
    return pd.DataFrame(rows)


def train_loop_no_wandb(
    model: ModularModel,
    cfg: TrainConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    label: str = "pan",
    log_to_dataframe: bool = False,
) -> LocalTrainingResult:
    """Train a model without W&B; optionally return step metrics as a DataFrame."""
    if cfg.log_console:
        console.print(f"\n  [bold cyan]┌─ {label} (no-wandb) ─────────────────────────────[/]")
        console.print(cfg.to_str() if hasattr(cfg, "to_str") else str(cfg))
        console.print("  [bold cyan]└──────────────────────────────────────────────────[/]")

    if cfg.dry_run:
        if cfg.log_console:
            console.print(f"  [yellow]dry-run[/] {label} — {cfg.n_steps:,} steps skipped")
        return LocalTrainingResult(
            grok_step=None,
            best_val_acc=float("-inf"),
            best_val_loss=float("inf"),
            final_val_acc=None,
            final_val_loss=None,
            elapsed_sec=0.0,
            metrics=[],
            metrics_df=_build_metrics_df([]) if log_to_dataframe else None,
        )

    torch.manual_seed(cfg.seed)

    compiled = _maybe_compile(model, getattr(cfg, "use_compile", False))
    raw = _unwrap(compiled)
    is_pan = isinstance(raw, PAN)

    weight_decay = cfg.weight_decay if is_pan else 1.0
    opt = torch.optim.AdamW(compiled.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    ex, ey = val_x, val_y
    if getattr(cfg, "val_samples", None) and cfg.val_samples < len(val_x):
        idx = torch.randperm(len(val_x), device=cfg.device)[: cfg.val_samples]
        ex, ey = val_x[idx], val_y[idx]

    n_train = len(train_x)
    grok_step: Optional[int] = None
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    val_acc_history: list[float] = []
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    console_every = cfg.log_every * 5

    last_vl: Optional[float] = None
    last_va: Optional[float] = None

    for step in range(cfg.n_steps):
        step_t0 = time.time()

        compiled.train()
        idx = torch.randperm(n_train, device=cfg.device)[: cfg.batch_size]
        batch_x, batch_y = train_x[idx], train_y[idx]

        logits = compiled(batch_x)
        loss = F.cross_entropy(logits, batch_y)

        if cfg.diversity_weight > 0 and hasattr(raw, "auxiliary_loss"):
            loss = loss + cfg.diversity_weight * raw.auxiliary_loss(batch_x, logits)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _grad_norm(raw)
        opt.step()

        step_time = time.time() - step_t0
        examples_per_sec = cfg.batch_size / max(step_time, 1e-9)

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            compiled.eval()
            with torch.no_grad():
                val_logits = compiled(ex)
                vl = F.cross_entropy(val_logits, ey).item()
                va = (val_logits.argmax(-1) == ey).float().mean().item()

            last_vl = vl
            last_va = va
            best_val_acc = max(best_val_acc, va)
            best_val_loss = min(best_val_loss, vl)
            val_acc_history.append(va)

            if va > 0.99 and grok_step is None:
                grok_step = step

            metrics: dict[str, Any] = {
                "label": label,
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(vl),
                "val_acc": float(va),
                "best_val_acc": float(best_val_acc),
                "best_val_loss": float(best_val_loss),
                "grok": float(va > 0.99),
                "grok_step": grok_step if grok_step is not None else -1,
                "lr": float(opt.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm),
                "step_time_sec": float(step_time),
                "examples_per_sec": float(examples_per_sec),
                "elapsed_sec": float(time.time() - t0),
                "plateau_steps": _plateau_depth(val_acc_history),
            }

            if is_pan:
                freq_info = raw.get_learned_frequencies()
                err_a = freq_info["error_a"]
                err_b = freq_info["error_b"]
                metrics["freq_err_a_mean"] = float(err_a.mean())
                metrics["freq_err_b_mean"] = float(err_b.mean())
                metrics["freq_err_max"] = float(max(err_a.max(), err_b.max()))

                k_count = getattr(raw, "k", None) or getattr(raw, "k_freqs", 0)
                for i in range(k_count):
                    metrics[f"freq_err_a_{i}"] = float(err_a[i])
                    metrics[f"freq_err_b_{i}"] = float(err_b[i])

                dec = raw.dec if hasattr(raw, "dec") else raw.decoder
                metrics["fourier_conc"] = float(_fourier_concentration(dec.weight.detach()))

                mixer = getattr(raw, "mix", None) or getattr(raw, "phase_mix", None)
                if mixer is not None:
                    ent_mean, ent_min = _mixing_weight_entropy(mixer.weight)
                    metrics["mix_entropy_mean"] = ent_mean
                    metrics["mix_entropy_min"] = ent_min

            rows.append(_to_native_metrics(metrics))

            if cfg.log_console and step % console_every == 0:
                console.print(
                    f"  [{label}] step={step:>6,} | "
                    f"train_loss={loss.item():.4f} | "
                    f"val_loss={vl:.4f} | "
                    f"val_acc={va:.4f} | "
                    f"grad_norm={grad_norm:.3f} | "
                    f"{examples_per_sec:.1f} ex/s"
                )

            if grok_step is not None and getattr(cfg, "early_stop", False):
                break

    elapsed = time.time() - t0
    if cfg.log_console:
        status = f"grokked @ {grok_step:,}" if grok_step is not None else "did not grok"
        console.print(f"  [dim]{label} done[/] — {status} — {elapsed:.0f}s total")

    metrics_df = _build_metrics_df(rows) if log_to_dataframe else None
    return LocalTrainingResult(
        grok_step=grok_step,
        best_val_acc=float(best_val_acc),
        best_val_loss=float(best_val_loss),
        final_val_acc=float(last_va) if last_va is not None else None,
        final_val_loss=float(last_vl) if last_vl is not None else None,
        elapsed_sec=float(elapsed),
        metrics=rows,
        metrics_df=metrics_df,
    )


def run_training_no_wandb(
    train_cfg: TrainConfig,
    run_cfg: RunConfig,
    *,
    log_to_dataframe: bool = False,
) -> LocalTrainingResult:
    """Build data/model from configs and train without W&B."""
    model = build(run_cfg.arch, train_cfg)
    tx, ty, vx, vy = make_modular_dataset(
        train_cfg.p,
        seed=train_cfg.seed,
        device=train_cfg.device,
    )
    return train_loop_no_wandb(
        model,
        train_cfg,
        tx,
        ty,
        vx,
        vy,
        label=run_cfg.label or run_cfg.arch,
        log_to_dataframe=log_to_dataframe,
    )

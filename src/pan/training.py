"""Training loop — logs everything to wandb, returns grok_step."""

from __future__ import annotations

import time
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.console import Console

from pan.config import TrainConfig
from pan.constants import DEVICE
from pan.models.base import ModularModel
from pan.models.pan import PAN

console = Console()


def _maybe_compile(model: nn.Module, use: bool) -> nn.Module:
    if not use or not hasattr(torch, "compile"):
        return model
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
        return torch.compile(model, backend="aot_eager")
    except Exception as e:
        warnings.warn(f"torch.compile failed ({e}); eager mode.")
        return model


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _fourier_concentration(W: torch.Tensor, top_k: int = 10) -> float:
    energy = torch.fft.fft(W.float(), dim=0).abs() ** 2
    total = energy.sum().item()
    return energy.reshape(-1).topk(min(top_k, energy.numel())).values.sum().item() / total if total > 1e-10 else 0.0


_defined_labels: set[str] = set()


def reset_metrics():
    """Clear the defined-labels cache. Call when starting a new wandb run."""
    _defined_labels.clear()


def define_wandb_metrics(label: str):
    """Register a label's metric namespace with wandb. Idempotent within a run."""
    if label in _defined_labels:
        return
    step_key = f"{label}_step"
    wandb.define_metric(step_key, hidden=True)
    wandb.define_metric(f"{label}/*", step_metric=step_key)
    _defined_labels.add(label)


def train(
    model: ModularModel,
    cfg: TrainConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    label: str = "model",
) -> Optional[int]:
    """
    Train model on modular addition. Logs to active wandb run.

    Returns grok_step (or None).
    """
    # Print config to console
    if cfg.log_console:
        console.print(f"\n  [bold cyan]┌─ {label} ─────────────────────────────────────[/]")
        console.print(cfg.to_str())
        console.print(f"  [bold cyan]└──────────────────────────────────────────────[/]")

    if cfg.dry_run:
        console.print(f"  [yellow]dry-run[/] {label} — {cfg.n_steps:,} steps skipped")
        return None

    # Register this label's metrics with wandb (idempotent)
    define_wandb_metrics(label)
    step_key = f"{label}_step"

    torch.manual_seed(cfg.seed)
    compiled = _maybe_compile(model, cfg.use_compile)
    opt = torch.optim.AdamW(compiled.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Subsample val set if requested
    ex, ey = val_x, val_y
    if cfg.val_samples and cfg.val_samples < len(val_x):
        idx = torch.randperm(len(val_x), device=DEVICE)[: cfg.val_samples]
        ex, ey = val_x[idx], val_y[idx]

    n_train = len(train_x)
    grok_step = None
    t0 = time.time()
    # How often to print a console progress line (every 5 log intervals)
    console_every = cfg.log_every * 5

    for step in range(cfg.n_steps):
        compiled.train()
        idx = torch.randperm(n_train, device=DEVICE)[: cfg.batch_size]
        batch_x, batch_y = train_x[idx], train_y[idx]
        logits = compiled(batch_x)
        loss = F.cross_entropy(logits, batch_y)

        raw = _unwrap(compiled)
        if cfg.diversity_weight > 0:
            loss = loss + cfg.diversity_weight * raw.auxiliary_loss(batch_x, logits)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            compiled.eval()
            with torch.no_grad():
                vl = F.cross_entropy(compiled(ex), ey).item()
                va = (compiled(ex).argmax(-1) == ey).float().mean().item()

            metrics = {
                step_key: step,
                f"{label}/train_loss": loss.item(),
                f"{label}/val_loss": vl,
                f"{label}/val_acc": va,
            }

            # Mechanistic checkpoints for PAN
            if cfg.record_checkpoints and isinstance(raw, PAN):
                info = raw.get_learned_frequencies()
                for i in range(raw.k):
                    metrics[f"{label}/freq_a_{i}"] = float(info["learned_a"][i])
                    metrics[f"{label}/freq_b_{i}"] = float(info["learned_b"][i])
                    metrics[f"{label}/err_a_{i}"] = float(info["error_a"][i])
                    metrics[f"{label}/err_b_{i}"] = float(info["error_b"][i])
                metrics[f"{label}/fourier_conc"] = _fourier_concentration(raw.dec.weight.detach())

            wandb.log(metrics)

            # Periodic console progress
            if cfg.log_console and step % console_every == 0:
                elapsed = time.time() - t0
                console.print(
                    f"  [{label}] step={step:>6,} | "
                    f"train_loss={loss.item():.3f} | "
                    f"val_loss={vl:.3f} | "
                    f"val_acc={va:.3f} | "
                    f"{elapsed:.0f}s"
                )

            if va > 0.99 and grok_step is None:
                grok_step = step
                console.print(
                    f"  [bold green]★ {label} GROKKED[/] step={step:,} "
                    f"acc={va:.3f} ({time.time() - t0:.0f}s)"
                )
                if cfg.early_stop:
                    break

    # Final summary
    elapsed = time.time() - t0
    if cfg.log_console:
        status = f"grokked @ {grok_step:,}" if grok_step else "did not grok"
        console.print(f"  [dim]{label} done[/] — {status} — {elapsed:.0f}s total")

    return grok_step

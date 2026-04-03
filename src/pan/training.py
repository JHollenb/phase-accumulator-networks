"""W&B-friendly training loop for PAN experiments.

Pattern:
- train(..., config=None) is the main entrypoint
- wandb.init(config=...) happens inside train()
- wandb.config is treated as the source of truth after init()
- works for:
    1) normal runs
    2) wandb sweeps via wandb.agent(..., function=train_wrapper)

Returns grok_step (or None).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Optional

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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _maybe_compile(model: nn.Module, use: bool) -> nn.Module:
    if not use or not hasattr(torch, "compile"):
        return model
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
        return torch.compile(model, backend="aot_eager")
    except Exception as e:
        warnings.warn(f"torch.compile failed ({e}); falling back to eager mode.")
        return model


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _fourier_concentration(W: torch.Tensor, top_k: int = 10) -> float:
    energy = torch.fft.fft(W.float(), dim=0).abs() ** 2
    total = energy.sum().item()
    if total <= 1e-10:
        return 0.0
    top = energy.reshape(-1).topk(min(top_k, energy.numel())).values.sum().item()
    return top / total


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _param_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return math.sqrt(total)


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        return dict(cfg.to_dict())
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {
            k: v
            for k, v in vars(cfg).items()
            if not k.startswith("_") and not callable(v)
        }
    raise TypeError(f"Cannot convert config of type {type(cfg)!r} to dict")


def _coerce_like(value: Any, like: Any) -> Any:
    if like is None:
        return value

    if isinstance(like, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)

    if isinstance(like, float):
        return float(value)

    if isinstance(like, str):
        return str(value)

    return value


def _merge_cfg_from_wandb(base_cfg: TrainConfig, wb_cfg: Any) -> TrainConfig:
    """Return a cfg with fields overridden by wandb.config when names match."""
    updates = {}
    for key, value in dict(wb_cfg).items():
        if hasattr(base_cfg, key):
            current = getattr(base_cfg, key)
            updates[key] = _coerce_like(value, current)

    if is_dataclass(base_cfg):
        return replace(base_cfg, **updates)

    # fallback for non-dataclass configs
    for key, value in updates.items():
        setattr(base_cfg, key, value)
    return base_cfg


def build_config_defaults(cfg: TrainConfig, *, label: str = "model") -> dict[str, Any]:
    """Config passed to wandb.init(config=...). Safe for both regular runs and sweeps."""
    defaults = _cfg_to_dict(cfg)

    # Common experiment metadata fields; sweeps can override these too.
    defaults.setdefault("label", label)
    defaults.setdefault("project", "pan")
    defaults.setdefault("group", None)
    defaults.setdefault("job_type", "train")
    defaults.setdefault("tags", None)

    # W&B behavior toggles
    defaults.setdefault("watch_model", False)
    defaults.setdefault("watch_log_freq", max(getattr(cfg, "log_every", 50), 50))

    return defaults


_defined_metric_namespaces: set[str] = set()


def reset_metrics() -> None:
    _defined_metric_namespaces.clear()


def define_wandb_metrics(label: str) -> None:
    """Define metrics once per run/namespace."""
    if label in _defined_metric_namespaces:
        return

    step_key = f"{label}/step"
    wandb.define_metric(step_key)
    wandb.define_metric(f"{label}/*", step_metric=step_key)

    wandb.define_metric(f"{label}/train_loss", summary="min")
    wandb.define_metric(f"{label}/val_loss", summary="min")
    wandb.define_metric(f"{label}/val_acc", summary="max")
    wandb.define_metric(f"{label}/best_val_acc", summary="max")
    wandb.define_metric(f"{label}/best_val_loss", summary="min")
    wandb.define_metric(f"{label}/grok_step", summary="min")
    wandb.define_metric(f"{label}/examples_per_sec", summary="mean")
    wandb.define_metric(f"{label}/step_time_sec", summary="mean")
    wandb.define_metric(f"{label}/grad_norm", summary="last")
    wandb.define_metric(f"{label}/param_norm", summary="last")

    _defined_metric_namespaces.add(label)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    model: ModularModel,
    cfg: TrainConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    config: Optional[dict[str, Any]] = None,
) -> Optional[int]:
    """
    Sweep-friendly training entrypoint.

    Pattern:
        grok_step = train(model, cfg, train_x, train_y, val_x, val_y)

    or for sweeps:
        wandb.agent(sweep_id, function=lambda: train(...))

    Args:
        model: already-constructed model
        cfg: baseline TrainConfig
        train_x/train_y/val_x/val_y: tensors on the correct device
        config: optional overrides passed to wandb.init(config=config)

    Returns:
        grok_step if val_acc first exceeds 0.99 else None
    """
    config_defaults = build_config_defaults(cfg)

    if config is not None:
        config_defaults.update(config)

    with wandb.init(
        project=config_defaults.get("project"),
        group=config_defaults.get("group"),
        job_type=config_defaults.get("job_type", "train"),
        tags=config_defaults.get("tags"),
        config=config_defaults,
    ):
        wb_cfg = wandb.config
        label = wb_cfg.label

        # Merge sweep-sampled params back into TrainConfig
        cfg = _merge_cfg_from_wandb(cfg, wb_cfg)

        define_wandb_metrics(label)

        if cfg.log_console:
            console.print(f"\n  [bold cyan]┌─ {label} ─────────────────────────────────────[/]")
            if hasattr(cfg, "to_str") and callable(cfg.to_str):
                console.print(cfg.to_str())
            else:
                console.print(str(_cfg_to_dict(cfg)))
            console.print(f"  [bold cyan]└──────────────────────────────────────────────[/]")

        if cfg.dry_run:
            console.print(f"  [yellow]dry-run[/] {label} — {cfg.n_steps:,} steps skipped")
            wandb.log({f"{label}/step": 0, f"{label}/dry_run": 1})
            return None

        torch.manual_seed(cfg.seed)

        compiled = _maybe_compile(model, cfg.use_compile)
        raw = _unwrap(compiled)

        opt = torch.optim.AdamW(
            compiled.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        if bool(wb_cfg.get("watch_model", False)):
            wandb.watch(raw, log="all", log_freq=int(wb_cfg.get("watch_log_freq", cfg.log_every)))

        # Subsample val set if requested
        ex, ey = val_x, val_y
        if cfg.val_samples and cfg.val_samples < len(val_x):
            idx = torch.randperm(len(val_x), device=DEVICE)[: cfg.val_samples]
            ex, ey = val_x[idx], val_y[idx]

        n_train = len(train_x)
        grok_step: Optional[int] = None
        best_val_acc = float("-inf")
        best_val_loss = float("inf")
        t0 = time.time()
        console_every = cfg.log_every * 5

        for step in range(cfg.n_steps):
            step_t0 = time.time()

            compiled.train()
            idx = torch.randperm(n_train, device=DEVICE)[: cfg.batch_size]
            batch_x, batch_y = train_x[idx], train_y[idx]

            logits = compiled(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            if cfg.diversity_weight > 0:
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

                best_val_acc = max(best_val_acc, va)
                best_val_loss = min(best_val_loss, vl)

                if va > 0.99 and grok_step is None:
                    grok_step = step
                    if cfg.log_console:
                        console.print(
                            f"  [bold green]★ {label} GROKKED[/] step={step:,} "
                            f"acc={va:.3f} ({time.time() - t0:.0f}s)"
                        )

                metrics = {
                    f"{label}/step": step,
                    f"{label}/train_loss": float(loss.item()),
                    f"{label}/val_loss": float(vl),
                    f"{label}/val_acc": float(va),
                    f"{label}/best_val_acc": float(best_val_acc),
                    f"{label}/best_val_loss": float(best_val_loss),
                    f"{label}/grok": float(va > 0.99),
                    f"{label}/grok_step": grok_step if grok_step is not None else -1,
                    f"{label}/lr": float(opt.param_groups[0]["lr"]),
                    f"{label}/grad_norm": float(grad_norm),
                    f"{label}/param_norm": float(_param_norm(raw)),
                    f"{label}/step_time_sec": float(step_time),
                    f"{label}/examples_per_sec": float(examples_per_sec),
                    f"{label}/elapsed_sec": float(time.time() - t0),
                }

                # PAN-specific mechanistic metrics
                if cfg.record_checkpoints and isinstance(raw, PAN):
                    info = raw.get_learned_frequencies()
                    for i in range(raw.k):
                        metrics[f"{label}/freq_a_{i}"] = float(info["learned_a"][i])
                        metrics[f"{label}/freq_b_{i}"] = float(info["learned_b"][i])
                        metrics[f"{label}/err_a_{i}"] = float(info["error_a"][i])
                        metrics[f"{label}/err_b_{i}"] = float(info["error_b"][i])
                    metrics[f"{label}/fourier_conc"] = float(
                        _fourier_concentration(raw.dec.weight.detach())
                    )

                wandb.log(metrics)

                if cfg.log_console and step % console_every == 0:
                    elapsed = time.time() - t0
                    console.print(
                        f"  [{label}] step={step:>6,} | "
                        f"train_loss={loss.item():.4f} | "
                        f"val_loss={vl:.4f} | "
                        f"val_acc={va:.4f} | "
                        f"grad_norm={grad_norm:.3f} | "
                        f"{examples_per_sec:.1f} ex/s | "
                        f"{elapsed:.0f}s"
                    )

                if grok_step is not None and cfg.early_stop:
                    break

        elapsed = time.time() - t0
        wandb.summary[f"{label}/elapsed_sec"] = float(elapsed)
        wandb.summary[f"{label}/final_val_acc"] = float(va)
        wandb.summary[f"{label}/final_val_loss"] = float(vl)
        wandb.summary[f"{label}/best_val_acc"] = float(best_val_acc)
        wandb.summary[f"{label}/best_val_loss"] = float(best_val_loss)
        wandb.summary[f"{label}/grok_step"] = grok_step

        if cfg.log_console:
            status = f"grokked @ {grok_step:,}" if grok_step is not None else "did not grok"
            console.print(f"  [dim]{label} done[/] — {status} — {elapsed:.0f}s total")

        return grok_step

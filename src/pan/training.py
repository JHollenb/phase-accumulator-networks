"""Generic training loop for PAN and Transformer."""

from __future__ import annotations

import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from pan.config import TrainConfig, TrainHistory
from pan.constants import DEVICE
from pan.analysis.mechanistic import fourier_concentration

console = Console()


# ── Helpers ─────────────────────────────────────────────────────────────────


def maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    """Wrap with torch.compile when available; silent fallback otherwise."""
    if not use_compile or not hasattr(torch, "compile"):
        return model
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.recompile_limit = 64
        return torch.compile(model, backend="aot_eager")
    except Exception as e:
        warnings.warn(f"torch.compile failed ({e}); running eager.")
        return model


def _subsample_val(val_x, val_y, n: int | None):
    if n is None or n >= len(val_x):
        return val_x, val_y
    idx = torch.randperm(len(val_x), device=val_x.device)[:n]
    return val_x[idx], val_y[idx]


def unwrap(model: nn.Module) -> nn.Module:
    """Unwrap a torch.compiled model to access raw parameters."""
    return getattr(model, "_orig_mod", model)


# ── Main loop ───────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    cfg: TrainConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    label: str = "model",
    record_checkpoints: bool = False,
) -> TrainHistory:
    """
    Train a model on modular addition.  Identical loop for PAN and Transformer.

    Uses AdamW + full-batch val eval.  Returns a populated TrainHistory.
    """
    console.print(f"  [dim]train[/] {label}  p={cfg.p} steps={cfg.n_steps:,} "
                  f"wd={cfg.weight_decay} seed={cfg.seed}")

    if cfg.dry_run:
        console.print(f"  [yellow]dry-run — skipping {cfg.n_steps:,} steps[/]")
        return TrainHistory()

    torch.manual_seed(cfg.seed)
    model = maybe_compile(model, cfg.use_compile)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    eval_x, eval_y = _subsample_val(val_x, val_y, cfg.val_samples)

    history = TrainHistory()
    n_train = len(train_x)
    t0 = time.time()

    with Progress(SpinnerColumn(), *Progress.get_default_columns(),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        task = prog.add_task(f"[cyan]{label}", total=cfg.n_steps)

        for step in range(cfg.n_steps):
            model.train()
            idx = torch.randperm(n_train, device=DEVICE)[: cfg.batch_size]
            logits = model(train_x[idx])
            loss = F.cross_entropy(logits, train_y[idx])

            # Diversity regularisation (PAN only)
            if cfg.diversity_weight > 0 and hasattr(unwrap(model), "phase_mix"):
                raw = unwrap(model)
                with torch.no_grad():
                    phi_a = raw.encoder_a(train_x[idx][:, 0])
                    phi_b = raw.encoder_b(train_x[idx][:, 1])
                mix = raw.phase_mix(torch.cat([phi_a, phi_b], dim=-1))
                mix = mix - mix.mean(0, keepdim=True)
                mix = mix / mix.norm(dim=0, keepdim=True).clamp(min=1e-6)
                gram = mix.T @ mix / mix.shape[0]
                eye = torch.eye(gram.shape[0], device=gram.device)
                loss = loss + cfg.diversity_weight * (gram - eye).pow(2).sum() / gram.shape[0]

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if step % cfg.log_every == 0:
                model.eval()
                with torch.no_grad():
                    vl = F.cross_entropy(model(eval_x), eval_y).item()
                    va = (model(eval_x).argmax(-1) == eval_y).float().mean().item()

                history.steps.append(step)
                history.train_loss.append(loss.item())
                history.val_loss.append(vl)
                history.val_acc.append(va)

                if record_checkpoints and hasattr(unwrap(model), "get_learned_frequencies"):
                    raw = unwrap(model)
                    history.freq_checkpoints[step] = raw.get_learned_frequencies()
                    dec_w = raw.decoder.weight.detach().float()
                    history.fourier_conc_steps.append(step)
                    history.fourier_conc_values.append(
                        fourier_concentration(dec_w, top_k=min(10, dec_w.shape[0]))
                    )

                if va > 0.99 and history.grok_step is None:
                    history.grok_step = step
                    console.print(
                        f"  [bold green]★ {label} GROKKED[/] step={step:,} "
                        f"acc={va:.3f} ({time.time() - t0:.0f}s)"
                    )
                    if cfg.early_stop:
                        break

            prog.update(task, completed=step + 1)

    return history

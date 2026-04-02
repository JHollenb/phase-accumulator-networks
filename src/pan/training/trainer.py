import warnings
import torch
import torch.nn as nn

# compile
def _maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    """
    Wrap model with torch.compile if available and requested.

    On M-series Macs with PyTorch ≥ 2.3 this gives ~1.5–2× speedup on the
    Transformer. Silently falls back to eager mode if compile fails — some
    MPS ops are not yet supported by the inductor backend.

    NOTE: do NOT use compile in primes mode. Each prime P produces a
    differently-shaped decoder (bias dim = P), which triggers a recompile
    per prime — noisy W0328 warnings and no speedup benefit since PAN is
    tiny. primes mode passes use_compile=False explicitly.
    """
    if not use_compile:
        return model
    if not hasattr(torch, 'compile'):
        warnings.warn("torch.compile not available (need PyTorch ≥ 2.0). "
                      "Running in eager mode.")
        return model
    try:
        # Silence dynamo recompile chatter — we handle shape changes by
        # disabling compile where shapes vary (primes mode).
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.recompile_limit = 64
        return torch.compile(model, backend='aot_eager')
    except Exception as e:
        warnings.warn(f"torch.compile failed ({e}). Running in eager mode.")
        return model


# TODO - losses
import torch

def pan_diversity_loss(model, x_batch: torch.Tensor) -> torch.Tensor:
    """
    Penalize mode collapse in PAN phase mixing outputs.
    Assumes model has encoder_a, encoder_b, phase_mix.
    """
    with torch.no_grad():
        phi_a = model.encoder_a(x_batch[:, 0])
        phi_b = model.encoder_b(x_batch[:, 1])

    mix_out = model.phase_mix(torch.cat([phi_a, phi_b], dim=-1))
    mix_norm = mix_out - mix_out.mean(0, keepdim=True)
    norms = mix_norm.norm(dim=0, keepdim=True).clamp(min=1e-6)
    mix_norm = mix_norm / norms
    gram = mix_norm.T @ mix_norm / mix_out.shape[0]
    eye = torch.eye(gram.shape[0], device=gram.device)
    return (gram - eye).pow(2).sum() / gram.shape[0]


def auxiliary_loss(model, cfg, x_batch: torch.Tensor) -> torch.Tensor:
    if cfg.diversity_weight > 0 and hasattr(model, "phase_mix"):
        return cfg.diversity_weight * pan_diversity_loss(model, x_batch)
    return torch.tensor(0.0, device=x_batch.device)

# trainer
import time
import torch
import torch.nn.functional as F

from pan.constants import DEVICE
from pan.training.compile import maybe_compile
from pan.training.config import TrainConfig, TrainHistory
from pan.training.losses import auxiliary_loss


def _make_val_loader(val_x, val_y, val_samples):
    if val_samples is None or val_samples >= len(val_x):
        return val_x, val_y
    idx = torch.randperm(len(val_x), device=val_x.device)[:val_samples]
    return val_x[idx], val_y[idx]


def _print_run_config(cfg: TrainConfig, label: str) -> None:
    print(f"  ┌─ config [{label}] ─────────────────────────────────────")
    print(f"  │  p={cfg.p}  k={cfg.k_freqs}  seed={cfg.seed}  steps={cfg.n_steps:,}")
    print(
        f"  │  lr={cfg.lr}  weight_decay={cfg.weight_decay}  "
        f"diversity_weight={cfg.diversity_weight}"
    )
    print(
        f"  │  batch={cfg.batch_size}  log_every={cfg.log_every}  "
        f"early_stop={cfg.early_stop}"
    )
    print(f"  │  val_samples={cfg.val_samples or 'full'}  compile={cfg.use_compile}")
    print(f"  └────────────────────────────────────────────────────────")


def train(
    model,
    cfg: TrainConfig,
    train_x,
    train_y,
    val_x,
    val_y,
    label: str = "model",
) -> TrainHistory:
    _print_run_config(cfg, label)

    if cfg.dry_run:
        print(f"  [dry-run] would train {cfg.n_steps:,} steps — skipping")
        return TrainHistory()

    torch.manual_seed(cfg.seed)
    model = maybe_compile(model, cfg.use_compile)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    eval_x, eval_y = _make_val_loader(val_x, val_y, cfg.val_samples)

    history = TrainHistory()
    n_train = len(train_x)
    t0 = time.time()

    for step in range(cfg.n_steps):
        model.train()

        idx = torch.randperm(n_train, device=DEVICE)[:cfg.batch_size]
        x_batch = train_x[idx]
        y_batch = train_y[idx]

        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss = loss + auxiliary_loss(model, cfg, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(eval_x)
                val_loss = F.cross_entropy(val_logits, eval_y).item()
                val_acc = (val_logits.argmax(-1) == eval_y).float().mean().item()

            history.steps.append(step)
            history.train_loss.append(float(loss.item()))
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)

            if val_acc > 0.99 and history.grok_step is None:
                history.grok_step = step
                elapsed = time.time() - t0
                print(
                    f"  ★ [{label}] GROKKED at step {step} "
                    f"(val_acc={val_acc:.3f}, {elapsed:.0f}s elapsed)"
                )
                if cfg.early_stop:
                    print("  → early stopping")
                    break

            if step % (cfg.log_every * 5) == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{label}] step={step:6d} | "
                    f"train_loss={loss.item():.3f} | "
                    f"val_acc={val_acc:.3f} | "
                    f"{elapsed:.0f}s"
                )

    return history

from pan.config import get_settings


def say_hello(name: str | None = None) -> str:
    settings = get_settings()
    target = name or settings.greeting_target
    return f"Hello, {target}! (app={settings.app_name}, debug={settings.debug})"

import argparse
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Device selection ─────────────────────────────────────────────────────────
# MPS note: Apple Silicon MPS is 3–5× faster than CPU for the Transformer
# (the slow part of this experiment). PAN is so small it barely matters.
# MPS caveat: torch.compile has limited MPS support prior to PyTorch 2.3.
# We handle this gracefully in _maybe_compile().

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps'  if torch.backends.mps.is_available()
    else 'cpu'
)

# ── Constants ────────────────────────────────────────────────────────────────
PHASE_SCALE   = 65536
PHASE_SCALE_F = 65536.0
TWO_PI        = 2.0 * math.pi


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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA
# ═══════════════════════════════════════════════════════════════════════════

def make_modular_dataset(p: int, train_frac: float = 0.4, seed: int = 42):
    """
    Generate all (a, b, (a+b) mod p) triples.

    Note: Nanda uses 40% training data — counterintuitively, *less* training
    data makes grokking more pronounced because the network must generalize
    rather than memorize. We match this setting for fair comparison.

    Returns: (train_inputs, train_labels, val_inputs, val_labels)
    All integer tensors on DEVICE.
    """
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b, (a + b) % p)
                      for a in range(p) for b in range(p)], dtype=np.int64)
    perm  = rng.permutation(len(pairs))
    pairs = pairs[perm]

    n_train = int(train_frac * len(pairs))
    train   = torch.tensor(pairs[:n_train], device=DEVICE)
    val     = torch.tensor(pairs[n_train:], device=DEVICE)

    return train[:, :2], train[:, 2], val[:, :2], val[:, 2]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: PHASE ACCUMULATOR NETWORK
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TRANSFORMER BASELINE
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def _make_val_loader(val_x, val_y, val_samples: Optional[int]):
    """
    Optionally subsample the validation set.

    WHY THIS MATTERS:
    Full val eval on P=113 = 7,661 samples takes ~338ms on CPU per call.
    At log_every=200 and 50K steps that's 250 evals × 338ms = ~85s overhead.
    Subsampling to 1024 cuts this to ~11s. The grokking transition is sharp
    (0% → 99% in a few hundred steps) so 1024 samples is enough to detect it.
    Use val_samples=None for final publication-quality runs.
    """
    if val_samples is None or val_samples >= len(val_x):
        return val_x, val_y
    idx = torch.randperm(len(val_x), device=val_x.device)[:val_samples]
    return val_x[idx], val_y[idx]


def _print_run_config(cfg: TrainConfig, label: str) -> None:
    """
    Print the exact TrainConfig used for this training run.
    Called at the start of every train() call so logs are self-contained
    and sub-run configs (sweep/primes) are verifiable.
    """
    print(f"  ┌─ config [{label}] ─────────────────────────────────────")
    print(f"  │  p={cfg.p}  k={cfg.k_freqs}  seed={cfg.seed}  steps={cfg.n_steps:,}")
    print(f"  │  lr={cfg.lr}  weight_decay={cfg.weight_decay}  diversity_weight={cfg.diversity_weight}")
    print(f"  │  batch={cfg.batch_size}  log_every={cfg.log_every}  early_stop={cfg.early_stop}")
    print(f"  │  val_samples={cfg.val_samples or 'full'}  compile={cfg.use_compile}")
    print(f"  └────────────────────────────────────────────────────────")


def train(model: nn.Module, cfg: TrainConfig,
          train_x, train_y, val_x, val_y,
          label: str = "model",
          record_checkpoints: bool = False) -> TrainHistory:
    """
    Training loop — identical for PAN and Transformer.
    Uses AdamW with weight decay, matching Nanda's setup.

    Key optimizations vs naive loop:
    - val_samples: subsample val set to avoid 338ms/eval overhead
    - early_stop: halt once grokked, saves remaining steps in sweep mode
    - torch.compile: 1.5–2× speedup on MPS for Transformer (aot_eager backend)
    """
    _print_run_config(cfg, label)
    if cfg.dry_run:
        print(f"  [dry-run] would train {cfg.n_steps:,} steps — skipping")
        return TrainHistory()
    torch.manual_seed(cfg.seed)

    # Compile after seeding (compile is deterministic given the same seed)
    model = _maybe_compile(model, cfg.use_compile)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Fix val loader once — don't re-sample every eval (reproducible)
    eval_x, eval_y = _make_val_loader(val_x, val_y, cfg.val_samples)

    history = TrainHistory()
    n_train = len(train_x)
    t0      = time.time()

    for step in range(cfg.n_steps):
        model.train()

        idx      = torch.randperm(n_train, device=DEVICE)[:cfg.batch_size]
        x_batch  = train_x[idx]
        y_batch  = train_y[idx]

        logits = model(x_batch)
        loss   = F.cross_entropy(logits, y_batch)

        # Diversity regularization (PAN only) — penalises mode collapse where
        # all K phase-mixing outputs converge to the same frequency.
        # We compute the Gram matrix of the K output phase channels and add a
        # penalty on the off-diagonal entries. When all channels are identical
        # the off-diagonal entries are 1; when they are orthogonal they are 0.
        # Only applied when diversity_weight > 0 and model has phase_mix.
        if cfg.diversity_weight > 0 and hasattr(model, 'phase_mix'):
            with torch.no_grad():
                phi_a = model.encoder_a(x_batch[:, 0])
                phi_b = model.encoder_b(x_batch[:, 1])
            mix_out = model.phase_mix(torch.cat([phi_a, phi_b], dim=-1))
            # Normalise each channel to zero-mean unit-variance before Gram
            mix_norm = mix_out - mix_out.mean(0, keepdim=True)
            norms    = mix_norm.norm(dim=0, keepdim=True).clamp(min=1e-6)
            mix_norm = mix_norm / norms
            gram     = mix_norm.T @ mix_norm / mix_out.shape[0]  # (K, K)
            eye      = torch.eye(gram.shape[0], device=gram.device)
            div_loss = (gram - eye).pow(2).sum() / gram.shape[0]
            loss     = loss + cfg.diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(eval_x)
                val_loss   = F.cross_entropy(val_logits, eval_y).item()
                val_acc    = (val_logits.argmax(-1) == eval_y).float().mean().item()

            history.steps.append(step)
            history.train_loss.append(loss.item())
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)

            # Tier 3: record frequency snapshots and Fourier concentration
            if record_checkpoints and hasattr(model, 'get_learned_frequencies'):
                # Unwrap compiled model to access raw parameters
                raw_model = model
                if hasattr(model, '_orig_mod'):
                    raw_model = model._orig_mod
                history.freq_checkpoints[step] = raw_model.get_learned_frequencies()
                # Fourier concentration of decoder weight rows (K→P projection)
                dec_w = raw_model.decoder.weight.detach().float()  # (P, K)
                history.fourier_conc_steps.append(step)
                history.fourier_conc_values.append(
                    fourier_concentration(dec_w, top_k=min(10, dec_w.shape[0]))
                )

            if val_acc > 0.99 and history.grok_step is None:
                history.grok_step = step
                elapsed = time.time() - t0
                print(f"  ★ [{label}] GROKKED at step {step} "
                      f"(val_acc={val_acc:.3f}, {elapsed:.0f}s elapsed)")
                if cfg.early_stop:
                    print(f"  → early stopping (--early-stop)")
                    break

            if step % (cfg.log_every * 5) == 0:
                elapsed = time.time() - t0
                print(f"  [{label}] step={step:6d} | "
                      f"train_loss={loss.item():.3f} | "
                      f"val_acc={val_acc:.3f} | "
                      f"{elapsed:.0f}s")

    return history



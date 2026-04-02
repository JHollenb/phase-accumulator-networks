import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


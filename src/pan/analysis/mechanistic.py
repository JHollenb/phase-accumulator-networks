"""Mechanistic interpretability tools for PAN."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pan.constants import TWO_PI, SIFP_QUANT_ERROR


def fourier_concentration(embed_W: torch.Tensor, top_k: int = 10) -> float:
    """Fraction of total FFT energy captured by the top-k frequency bins."""
    W = embed_W.float()
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    energy = torch.fft.fft(W, dim=0).abs() ** 2
    total = energy.sum().item()
    if total < 1e-10:
        return 0.0
    return energy.reshape(-1).topk(top_k).values.sum().item() / total


def ablation_test(model: nn.Module, val_x, val_y) -> dict[str, float]:
    """Zero-out each PAN component; measure accuracy drop."""
    from pan.models.pan import PAN

    model.eval()
    results = {}

    with torch.no_grad():
        base = (model(val_x).argmax(-1) == val_y).float().mean().item()
        results["baseline"] = base

        if not isinstance(model, PAN):
            return results

        # Zero phase mixing
        orig_w = model.phase_mix.weight.data.clone()
        model.phase_mix.weight.data.zero_()
        results["zero_mixing"] = (model(val_x).argmax(-1) == val_y).float().mean().item()
        model.phase_mix.weight.data.copy_(orig_w)

        # Randomise frequencies
        orig_fa, orig_fb = model.encoder_a.freq.data.clone(), model.encoder_b.freq.data.clone()
        model.encoder_a.freq.data = torch.rand_like(orig_fa) * TWO_PI
        model.encoder_b.freq.data = torch.rand_like(orig_fb) * TWO_PI
        results["random_freqs"] = (model(val_x).argmax(-1) == val_y).float().mean().item()
        model.encoder_a.freq.data.copy_(orig_fa)
        model.encoder_b.freq.data.copy_(orig_fb)

        # Zero reference phases
        orig_ref = model.phase_gate.ref_phase.data.clone()
        model.phase_gate.ref_phase.data.zero_()
        results["zero_gates"] = (model(val_x).argmax(-1) == val_y).float().mean().item()
        model.phase_gate.ref_phase.data.copy_(orig_ref)

    return results


def detect_mode_collapse(model) -> bool:
    """True if all K mixing outputs are dominated by the same input slot."""
    W = model.phase_mix.weight.detach().cpu().numpy()
    dominant = [int(np.argmax(np.abs(row))) for row in W]
    return len(set(dominant)) == 1

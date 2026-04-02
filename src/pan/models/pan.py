"""Phase Accumulator Network — the core architecture."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pan.constants import TWO_PI


class PhaseEncoder(nn.Module):
    """
    Encode integer token a ∈ [0, P) as K phases: φ_k(a) = a × freq_k  mod 2π.

    Initialised to the natural Fourier basis of ℤ_P so training starts from
    the theoretically optimal point.  In SIFP-32 hardware this is a single
    integer multiply-mod (~2 cycles).
    """

    def __init__(self, p: int, k_freqs: int):
        super().__init__()
        self.p, self.k_freqs = p, k_freqs
        init = torch.tensor([(k + 1) * TWO_PI / p for k in range(k_freqs)])
        self.freq = nn.Parameter(init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens.float().unsqueeze(-1) * self.freq.unsqueeze(0)) % TWO_PI


class PhaseMixer(nn.Module):
    """Linear mix of N input phases → M output phases, wrapped mod 2π."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * 0.1 + 1.0 / n_in)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        return F.linear(phases, self.weight) % TWO_PI


class PhaseGate(nn.Module):
    """
    Phase-selective activation: gate = (1 + cos(φ − φ_ref)) / 2.

    ref_phase is wrapped inside forward() so Adam's momentum can push the
    stored parameter outside [0, 2π) without gradient blow-up.
    """

    def __init__(self, n_phases: int):
        super().__init__()
        self.ref_phase = nn.Parameter(torch.rand(n_phases) * TWO_PI)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        ref = torch.remainder(self.ref_phase, TWO_PI)
        return (1.0 + torch.cos(phases - ref.unsqueeze(0))) / 2.0


class PAN(nn.Module):
    """
    Phase Accumulator Network for modular arithmetic.

    Pipeline:  encode(a,b) → mix(2K→K) → gate(K) → decode(K→P)

    ~743 params for P=113 K=5  vs  ~227 K for Nanda's transformer.
    """

    def __init__(self, p: int, k_freqs: int = 5):
        super().__init__()
        self.p, self.k_freqs = p, k_freqs

        self.encoder_a = PhaseEncoder(p, k_freqs)
        self.encoder_b = PhaseEncoder(p, k_freqs)
        self.phase_mix = PhaseMixer(2 * k_freqs, k_freqs)
        self.phase_gate = PhaseGate(k_freqs)
        self.decoder = nn.Linear(k_freqs, p, bias=True)

        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phi_a = self.encoder_a(inputs[:, 0])
        phi_b = self.encoder_b(inputs[:, 1])
        mixed = self.phase_mix(torch.cat([phi_a, phi_b], dim=-1))
        gated = self.phase_gate(mixed)
        return self.decoder(gated)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learned_frequencies(self) -> dict:
        """Learned vs theoretical frequencies with angular error."""
        fa_raw = self.encoder_a.freq.detach().cpu().numpy()
        fb_raw = self.encoder_b.freq.detach().cpu().numpy()
        fa, fb = fa_raw % TWO_PI, fb_raw % TWO_PI
        theory = np.array([(k + 1) * TWO_PI / self.p for k in range(self.k_freqs)])

        def _err(learned, theory):
            d = np.abs(learned - theory) % TWO_PI
            return np.minimum(d, TWO_PI - d)

        return dict(
            learned_a=fa, learned_b=fb,
            learned_a_raw=fa_raw, learned_b_raw=fb_raw,
            theoretical=theory,
            error_a=_err(fa, theory), error_b=_err(fb, theory),
        )

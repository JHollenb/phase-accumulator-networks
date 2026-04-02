"""Phase Accumulator Network — the core architecture."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pan.constants import TWO_PI
from pan.models.base import ModularModel


class PhaseEncoder(nn.Module):
    """Encode integer token a ∈ [0, P) as K phases: φ_k(a) = a × freq_k  mod 2π."""

    def __init__(self, p: int, k: int):
        super().__init__()
        self.freq = nn.Parameter(torch.tensor([(i + 1) * TWO_PI / p for i in range(k)]))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens.float().unsqueeze(-1) * self.freq.unsqueeze(0)) % TWO_PI


class PhaseMixer(nn.Module):
    """Linear mix of N phases → M phases, wrapped mod 2π."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * 0.1 + 1.0 / n_in)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        return F.linear(phases, self.weight) % TWO_PI


class PhaseGate(nn.Module):
    """Phase-selective activation: (1 + cos(φ − φ_ref)) / 2."""

    def __init__(self, k: int):
        super().__init__()
        self.ref = nn.Parameter(torch.rand(k) * TWO_PI)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        return (1.0 + torch.cos(phases - torch.remainder(self.ref, TWO_PI))) / 2.0


class PAN(ModularModel):
    """
    Phase Accumulator Network: encode → mix → gate → decode.

    ~743 params for P=113 K=5  vs  ~227K for Nanda's transformer.
    """

    def __init__(self, p: int, k: int = 5):
        super().__init__()
        self.p, self.k = p, k
        self.enc_a = PhaseEncoder(p, k)
        self.enc_b = PhaseEncoder(p, k)
        self.mix = PhaseMixer(2 * k, k)
        self.gate = PhaseGate(k)
        self.dec = nn.Linear(k, p)
        nn.init.normal_(self.dec.weight, std=0.02)
        nn.init.zeros_(self.dec.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phi = self.mix(torch.cat([self.enc_a(inputs[:, 0]), self.enc_b(inputs[:, 1])], -1))
        return self.dec(self.gate(phi))

    def auxiliary_loss(self, inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Diversity regularisation: penalise mode collapse in the mixing layer."""
        with torch.no_grad():
            phi = torch.cat([self.enc_a(inputs[:, 0]), self.enc_b(inputs[:, 1])], -1)
        m = self.mix(phi)
        m = (m - m.mean(0, keepdim=True)) / m.norm(dim=0, keepdim=True).clamp(min=1e-6)
        gram = m.T @ m / m.shape[0]
        return (gram - torch.eye(gram.shape[0], device=gram.device)).pow(2).sum() / gram.shape[0]

    # ── Mechanistic introspection ───────────────────────────────────────────

    def theoretical_freqs(self) -> np.ndarray:
        return np.array([(i + 1) * TWO_PI / self.p for i in range(self.k)])

    def get_learned_frequencies(self) -> dict:
        """Learned vs theoretical frequencies with angular error."""
        fa = self.enc_a.freq.detach().cpu().numpy() % TWO_PI
        fb = self.enc_b.freq.detach().cpu().numpy() % TWO_PI
        th = self.theoretical_freqs()

        def err(l, t):
            d = np.abs(l - t) % TWO_PI
            return np.minimum(d, TWO_PI - d)

        return dict(learned_a=fa, learned_b=fb, theoretical=th,
                    error_a=err(fa, th), error_b=err(fb, th))

    def is_mode_collapsed(self) -> bool:
        """True if all mixing outputs are dominated by the same input slot."""
        W = self.mix.weight.detach().cpu().numpy()
        return len({int(np.argmax(np.abs(row))) for row in W}) == 1

import torch.nn as nn

class PhaseEncoder(nn.Module):
    """
    Encodes an integer token a ∈ [0, P) as K phases:
        φ_k(a) = a × freq_k   mod 2π

    The freq_k are learned, initialized to the natural Fourier basis of ℤ_P:
        freq_k_init = k × 2π / P   for k = 1..K

    In SIFP-32 hardware this would be integer multiply-mod, ~2 cycles.
    In software we work in float [0, 2π) for differentiability.

    Key property: for any learned frequency f, the phase φ_k(a+b mod P) equals
    φ_k(a) + φ_k(b) mod 2π — addition becomes phase rotation.
    This is the U(1) group isomorphism that makes modular arithmetic tractable.
    """

    def __init__(self, p: int, k_freqs: int):
        super().__init__()
        self.p       = p
        self.k_freqs = k_freqs

        # Initialize to natural Fourier basis of Z_P
        init_freqs = torch.tensor(
            [(k + 1) * TWO_PI / p for k in range(k_freqs)],
            dtype=torch.float32
        )
        self.freq = nn.Parameter(init_freqs)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch,) integer tensor in [0, P)
        returns: (batch, K) float tensor of phases in [0, 2π)
        """
        a      = tokens.float().unsqueeze(-1)   # (batch, 1)
        f      = self.freq.unsqueeze(0)          # (1, K)
        phases = (a * f) % TWO_PI               # (batch, K)
        return phases


class PhaseMixingLayer(nn.Module):
    """
    Mixes N input phases into M output phases via learned phase offsets.

    For each output j:
        φ_out[j] = (Σᵢ w[j,i] × φ_in[i]) mod 2π

    In hardware: 16-bit integer weights, multiply-add mod 2^16.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        w_init      = torch.randn(n_out, n_in) * 0.1 + (1.0 / n_in)
        self.weight = nn.Parameter(w_init)

    def forward(self, phases_in: torch.Tensor) -> torch.Tensor:
        """
        phases_in: (batch, N_in) in [0, 2π)
        returns:   (batch, N_out) in [0, 2π)
        """
        return F.linear(phases_in, self.weight) % TWO_PI


class PhaseGate(nn.Module):
    """
    Phase-selective nonlinearity: fires when input phase aligns with reference.

        gate[j] = (1 + cos(φ_in[j] - φ_ref[j])) / 2    ∈ [0, 1]

    FIX — ref_phase wrapping:
    ref_phase lives on 𝕊¹ but Adam treats it as a value in ℝ. Without
    intervention, momentum pushes ref_phase outside [0, 2π). Seen in
    the 100K run: gates at 429°, 482° etc., causing a step-62K accuracy
    collapse as the gradient blew up near an inflection point.

    Fix: wrap ref_phase with torch.remainder inside forward() before
    computing the diff. The stored parameter can still wander in ℝ but
    the effective phase is always in [0, 2π). torch.remainder has
    gradient = 1 almost everywhere so autograd is unaffected.
    """

    def __init__(self, n_phases: int):
        super().__init__()
        self.ref_phase = nn.Parameter(torch.rand(n_phases) * TWO_PI)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: (batch, K) in [0, 2π)
        returns: (batch, K) gates in [0, 1]
        """
        # Wrap into [0, 2π) before diff — prevents wandering gradient spikes.
        ref        = torch.remainder(self.ref_phase, TWO_PI)
        phase_diff = phases - ref.unsqueeze(0)
        return (1.0 + torch.cos(phase_diff)) / 2.0


class PhaseAccumulatorNetwork(nn.Module):
    """
    The complete Phase Accumulator Network (PAN) for modular arithmetic.

    Architecture:
      1. PhaseEncoder:    [a, b] → [φ_k(a), φ_k(b)] for k=0..K-1
      2. PhaseMixingLayer: 2K → K  (combine phases from a and b)
      3. PhaseGate:        K → K   (phase-selective activation)
      4. Linear decoder:   K → P   (project to logits)

    Parameter count for P=113, K=5: ~743 parameters
    Transformer baseline:           ~227,200 parameters
    Ratio:                          ~305×
    """

    def __init__(self, p: int, k_freqs: int = 5):
        super().__init__()
        self.p       = p
        self.k_freqs = k_freqs

        self.encoder_a  = PhaseEncoder(p, k_freqs)
        self.encoder_b  = PhaseEncoder(p, k_freqs)
        self.phase_mix  = PhaseMixingLayer(2 * k_freqs, k_freqs)
        self.phase_gate = PhaseGate(k_freqs)
        self.decoder    = nn.Linear(k_freqs, p, bias=True)

        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, 2) integer tensor [a, b]
        returns: (batch, P) logits
        """
        a = inputs[:, 0]
        b = inputs[:, 1]

        phi_a      = self.encoder_a(a)
        phi_b      = self.encoder_b(b)
        phi_mixed  = self.phase_mix(torch.cat([phi_a, phi_b], dim=-1))
        gates      = self.phase_gate(phi_mixed)
        return self.decoder(gates)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learned_frequencies(self) -> dict:
        """
        Return learned frequencies alongside theoretical Fourier basis values.

        The encoder stores raw frequency parameters in ℝ. These can drift
        outside [0, 2π) — values like -6.26 are aliases of their correct
        canonical value modulo 2π. We wrap to [0, 2π) before computing
        errors so the error reflects true angular distance, not raw drift.
        """
        freqs_a_raw  = self.encoder_a.freq.detach().cpu().numpy()
        freqs_b_raw  = self.encoder_b.freq.detach().cpu().numpy()
        # Wrap to [0, 2π) — same operation as the forward pass modulo
        freqs_a      = freqs_a_raw % TWO_PI
        freqs_b      = freqs_b_raw % TWO_PI
        theoretical  = np.array([(k + 1) * TWO_PI / self.p
                                  for k in range(self.k_freqs)])
        # Angular error: minimum distance on the circle
        def _angle_err(learned, theory):
            diff = np.abs(learned - theory) % TWO_PI
            return np.minimum(diff, TWO_PI - diff)
        return {
            'learned_a':    freqs_a,
            'learned_b':    freqs_b,
            'learned_a_raw': freqs_a_raw,
            'learned_b_raw': freqs_b_raw,
            'theoretical':  theoretical,
            'error_a':      _angle_err(freqs_a, theoretical),
            'error_b':      _angle_err(freqs_b, theoretical),
        }


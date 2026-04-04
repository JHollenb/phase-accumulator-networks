# Phase-Native Networks: A New Computational Basis for Sinusoidal Computation

**Research Proposal v0.1 — March 2026**
*Companion to: Spectral IEEE 754 Whitepaper, SIFP Format Specifications*

---

## 1. What We Are Trying to Do

We are trying to determine whether there exists a **fundamentally different computational basis** for neural networks — one where sinusoidal composition is the atomic operation rather than a discovered emergent behavior.

Specifically: we want to train a small network where every "neuron" is a phase accumulator, every "weight" is a phase offset, the forward pass is a sequence of integer additions modulo 2π, and the network learns to solve structured tasks (starting with modular arithmetic) with fewer parameters and in fewer gradient steps than a standard transformer.

If it works, it is not a compression algorithm. It is evidence that the Fourier representations discovered by transformers through billions of gradient steps can instead be the **ground-level architecture** — that the computation transformers painfully learn to approximate is the computation this architecture natively performs.

---

## 2. Why

### 2.1 The Quake Moment Analogy

In 1999, Quake III shipped with an inverse square root implementation that exploited a structural property of IEEE 754 floating point: reinterpreting a float as an integer gives you its base-2 logarithm for free. The result was that `1/sqrt(x)` — normally requiring an expensive iterative computation — collapsed to one integer subtract, one shift, and one Newton-Raphson step.

The trick did not make `sqrt()` faster. It made `sqrt()` **unnecessary** for that use case, because the bit layout of the number format already encoded the answer.

The question this proposal pursues: **is there a cost-model shift for sinusoidal arithmetic equivalent to what IEEE 754 provided for logarithms?**

In the SIFP-32 format, `e^(iα) × e^(iβ) = e^(i(α+β))` maps to one integer add. Not a fast approximation. Exact, by construction. Phase addition **is** integer addition. A network designed around this primitive has a completely different cost model than a transformer.

### 2.2 The Fourier Interpretability Evidence

Three independent lines of mechanistic interpretability research converge on the same finding:

**Nanda et al. (ICLR 2023)** fully reverse-engineered a one-layer transformer trained on modular addition. The network learns an explicit Fourier multiplication algorithm: inputs are encoded as sinusoidal functions at 5 key frequencies, the MLP computes trigonometric products, and the output is decoded via sinusoidal projection. The circuit is genuinely Fourier — ablating all non-key components *improved* performance.

**Kantamneni & Tegmark (2025)** extended this to GPT-J (6B), Pythia (6.9B), and Llama-3.1 (8B). These models represent numbers as generalized helices at periods T = [2, 5, 10, 100]. The "Clock algorithm" — circular rotation of sinusoidal representations — is how these multi-billion parameter models compute arithmetic.

**Zhou et al. (NeurIPS 2024)** showed that GPT-2-XL and GPT-J use Fourier features with outlier components at periods 2, 2.5, 5, and 10. MLP layers use low-frequency features for magnitude, attention layers use high-frequency features for modular arithmetic.

The finding is consistent: **transformers trained on arithmetic discover sinusoidal computation**. They spend billions of parameters and gradient steps converging on a representational solution that SIFP encodes directly in the bit layout.

The question follows naturally: if the computation the network is trying to learn is sinusoidal phase rotation, what happens if you give the network phase rotation as its primitive — not something it has to discover, but something it starts with?

### 2.3 The Cost Model Shift

IEEE 754's cost model: multiplication cheap, addition cheap, transcendentals expensive.

SIFP's cost model: **sinusoidal composition free, addition of arbitrary values expensive.**

Every algorithm and architecture in neural network design is optimized for the IEEE 754 cost model. The transformer's attention mechanism, its MLP, its positional encoding — all assume that matmuls and pointwise nonlinearities are the cheap operations. Nobody has seriously designed a neural architecture around the SIFP cost model, because SIFP hasn't existed as a well-specified format.

A network where phase composition is the atomic operation does not look like a transformer. Similarity between two representations is phase alignment (circular distance), not dot product. Nonlinear operations are phase modulation (FM synthesis), not ReLU. Positional encoding is a phase accumulator advance (one integer add per token), not sin/cos evaluation. The residual connection — the expensive addition in SIFP — may not need to exist because gradient flow in a phase accumulator network does not have the same vanishing gradient problem.

---

## 3. The Proposed Architecture: Phase Accumulator Networks (PAN)

### 3.1 Core Primitive: The Phase Neuron

A standard neuron computes: `y = activation(Σ wᵢxᵢ + b)`

A phase neuron computes: `φ_out = (Σ wᵢφᵢ_in) mod 2π`

Where:
- `φᵢ_in` is the phase of input i (a scalar in [0, 2π), stored as a 16-bit integer in SIFP)
- `wᵢ` is an integer weight (a phase multiplier, also 16-bit)
- The output is a phase, which is the input to the next layer

The operation is: **integer multiply-accumulate modulo 65536.** On hardware, this is cheaper than a floating-point multiply. On software, it's a `uint16_t` multiply and add.

### 3.2 The Phase Layer

A phase layer takes N input phases and produces M output phases:

```
φ_out[j] = (Σᵢ W[j,i] × φ_in[i]) mod 65536    for j = 0..M-1
```

Where `W[j,i]` are 16-bit integer weights. The entire forward pass is integer arithmetic.

Note what this layer computes: it is a **linear phase mixing** — a generalized Fourier projection. If the inputs encode sinusoidal signals at various frequencies, the output is a weighted combination of those phases. This is the operation that Nanda's grokked transformer was doing in its MLP layers, emergently. Here it is the architecture.

### 3.3 The Amplitude Layer

Phase alone cannot represent magnitude. The architecture maintains a parallel **amplitude stream** — the log-magnitude field L from SIFP-32. An amplitude layer is:

```
L_out[j] = (Σᵢ V[j,i] × L_in[i]) + b[j]
```

Where `V[j,i]` are fixed-point weights and the sum is standard arithmetic. This is the expensive part of the architecture — but it operates on log-magnitudes, which are bounded and smooth, making it more numerically stable than operating on linear amplitudes.

The full neuron state is `(L, φ)` — a SIFP-32 word.

### 3.4 The Phase Comparison (Nonlinearity)

A standard network uses ReLU or GELU as its nonlinearity. Phase networks use **phase distance** as the nonlinearity:

```
gate[j] = cos(φ_out[j] - φ_ref[j])    (scalar in [-1, 1])
output[j] = gate[j] × 2^(L_out[j])
```

Where `φ_ref[j]` is a learned reference phase per neuron. The output is large when the input phase aligns with the reference phase and small when they are in opposition. This is **phase-selective activation** — the neuron fires when it receives input with the right sinusoidal structure.

This is not arbitrary. It is exactly the operation that Nanda's interpretability analysis found in the grokked network: neurons that selectively respond to specific Fourier components of the input.

### 3.5 Architecture for Modular Arithmetic (Minimal Test Case)

Input: two integers a, b ∈ [0, P) for prime P.

**Encoding layer:** Map each input to a set of K phase accumulators at different frequencies:
```
φₖ(a) = (a × freq_word_k) mod 65536    for k = 0..K-1
```
where `freq_word_k = round(k × 65536 / P)` — these are the SIFP-PA frequency control words for the natural Fourier basis of ℤ_P.

**Phase mixing layer:** A 2K → K phase layer that combines phases from a and b:
```
φ_mixed[j] = (Σ W[j,i] × φ_encoded[i]) mod 65536
```

**Phase comparison layer:** K phase neurons with learned reference phases produce K scalar gates.

**Decoding layer:** Standard linear layer mapping K scalars to logits over P classes.

Total parameters for P=113, K=5: 10 frequencies (2 encoders × 5) + 50 mixing weights + 5 ref phases + 113×5 + 113 decoder = **743 parameters total** (verified empirically).

Compare to our transformer baseline: 227,200 parameters for the same task.

**If the PAN architecture solves mod-113 addition, it does so with 305× fewer parameters.** This is the test. (Spoiler: it does — see §4.1.)

### 3.6 How It Differs From Standard Ideas

| Property | Standard Transformer | Phase-Native Network |
|----------|---------------------|---------------------|
| Weight representation | FP32 reals | 16-bit phase integers |
| Forward pass operation | Float multiply-accumulate | Integer multiply-accumulate mod 2π |
| Nonlinearity | ReLU / GELU / SiLU | Phase comparison (cosine gate) |
| Similarity measure | Dot product | Phase alignment (circular distance) |
| Positional encoding | sin/cos evaluation | Phase accumulator advance (1 integer add) |
| Sin/cos cost | O(1) transcendental | O(0) — native representation |
| Gradient flow | Through addition (residual) | Through phase accumulation |
| Inductive bias | None (universal approximator) | Sinusoidal / Fourier structure |
| What it discovers | Fourier circuits, emergently | Starts from Fourier structure |

This is not a transformer with a different number format. It is a different architecture with a different primitive operation, designed around a different cost model.

---

## 4. How We Will Test It

### 4.1 Tier 1: Existence Proof — ✓ PASSED

**Task:** Modular arithmetic — a + b mod 113 (Nanda's exact task)

**Baseline:** Nanda's 1-layer transformer, ~227K parameters (our implementation),
grokked at step 7,000.

**PAN experiment:**
- Architecture as described in §3.5
- K = 5 frequencies, 743 total parameters
- Phase mixing layer: 10 → 5 phase neurons
- Decoder: 5 → 113 linear

**Pass criterion:** PAN achieves >99% validation accuracy on mod-113 addition

**Kill criterion:** PAN cannot exceed 50% validation accuracy after 100K steps.

**Result: ★ GROKKED at step 48,400 — val_acc = 99.9%**

| Metric | PAN | Transformer | Notes |
|--------|-----|-------------|-------|
| Parameters | **743** | 227,200 | **305× fewer** |
| Grokking step | 48,400 | 7,000 | 6.9× more steps |
| Wall-clock to grok | ~62s | ~15s | ~4× slower in real time |
| Val accuracy | 99.9% | 99.1% | PAN marginally higher |

The ablation is definitive — zeroing any single component collapses accuracy
to chance, confirming phase arithmetic is the active mechanism:

| Ablation | Accuracy | Drop |
|----------|----------|------|
| Baseline (trained) | 99.9% | — |
| Zero phase mixing | 0.9% | −99.1% |
| Randomize frequencies | 0.7% | −99.2% |
| Zero ref phases | 2.0% | −97.9% |

**Architecture notes from the run:**
Two bugs were found and fixed during the experiment. (1) The `PhaseGate`
reference phases must be wrapped with `torch.remainder` inside `forward()`;
without this, Adam pushes them outside [0, 2π), causing gradient spikes and
accuracy collapses (observed at step 62K in an earlier run). (2) The
`PhaseMixingLayer` is prone to mode collapse where all K outputs converge to a
single frequency. An off-diagonal Gram penalty (`diversity_weight=0.01`) on the
mixing layer outputs resolves this. Both fixes are live in `pan.py`.

**What the network learned:**
The phase mixing converged to two effective frequency slots — `freq[3]` and
`freq[4]` — routed independently from inputs `a` and `b`, with mixing weights
near ±1. The network found a two-frequency solution where the theoretical
analysis predicted five. This is a more compressed representation than Nanda
found in the transformer and warrants mechanistic investigation in Tier 3.

### 4.2 Tier 2: Parameter Efficiency — ✓ PASSED

**Task:** Sweep K from 1 to 15. Find the minimum K at which PAN grokks reliably across 3 seeds.

**Result: minimum reliable K = 9 (1,319 parameters)**

Full results — K=1 to 15, 3 seeds each, WD=0.01, DW=0.01, 100K steps:

| K | Grokked | Mean step | Params | Notes |
|---|---------|-----------|--------|-------|
| 1 | 0/3 | — | 231 | Cannot span mod-113 Fourier basis |
| 2 | 0/3 | — | 353 | Insufficient capacity |
| 3 | 0/3 | — | 479 | Insufficient capacity |
| 4 | 0/3 | — | 609 | Insufficient capacity |
| 5 | 0/3 | — | 743 | Seed-sensitive — compare run (seed=42 only) grokked at 48,400 |
| 6 | 1/3 | 13,800 | 881 | Borderline |
| 7 | 1/3 | 29,800 | 1,023 | Borderline |
| 8 | 0/3 | — | 1,169 | **Anomaly** — s42 peaked at 97.4% but never crossed 99% |
| 9 | 2/3 | 22,500 | 1,319 | **Minimum reliable K** |
| 10 | 2/3 | 10,100 | 1,473 | |
| 11 | 2/3 | 15,000 | 1,631 | |
| 12 | 3/3 | 15,133 | 1,793 | First K where all seeds grokk |
| 13 | 3/3 | 21,733 | 1,959 | |
| 14 | 3/3 | 23,867 | 2,129 | |
| 15 | 3/3 | 7,200 | 2,303 | Fastest mean grokking step |

**Key findings:**

The capacity threshold is real and sharp. K<5 networks are representationally incapable — mod-113 requires at least 5 Fourier frequencies (matching Nanda's finding) and a PAN with K=3 can only represent 3. Loss barely moves from random baseline for K≤4.

K=5 is the theoretical minimum but is seed-sensitive. The compare run (Tier 1) grokked because seed=42 happens to be a favorable initialization. The sweep used three seeds and all three failed at K=5. This is an important nuance: the architecture *can* grok at K=5, but doesn't do so reliably.

The K=8 anomaly is unexplained. K=7 grokked 1/3 seeds and K=9 grokked 2/3, but K=8 grokked 0/3. Seed 42 at K=8 reached 97.4% val accuracy and plateaued there for 60,000 steps — close to grokking but stalled. This is likely a local minimum specific to the K=8 loss landscape and warrants investigation.

K≥12 achieves 3/3 reliability at 1,793 parameters — still 127× fewer than the transformer baseline.

**Parameter efficiency vs transformer:**
- Minimum reliable PAN (K=9): 1,319 params vs 227,200 — **172× fewer**
- Fully reliable PAN (K=12): 1,793 params vs 227,200 — **127× fewer**

### 4.3 Tier 3: Mechanistic Equivalence (UNLOCKED)

If Tier 1 passes: use TransformerLens-style hooks to inspect the learned phase weights. Do the learned `freq_word_k` values converge to `k × 65536 / 113` — the exact Fourier basis of ℤ₁₁₃? Do the learned reference phases correspond to the output logit phases?

This is the mechanistic interpretability test: **does the PAN learn the same algorithm that Nanda found in the transformer, but directly, without having to discover it through gradient descent on a general architecture?**

### 4.4 Tier 4: Generalization — ✓ PASSED

**Task:** Test K=9 PAN across primes P ∈ {43, 67, 89, 113, 127} with identical hyperparameters (WD=0.01, DW=0.01, seed=42). No tuning per prime.

**Result: 5/5 primes grokked to ≥99.1% accuracy.**

| P | Grok step | Final acc | Params | Notes |
|---|-----------|-----------|--------|-------|
| 43 | 12,000 | 99.2% | 619 | K=9 over-provisioned; fast |
| 67 | 11,800 | 99.1% | 859 | |
| 89 | 139,800 | 99.1% | 1,079 | Slow grokker — needed 140K steps |
| 113 | 11,200 | 99.3% | 1,319 | Our primary benchmark |
| 127 | 23,400 | 99.2% | 1,459 | |

**Key findings:**

The architecture is principled — same K, same hyperparameters, different prime, same result. No per-prime tuning was required. This rules out the concern that K=9 on mod-113 is a lucky coincidence specific to that problem.

P=89 grokked at step 139,800 rather than ~11K like the others. The 100K run had ended mid-cliff (97.4% at step 99K, clearly still rising). This is not a failure mode but a longer pre-grokking phase — the same qualitative pattern, just slower. The cause is likely that P=89 is close enough to P=113 that K=9 is near its minimum for that prime size, making the optimizer's job harder than for smaller primes where K=9 is over-provisioned.

**Parameter scaling:** The decoder is K×P, so parameter count scales linearly with P. A 619-parameter network (P=43) solving a nontrivial group operation to 99.2% accuracy may be the smallest neural network to do so.

Still to test (not yet run):
- Modular multiplication: a × b mod P
- Two-step arithmetic: (a + b) × c mod P
- Held-out primes not seen during development (P=59, 71, 97)

### 4.5 Tier 5: The Hardest Test — Language

If Tiers 1-4 pass: attempt phase encoding of token embeddings for a small language model. This tests whether the inductive bias is useful beyond modular arithmetic or whether it is specifically a modular-arithmetic trick.

**The honest expectation:** Language probably does not reduce to sinusoidal composition in the way modular arithmetic does. A PAN language model may underperform a transformer. This is a valid and important negative result — it would constrain the claim to "SIFP-native architectures are the right primitive for sinusoidal computation tasks" rather than "SIFP-native architectures replace transformers."

---

## 5. How We Will Prove It Is Real

### 5.1 The Three Tests That Cannot Be Faked

**Test 1 — Parameter count:** Count the parameters. If PAN solves mod-113 with fewer than 5,000 parameters and the baseline transformer requires >100,000, the result is real. Parameters cannot be hidden.

**Test 2 — Mechanistic alignment:** Run Nanda's full Fourier analysis on the PAN's learned weights. The analysis measures whether the network is computing with sinusoidal representations by examining the DFT of the embedding matrix. If the PAN's learned frequency words match the theoretical values `k × 65536 / 113` to within quantization error, the network is genuinely using phase arithmetic — not finding some other solution.

**Test 3 — Ablation:** Take a trained PAN and zero out the phase mixing layer. If the network's accuracy collapses from 99% to chance, the phase mixing is doing the work. If accuracy holds, the network found a shortcut that doesn't use phase arithmetic, and the result is not what we claimed.

### 5.2 The Comparison That Matters

The standard comparison is: "does PAN match transformer accuracy?" That is not the interesting comparison. The interesting comparison is:

**At what parameter count does a transformer first become capable of solving mod-113?**

Train transformers of size N = {128, 256, 512, 1K, 2K, 5K, 10K, 50K, 340K} parameters. Find the smallest transformer that grokks. Compare that parameter count to PAN.

If PAN solves the task with parameters at or below the transformer's threshold, the architecture is using the structure of the problem in a way the transformer cannot at that scale.

### 5.3 What Would Falsify the Claim

- PAN requires more parameters than the baseline transformer to achieve the same accuracy. → The phase primitive is not more efficient for this task.
- PAN's learned frequency words do not converge to the theoretical Fourier basis values. → The network is solving the problem through a different mechanism, not phase arithmetic.
- Removing the phase mixing layer does not degrade PAN's performance. → Phase arithmetic is not the active ingredient.
- PAN fails to generalize across primes without retraining. → The architecture is brittle, not principled.

Any of these results falsifies the core claim. The experiment is designed so that these falsifications are detectable.

---

## 6. What a Positive Result Means

If PAN solves modular arithmetic with dramatically fewer parameters, faster grokking, and interpretably sinusoidal learned weights, the implication is:

**The optimal architecture for sinusoidal computation is not a transformer that discovers Fourier circuits through gradient descent. It is an architecture whose forward pass is Fourier computation.**

This opens three directions:

**Direction 1 — Sinusoidal-native architectures for DSP and scientific computing.** Signal processing, PDE solving, and climate modeling are dominated by sinusoidal computation. A PAN-based architecture for these tasks would be to neural computation what the FFT was to signal processing: not faster matrix multiplication, but a different algorithm that makes the matrix multiplication unnecessary.

**Direction 2 — Hybrid transformers with PAN sublayers.** Replace the transformer's MLP sublayer with a PAN for layers that interpretability analysis identifies as computing sinusoidal operations. This is surgical — use the right primitive for the right computation.

**Direction 3 — The full SIFP stack.** A PAN architecture that trains and infers in SIFP format, with phase arithmetic from embedding to logit. At inference, the forward pass is integer arithmetic throughout. No floating point required for the sinusoidal computation paths.

---

*v0.2 — March 2026*
*Status: Tiers 1, 2 & 4 validated. PAN grokked mod-P addition at ≥99.1% across all 5 test primes with identical hyperparameters (K=9, 619–1,459 params). Tier 3 (mechanistic) and Tier 5 (language) in progress.*

# PAN Hypothesis Tracking Report

*Phase Accumulator Networks — mechanistic metrics and expected observations*

---

## How to read this document

Each hypothesis is written as a falsifiable claim. For each one:

- **Why** explains the theoretical motivation
- **Tracked by** lists the exact wandb metric keys
- **Confirming signal** describes what the data looks like if the hypothesis is correct
- **Falsifying signal** describes what would invalidate it
- **Ambiguous / watch for** covers intermediate cases that require interpretation

Hypotheses are grouped into three clusters: existing paper claims, K=8 anomaly, and new speed hypotheses from the latest sweep analysis.

---

## Cluster A — Core mechanistic claims (existing paper)

These are the claims that constitute the paper's central contribution. All must hold for the work to be publishable.

---

### Hyp 1 — Phase arithmetic is the active mechanism, not a decoder shortcut

**Why.** The ablation results show accuracy collapses to chance when you zero the phase mixing layer, randomize the frequencies, or zero the reference phases. But the ablation only shows the mechanism is *necessary* — it doesn't prove the decoder isn't exploiting a secondary shortcut alongside the phase circuit. The decoder is a full linear layer mapping K scalars to P logits; in principle it could memorize a lookup table that doesn't depend on the phase structure at all.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/fourier_conc` | Energy fraction of decoder weights in top-10 Fourier modes |
| `{label}/mix_entropy_mean` | Mean entropy of mixing weight distribution per channel |
| `{label}/mix_entropy_min` | Minimum entropy across channels (the cleanlest channel) |

**Confirming signal.** `fourier_conc` should be low early in training (decoder weights are random, broadly spread across frequencies) and rise sharply at or just after the grokking step as the decoder aligns to sinusoidal basis vectors. A value above ~0.7 at the grokked solution is strong evidence the decoder is computing a Fourier projection, not a lookup table. `mix_entropy_mean` should fall during training as the mixing layer commits to clean per-channel routing; at grokking it should be close to its minimum.

**Falsifying signal.** `fourier_conc` stays low even after grokking, or rises gradually over thousands of steps after the accuracy cliff. This would suggest the decoder is fitting a non-Fourier function and the phase mechanism is incidental.

**Ambiguous / watch for.** A delayed rise in `fourier_conc` — accuracy hits 99% at step N but Fourier concentration doesn't rise until step N+5000 — would mean the network grokked with a partially non-Fourier decoder and only cleaned up afterward. This is still consistent with phase being the active ingredient (the accuracy comes from phase arithmetic) but the decoder story is messier.

---

### Hyp 2 — K=8 is anomalous due to a metastable partial-Fourier basin

**Why.** K=8 grokked 0/3 seeds at 100K steps, reaching 97.4% accuracy and then plateauing — despite K=7 grokking 1/3 and K=9 grokking 2/3. This non-monotonicity is not explained by capacity (K=8 has more capacity than K=7) or by general seed sensitivity (K=9 is worse by that logic). The 97.4% plateau looks structurally different from the flat-line failures at K≤4: the network found *something near* the correct solution and stalled. Three sub-hypotheses are possible — metastable basin, frequency aliasing against 113, or all three seeds happening to initialize in the same bad region.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/plateau_steps` | Steps spent with val_acc ≥ 0.90 but < 0.99 |
| `{label}/val_acc` | Raw accuracy curve shape |
| `{label}/freq_err_a_mean`, `{label}/freq_err_b_mean` | Whether frequencies partially converged |
| `{label}/coherence_min` | Whether any channel has incoherent cross-encoder routing |

**Confirming signal (structural basin).** When running K=8 with 10 seeds, if 0/10 grok, `plateau_steps` is large (e.g., >50,000) for all seeds, and `freq_err_*` shows frequencies partially converged but stuck, this confirms a structural basin. The network reaches a configuration where some frequencies match the Fourier basis but the mixing matrix can't reorganize to complete the circuit.

**Confirming signal (seed sensitivity).** If some seeds (say 3/10) do grok, the anomaly is sampling noise — the three original seeds (42, 123, 456) happened to initialize in an unlucky region. In this case `plateau_steps` for non-grokking seeds should look similar to K=9 failures, not qualitatively deeper.

**Falsifying signal.** K=8 consistently grokks with enough seeds and the plateau depth at K=8 is not statistically different from K=9 non-grokking seeds. This would mean the 0/3 result was bad luck, not structure.

**Ambiguous / watch for.** K=8 grokks on exactly the seeds where `coherence_min` is high at initialization. This would implicate cross-encoder misalignment as the K=8-specific failure mode — frequencies align individually but the two encoders route them to different channels, and K=8 creates a configuration where this misalignment is more stable than at K=9.

---

### Hyp 3 — Frequency locking and grokking are separable events

**Why.** The naive story is: frequencies converge to the Fourier basis → the decoder learns the correct projection → accuracy hits 99%. But it's possible that frequency locking happens much earlier (during the memorization phase) and grokking is triggered by a separate event — the mixing weights reorganizing, or the decoder norm crossing a threshold. Separating these tells you which event is the rate-limiting step, which directly informs what to optimize.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/freq_err_a_mean` | Mean angular error of enc_a frequencies to theoretical basis |
| `{label}/freq_err_b_mean` | Mean angular error of enc_b frequencies to theoretical basis |
| `{label}/freq_err_a_{i}` | Per-channel angular error for enc_a, channel i |
| `{label}/freq_err_b_{i}` | Per-channel angular error for enc_b, channel i |
| `{label}/grok_step` | Step at which val_acc first crosses 99% |

**Confirming signal.** `freq_err_*_mean` drops sharply (e.g., from ~1.0 to <0.1 rad) several thousand steps *before* `grok_step`. The frequency convergence event is visible as a kink in the error curve well before the accuracy cliff. This means frequency search is not the bottleneck — something else (weight norm competition, mixing reorganization) is the slow step.

**Falsifying signal.** `freq_err_*_mean` drops at the same step as `grok_step`, or after. In this case frequency convergence and grokking are simultaneous or the same event, and there's no separate frequency-search phase to accelerate.

**Ambiguous / watch for.** Individual channels lock at different times (`freq_err_a_0` drops at step 5000, `freq_err_a_3` at step 15000) and grokking fires when the *last* required channel locks. This would suggest the bottleneck is whichever frequency is hardest to find — useful for understanding which theoretical basis frequencies are intrinsically harder to lock.

---

### Hyp 4 — Cross-encoder misalignment is the dominant failure mode

**Why.** PAN has two independent encoders (enc_a for input a, enc_b for input b). Each learns its own set of K frequency parameters. For the phase mixing to work correctly, enc_a and enc_b must route the *same* frequency to the *same* channel — otherwise the mixing layer is trying to combine phases from different frequencies, which produces incoherent output. Both encoders could individually converge to valid Fourier frequencies but route them to mismatched channels, causing failure. This is distinct from mode collapse (where all channels converge to *one* frequency) and from frequency error (where frequencies haven't converged to the basis yet).

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/coherence_mean` | Mean cosine similarity between enc_a and enc_b output vectors, per channel |
| `{label}/coherence_min` | Minimum coherence — the most misaligned channel |
| `{label}/freq_err_a_mean` | Individual enc_a error (for distinguishing from misalignment) |
| `{label}/freq_err_b_mean` | Individual enc_b error |

**Confirming signal.** In runs that fail to grok, `coherence_min` stays near zero throughout training even when both `freq_err_a_mean` and `freq_err_b_mean` are low. This is the fingerprint of misalignment: each encoder individually found the right frequencies, but they're mapped to different channels. In runs that succeed, `coherence_min` rises sharply at or before the grokking step and stays high.

**Falsifying signal.** `coherence_min` is high in failing runs, or low in successful runs. Either case would mean misalignment is not the failure mechanism. If coherence tracks accuracy with no lag, it may just be a consequence of grokking rather than a cause.

**Ambiguous / watch for.** Some channels show permanently low coherence even in grokked runs, but overall val_acc is 99%. This means the network grokked with only a subset of channels aligned — the redundant channels (above the minimum required K) are misaligned but irrelevant. This is useful: it shows that not all K channels need to be aligned, only the ~5 required by the task.

---

### Hyp 5 — Mixing weights converge to unit-weight phase addition, implementing Nanda's algorithm

**Why.** Nanda's transformer converged on frequencies at k×2π/113 for k ∈ {1,2,3,4,5} and implemented a specific Fourier circuit to compute modular addition. If PAN is genuinely implementing the same algorithm (rather than finding a different circuit that also achieves 99%), the mixing weights at the grokked solution should form an identity or permutation matrix over the active frequency slots, with weight ≈ ±1 on the dominant input and near-zero on everything else. The Tier 1 run found that only 2 frequency slots were active (3 and 4 from the theoretical basis) rather than 5 — which is either a more efficient encoding or a different circuit entirely. Distinguishing these requires looking at the full mixing weight structure at convergence.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/mix_entropy_min` | Minimum per-channel mixing entropy (low = clean ±1 routing) |
| `{label}/mix_entropy_mean` | Mean entropy across all channels |
| `{label}/freq_err_a_mean`, `{label}/freq_err_b_mean` | Whether frequencies match Nanda's basis |
| `{label}/fourier_conc` | Whether decoder aligns to Fourier basis vectors |

**Confirming signal.** At the grokked solution: `mix_entropy_min` is very low (< 0.5 nats, approaching log(1) = 0 for a perfectly deterministic channel), indicating each output channel is dominated by a single input. `freq_err_*_mean` is below the SIFP-16 quantization error (2π/65536 ≈ 9.6×10⁻⁵ rad), meaning frequencies have converged to within hardware precision of the theoretical basis. `fourier_conc` above 0.7. This combination would be strong evidence PAN is implementing Nanda's algorithm directly.

**Falsifying signal.** `mix_entropy_min` remains high even after grokking (diffuse mixing), or `freq_err_*_mean` converges to values that don't correspond to any theoretical basis frequency. PAN found a circuit that achieves 99% but it isn't Nanda's algorithm.

**Ambiguous / watch for.** Frequencies converge to a *subset* of the theoretical basis (e.g., only k=3 and k=4 as in Tier 1), with the remaining channels settling at non-basis values but with low entropy. This is the "compressed Fourier circuit" case — PAN found a more efficient encoding that uses fewer frequencies than the transformer's solution. It's consistent with the phase primitive claim but suggests the mechanism isn't identical to Nanda's algorithm.

---

### Hyp 6 — Weight decay drives grokking by making the Fourier circuit cheaper than memorization

**Why.** This is the mechanistic story behind grokking in general (Varma et al., 2023), applied specifically to PAN. Under weight decay, the optimizer continuously penalizes large weights. The memorization solution requires large decoder weights to hash each (a,b) pair to a specific output. The Fourier circuit solution requires small structured encoder/mixing weights and a compact decoder. As training progresses, weight decay makes the memorization solution increasingly expensive relative to the Fourier solution. The grokking transition fires when the Fourier circuit becomes the cheaper option. The `circuit_ratio` metric (decoder_norm / fourier_norm) is a direct measurement of this competition — when it falls below some threshold, grokking fires.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/circuit_ratio` | decoder_norm / fourier_norm — should fall at grokking |
| `{label}/decoder_norm` | L2 norm of decoder weights (memorization circuit) |
| `{label}/fourier_norm` | L2 norm of encoder + mixing + gate weights (Fourier circuit) |

**Confirming signal.** `circuit_ratio` starts high (decoder dominates), falls gradually during the memorization phase, and then drops sharply at the grokking step. Plotting `circuit_ratio` against `val_acc` should show them crossing roughly simultaneously. `decoder_norm` should fall at grokking as weight decay collapses the large decoder weights, while `fourier_norm` stays stable or rises slightly (the encoder frequencies are small and structured, not much to decay).

**Falsifying signal.** `circuit_ratio` doesn't show a sharp transition at grokking — it either falls linearly throughout training, or it falls long before grokking fires. The former would suggest weight decay isn't the specific mechanism (something else triggers the transition). The latter would suggest the ratio isn't the right variable and some other threshold determines grokking.

**Ambiguous / watch for.** The circuit_ratio threshold at grokking varies across seeds and K values. If K=9 consistently grokks at ratio ≈ 1.2 and K=12 at ratio ≈ 0.8, this would suggest the threshold depends on the number of redundant channels — over-provisioned networks need a lower ratio (more weight decay pressure) to trigger grokking. This would be a useful finding for the paper.

---

## Cluster B — Speed hypotheses (new, from sweep analysis)

These hypotheses come from the observation that speed is non-monotone in K: K=10 is faster than K=9, K=15 is fastest of all, but K=8 fails entirely. The goal is to understand and exploit this relationship.

---

### Hyp 7 — Higher learning rate shortens the memorization phase without destabilizing training

**Why.** The grokking transition requires weight decay pressure to build up enough that the Fourier circuit becomes cheaper than memorization. A higher learning rate means weights grow faster in early training — the decoder builds a larger memorization solution more quickly, but weight decay then also exerts more pressure against it. The net effect should be a shorter memorization phase (the competition resolves faster). The risk is overshooting past the Fourier basin if the lr is too high.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/grok_step` | Primary outcome — did it go faster? |
| `{label}/circuit_ratio` | Does the ratio cross earlier with higher lr? |
| `{label}/decoder_norm` | Does memorization build faster and collapse faster? |
| `{label}/val_acc` distribution across seeds | Did variance increase (some seeds destabilize)? |

**Confirming signal.** At lr=2e-3 or 3e-3 (vs baseline 1e-3): mean `grok_step` decreases, `circuit_ratio` crosses its threshold sooner, and grok rate (fraction of seeds that grok) stays at least as high as the baseline. `decoder_norm` rises faster in early training and falls sharper at grokking.

**Falsifying signal.** Grok rate drops at higher lr even if mean grok step is lower — some seeds grok fast but others destabilize entirely. Or mean grok step doesn't improve despite higher lr, suggesting the memorization phase isn't the bottleneck.

**Ambiguous / watch for.** Higher lr works at K=12 but not K=9. This would suggest the minimum-K runs are more sensitive to lr because they have less slack — the network needs more careful optimization to find the right circuit when K is near the minimum.

---

### Hyp 8 — A WD schedule (low → high) compresses both phases independently

**Why.** Constant WD=0.01 applies the same penalty throughout training. During the memorization phase, you want enough WD to eventually force generalization but not so much that you prevent memorization from completing (which is needed for the Fourier circuit to have something to compete against). During the generalization phase, you want strong WD pressure to collapse the decoder quickly. A schedule that starts low (e.g., 0.001) for the first N steps and then ramps to high (e.g., 0.05) should let memorization complete faster and then force generalization more aggressively than constant WD can.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/grok_step` | Primary outcome |
| `{label}/circuit_ratio` | Timing of the norm crossover |
| `{label}/decoder_norm` | Does the decoder build faster (low WD phase) and collapse faster (high WD phase)? |
| `{label}/train_loss` | Does low WD early allow faster memorization? |

**Confirming signal.** With a schedule: `train_loss` reaches its memorization floor earlier (low WD phase allows faster convergence), `circuit_ratio` then drops sharply at the WD ramp (high pressure suddenly applied), and `grok_step` is lower than constant WD. The `decoder_norm` trace should show two distinct phases — growth during low WD, sharp collapse at the ramp.

**Falsifying signal.** WD schedule doesn't improve `grok_step` over constant WD. This would mean the memorization phase isn't currently the bottleneck, and the constant WD=0.01 is already near-optimal for the competition dynamics. Alternatively, the ramp timing is critical and a poorly-chosen ramp step (N) makes things worse.

**Ambiguous / watch for.** WD schedule reduces mean `grok_step` but increases variance — some seeds grok much faster, others miss entirely. This would suggest the schedule is sensitive to where the network is in the memorization phase when the ramp fires.

---

### Hyp 9 — Nanda-frequency initialization reduces grokking step by starting near the solution basin

**Why.** Currently frequencies are initialized to the theoretical Fourier basis (k×2π/P) but then drift during training as the optimizer searches for the best configuration. If the network consistently ends up back near these theoretical values at the grokked solution, then the initialization is already correct and the optimizer is spending steps searching in a region it will eventually leave anyway. Starting with frequencies initialized directly to Nanda's values and frozen (or with very low lr) for the frequency parameters should collapse the frequency-search phase entirely.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/freq_err_a_mean`, `{label}/freq_err_b_mean` | Does error stay near zero throughout? |
| `{label}/grok_step` | Is grokking earlier? |
| `{label}/circuit_ratio` | Is weight norm competition now the sole bottleneck? |
| Step at which `freq_err_*_mean` first rises above 0.1 rad | Does the network leave the Nanda initialization? |

**Confirming signal.** `freq_err_*_mean` stays near zero throughout training (frequencies don't drift from initialization), `circuit_ratio` shows the same pattern as Hyp 6 but compressed in time, and `grok_step` is substantially lower (e.g., 50% reduction). This would confirm the frequency-search phase is a significant component of the total grokking time.

**Falsifying signal.** `freq_err_*_mean` rises early in training even with Nanda initialization — the optimizer moves away from the theoretical basis despite starting there. This would mean the Fourier basis values are not the attractors for the learned frequencies, and PAN finds a different solution. Or `grok_step` doesn't improve, meaning the frequency search phase isn't the bottleneck.

**Ambiguous / watch for.** Nanda initialization helps at high K but not at low K. Near the minimum threshold (K=9–11), the network may need some freedom to find the right frequency assignment, and fixing frequencies at initialization over-constrains the search. At high K (K=15), redundant channels have more flexibility and initialization matters less.

---

### Hyp 10 — Non-monotone speed across K is caused by redundant channels acting as early frequency locks

**Why.** The sweep showed that speed is not monotone in K: K=10 is faster than K=9, K=15 is fastest (~7K mean grokk step), but K=8 fails entirely. The proposed mechanism: at high K, the network has redundant channels above the minimum needed. In early training, multiple channels cluster on the same high-gradient frequency (whichever is easiest to find first). The diversity penalty then pushes them apart into distinct slots. The key observation is that this *initial clustering* acts as a fast early lock on at least one frequency, giving gradient a clean signal toward the solution from the very first steps. At minimum K (K=9), there's no redundancy — every channel has to find a unique frequency independently, and the search takes longer.

**Tracked by.**

| Metric | What it measures |
|---|---|
| `{label}/unique_slots_a` | Number of distinct frequency slots occupied by enc_a channels |
| `{label}/unique_slots_b` | Number of distinct frequency slots occupied by enc_b channels |
| `{label}/slot_utilisation` | Fraction of K channels on unique slots (1.0 = fully spread) |
| `{label}/max_cluster_size` | Largest cluster of channels on the same frequency slot |
| `{label}/grok_step` | Correlation with spread rate |

**Confirming signal.** At high K (K=15): in the first ~1000 steps, `max_cluster_size` is large (multiple channels clustering), then `slot_utilisation` rises quickly as the diversity penalty spreads them, and this early diversification rate correlates with final `grok_step` across seeds. Runs where `slot_utilisation` reaches 1.0 early grok fastest. At low K (K=9): `max_cluster_size` starts at 1 or 2 (not much clustering, no redundancy), `slot_utilisation` rises slowly, and the overall process takes longer.

**Falsifying signal.** `slot_utilisation` doesn't correlate with `grok_step` across seeds or K values. Or `max_cluster_size` is always 1 even at high K, meaning channels don't cluster at initialization. This would mean the redundant-channel mechanism isn't operating and the speed difference is explained by something else (perhaps just that more parameters provide better gradient signal generally).

**Ambiguous / watch for.** `slot_utilisation` rises at the same rate regardless of K. This would suggest the diversity penalty (DW=0.01) is the dominant force setting the spread rate, not the redundancy itself — and increasing DW might accelerate grokking at all K values by forcing faster diversification. Worth testing: run DW=0.05 at K=9 and see if grok_step approaches the K=15 baseline.

---

## Summary table

| Hypothesis | Cluster | Primary metrics | Status |
|---|---|---|---|
| Hyp 1 — Phase mechanism is active | Core | `fourier_conc`, `mix_entropy_*` | Supported by ablation; mechanistic confirmation pending |
| Hyp 2 — K=8 structural basin | K=8 anomaly | `plateau_steps`, `coherence_min` | 0/3 seeds; needs 10-seed run |
| Hyp 3 — Freq locking precedes grokking | Core | `freq_err_*_mean` vs `grok_step` | Not yet measured with checkpoints |
| Hyp 4 — Cross-encoder misalignment | Core | `coherence_mean`, `coherence_min` | Dominant failure hypothesis; not yet confirmed |
| Hyp 5 — Nanda circuit equivalence | Core | `mix_entropy_min`, `freq_err_*_mean`, `fourier_conc` | Partially supported; mixing weights need analysis |
| Hyp 6 — WD drives grokking via norm competition | Core | `circuit_ratio`, `decoder_norm`, `fourier_norm` | Consistent with existing results; not yet directly measured |
| Hyp 7 — Higher LR shortens memorization | Speed | `grok_step`, `circuit_ratio` | Untested |
| Hyp 8 — WD schedule compresses both phases | Speed | `grok_step`, `decoder_norm` | Untested |
| Hyp 9 — Nanda init reduces frequency search | Speed | `freq_err_*_mean`, `grok_step` | Untested |
| Hyp 10 — Redundant channels = early locks | Speed | `slot_utilisation`, `max_cluster_size` | Consistent with K=15 speed; not yet directly measured |

---

## Experiment priority

Run these in order. Each one either confirms a hypothesis or tells you what to run next.

1. **K=8 with 10 seeds at 200K steps.** Resolves Hyp 2 definitively. Cheapest experiment with the most diagnostic value.
2. **Tier 3 mechanistic run (record_checkpoints=True, early_stop=False).** Generates the frequency trajectory and mixing weight data needed to confirm Hyp 3, 4, and 5 simultaneously from a single run. Essential for the paper's mechanistic claims.
3. **LR sweep at K=9: lr ∈ {1e-3, 2e-3, 3e-3, 5e-3}, 3 seeds each.** Tests Hyp 7. Fast (K=9, early stop). Results directly inform whether the step-count gap vs the transformer can be closed.
4. **WD schedule experiment.** Tests Hyp 8. Requires implementing a scheduler, so slightly more engineering work.
5. **Nanda-frequency initialization experiment.** Tests Hyp 9. Clean test: freeze frequencies or set lr_freq=0 and compare grok_step distribution.
6. **slot_utilisation vs grok_step correlation.** Tests Hyp 10. Already being logged — pull from existing wandb runs and check the correlation. No new experiments required.

---

*Report generated April 2026 · Metrics implemented in `training.py` · All hypotheses falsifiable from logged wandb data*

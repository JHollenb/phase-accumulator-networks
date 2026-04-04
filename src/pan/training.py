"""Sweep-friendly training entrypoint for PAN.

Pattern:
- experiment.py defines sweep_configuration and starts wandb.agent(...)
- this file owns wandb.init(...)
- wandb.config becomes the source of truth for the sampled hyperparameters
- model/data are built after config is resolved
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.console import Console

from pan.config import TrainConfig
from pan.data import make_modular_dataset
from pan.models import build
from pan.models.base import ModularModel
from pan.models.pan import PAN

console = Console()

TWO_PI = 2.0 * math.pi


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _maybe_compile(model: nn.Module, use: bool) -> nn.Module:
    if not use or not hasattr(torch, "compile"):
        return model
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
        return torch.compile(model, backend="aot_eager")
    except Exception as e:
        warnings.warn(f"torch.compile failed ({e}); falling back to eager mode.")
        return model


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _fourier_concentration(W: torch.Tensor, top_k: int = 10) -> float:
    """Energy fraction in top-k Fourier modes of decoder weight matrix.

    Proves hyp 1 (phase mechanism) and 5 (Nanda-circuit equivalence):
    rises sharply at grokking as the decoder aligns to sinusoidal basis vectors.
    """
    energy = torch.fft.fft(W.float(), dim=0).abs() ** 2
    total = energy.sum().item()
    if total <= 1e-10:
        return 0.0
    top = energy.reshape(-1).topk(min(top_k, energy.numel())).values.sum().item()
    return top / total


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _param_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return math.sqrt(total)


# --- Metric 2: Per-channel mixing weight entropy ---------------------------

def _mixing_weight_entropy(mix_weight: torch.Tensor) -> tuple[float, float]:
    """Entropy of softmax over each output channel's input weights.

    Low entropy = clean routing (one dominant input per channel).
    High entropy = diffuse / collapsed mixing.

    Proves hyp 1 (ablation: mixing matters) and hyp 4 (cross-encoder
    misalignment): at grokking, clean channels should have low entropy
    with bimodal weight distribution (one weight from enc_a, one from enc_b).

    Returns (mean_entropy, min_entropy) across output channels.
    """
    W = mix_weight.detach().float()          # (K, 2K)
    probs = torch.softmax(W.abs(), dim=-1)   # treat magnitude as unnorm prob
    log_p = torch.log(probs + 1e-10)
    entropy = -(probs * log_p).sum(dim=-1)   # (K,)  per-channel entropy
    return float(entropy.mean().item()), float(entropy.min().item())


# --- Metric 3: Circuit efficiency ratio ------------------------------------

def _circuit_efficiency(model: nn.Module) -> dict[str, float]:
    """Decompose parameter L2 norm into encoder/mixing vs decoder contributions.
 
    The memorisation solution lives in large decoder weights; the Fourier
    circuit solution lives in structured (small-norm) encoder + mixing weights.
    Weight decay makes the Fourier circuit cheaper as training progresses.
 
    The ratio  decoder_norm / encoder_norm  should *fall* at grokking as the
    decoder shrinks relative to the phase weights.
 
    Proves hyp 6 (WD drives grokking) and hyp 7/8 (LR and WD schedule speed).
    The crossing of this ratio is the mechanistic event we're accelerating.
    """
    enc_norm = dec_norm = mix_norm = gate_norm = 0.0
 
    # Encoder frequencies — new API uses enc_a/enc_b, old used encoder_a/encoder_b
    enc_a = getattr(model, "enc_a", None) or getattr(model, "encoder_a", None)
    enc_b = getattr(model, "enc_b", None) or getattr(model, "encoder_b", None)
    if enc_a is not None:
        enc_norm += enc_a.freq.detach().pow(2).sum().item()
    if enc_b is not None:
        enc_norm += enc_b.freq.detach().pow(2).sum().item()
 
    # Mixing layer — new API uses .mix, old used .phase_mix
    mixer = getattr(model, "mix", None) or getattr(model, "phase_mix", None)
    if mixer is not None:
        mix_norm = mixer.weight.detach().pow(2).sum().item()
 
    # Gate — new API uses .gate with .ref, old used .phase_gate with .ref_phase
    gate = getattr(model, "gate", None) or getattr(model, "phase_gate", None)
    if gate is not None:
        ref = getattr(gate, "ref", None)
        if ref is None:
            ref = getattr(gate, "ref_phase", None)
        if ref is not None:
            gate_norm = ref.detach().pow(2).sum().item()
 
    # Decoder — new API uses .dec, old used .decoder
    decoder = getattr(model, "dec", None) or getattr(model, "decoder", None)
    if decoder is not None:
        dec_norm = decoder.weight.detach().pow(2).sum().item()
        if decoder.bias is not None:
            dec_norm += decoder.bias.detach().pow(2).sum().item()
 
    fourier_norm = math.sqrt(enc_norm + mix_norm + gate_norm)
    decoder_norm = math.sqrt(dec_norm)
    ratio = decoder_norm / (fourier_norm + 1e-10)
 
    return {
        "fourier_norm": fourier_norm,
        "decoder_norm": decoder_norm,
        "circuit_ratio": ratio,   # falls at grokking as decoder shrinks
    }


# --- Metric 4: Cross-encoder phase coherence -------------------------------

def _cross_encoder_coherence(
    model: nn.Module,
    val_x: torch.Tensor,
    max_samples: int = 512,
) -> dict[str, float]:
    """Per-channel cosine similarity between enc_a and enc_b phase vectors.

    If enc_a and enc_b routed the same frequency to channel j, the phases
    will be coherent (high similarity on correlated inputs). If they routed
    different frequencies the similarity is near-random (≈0).

    Direct test of hyp 4 (cross-encoder misalignment is the dominant failure):
    failing runs should show near-zero coherence on some channels even when
    individual frequency errors are low.

    Returns mean and min coherence across K channels.
    """
    enc_a = getattr(model, "enc_a", None) or getattr(model, "encoder_a", None)
    enc_b = getattr(model, "enc_b", None) or getattr(model, "encoder_b", None)
    mixer = getattr(model, "mix", None) or getattr(model, "phase_mix", None)
    if not (enc_a and enc_b and mixer):
        return {}

    model.eval()
    with torch.no_grad():
        x = val_x[:max_samples]
        phi_a = enc_a(x[:, 0])   # (N, K)
        phi_b = enc_b(x[:, 1])   # (N, K)

        # Mix each encoder separately through the mixing layer
        zeros = torch.zeros_like(phi_a)
        mixed_a = mixer(torch.cat([phi_a, zeros], dim=-1))   # (N, K)
        mixed_b = mixer(torch.cat([zeros, phi_b], dim=-1))   # (N, K)

        # Cosine similarity per channel: treat the N-dim phase vector as a vector
        # Normalise each column first
        na = mixed_a - mixed_a.mean(0, keepdim=True)
        nb = mixed_b - mixed_b.mean(0, keepdim=True)
        norm_a = na.norm(dim=0).clamp(min=1e-8)
        norm_b = nb.norm(dim=0).clamp(min=1e-8)
        coherence = (na * nb).sum(dim=0) / (norm_a * norm_b)   # (K,)

    return {
        "coherence_mean": float(coherence.mean().item()),
        "coherence_min":  float(coherence.min().item()),
    }


# --- Metric 7: Redundant channel utilisation rate --------------------------

def _redundant_channel_utilisation(
    model: nn.Module,
    p: int,
    slot_tolerance_rad: float = 0.2,
) -> dict[str, Any]:
    """Count distinct frequency slots occupied across K channels.

    Channels within slot_tolerance_rad of each other count as the same slot.
    In early training with high K, multiple channels cluster on the same
    high-gradient frequency before the diversity penalty spreads them apart.
    The rate at which they spread correlates with grokking speed (hyp 10).

    Returns:
      unique_slots_a / unique_slots_b  — distinct slots occupied
      slot_utilisation                 — fraction of K channels on unique slots
      max_cluster_size                 — largest cluster (1 = fully spread)
    """
    enc_a = getattr(model, "enc_a", None) or getattr(model, "encoder_a", None)
    enc_b = getattr(model, "enc_b", None) or getattr(model, "encoder_b", None)
    if not (enc_a and enc_b):
        return {}

    info = model.get_learned_frequencies()
    k = getattr(model, "k", None) or getattr(model, "k_freqs", None)
    if k is None:
        return {}

    def _unique_slots(freqs):
        """Greedy clustering: group frequencies within tolerance."""
        freqs_wrapped = freqs % TWO_PI
        assigned = [-1] * len(freqs_wrapped)
        slot_id = 0
        for i, f in enumerate(freqs_wrapped):
            if assigned[i] >= 0:
                continue
            assigned[i] = slot_id
            for j in range(i + 1, len(freqs_wrapped)):
                diff = abs(freqs_wrapped[j] - f) % TWO_PI
                dist = min(diff, TWO_PI - diff)
                if dist < slot_tolerance_rad and assigned[j] < 0:
                    assigned[j] = slot_id
            slot_id += 1
        n_unique = len(set(assigned))
        counts = [assigned.count(s) for s in set(assigned)]
        return n_unique, max(counts)

    ua, max_a = _unique_slots(info["learned_a"])
    ub, max_b = _unique_slots(info["learned_b"])
    utilisation = (ua + ub) / (2 * k)   # 1.0 = fully spread, <1 = clustering

    return {
        "unique_slots_a":    ua,
        "unique_slots_b":    ub,
        "slot_utilisation":  utilisation,
        "max_cluster_size":  max(max_a, max_b),
    }


# --- Grokking step distribution helpers ------------------------------------

def _plateau_depth(val_acc_history: list[float], threshold: float = 0.90) -> int:
    """Steps spent above threshold but below 0.99 — the near-grok plateau.

    Distinguishes 'never grokked' from 'almost grokked and stalled'.
    Critical for understanding K=8 anomaly (hyp 2): K=8 should show large
    plateau_depth while K=1-4 failures show near-zero.
    """
    return sum(1 for v in val_acc_history if threshold <= v < 0.99)


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "model_dump") and callable(cfg.model_dump):
        return dict(cfg.model_dump())
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        return dict(cfg.to_dict())
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {
            k: v
            for k, v in vars(cfg).items()
            if not k.startswith("_") and not callable(v)
        }
    raise TypeError(f"Cannot convert config of type {type(cfg)!r} to dict")


def _coerce_like(value: Any, like: Any) -> Any:
    if like is None:
        return value
    if isinstance(like, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, str):
        return str(value)
    return value


def _merge_cfg_from_wandb(base_cfg: TrainConfig, wb_cfg: Any) -> TrainConfig:
    """Overlay wandb.config values onto base_cfg when field names match."""
    updates = {}
    for key, value in dict(wb_cfg).items():
        if hasattr(base_cfg, key):
            updates[key] = _coerce_like(value, getattr(base_cfg, key))

    if hasattr(base_cfg, "overlay") and callable(base_cfg.overlay):
        return base_cfg.overlay(**updates)
    if is_dataclass(base_cfg):
        return replace(base_cfg, **updates)
    for key, value in updates.items():
        setattr(base_cfg, key, value)
    return base_cfg


def _build_init_config(
    base_cfg: TrainConfig,
    *,
    project: str,
    group: str,
    label: str,
    arch: str,
    extra_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = _cfg_to_dict(base_cfg)
    cfg.update(
        {
            "project": project,
            "group": group,
            "label": label,
            "arch": arch,
            "watch_model": False,
            "watch_log_freq": max(getattr(base_cfg, "log_every", 100), 100),
        }
    )
    if extra_config:
        cfg.update(extra_config)
    return cfg


_defined_metric_namespaces: set[str] = set()


def reset_metrics() -> None:
    _defined_metric_namespaces.clear()


def define_wandb_metrics(label: str) -> None:
    if label in _defined_metric_namespaces:
        return

    step_key = f"{label}/step"
    wandb.define_metric(step_key)
    wandb.define_metric(f"{label}/*", step_metric=step_key)

    # Core training
    wandb.define_metric(f"{label}/train_loss",        summary="min")
    wandb.define_metric(f"{label}/val_loss",          summary="min")
    wandb.define_metric(f"{label}/val_acc",           summary="max")
    wandb.define_metric(f"{label}/best_val_acc",      summary="max")
    wandb.define_metric(f"{label}/best_val_loss",     summary="min")
    wandb.define_metric(f"{label}/grok_step",         summary="min")
    wandb.define_metric(f"{label}/grad_norm",         summary="last")
    wandb.define_metric(f"{label}/examples_per_sec",  summary="mean")
    wandb.define_metric(f"{label}/step_time_sec",     summary="mean")

    # Metric 1 — frequency angular error (per-channel, proves hyp 3/4/5)
    wandb.define_metric(f"{label}/freq_err_a_mean",   summary="last")
    wandb.define_metric(f"{label}/freq_err_b_mean",   summary="last")
    wandb.define_metric(f"{label}/freq_err_max",      summary="last")

    # Metric 1 + Fourier concentration (proves hyp 1/5)
    wandb.define_metric(f"{label}/fourier_conc",      summary="last")

    # Metric 2 — mixing weight entropy (proves hyp 1/4)
    wandb.define_metric(f"{label}/mix_entropy_mean",  summary="last")
    wandb.define_metric(f"{label}/mix_entropy_min",   summary="last")

    # Metric 3 — circuit efficiency ratio (proves hyp 6/7/8)
    wandb.define_metric(f"{label}/fourier_norm",      summary="last")
    wandb.define_metric(f"{label}/decoder_norm",      summary="last")
    wandb.define_metric(f"{label}/circuit_ratio",     summary="last")

    # Metric 4 — cross-encoder coherence (proves hyp 4)
    wandb.define_metric(f"{label}/coherence_mean",    summary="last")
    wandb.define_metric(f"{label}/coherence_min",     summary="last")

    # Metric 5 — grokking distribution (proves hyp 2/K=8 anomaly)
    wandb.define_metric(f"{label}/plateau_steps",     summary="max")

    # Metric 7 — redundant channel utilisation (proves hyp 10)
    wandb.define_metric(f"{label}/unique_slots_a",    summary="last")
    wandb.define_metric(f"{label}/unique_slots_b",    summary="last")
    wandb.define_metric(f"{label}/slot_utilisation",  summary="last")
    wandb.define_metric(f"{label}/max_cluster_size",  summary="last")

    _defined_metric_namespaces.add(label)


# -----------------------------------------------------------------------------
# Core loop (no wandb.init here)
# -----------------------------------------------------------------------------

def train_loop(
    model: ModularModel,
    cfg: TrainConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    label: str = "pan",
) -> Optional[int]:
    """
    Train model and log to the active wandb run.
    Returns grok_step (or None).
    """
    define_wandb_metrics(label)

    if cfg.log_console:
        console.print(f"\n  [bold cyan]┌─ {label} ─────────────────────────────────────[/]")
        if hasattr(cfg, "to_str") and callable(cfg.to_str):
            console.print(cfg.to_str())
        else:
            console.print(str(_cfg_to_dict(cfg)))
        console.print(f"  [bold cyan]└──────────────────────────────────────────────[/]")

    if cfg.dry_run:
        console.print(f"  [yellow]dry-run[/] {label} — {cfg.n_steps:,} steps skipped")
        wandb.log({f"{label}/step": 0, f"{label}/dry_run": 1})
        return None

    torch.manual_seed(cfg.seed)

    compiled = _maybe_compile(model, getattr(cfg, "use_compile", False))
    raw = _unwrap(compiled)
    is_pan = isinstance(raw, PAN)

    if is_pan:
        wd=cfg.weight_decay
    else:
        console.print(f"  [yellow]dry-run[/] {label} — NON-PAN Arch, setting WD=1.0")
        wd=1.0

    opt = torch.optim.AdamW(
        compiled.parameters(),
        lr=cfg.lr,
        weight_decay=wd,
    )

    if bool(wandb.config.get("watch_model", False)):
        wandb.watch(raw, log="all", log_freq=int(wandb.config.get("watch_log_freq", cfg.log_every)))

    ex, ey = val_x, val_y
    if getattr(cfg, "val_samples", None) and cfg.val_samples < len(val_x):
        idx = torch.randperm(len(val_x), device=cfg.device)[: cfg.val_samples]
        ex, ey = val_x[idx], val_y[idx]

    n_train = len(train_x)
    grok_step: Optional[int] = None
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    val_acc_history: list[float] = []    # for plateau_depth (metric 5)
    t0 = time.time()
    console_every = cfg.log_every * 5

    last_vl = None
    last_va = None

    for step in range(cfg.n_steps):
        step_t0 = time.time()

        compiled.train()
        idx = torch.randperm(n_train, device=cfg.device)[: cfg.batch_size]
        batch_x, batch_y = train_x[idx], train_y[idx]

        logits = compiled(batch_x)
        loss = F.cross_entropy(logits, batch_y)

        if cfg.diversity_weight > 0:
            loss = loss + cfg.diversity_weight * raw.auxiliary_loss(batch_x, logits)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _grad_norm(raw)
        opt.step()

        step_time = time.time() - step_t0
        examples_per_sec = cfg.batch_size / max(step_time, 1e-9)

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            compiled.eval()
            with torch.no_grad():
                val_logits = compiled(ex)
                vl = F.cross_entropy(val_logits, ey).item()
                va = (val_logits.argmax(-1) == ey).float().mean().item()

            last_vl = vl
            last_va = va
            best_val_acc = max(best_val_acc, va)
            best_val_loss = min(best_val_loss, vl)
            val_acc_history.append(va)

            if va > 0.99 and grok_step is None:
                grok_step = step
                if cfg.log_console:
                    console.print(
                        f"  [bold green]★ {label} GROKKED[/] step={step:,} "
                        f"acc={va:.3f} ({time.time() - t0:.0f}s)"
                    )

            # ── Core training metrics ────────────────────────────────────
            metrics: dict[str, Any] = {
                f"{label}/step":             step,
                f"{label}/train_loss":       float(loss.item()),
                f"{label}/val_loss":         float(vl),
                f"{label}/val_acc":          float(va),
                f"{label}/best_val_acc":     float(best_val_acc),
                f"{label}/best_val_loss":    float(best_val_loss),
                f"{label}/grok":             float(va > 0.99),
                f"{label}/grok_step":        grok_step if grok_step is not None else -1,
                f"{label}/lr":               float(opt.param_groups[0]["lr"]),
                f"{label}/grad_norm":        float(grad_norm),
                f"{label}/step_time_sec":    float(step_time),
                f"{label}/examples_per_sec": float(examples_per_sec),
                f"{label}/elapsed_sec":      float(time.time() - t0),
                # Metric 5 — plateau depth (proves K=8 anomaly / hyp 2)
                f"{label}/plateau_steps":    _plateau_depth(val_acc_history),
            }

            if is_pan:
                # ── Metric 1: Per-channel frequency angular error ────────
                # Proves hyp 3 (freq locking precedes grokking),
                # hyp 4 (cross-encoder misalignment), hyp 5 (Nanda circuit).
                freq_info = raw.get_learned_frequencies()
                err_a = freq_info["error_a"]   # np.ndarray (K,)
                err_b = freq_info["error_b"]
                metrics[f"{label}/freq_err_a_mean"] = float(err_a.mean())
                metrics[f"{label}/freq_err_b_mean"] = float(err_b.mean())
                metrics[f"{label}/freq_err_max"]    = float(max(err_a.max(), err_b.max()))

                # Per-channel errors for detailed sweep analysis
                k_count = getattr(raw, "k", None) or getattr(raw, "k_freqs", 0)
                for i in range(k_count):
                    metrics[f"{label}/freq_err_a_{i}"] = float(err_a[i])
                    metrics[f"{label}/freq_err_b_{i}"] = float(err_b[i])

                # ── Metric 1 + Fourier concentration ────────────────────
                # Proves hyp 1 (phase mechanism active) and hyp 5.
                dec = raw.dec if hasattr(raw, "dec") else raw.decoder
                metrics[f"{label}/fourier_conc"] = float(
                    _fourier_concentration(dec.weight.detach())
                )

                # ── Metric 2: Mixing weight entropy ─────────────────────
                # Proves hyp 1 and hyp 4 (cross-encoder routing clarity).
                # Low entropy = clean per-channel routing = Nanda circuit.
                mixer = getattr(raw, "mix", None) or getattr(raw, "phase_mix", None)
                if mixer is not None:
                    ent_mean, ent_min = _mixing_weight_entropy(mixer.weight)
                    metrics[f"{label}/mix_entropy_mean"] = ent_mean
                    metrics[f"{label}/mix_entropy_min"]  = ent_min

                # ── Metric 3: Circuit efficiency ratio ──────────────────
                # Proves hyp 6 (WD drives grokking): circuit_ratio falls
                # at grokking as decoder shrinks relative to phase weights.
                # Also the key signal for hyp 7/8 (LR / WD schedule speed).
                circuit = _circuit_efficiency(raw)
                metrics[f"{label}/fourier_norm"]  = circuit["fourier_norm"]
                metrics[f"{label}/decoder_norm"]  = circuit["decoder_norm"]
                metrics[f"{label}/circuit_ratio"] = circuit["circuit_ratio"]

                # ── Metric 4: Cross-encoder phase coherence ──────────────
                # Direct test of hyp 4: failing runs show near-zero coherence
                # on some channels even when individual freq errors are low.
                coherence = _cross_encoder_coherence(raw, val_x)
                if coherence:
                    metrics[f"{label}/coherence_mean"] = coherence["coherence_mean"]
                    metrics[f"{label}/coherence_min"]  = coherence["coherence_min"]

                # ── Metric 7: Redundant channel utilisation ──────────────
                # Proves hyp 10 (redundant channels spread early = faster grok).
                p_val = getattr(raw, "p", cfg.p)
                slot_info = _redundant_channel_utilisation(raw, p_val)
                if slot_info:
                    metrics[f"{label}/unique_slots_a"]   = slot_info["unique_slots_a"]
                    metrics[f"{label}/unique_slots_b"]   = slot_info["unique_slots_b"]
                    metrics[f"{label}/slot_utilisation"] = slot_info["slot_utilisation"]
                    metrics[f"{label}/max_cluster_size"] = slot_info["max_cluster_size"]

            wandb.log(metrics)

            if cfg.log_console and step % console_every == 0:
                elapsed = time.time() - t0
                console.print(
                    f"  [{label}] step={step:>6,} | "
                    f"train_loss={loss.item():.4f} | "
                    f"val_loss={vl:.4f} | "
                    f"val_acc={va:.4f} | "
                    f"grad_norm={grad_norm:.3f} | "
                    f"{examples_per_sec:.1f} ex/s | "
                    f"{elapsed:.0f}s"
                )

            if grok_step is not None and getattr(cfg, "early_stop", False):
                break

    elapsed = time.time() - t0
    wandb.summary[f"{label}/elapsed_sec"]     = float(elapsed)
    wandb.summary[f"{label}/best_val_acc"]    = float(best_val_acc) if best_val_acc != float("-inf") else None
    wandb.summary[f"{label}/best_val_loss"]   = float(best_val_loss) if best_val_loss != float("inf") else None
    wandb.summary[f"{label}/final_val_acc"]   = float(last_va) if last_va is not None else None
    wandb.summary[f"{label}/final_val_loss"]  = float(last_vl) if last_vl is not None else None
    wandb.summary[f"{label}/grok_step"]       = grok_step
    wandb.summary[f"{label}/plateau_steps"]   = _plateau_depth(val_acc_history)

    if is_pan:
        wandb.summary["mode_collapsed"] = raw.is_mode_collapsed()

        # Final-state summary values for each mechanistic metric
        freq_info = raw.get_learned_frequencies()
        wandb.summary[f"{label}/final_freq_err_a_mean"] = float(freq_info["error_a"].mean())
        wandb.summary[f"{label}/final_freq_err_b_mean"] = float(freq_info["error_b"].mean())

        circuit = _circuit_efficiency(raw)
        wandb.summary[f"{label}/final_circuit_ratio"] = circuit["circuit_ratio"]
        wandb.summary[f"{label}/final_decoder_norm"]  = circuit["decoder_norm"]
        wandb.summary[f"{label}/final_fourier_norm"]  = circuit["fourier_norm"]

        coherence = _cross_encoder_coherence(raw, val_x)
        if coherence:
            wandb.summary[f"{label}/final_coherence_min"] = coherence["coherence_min"]

    if cfg.log_console:
        status = f"grokked @ {grok_step:,}" if grok_step is not None else "did not grok"
        console.print(f"  [dim]{label} done[/] — {status} — {elapsed:.0f}s total")

    return grok_step


# -----------------------------------------------------------------------------
# Sweep-friendly run entrypoint
# -----------------------------------------------------------------------------

def merge_wandb_overrides(base_cfg: TrainConfig, wb_cfg: dict[str, Any]) -> TrainConfig:
    allowed = set(base_cfg.model_fields.keys())
    updates = {k: v for k, v in wb_cfg.items() if k in allowed}
    return base_cfg.overlay(**updates)

def run_training(
    train_cfg: TrainConfig,
    run_cfg: RunConfig,
    *,
    extra_wandb_config: Optional[dict[str, Any]] = None,
) -> Optional[int]:
    """
    Start a W&B run, merge sampled hyperparameters from wandb.config into
    TrainConfig, then build model/data and execute training.

    Design:
    - TrainConfig is the source of truth for model/training/runtime fields.
    - RunConfig contains logging metadata like project/group/label/arch.
    - wandb.config may override TrainConfig fields and selected metadata fields.
    - If label_from_arch=True is present in wandb config, label defaults to arch.
    """
    init_config: dict[str, Any] = {
        **train_cfg.wandb_payload(),
        "arch": run_cfg.arch,
        "label": run_cfg.label,
        "label_from_arch": False,
        "watch_model": False,
        "watch_log_freq": max(train_cfg.log_every, 100),
    }
    if extra_wandb_config:
        init_config.update(extra_wandb_config)

    with wandb.init(
        project=run_cfg.project,
        group=run_cfg.group,
        config=init_config,
    ):
        wb_cfg = dict(wandb.config)

        resolved_cfg = merge_wandb_overrides(train_cfg, wb_cfg)

        resolved_arch = str(wb_cfg.get("arch") or run_cfg.arch)

        label_from_arch = bool(wb_cfg.get("label_from_arch", False))
        explicit_label = wb_cfg.get("label")

        if label_from_arch:
            resolved_label = resolved_arch
        else:
            resolved_label = str(explicit_label or resolved_arch or run_cfg.label)

        if resolved_cfg.log_console:
            console.print(
                f"[dim]run[/] "
                f"project={run_cfg.project} "
                f"group={run_cfg.group} "
                f"arch={resolved_arch} "
                f"label={resolved_label}"
            )

        model = build(resolved_arch, resolved_cfg)
        tx, ty, vx, vy = make_modular_dataset(
            resolved_cfg.p,
            seed=resolved_cfg.seed,
            device=resolved_cfg.device
        )

        result = train_loop(
            model,
            resolved_cfg,
            tx,
            ty,
            vx,
            vy,
            label=resolved_label,
        )

        wandb.summary["arch"] = resolved_arch
        wandb.summary["label"] = resolved_label
        wandb.summary["p"] = resolved_cfg.p
        wandb.summary["seed"] = resolved_cfg.seed

        return result

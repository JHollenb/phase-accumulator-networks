from pan.models.pan import *
from pan.models.baseline import *
from pan.core import *

def run_main_comparison(cfg: TrainConfig):
    """Tier 1: Head-to-head PAN vs Transformer on mod-P addition."""
    print(f"\n{'='*60}")
    print(f" Phase Accumulator Network — Main Comparison")
    print(f" Task: a + b mod {cfg.p}  |  Device: {DEVICE}")
    print(f" compile={cfg.use_compile}  div_weight={cfg.diversity_weight}  val_samples={cfg.val_samples}  "
          f"early_stop={cfg.early_stop}")
    print(f"{'='*60}")

    train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)
    print(f" Dataset: {len(train_x)} train, {len(val_x)} val pairs")

    pan = PhaseAccumulatorNetwork(cfg.p, cfg.k_freqs).to(DEVICE)
    tf  = TransformerBaseline(cfg.p, cfg.d_model, cfg.n_heads, cfg.d_mlp).to(DEVICE)

    print(f"\n PAN parameters:         {pan.count_parameters():>10,}")
    print(f" Transformer parameters: {tf.count_parameters():>10,}")
    print(f" Ratio:                  {tf.count_parameters()/pan.count_parameters():>10.1f}×")

    print(f"\n{'─'*60}\n Training PAN (K={cfg.k_freqs})...\n{'─'*60}")
    hist_pan = train(pan, cfg, train_x, train_y, val_x, val_y, label="PAN")

    print(f"\n{'─'*60}\n Training Transformer (d={cfg.d_model})...\n{'─'*60}")
    hist_tf = train(tf, cfg, train_x, train_y, val_x, val_y, label="TF")

    print(f"\n{'='*60}\n RESULTS\n{'='*60}")
    print(f" PAN grokking step:         {hist_pan.grok_step or 'did not grok':>15}")
    print(f" Transformer grokking step: {hist_tf.grok_step  or 'did not grok':>15}")
    if hist_pan.grok_step and hist_tf.grok_step:
        ratio   = hist_tf.grok_step / hist_pan.grok_step
        verdict = f"PAN {ratio:.1f}× faster" if ratio > 1 else f"PAN {1/ratio:.1f}× slower"
        print(f" Speed comparison:          {verdict:>15}")
    print(f" Parameter ratio:           "
          f"{tf.count_parameters()/pan.count_parameters():>14.1f}×")

    if not cfg.dry_run:
        analyze_pan(pan)
        ablation_test(pan, val_x, val_y, label="PAN")

    if cfg.save_model and not cfg.dry_run:
        save_model_weights(pan, cfg, 'PAN', hist_pan.grok_step, cfg.output_dir)
        save_model_weights(tf,  cfg, 'TF',  hist_tf.grok_step,  cfg.output_dir)

    if not cfg.dry_run:
        import os
        os.makedirs(cfg.output_dir, exist_ok=True)
        plot_comparison(hist_pan, hist_tf,
                        pan.count_parameters(), tf.count_parameters(), cfg.p,
                        save_path=os.path.join(cfg.output_dir, 'pan_vs_transformer.png'))
        plot_frequency_convergence(pan,
                        save_path=os.path.join(cfg.output_dir, 'pan_frequencies.png'))

    return pan, tf, hist_pan, hist_tf



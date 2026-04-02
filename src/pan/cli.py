import argparse

from pan.utils import *
from pan.core import say_hello

def main():
    parser = argparse.ArgumentParser(
        description='Phase Accumulator Network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode',
                        choices=['compare', 'sweep', 'primes', 'tier3',
                                 'dw_sweep', 'wd_sweep', 'k8_sweep',
                                 'tf_sweep', 'held_out_primes'],
                        default='compare')
    parser.add_argument('--p',     type=int,   default=113,
                        help='Prime for modular arithmetic')
    parser.add_argument('--k',     type=int,   default=5,
                        help='Phase frequencies (PAN only)')
    parser.add_argument('--steps', type=int,   default=50_000)
    parser.add_argument('--seed',  type=int,   default=42)
    parser.add_argument('--weight-decay', type=float, default=1.0,
                        help='AdamW weight decay. Nanda uses 1.0 for the '
                             'transformer. Try 0.1 for PAN — it has 300x '
                             'fewer params and 1.0 may suppress grokking.')
    parser.add_argument('--diversity-weight', type=float, default=0.01,
                        help='Off-diagonal Gram penalty to prevent PAN mode '
                             'collapse (all K gates converging to one frequency). '
                             '0.01 is a light nudge; 0.1 is strong. 0 disables.')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to write plot PNGs into. '
                             'Defaults to cwd. The bash script passes the '
                             'run dir here directly, avoiding any mv race.')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model state_dict to output_dir '
                             'as pan_<label>_<step>.pt after training. '
                             'Always enabled for tier3 mode.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print every sub-run config and total run count, '
                             'then exit without training. Use to verify overnight '
                             'sweep parameters before committing.')

    # ── Performance flags ────────────────────────────────────────────────
    perf = parser.add_argument_group('performance (macOS / MPS)')
    perf.add_argument('--no-compile', action='store_true',
                      help='Disable torch.compile (fallback for older PyTorch)')
    perf.add_argument('--val-samples', type=int, default=None,
                      metavar='N',
                      help='Subsample val set to N examples for faster evals. '
                           '1024 recommended for sweep mode. '
                           'None = full val set (default, use for final runs).')
    perf.add_argument('--log-every', type=int, default=200,
                      help='Log + eval every N steps. '
                           '500 saves ~50s per 50K run.')
    perf.add_argument('--no-early-stop', action='store_true',
                      help='Continue training after grokking '
                           '(use for full loss curve plots)')

    args = parser.parse_args()

    # TODO - Train

    # TODO - Improve header
    print(f"\n Phase Accumulator Network")
    print(f" {'─'*44}")
    print(f"  Device:       {DEVICE}")
    print(f"  Mode:         {args.mode}")
    # print(f"  P={cfg.p}  K={cfg.k_freqs}  steps={cfg.n_steps:,}  seed={cfg.seed}")
    # print(f"  weight_decay:     {cfg.weight_decay}")
    # print(f"  diversity_weight: {cfg.diversity_weight}")
    # compile_note = use_compile
    # if args.mode in ('sweep', 'primes', 'dw_sweep', 'wd_sweep', 'k8_sweep', 'tf_sweep'):
    #     compile_note = f"{cfg.use_compile} (outer); sub-runs use False"
    # elif args.mode == 'tier3':
    #     compile_note = f"{cfg.use_compile} (outer); tier3 sub-run uses False"
    # print(f"  compile:          {compile_note}")
    # if args.mode == 'tier3':
    #     print(f"  [tier3]  compile forced False; early_stop forced False; record_checkpoints=True; save_model=True")
    # print(f"  val_samples:  {cfg.val_samples or 'full'}")
    # print(f"  log_every:    {cfg.log_every}")
    # print(f"  early_stop:   {cfg.early_stop}")

    # if args.baseline_only:
    #     train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)
    #     tf = TransformerBaseline(cfg.p).to(DEVICE)
    #     print(f"\n Transformer parameters: {tf.count_parameters():,}")
    #     hist = train(tf, cfg, train_x, train_y, val_x, val_y, label="TF")
    #     print(f"\n Grokking step: {hist.grok_step or 'did not grok'}")
    #     return
    #
    # Dry-run: print summary counts before dispatching so user knows what's coming
    # if cfg.dry_run:
    #     run_counts = {
    #         'compare':         ('PAN + TF', 2, cfg.n_steps),
    #     }
    #     if args.mode in run_counts:
    #         desc, n_runs, steps = run_counts[args.mode]
    #         print(f"\n  [dry-run] mode={args.mode}")
    #         print(f"  [dry-run] {desc}  →  {n_runs} total runs  ×  {steps:,} steps each")
    #         print(f"  [dry-run] estimated wall time: "
    #               f"~{n_runs * steps / 50_000 * 1.2:.0f} min on MPS")
    #         print(f"  [dry-run] training will be skipped — each run prints config and exits\n")
    #
    # if args.mode == 'compare':
    #     run_main_comparison(cfg)
    print("running say_hello..")
    print(say_hello(DEVICE))

if __name__ == '__main__':
    main()

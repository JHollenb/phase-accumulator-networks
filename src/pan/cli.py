import argparse

from pan.experiments import EXPERIMENTS
from pan.training import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase Accumulator Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mode", choices=["compare", "sweep_k"], default="compare")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    perf = parser.add_argument_group("performance")
    perf.add_argument("--no-compile", action="store_true")
    perf.add_argument("--val-samples", type=int, default=None)
    perf.add_argument("--log-every", type=int, default=200)
    perf.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    perf.add_argument("--no-early-stop", dest="early_stop", action="store_false")

    return parser


def args_to_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        p=args.p,
        n_steps=args.steps,
        log_every=args.log_every,
        seed=args.seed,
        k_freqs=args.k,
        weight_decay=args.weight_decay,
        diversity_weight=args.diversity_weight,
        val_samples=args.val_samples,
        use_compile=not args.no_compile,
        early_stop=args.early_stop,
        output_dir=args.output_dir,
        save_model=args.save_model,
        dry_run=args.dry_run,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)

    runner = EXPERIMENTS[args.mode]
    result = runner(cfg)

    print("\nDone.")
    print(f"Experiment: {result.name}")
    print(f"Run dir:    {result.run_dir}")
    for k, v in result.artifacts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

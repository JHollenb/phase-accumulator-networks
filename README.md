# Phase Accumulator Networks (PAN)

Phase Accumulator Networks (PAN) are a neural architecture for modular arithmetic tasks where **phase addition** is the primary inductive bias, rather than relying only on generic MLP/attention circuits to discover Fourier structure during training.

This repository contains:
- A PAN model implementation.
- A Transformer baseline for side-by-side comparison.
- Training utilities with Weights & Biases logging.
- A CLI for running the core PAN-vs-Transformer experiment.

## What this project is

PAN is designed to study grokking-style behavior on modular addition and related algorithmic tasks.

Given inputs \((a, b)\), the training target is modular addition:
\[
(a + b) \mod p
\]

The PAN architecture explicitly parameterizes phase-like components and tracks metrics during training that help inspect whether and when Fourier-like circuits emerge.

## Installation

### Prerequisites
- Python 3.11+
- (Recommended) `uv` for environment management

### Setup

```bash
uv sync
```

If you want development tools (tests/lint):

```bash
uv sync --extra dev
```

If you use Weights & Biases logging, authenticate once:

```bash
wandb login
```

## Quick start: run the CLI comparison

The installed CLI entrypoint is `pan`.

```bash
uv run pan
```

Useful flags:

```bash
uv run pan --p 113 --k 5 --steps 50000 --seed 42
uv run pan --no-compile
uv run pan --dry-run
```

What `pan` does:
1. Builds a PAN model and a Transformer baseline.
2. Creates the modular-addition train/validation dataset.
3. Runs both models through the training loop.
4. Logs metrics to W&B (project: `pan`).

## Using PAN as a library

You can use the package directly in scripts and notebooks.

### Minimal example (single model)

```python
import wandb
from pan import DEVICE, PAN, TrainConfig, make_modular_dataset, train_loop

cfg = TrainConfig(p=67, k_freqs=5, n_steps=20_000)

train_x, train_y, val_x, val_y = make_modular_dataset(cfg.p, seed=cfg.seed)
model = PAN(cfg.p, k=cfg.k_freqs).to(DEVICE)

wandb.init(project="pan", name="pan-library-example", config=cfg.model_dump())
train_loop(model, cfg, train_x, train_y, val_x, val_y, label="PAN")
wandb.finish()
```

### Sweep/automation-style entrypoint

For configurable experiment runners, use `run_training` with `RunConfig`:

```python
from pan import TrainConfig, RunConfig
from pan.training import run_training

train_cfg = TrainConfig(p=113, k_freqs=9, n_steps=100_000)
run_cfg = RunConfig(project="pan", group="my-experiments", label="PAN", arch="pan")

run_training(train_cfg, run_cfg)
```

## Creating new experiments

A good pattern is to create small scripts under `scripts/` that compose:
- `TrainConfig` (hyperparameters)
- `RunConfig` (W&B metadata/grouping)
- `run_training(...)` (execution)

### Example: new experiment script

Create `scripts/exp_pan_k_sweep.py`:

```python
from pan import RunConfig, TrainConfig
from pan.training import run_training


def main() -> None:
    for k in [3, 5, 7, 9]:
        cfg = TrainConfig(p=113, k_freqs=k, n_steps=50_000)
        run = RunConfig(project="pan", group="k-sweep", label=f"PAN-k{k}", arch="pan")
        run_training(cfg, run, extra_wandb_config={"k_freqs": k})


if __name__ == "__main__":
    main()
```

Run it with:

```bash
uv run python scripts/exp_pan_k_sweep.py
```

### Tips for experiment design

- Keep one script per experiment family (`k_sweep`, `wd_sweep`, `ablation_*`, etc.).
- Use W&B `group` to cluster related runs.
- Keep labels concise; include key knobs in `label`.
- Prefer immutable config updates via `TrainConfig(...).overlay(...)` when deriving variants.

## Project structure

```text
src/pan/
├── __init__.py
├── cli.py                # `pan`
├── config.py             # TrainConfig + RunConfig
├── constants.py          # DEVICE and shared constants
├── data.py               # modular dataset generation
├── training.py           # train_loop + run_training
└── models/
    ├── base.py
    ├── factory.py        # build("pan" | "transformer", cfg)
    ├── pan.py
    └── transformer.py

tests/
└── test_core.py
```

## Development

Run tests:

```bash
uv run pytest
```

Optional linting (if installed):

```bash
uv run ruff check .
```

## License

MIT (see `LICENSE`).

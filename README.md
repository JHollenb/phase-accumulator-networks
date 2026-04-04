# Phase Accumulator Network (PAN) v2

A neural architecture where sinusoidal phase addition replaces multiply-accumulate.

> Transformers painfully discover Fourier circuits through training.  
> PAN makes them native. SIFP makes the hardware O(1).

## Quick Start

```bash
uv sync
wandb login          # one-time setup

uv run experiment.py exp1-k8-anomaly
uv run experiment.py exp2-tier3
uv run experiment.py exp3-lr-sweep
uv run experiment.py exp4-wd-schedule
uv run experiment.py exp5-nanda-init
uv run experiment.py exp6-slot-util-phase2
```

Every run logs to wandb automatically. View live training curves, frequency
trajectories, Fourier concentration, and sweep comparisons at wandb.ai.

## Using as a Library

```python
import wandb
from pan import PAN, TrainConfig, make_modular_dataset, train, DEVICE

wandb.init(project="pan", name="quick-test")
cfg = TrainConfig(p=67, k_freqs=5, n_steps=20_000)
tx, ty, vx, vy = make_modular_dataset(cfg.p)
model = PAN(cfg.p, k=cfg.k_freqs).to(DEVICE)
grok_step = train(model, cfg, tx, ty, vx, vy, label="test")
wandb.finish()
```

## Adding a New Experiment

Review experiments.py and follow the format.

Or skip the CLI entirely and use the library API in a notebook.

## Project Structure

```
experiments.py         # Experiments to perform
src/pan/
├── __init__.py        # Public API: PAN, Transformer, train, grid_search
├── cli.py             # Typer commands (one per experiment)
├── config.py          # TrainConfig (Pydantic)
├── constants.py       # DEVICE, TWO_PI, SIFP_QUANT_ERROR
├── data.py            # make_modular_dataset
├── training.py        # train() — logs to wandb, returns grok_step
└── models/
    ├── base.py        # ModularModel (count_parameters, auxiliary_loss)
    ├── pan.py         # PAN, PhaseEncoder, PhaseMixer, PhaseGate
    ├── transformer.py # Nanda-architecture baseline
    └── factory.py     # build("pan", cfg) / build("transformer", cfg)
```

## Design Principles

- **wandb is the history layer.** No `TrainHistory` dataclass, no manual JSON
  serialisation, no matplotlib. Every metric, checkpoint, and sweep result
  lives in wandb. Paper figures come from wandb exports or Plotly post-hoc.
- **Models own their own logic.** `auxiliary_loss()` lives on `PAN`, not in the
  training loop. `is_mode_collapsed()` lives on `PAN`, not in a sweep helper.
- **One sweep primitive.** `grid_search(cfg, vary={...}, seeds=[...])` replaces
  five separate experiment files. Each CLI command is ~10 lines that call it.
- **Config is immutable.** `cfg.overlay(seed=99)` returns a new config.
  Serialises to JSON/wandb with `.model_dump()`.


# Phase Accumulator Network (PAN) v2

A neural architecture where sinusoidal phase addition replaces multiply-accumulate.

> Transformers painfully discover Fourier circuits through training.  
> PAN makes them native. SIFP makes the hardware O(1).

## Quick Start

```bash
uv sync
wandb login          # one-time setup

# Head-to-head comparison
uv run pan compare --p 113 --k 5

# Parameter sweep
uv run pan sweep --p 113 --steps 50000

# Cross-prime generalisation
uv run pan primes --k 9

# Mechanistic analysis with frequency checkpoints
uv run pan tier3 --p 113 --k 9 --steps 100000

# Overnight sweeps
uv run pan dw-sweep --steps 100000
uv run pan wd-sweep --steps 100000
uv run pan k8 --steps 200000
uv run pan tf-sweep --steps 100000

# Held-out primes
uv run pan held-out --steps 200000
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

## Grid Search API

```python
from pan import TrainConfig, grid_search

cfg = TrainConfig(p=113, n_steps=50_000)
results = grid_search(
    cfg,
    vary={"k_freqs": [3, 5, 7, 9, 11]},
    seeds=[42, 123, 456],
)
# results[5] = {"n_grokked": 3, "mean_step": 1200.0, ...}
```

## Adding a New Experiment

Add a command to `cli.py`:

```python
@app.command()
def my_experiment(p: int = 113, steps: int = 50_000, ...):
    """One-line description."""
    cfg = _cfg(p=p, steps=steps, ...)
    wandb.init(project="pan", name="my-experiment", config=cfg.model_dump())
    results = grid_search(cfg, vary={"lr": [1e-4, 1e-3, 1e-2]})
    print_results(results, "lr", "Learning Rate Sweep")
    wandb.finish()
```

Or skip the CLI entirely and use the library API in a notebook.

## Project Structure

```
src/pan/
├── __init__.py        # Public API: PAN, Transformer, train, grid_search
├── cli.py             # Typer commands (one per experiment)
├── config.py          # TrainConfig (Pydantic)
├── constants.py       # DEVICE, TWO_PI, SIFP_QUANT_ERROR
├── data.py            # make_modular_dataset
├── training.py        # train() — logs to wandb, returns grok_step
├── sweep.py           # grid_search() + print_results()
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

## Tests

```bash
uv run pytest
```

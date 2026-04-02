"""
Experiment registry — the extension point for new experiments.

To add a new experiment, create a file in pan/experiments/ and decorate a
function with @experiment:

    from pan.experiments import experiment

    @experiment("my_sweep", help="My custom sweep over X")
    def run(cfg: TrainConfig) -> dict:
        ...
        return results

It will automatically appear as `pan run my_sweep` in the CLI.
"""

from __future__ import annotations

from typing import Callable

from pan.config import TrainConfig

# name → (function, help_text)
_REGISTRY: dict[str, tuple[Callable[[TrainConfig], dict], str]] = {}


def experiment(name: str, *, help: str = ""):
    """Register a function as a runnable experiment."""
    def decorator(fn: Callable[[TrainConfig], dict]):
        _REGISTRY[name] = (fn, help)
        return fn
    return decorator


def get(name: str) -> Callable[[TrainConfig], dict]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}")
    return _REGISTRY[name][0]


def list_experiments() -> dict[str, str]:
    return {name: help_text for name, (_, help_text) in sorted(_REGISTRY.items())}

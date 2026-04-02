"""Modular arithmetic dataset generation."""

import numpy as np
import torch

from pan.constants import DEVICE


def make_modular_dataset(
    p: int, train_frac: float = 0.4, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    All (a, b, (a+b) mod p) triples, shuffled and split.

    Returns (train_x, train_y, val_x, val_y) on DEVICE.
    """
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b, (a + b) % p) for a in range(p) for b in range(p)], dtype=np.int64)
    pairs = pairs[rng.permutation(len(pairs))]

    n = int(train_frac * len(pairs))
    train = torch.tensor(pairs[:n], device=DEVICE)
    val = torch.tensor(pairs[n:], device=DEVICE)
    return train[:, :2], train[:, 2], val[:, :2], val[:, 2]

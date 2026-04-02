import numpy as np
import torch

from pan.constants import DEVICE


def make_modular_dataset(p: int, train_frac: float = 0.4, seed: int = 42):
    """
    Generate all (a, b, (a+b) mod p) triples and split into train/val.
    """
    rng = np.random.default_rng(seed)
    pairs = np.array(
        [(a, b, (a + b) % p) for a in range(p) for b in range(p)],
        dtype=np.int64,
    )
    perm = rng.permutation(len(pairs))
    pairs = pairs[perm]

    n_train = int(train_frac * len(pairs))
    train = torch.tensor(pairs[:n_train], device=DEVICE)
    val = torch.tensor(pairs[n_train:], device=DEVICE)

    return train[:, :2], train[:, 2], val[:, :2], val[:, 2]

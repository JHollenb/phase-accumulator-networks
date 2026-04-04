"""Phase Accumulator Network — sinusoidal phase arithmetic for modular addition."""

from pan.models import PAN, Transformer, build
from pan.config import TrainConfig, RunConfig
from pan.data import make_modular_dataset
from pan.training import train_loop, run_training
#from pan.sweep import grid_search
from pan.constants import DEVICE


__all__ = ["PAN", "Transformer", "build", "TrainConfig", "RunConfig", "make_modular_dataset",
           "train_loop", "run_training", "DEVICE"]

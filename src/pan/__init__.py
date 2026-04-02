"""Phase Accumulator Network — sinusoidal phase arithmetic for modular addition."""

from pan.models import PAN, Transformer
from pan.config import TrainConfig, TrainHistory
from pan.data import make_modular_dataset
from pan.training import train
from pan.constants import DEVICE, TWO_PI, PHASE_SCALE

__all__ = [
    "PAN", "Transformer", "TrainConfig", "TrainHistory",
    "make_modular_dataset", "train", "DEVICE",
]

import math
import torch

PHASE_SCALE = 65536
PHASE_SCALE_F = 65536.0
TWO_PI = 2.0 * math.pi

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

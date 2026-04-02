"""Global constants and device auto-detection."""

import math
import torch

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

TWO_PI = 2.0 * math.pi
PHASE_SCALE = 65536
SIFP_QUANT_ERROR = TWO_PI / PHASE_SCALE

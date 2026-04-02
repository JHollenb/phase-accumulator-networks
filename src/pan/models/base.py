"""Base class for PAN and Transformer models."""

import torch
import torch.nn as nn


class ModularModel(nn.Module):
    """Shared interface for models that solve modular arithmetic."""

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def auxiliary_loss(self, inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Extra regularisation loss. Override in subclasses. Default: 0."""
        return torch.tensor(0.0, device=logits.device)

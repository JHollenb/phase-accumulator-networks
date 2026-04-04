"""One-layer transformer baseline — Nanda et al. architecture."""

import torch
import torch.nn as nn

from pan.models.base import ModularModel


class Transformer(ModularModel):

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p, self.d_model = p, d_model
        self.tok_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d_model))
        self.unembed = nn.Linear(d_model, p, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B = inputs.shape[0]
        seq = torch.cat([inputs, torch.full((B, 1), self.p, dtype=torch.long, device=inputs.device)], 1)
        x = self.tok_embed(seq) + self.pos_embed(torch.arange(3, device=inputs.device))
        mask = torch.triu(torch.ones(3, 3, device=inputs.device), diagonal=1).bool()
        x = x + self.attn(x, x, x, attn_mask=mask)[0]
        return self.unembed((x + self.mlp(x))[:, -1])

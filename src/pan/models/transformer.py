"""One-layer transformer baseline — Nanda et al. architecture."""

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    One-layer causal transformer for modular arithmetic.

    Matches Nanda's setup: token + positional embeddings, single-layer
    multi-head self-attention with causal mask, one-hidden-layer MLP,
    linear unembed from the '=' position.

    ~227 K params for P=113, d=128.
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p, self.d_model = p, d_model

        self.tok_embed = nn.Embedding(p + 1, d_model)  # +1 for '=' token
        self.pos_embed = nn.Embedding(3, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d_model)
        )
        self.unembed = nn.Linear(d_model, p, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B = inputs.shape[0]
        eq = torch.full((B, 1), self.p, dtype=torch.long, device=inputs.device)
        seq = torch.cat([inputs, eq], dim=1)
        pos = torch.arange(3, device=inputs.device).unsqueeze(0)
        x = self.tok_embed(seq) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(3, 3, device=inputs.device), diagonal=1).bool()
        x = x + self.attn(x, x, x, attn_mask=mask)[0]
        x = x + self.mlp(x)
        return self.unembed(x[:, -1, :])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

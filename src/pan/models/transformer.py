import torch.nn as nn

class TransformerBaseline(nn.Module):
    """
    One-layer transformer for modular arithmetic — Nanda et al. architecture.
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int = 512):
        super().__init__()
        self.p      = p
        self.d_model = d_model

        self.tok_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp       = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )
        self.unembed = nn.Linear(d_model, p, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch     = inputs.shape[0]
        eq_token  = torch.full((batch, 1), self.p,
                                dtype=torch.long, device=inputs.device)
        seq       = torch.cat([inputs, eq_token], dim=1)
        positions = torch.arange(3, device=inputs.device).unsqueeze(0)
        x         = self.tok_embed(seq) + self.pos_embed(positions)
        mask      = torch.triu(
            torch.ones(3, 3, device=inputs.device), diagonal=1).bool()
        x_attn, _ = self.attn(x, x, x, attn_mask=mask)
        x         = x + x_attn
        x         = x + self.mlp(x)
        return self.unembed(x[:, -1, :])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



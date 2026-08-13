import torch.nn as nn
from .sparse_attention import SparseMultiheadSelfAttention


class SparseTransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8, ff_dim=512, dropout=0.2, local_radius=64, global_stride=32):
        super().__init__()
        self.attn = SparseMultiheadSelfAttention(dim, heads, dropout, local_radius, global_stride)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, variant_mask=None, return_attention=False):
        a, weights = self.attn(x, variant_mask=variant_mask, return_attention=return_attention)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.ff(x))
        return x, weights


class SparseTransformerEncoder(nn.Module):
    def __init__(self, layers=4, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([SparseTransformerBlock(**kwargs) for _ in range(layers)])

    def forward(self, x, variant_mask=None, return_attention=False):
        attentions = []
        for layer in self.layers:
            x, a = layer(x, variant_mask=variant_mask, return_attention=return_attention)
            if return_attention:
                attentions.append(a)
        return x, attentions

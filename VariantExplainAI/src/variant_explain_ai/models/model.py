import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import SparseTransformerEncoder


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VariantExplainAI(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, cnn_filters=(128,256), cnn_kernels=(5,3),
                 transformer_layers=4, attention_heads=8, ff_dim=512, local_radius=64, global_stride=32,
                 dropout=0.2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.type_embedding = nn.Embedding(4, 16)
        self.meta_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16))
        self.mask_embedding = nn.Embedding(2, 8)
        fusion_in = embedding_dim * 3 + 16 + 16 + 8
        self.fusion = nn.Sequential(nn.Linear(fusion_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.posenc = SinusoidalPositionalEncoding(hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_filters[0], kernel_size=cnn_kernels[0], padding=cnn_kernels[0]//2),
            nn.BatchNorm1d(cnn_filters[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=cnn_kernels[1], padding=cnn_kernels[1]//2),
            nn.BatchNorm1d(cnn_filters[1]), nn.ReLU(), nn.Dropout(dropout),
        )
        if cnn_filters[1] != hidden_dim:
            self.cnn_proj = nn.Linear(cnn_filters[1], hidden_dim)
        else:
            self.cnn_proj = nn.Identity()
        self.transformer = SparseTransformerEncoder(
            layers=transformer_layers, dim=hidden_dim, heads=attention_heads, ff_dim=ff_dim,
            dropout=dropout, local_radius=local_radius, global_stride=global_stride
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def fuse_inputs(self, ref_ids, alt_ids, variant_mask, metadata):
        er = self.embedding(ref_ids)
        ea = self.embedding(alt_ids)
        diff = ea - er
        vtype = metadata[:, 0].long().clamp(0, 3)
        lengths = metadata[:, 1:4]
        t = self.type_embedding(vtype)
        m = self.meta_mlp(lengths)
        meta = torch.cat([t, m], dim=-1)[:, None, :].expand(-1, er.size(1), -1)
        vm = self.mask_embedding(variant_mask.long())
        h = self.fusion(torch.cat([er, ea, diff, meta, vm], dim=-1))
        return self.posenc(h)

    def forward_from_fused(self, fused, variant_mask, return_attention=False):
        B = fused.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, fused], dim=1)
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        x = self.cnn_proj(x)
        z, attn = self.transformer(x, variant_mask=variant_mask, return_attention=return_attention)
        logits = self.classifier(z[:, 0])
        return {"logits": logits, "hidden": z, "attentions": attn}

    def forward(self, ref_ids, alt_ids, variant_mask, metadata, return_attention=False, return_fused=False):
        fused = self.fuse_inputs(ref_ids, alt_ids, variant_mask, metadata)
        out = self.forward_from_fused(fused, variant_mask, return_attention=return_attention)
        if return_fused:
            out["fused"] = fused
        return out

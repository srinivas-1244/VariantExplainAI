import torch
import torch.nn as nn
from .model import VariantExplainAI


class CNNOnly(VariantExplainAI):
    def forward_from_fused(self, fused, variant_mask, return_attention=False):
        B = fused.size(0); cls = self.cls.expand(B,-1,-1)
        x = torch.cat([cls, fused], dim=1)
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        x = self.cnn_proj(x)
        pooled = x.mean(dim=1)
        return {"logits": self.classifier(pooled), "hidden": x, "attentions": []}


class SparseTransformerOnly(VariantExplainAI):
    def forward_from_fused(self, fused, variant_mask, return_attention=False):
        B = fused.size(0); cls = self.cls.expand(B,-1,-1)
        x = torch.cat([cls, fused], dim=1)
        z, attn = self.transformer(x, variant_mask=variant_mask, return_attention=return_attention)
        return {"logits": self.classifier(z[:,0]), "hidden": z, "attentions": attn}

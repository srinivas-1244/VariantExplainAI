import torch
import torch.nn.functional as F


def normalized_importance(attributions, eps=1e-8):
    # [B,T,D] -> [B,T]
    s = attributions.abs().sum(dim=-1)
    return s / (s.sum(dim=-1, keepdim=True) + eps)


def expanded_target_mask(variant_mask, radius=10):
    B, T = variant_mask.shape
    out = torch.zeros_like(variant_mask, dtype=torch.float32)
    for b in range(B):
        pos = torch.where(variant_mask[b])[0]
        for p in pos.tolist():
            lo=max(0,p-radius); hi=min(T,p+radius+1)
            out[b,lo:hi] = 1.0
        if out[b].sum() == 0:
            out[b, T//2] = 1.0
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def cosine_alignment_loss(attributions, target_distribution):
    imp = normalized_importance(attributions)
    cos = F.cosine_similarity(imp, target_distribution, dim=-1)
    return (1.0 - cos).mean()


def differentiable_integrated_gradients(model, fused, variant_mask, target_labels, n_steps=8):
    """Differentiable IG on fused allele-aware representations for training-time alignment."""
    baseline = torch.zeros_like(fused)
    total_grad = torch.zeros_like(fused)
    alphas = torch.linspace(1.0/n_steps, 1.0, n_steps, device=fused.device, dtype=fused.dtype)
    for alpha in alphas:
        x = baseline + alpha * (fused - baseline)
        x.requires_grad_(True)
        out = model.forward_from_fused(x, variant_mask, return_attention=False)
        selected = out["logits"].gather(1, target_labels[:,None]).sum()
        grad = torch.autograd.grad(selected, x, create_graph=True, retain_graph=True)[0]
        total_grad = total_grad + grad
    avg_grad = total_grad / float(n_steps)
    return (fused - baseline) * avg_grad

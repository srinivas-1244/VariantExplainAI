import torch


def integrated_gradients_fused(model, fused, variant_mask, target_labels, n_steps=50):
    model.eval()
    baseline = torch.zeros_like(fused)
    total = torch.zeros_like(fused)
    for alpha in torch.linspace(1.0/n_steps, 1.0, n_steps, device=fused.device):
        x = (baseline + alpha * (fused-baseline)).detach().requires_grad_(True)
        logits = model.forward_from_fused(x, variant_mask)["logits"]
        sel = logits.gather(1, target_labels[:,None]).sum()
        grad = torch.autograd.grad(sel, x)[0]
        total += grad.detach()
    return (fused-baseline) * (total / n_steps)

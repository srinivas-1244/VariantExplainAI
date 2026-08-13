import torch
import torch.nn.functional as F


def class_weights_from_labels(labels, device=None):
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(y, minlength=2).float().clamp_min(1)
    inv = counts.sum() / counts
    w = inv / inv.mean()
    return w.to(device) if device is not None else w


def weighted_cross_entropy(logits, labels, weights=None):
    return F.cross_entropy(logits, labels, weight=weights)


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p.gather(1, labels[:,None]).squeeze(1)
    logpt = logp.gather(1, labels[:,None]).squeeze(1)
    at = torch.where(labels == 1, torch.tensor(alpha, device=logits.device), torch.tensor(1-alpha, device=logits.device))
    return (-at * (1-pt).pow(gamma) * logpt).mean()

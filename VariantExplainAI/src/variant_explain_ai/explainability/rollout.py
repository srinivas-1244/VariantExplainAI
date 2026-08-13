import torch


def attention_rollout(attentions):
    """attentions: list of [B,H,T,T]. Returns CLS-to-token rollout [B,T]."""
    if not attentions:
        return None
    result = None
    for a in attentions:
        a = a.mean(dim=1)
        I = torch.eye(a.size(-1), device=a.device, dtype=a.dtype).unsqueeze(0)
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = a if result is None else torch.bmm(a, result)
    return result[:,0,:]

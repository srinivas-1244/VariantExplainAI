import torch


def mask_top_attribution(ref_ids, alt_ids, scores, fraction=0.10, pad_id=0):
    k = max(1, int(scores.size(1) * fraction))
    idx = torch.topk(scores, k=k, dim=1).indices
    r = ref_ids.clone(); a = alt_ids.clone()
    for b in range(r.size(0)):
        r[b, idx[b]] = pad_id; a[b, idx[b]] = pad_id
    return r, a


def mask_variant_center(ref_ids, alt_ids, variant_mask, pad_id=0):
    r=ref_ids.clone(); a=alt_ids.clone()
    r[variant_mask] = pad_id; a[variant_mask] = pad_id
    return r,a


def allele_swap(ref_ids, alt_ids):
    return alt_ids.clone(), ref_ids.clone()

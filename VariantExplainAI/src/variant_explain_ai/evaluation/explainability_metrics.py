import numpy as np


def biological_annotation_overlap(scores, annotation_mask, top_fraction=0.10):
    """Fraction of top-attribution token positions overlapping an independent binary annotation mask."""
    s=np.asarray(scores); m=np.asarray(annotation_mask).astype(bool); k=max(1,int(len(s)*top_fraction)); idx=np.argpartition(s,-k)[-k:]
    return float(m[idx].mean())


def topk_functional_hit(scores, annotation_mask, k=10):
    s=np.asarray(scores); m=np.asarray(annotation_mask).astype(bool); k=min(k,len(s)); idx=np.argpartition(s,-k)[-k:]
    return float(m[idx].any())


def attribution_stability(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.std()==0 or b.std()==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

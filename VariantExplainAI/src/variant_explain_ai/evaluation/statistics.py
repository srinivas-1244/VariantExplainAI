import numpy as np
from scipy.stats import wilcoxon


def bootstrap_mean_ci(values, confidence=0.95, n_boot=2000, seed=42):
    x=np.asarray(values,float); rng=np.random.default_rng(seed)
    means=np.asarray([rng.choice(x,size=len(x),replace=True).mean() for _ in range(n_boot)])
    a=(1-confidence)/2
    return float(x.mean()), float(np.quantile(means,a)), float(np.quantile(means,1-a))


def holm_adjust(pvalues):
    p=np.asarray(pvalues,float); m=len(p); order=np.argsort(p); adj=np.empty(m)
    running=0.0
    for rank, idx in enumerate(order):
        val=(m-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj


def paired_wilcoxon(a,b):
    return wilcoxon(np.asarray(a),np.asarray(b),zero_method="wilcox",alternative="two-sided").pvalue

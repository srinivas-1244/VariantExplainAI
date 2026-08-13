from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def save_roc_pr(y, p, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fpr,tpr,_=roc_curve(y,p)
    fig=plt.figure(); plt.plot(fpr,tpr); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.tight_layout(); fig.savefig(out/"roc_curve.png",dpi=300); plt.close(fig)
    prec,rec,_=precision_recall_curve(y,p)
    fig=plt.figure(); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve"); plt.tight_layout(); fig.savefig(out/"pr_curve.png",dpi=300); plt.close(fig)

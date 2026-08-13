import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


def binary_metrics(y_true, p_pathogenic, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_pathogenic, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / max(1, tn+fp),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auroc": roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": average_precision_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

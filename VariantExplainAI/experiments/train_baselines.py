#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from variant_explain_ai.data.kmer import KmerTokenizer
from variant_explain_ai.evaluation.metrics import binary_metrics

p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--test',required=True); p.add_argument('--k',type=int,default=5); p.add_argument('--out',default='outputs/baselines'); a=p.parse_args()
tr=pd.read_csv(a.train); te=pd.read_csv(a.test); tok=KmerTokenizer(a.k); V=tok.vocab_size

def feat(seq):
    ids=tok.encode(seq); x=np.bincount(ids,minlength=V).astype(np.float32); return x/max(1,x.sum())
def make(df):
    # REF, ALT, and explicit frequency difference preserve allele specificity.
    xr=np.stack([feat(s) for s in df.ref_seq]); xa=np.stack([feat(s) for s in df.alt_seq]); return np.concatenate([xr,xa,xa-xr],axis=1)
Xtr,Xte=make(tr),make(te); ytr=tr.label.values; yte=te.label.values
models={'RandomForest':RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=-1),'SVM_RBF':SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=42)}
out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=[]
for name,m in models.items():
    m.fit(Xtr,ytr); pth=m.predict_proba(Xte)[:,1]; met=binary_metrics(yte,pth); rows.append({'model':name,**met}); pd.DataFrame({'variant_id':te.variant_id,'true_label':yte,'p_pathogenic':pth}).to_csv(out/f'{name}_predictions.csv',index=False)
pd.DataFrame(rows).to_csv(out/'baseline_metrics.csv',index=False); print(pd.DataFrame(rows))

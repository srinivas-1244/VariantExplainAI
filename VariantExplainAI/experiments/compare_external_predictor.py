#!/usr/bin/env python
"""Fair common-subset comparison with any external pathogenicity score file.
Expected external CSV columns: variant_id, score. Higher score must mean more pathogenic.
"""
import argparse
from pathlib import Path
import pandas as pd
from variant_explain_ai.evaluation.metrics import binary_metrics
p=argparse.ArgumentParser(); p.add_argument('--variant-predictions',required=True); p.add_argument('--external',required=True); p.add_argument('--name',required=True); p.add_argument('--score-column',default='score'); p.add_argument('--threshold',type=float,default=0.5); p.add_argument('--out',default='outputs/external_comparison.csv'); a=p.parse_args()
v=pd.read_csv(a.variant_predictions); e=pd.read_csv(a.external); x=v.merge(e[['variant_id',a.score_column]],on='variant_id',how='inner')
if len(x)==0: raise SystemExit('No common eligible variants.')
m1=binary_metrics(x.true_label,x.p_pathogenic,a.threshold); m2=binary_metrics(x.true_label,x[a.score_column],a.threshold)
res=pd.DataFrame([{'model':'VariantExplainAI','n_common':len(x),**m1},{'model':a.name,'n_common':len(x),**m2}]); Path(a.out).parent.mkdir(parents=True,exist_ok=True); res.to_csv(a.out,index=False); print(res)

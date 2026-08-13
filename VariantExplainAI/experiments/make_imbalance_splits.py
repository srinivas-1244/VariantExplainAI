#!/usr/bin/env python
import argparse,pandas as pd
from pathlib import Path
from variant_explain_ai.data.splits import make_imbalance_training_set
p=argparse.ArgumentParser(); p.add_argument('--train',default='data/splits/variant_train.csv'); p.add_argument('--out',default='data/splits/imbalance'); p.add_argument('--ratios',nargs='+',default=['1:1','1:2','1:4']); a=p.parse_args(); df=pd.read_csv(a.train); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for r in a.ratios: make_imbalance_training_set(df,r).to_csv(out/f'train_{r.replace(":","_")}.csv',index=False)
print('Saved:',a.ratios)

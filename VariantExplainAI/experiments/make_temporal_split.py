#!/usr/bin/env python
import argparse,pandas as pd
from variant_explain_ai.data.splits import temporal_holdout_split,save_splits
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--cutoff',required=True); p.add_argument('--date-column',default='first_seen_date'); p.add_argument('--out',default='data/splits'); a=p.parse_args(); df=pd.read_csv(a.data); save_splits(temporal_holdout_split(df,a.cutoff,a.date_column),a.out,'temporal'); print('Temporal splits saved.')

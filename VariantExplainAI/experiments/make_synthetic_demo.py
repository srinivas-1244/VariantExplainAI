#!/usr/bin/env python
import argparse, random
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--out',default='data/demo'); p.add_argument('--n',type=int,default=240); p.add_argument('--length',type=int,default=101); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
rng=random.Random(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(a.n):
    y=i%2
    seq=[rng.choice('ACGT') for _ in range(a.length)]
    center=a.length//2
    ref=seq[center]
    # Synthetic signal: pathogenic ALT and nearby motif differ systematically.
    alt='G' if y==1 and ref!='G' else ('A' if ref!='A' else 'C')
    altseq=seq.copy(); altseq[center]=alt
    if y==1:
        motif='GGGTT'; lo=center-2; altseq[lo:lo+5]=list(motif)
    mask=['0']*a.length; mask[center]='1'
    rows.append({'variant_id':f'demo:{i}:{ref}:{alt}','chrom':'chr1','pos':i+1,'ref':ref,'alt':alt,'label':y,'label4':'Pathogenic' if y else 'Benign','stars':2,'gene':f'GENE{i%20}','ref_seq':''.join(seq),'alt_seq':''.join(altseq),'variant_mask':''.join(mask),'variant_type':'SNV','ref_len':1,'alt_len':1,'delta_len':0,'region':'coding' if i%3 else 'noncoding'})
df=pd.DataFrame(rows).sample(frac=1,random_state=a.seed).reset_index(drop=True)
tr=df.iloc[:int(.7*a.n)]; va=df.iloc[int(.7*a.n):int(.85*a.n)]; te=df.iloc[int(.85*a.n):]
tr.to_csv(out/'train.csv',index=False); va.to_csv(out/'val.csv',index=False); te.to_csv(out/'test.csv',index=False)
print(out)

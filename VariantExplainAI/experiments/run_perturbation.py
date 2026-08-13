#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from variant_explain_ai.utils.config import load_config
from variant_explain_ai.data.dataset import VariantDataset
from variant_explain_ai.models.model import VariantExplainAI
from variant_explain_ai.training.trainer import get_device,move_batch
from variant_explain_ai.explainability.ig import integrated_gradients_fused
from variant_explain_ai.explainability.perturbation import mask_top_attribution,mask_variant_center,allele_swap
from variant_explain_ai.evaluation.statistics import bootstrap_mean_ci,paired_wilcoxon,holm_adjust
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/config.yaml'); p.add_argument('--data',default='data/splits/variant_test.csv'); p.add_argument('--checkpoint',default='outputs/main/best.pt'); p.add_argument('--out',default='outputs/main/perturbation.csv'); p.add_argument('--limit',type=int,default=200); a=p.parse_args()
c=load_config(a.config); ds=VariantDataset(a.data,c['model']['kmer_size']); m=c['model']; model=VariantExplainAI(ds.tokenizer.vocab_size,m['embedding_dim'],m['hidden_dim'],tuple(m['cnn_filters']),tuple(m['cnn_kernels']),m['transformer_layers'],m['attention_heads'],m['ff_dim'],m['local_radius'],m['global_stride'],m['dropout'],m['num_classes']); dev=get_device(c['training'].get('device','auto')); model.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)['model_state']); model.to(dev).eval(); rows=[]
for batch in DataLoader(ds,batch_size=1,shuffle=False):
    b=move_batch(batch,dev)
    with torch.no_grad():
        base=model(b['ref_ids'],b['alt_ids'],b['variant_mask'],b['metadata']); target=base['logits'].argmax(1); p0=torch.softmax(base['logits'],1).gather(1,target[:,None]).item()
    fused=model.fuse_inputs(b['ref_ids'],b['alt_ids'],b['variant_mask'],b['metadata']); ig=integrated_gradients_fused(model,fused,b['variant_mask'],target,c['explainability']['ig_steps_eval']); scores=ig.abs().sum(-1)
    variants={}; variants['top_attr']=mask_top_attribution(b['ref_ids'],b['alt_ids'],scores,c['explainability']['topk_fraction']); variants['variant_center']=mask_variant_center(b['ref_ids'],b['alt_ids'],b['variant_mask']); variants['allele_swap']=allele_swap(b['ref_ids'],b['alt_ids'])
    rec={'variant_id':batch['variant_id'][0]}
    for name,(rr,aa) in variants.items():
        with torch.no_grad(): pp=torch.softmax(model(rr,aa,b['variant_mask'],b['metadata'])['logits'],1).gather(1,target[:,None]).item()
        rec[name+'_drop']=p0-pp
    rows.append(rec)
    if len(rows)>=a.limit: break
Path(a.out).parent.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(a.out,index=False); print(df.mean(numeric_only=True))

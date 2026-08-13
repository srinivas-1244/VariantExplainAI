#!/usr/bin/env python
"""Run principal architecture ablations using common data partitions.
Ablations supported here are architecture-level models; representation-level ablations are exposed by input mutation below.
"""
import argparse
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
from variant_explain_ai.utils.config import load_config
from variant_explain_ai.utils.seed import seed_everything
from variant_explain_ai.data.dataset import VariantDataset
from variant_explain_ai.models.model import VariantExplainAI
from variant_explain_ai.models.baselines import CNNOnly,SparseTransformerOnly
from variant_explain_ai.training.trainer import train,predict,get_device
from variant_explain_ai.evaluation.metrics import binary_metrics

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/config.yaml'); p.add_argument('--train',default='data/splits/variant_train.csv'); p.add_argument('--val',default='data/splits/variant_val.csv'); p.add_argument('--test',default='data/splits/variant_test.csv'); p.add_argument('--ablation',choices=['full','cnn_only','sparse_transformer_only','no_alignment','no_alt_change'],default='no_alignment'); p.add_argument('--out',default='outputs/ablation'); a=p.parse_args()
c=load_config(a.config); seed_everything(c['seed']); k=c['model']['kmer_size']; trdf=pd.read_csv(a.train); vadf=pd.read_csv(a.val); tedf=pd.read_csv(a.test)
if a.ablation=='no_alt_change':
    for d in (trdf,vadf,tedf): d['alt_seq']=d['ref_seq']
tr,va,te=VariantDataset(trdf,k),VariantDataset(vadf,k),VariantDataset(tedf,k); m=c['model']; klass={'cnn_only':CNNOnly,'sparse_transformer_only':SparseTransformerOnly}.get(a.ablation,VariantExplainAI)
model=klass(tr.tokenizer.vocab_size,m['embedding_dim'],m['hidden_dim'],tuple(m['cnn_filters']),tuple(m['cnn_kernels']),m['transformer_layers'],m['attention_heads'],m['ff_dim'],m['local_radius'],m['global_stride'],m['dropout'],m['num_classes'])
if a.ablation=='no_alignment': c['training']['alignment_weight']=0.0
od=Path(a.out)/a.ablation; model,_=train(model,tr,va,c,od); dev=get_device(c['training'].get('device','auto')); pred=predict(model,DataLoader(te,batch_size=c['training']['batch_size']),dev); met=binary_metrics(pred.true_label,pred.p_pathogenic,c['training']['threshold']); pd.DataFrame([{'ablation':a.ablation,**met}]).to_csv(od/'metrics.csv',index=False); print(met)

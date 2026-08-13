#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from variant_explain_ai.utils.config import load_config
from variant_explain_ai.utils.seed import seed_everything
from variant_explain_ai.data.dataset import VariantDataset
from variant_explain_ai.models.model import VariantExplainAI
from variant_explain_ai.training.trainer import train,predict,get_device
from variant_explain_ai.evaluation.metrics import binary_metrics
from variant_explain_ai.evaluation.plots import save_roc_pr

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--train",default="data/splits/variant_train.csv"); p.add_argument("--val",default="data/splits/variant_val.csv"); p.add_argument("--test",default="data/splits/variant_test.csv"); p.add_argument("--out",default="outputs/main"); a=p.parse_args()
c=load_config(a.config); seed_everything(c["seed"]); k=c["model"]["kmer_size"]
tr,va,te=VariantDataset(a.train,k),VariantDataset(a.val,k),VariantDataset(a.test,k)
m=c["model"]; model=VariantExplainAI(vocab_size=tr.tokenizer.vocab_size,embedding_dim=m["embedding_dim"],hidden_dim=m["hidden_dim"],cnn_filters=tuple(m["cnn_filters"]),cnn_kernels=tuple(m["cnn_kernels"]),transformer_layers=m["transformer_layers"],attention_heads=m["attention_heads"],ff_dim=m["ff_dim"],local_radius=m["local_radius"],global_stride=m["global_stride"],dropout=m["dropout"],num_classes=m["num_classes"])
model,_=train(model,tr,va,c,a.out); device=get_device(c["training"].get("device","auto")); loader=DataLoader(te,batch_size=c["training"]["batch_size"],shuffle=False)
pred=predict(model,loader,device); pred["predicted_label"]=(pred.p_pathogenic>=c["training"]["threshold"]).astype(int); pred["p_benign"]=1-pred.p_pathogenic
Path(a.out).mkdir(parents=True,exist_ok=True); pred.to_csv(Path(a.out)/"test_predictions.csv",index=False); metrics=binary_metrics(pred.true_label,pred.p_pathogenic,c["training"]["threshold"]); pd.DataFrame([metrics]).to_csv(Path(a.out)/"test_metrics.csv",index=False); save_roc_pr(pred.true_label,pred.p_pathogenic,Path(a.out)/"figures"); print(metrics)

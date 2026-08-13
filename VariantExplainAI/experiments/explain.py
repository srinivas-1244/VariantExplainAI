#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from variant_explain_ai.utils.config import load_config
from variant_explain_ai.data.dataset import VariantDataset
from variant_explain_ai.models.model import VariantExplainAI
from variant_explain_ai.training.trainer import get_device,move_batch
from variant_explain_ai.explainability.ig import integrated_gradients_fused
from variant_explain_ai.explainability.rollout import attention_rollout

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--data",default="data/splits/variant_test.csv"); p.add_argument("--checkpoint",default="outputs/main/best.pt"); p.add_argument("--out",default="outputs/main/explanations.csv"); p.add_argument("--limit",type=int,default=100); a=p.parse_args()
c=load_config(a.config); ds=VariantDataset(a.data,c["model"]["kmer_size"]); m=c["model"]; model=VariantExplainAI(ds.tokenizer.vocab_size,m["embedding_dim"],m["hidden_dim"],tuple(m["cnn_filters"]),tuple(m["cnn_kernels"]),m["transformer_layers"],m["attention_heads"],m["ff_dim"],m["local_radius"],m["global_stride"],m["dropout"],m["num_classes"])
device=get_device(c["training"].get("device","auto")); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck["model_state"]); model.to(device).eval(); rows=[]
for batch in DataLoader(ds,batch_size=1,shuffle=False):
    b=move_batch(batch,device); fused=model.fuse_inputs(b["ref_ids"],b["alt_ids"],b["variant_mask"],b["metadata"]); out=model.forward_from_fused(fused,b["variant_mask"],return_attention=True); target=out["logits"].argmax(1); ig=integrated_gradients_fused(model,fused,b["variant_mask"],target,c["explainability"]["ig_steps_eval"]); igs=ig.abs().sum(-1); roll=attention_rollout(out["attentions"])
    rows.append({"variant_id":batch["variant_id"][0],"predicted_class":int(target.item()),"ig_top_token":int(igs.argmax(1).item()),"rollout_top_token":int(roll[:,1:].argmax(1).item()) if roll is not None else -1});
    if len(rows)>=a.limit: break
Path(a.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False); print(f"Saved {len(rows)} explanations")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .losses import weighted_cross_entropy, class_weights_from_labels
from ..explainability.alignment import differentiable_integrated_gradients, expanded_target_mask, cosine_alignment_loss
from ..evaluation.metrics import binary_metrics


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); rows=[]
    for batch in loader:
        ids=batch["variant_id"]
        b=move_batch(batch,device)
        out=model(b["ref_ids"],b["alt_ids"],b["variant_mask"],b["metadata"])
        probs=torch.softmax(out["logits"],dim=1)[:,1].cpu().numpy()
        labels=b["label"].cpu().numpy()
        for vid,y,p in zip(ids,labels,probs): rows.append((vid,int(y),float(p)))
    return pd.DataFrame(rows,columns=["variant_id","true_label","p_pathogenic"])


def train(model, train_ds, val_ds, cfg, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    tc=cfg["training"]; ec=cfg.get("explainability",{})
    device=get_device(tc.get("device","auto")); model=model.to(device)
    train_loader=DataLoader(train_ds,batch_size=tc["batch_size"],shuffle=True,num_workers=tc.get("num_workers",0))
    val_loader=DataLoader(val_ds,batch_size=tc["batch_size"],shuffle=False,num_workers=tc.get("num_workers",0))
    weights=class_weights_from_labels(train_ds.df["label"].values,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=tc["learning_rate"],weight_decay=tc["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=tc["epochs"],eta_min=tc["min_learning_rate"])
    best=-np.inf; bad=0; history=[]
    for epoch in range(1,tc["epochs"]+1):
        model.train(); losses=[]
        for bi,batch in enumerate(tqdm(train_loader,desc=f"Epoch {epoch}",leave=False)):
            b=move_batch(batch,device); opt.zero_grad(set_to_none=True)
            fused=model.fuse_inputs(b["ref_ids"],b["alt_ids"],b["variant_mask"],b["metadata"])
            outm=model.forward_from_fused(fused,b["variant_mask"])
            lcls=weighted_cross_entropy(outm["logits"],b["label"],weights)
            lalign=torch.zeros((),device=device)
            if tc.get("alignment_weight",0)>0 and (bi % tc.get("ig_every_n_batches",1)==0):
                attr=differentiable_integrated_gradients(model,fused,b["variant_mask"],b["label"],n_steps=tc.get("ig_steps_train",8))
                target=expanded_target_mask(b["variant_mask"],radius=ec.get("alignment_radius_tokens",10))
                lalign=cosine_alignment_loss(attr,target)
            loss=lcls+tc.get("alignment_weight",0.2)*lalign
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get("grad_clip",1.0)); opt.step(); losses.append(float(loss.detach().cpu()))
        scheduler.step()
        pred=predict(model,val_loader,device)
        m=binary_metrics(pred.true_label,pred.p_pathogenic,tc.get("threshold",0.5)); au=m["auroc"]
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),**m,"lr":opt.param_groups[0]["lr"]})
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        if au>best:
            best=au; bad=0; torch.save({"model_state":model.state_dict(),"epoch":epoch,"val_metrics":m},out/"best.pt")
        else: bad+=1
        if bad>=tc.get("patience",10): break
    ck=torch.load(out/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model_state"])
    return model,history

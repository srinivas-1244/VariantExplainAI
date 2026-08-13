from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def variant_wise_split(df, seed=42):
    train, tmp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
    val, test = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def gene_wise_split(df, seed=42):
    x = df.copy()
    x["gene_group"] = x["gene"].fillna("").replace("", pd.NA)
    x["gene_group"] = x["gene_group"].fillna(x["variant_id"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, tmp_idx = next(gss.split(x, groups=x["gene_group"]))
    train = x.iloc[tr_idx]
    tmp = x.iloc[tmp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_idx, te_idx = next(gss2.split(tmp, groups=tmp["gene_group"]))
    return train.drop(columns="gene_group").reset_index(drop=True), tmp.iloc[va_idx].drop(columns="gene_group").reset_index(drop=True), tmp.iloc[te_idx].drop(columns="gene_group").reset_index(drop=True)


def save_splits(splits, out_dir, prefix="variant"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(["train", "val", "test"], splits):
        frame.to_csv(out / f"{prefix}_{name}.csv", index=False)


def temporal_holdout_split(df, cutoff, date_column="first_seen_date"):
    """Earlier records -> development; records after cutoff -> temporal test.
    Development is split 85/15 train/validation, preserving labels.
    """
    x=df.copy(); x[date_column]=pd.to_datetime(x[date_column])
    cutoff=pd.Timestamp(cutoff)
    dev=x[x[date_column] <= cutoff].copy(); test=x[x[date_column] > cutoff].copy()
    if len(test)==0: raise ValueError("Temporal test is empty; choose an earlier cutoff or provide first-seen dates.")
    train,val=train_test_split(dev,test_size=0.15,random_state=42,stratify=dev["label"])
    return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)


def make_imbalance_training_set(train_df, pathogenic_to_benign="1:2", seed=42):
    """Undersample the training partition only to a requested P:B ratio."""
    p,b=map(int,pathogenic_to_benign.split(':')); x=train_df.copy()
    pos=x[x.label==1]; neg=x[x.label==0]
    unit=min(len(pos)/p,len(neg)/b)
    np_=int(unit*p); nn_=int(unit*b)
    return pd.concat([pos.sample(np_,random_state=seed),neg.sample(nn_,random_state=seed)]).sample(frac=1,random_state=seed).reset_index(drop=True)

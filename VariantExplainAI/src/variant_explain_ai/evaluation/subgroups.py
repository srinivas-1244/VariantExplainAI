from .metrics import binary_metrics


def subgroup_metrics(pred_df, metadata_df, columns=("variant_type","region"), threshold=0.5):
    x=pred_df.merge(metadata_df,on="variant_id",how="left"); rows=[]
    for col in columns:
        if col not in x.columns: continue
        for val,g in x.groupby(col,dropna=False):
            if len(g)<2: continue
            m=binary_metrics(g.true_label,g.p_pathogenic,threshold)
            rows.append({"group":col,"value":str(val),"n":len(g),**m})
    return rows

#!/usr/bin/env python
import argparse
from pathlib import Path
from variant_explain_ai.utils.config import load_config
from variant_explain_ai.data.clinvar import read_clinvar_vcf, normalize_and_deduplicate
from variant_explain_ai.data.preprocess import build_sequence_table
from variant_explain_ai.data.splits import variant_wise_split, gene_wise_split, save_splits

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--vcf"); p.add_argument("--fasta"); p.add_argument("--out"); a=p.parse_args()
c=load_config(a.config); vcf=a.vcf or c["paths"]["clinvar_vcf"]; fasta=a.fasta or c["paths"]["grch38_fasta"]; out=a.out or c["paths"]["processed_csv"]
df=normalize_and_deduplicate(read_clinvar_vcf(vcf,min_stars=1)); seq,errors=build_sequence_table(df,fasta,c["model"]["half_window"])
Path(out).parent.mkdir(parents=True,exist_ok=True); seq.to_csv(out,index=False); errors.to_csv(Path(out).with_name("preprocess_errors.csv"),index=False)
save_splits(variant_wise_split(seq,c["seed"]),c["paths"]["split_dir"],"variant"); save_splits(gene_wise_split(seq,c["seed"]),c["paths"]["split_dir"],"gene")
print(f"Saved {len(seq)} variants to {out}; errors={len(errors)}")

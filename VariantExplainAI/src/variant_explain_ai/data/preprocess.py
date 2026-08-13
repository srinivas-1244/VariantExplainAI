import pandas as pd
from tqdm import tqdm
from .haplotype import HaplotypeBuilder


def build_sequence_table(df, fasta_path, half_window=1000, skip_errors=True):
    hb = HaplotypeBuilder(fasta_path, half_window=half_window)
    rows = []
    errors = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building haplotypes"):
        try:
            pair = hb.build(r.chrom, r.pos, r.ref, r.alt, validate=True)
            d = r.to_dict()
            d.update({
                "ref_seq": pair.ref_seq, "alt_seq": pair.alt_seq,
                "variant_mask": "".join(map(str, pair.variant_mask)),
                "variant_type": pair.variant_type, "ref_len": pair.ref_len,
                "alt_len": pair.alt_len, "delta_len": pair.delta_len
            })
            rows.append(d)
        except Exception as e:
            errors.append({"variant_id": r.get("variant_id", ""), "error": str(e)})
            if not skip_errors:
                raise
    return pd.DataFrame(rows), pd.DataFrame(errors)

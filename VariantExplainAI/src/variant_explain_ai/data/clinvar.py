from pathlib import Path
import re
import pandas as pd
import pysam

TARGET_SIG = {"Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"}
CONFLICT_TERMS = {"Conflicting_interpretations_of_pathogenicity", "conflicting_data_from_submitters"}


def _star_count(revstat):
    s = str(revstat).lower().replace("_", " ")
    if "practice guideline" in s:
        return 4
    if "expert panel" in s:
        return 3
    if "multiple submitters" in s and "no conflicts" in s:
        return 2
    if "criteria provided" in s and "single submitter" in s:
        return 1
    return 0


def _clean_sig(sig):
    if isinstance(sig, (tuple, list)):
        parts = list(sig)
    else:
        parts = re.split(r"[|/,]", str(sig))
    return [p.strip().replace(" ", "_") for p in parts if p]


def _gene_from_info(info):
    gi = info.get("GENEINFO")
    if not gi:
        return ""
    if isinstance(gi, tuple):
        gi = gi[0]
    return str(gi).split(":")[0]


def read_clinvar_vcf(vcf_path, min_stars=1):
    rows = []
    vf = pysam.VariantFile(str(vcf_path))
    for rec in vf.fetch():
        sigs = _clean_sig(rec.info.get("CLNSIG", ""))
        if any(x in CONFLICT_TERMS for x in sigs):
            continue
        retained = [x for x in sigs if x in TARGET_SIG]
        if len(set(retained)) != 1:
            continue
        rev = rec.info.get("CLNREVSTAT", "")
        if isinstance(rev, tuple):
            rev = ",".join(map(str, rev))
        stars = _star_count(rev)
        if stars < min_stars:
            continue
        label4 = retained[0]
        label = 1 if label4 in {"Pathogenic", "Likely_pathogenic"} else 0
        for alt in rec.alts or []:
            ref = rec.ref.upper(); alt = alt.upper()
            if not set(ref+alt) <= set("ACGT"):
                continue
            if max(len(ref), len(alt)) > 50:
                continue
            rows.append({
                "chrom": str(rec.chrom), "pos": int(rec.pos), "ref": ref, "alt": alt,
                "label4": label4, "label": label, "stars": stars,
                "gene": _gene_from_info(rec.info),
                "vcv": str(rec.info.get("ALLELEID", rec.id or ""))
            })
    return pd.DataFrame(rows)


def minimal_normalize_variant(pos, ref, alt):
    """Trim common suffix/prefix while preserving at least one base per allele."""
    pos = int(pos); ref = ref.upper(); alt = alt.upper()
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        pos += 1
    return pos, ref, alt


def normalize_and_deduplicate(df):
    out = df.copy()
    norm = out.apply(lambda r: minimal_normalize_variant(r.pos, r.ref, r.alt), axis=1)
    out[["pos", "ref", "alt"]] = pd.DataFrame(norm.tolist(), index=out.index)
    out["variant_id"] = out.apply(lambda r: f"{r.chrom}:{r.pos}:{r.ref}:{r.alt}", axis=1)
    out = out.drop_duplicates("variant_id").reset_index(drop=True)
    return out

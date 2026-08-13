from dataclasses import dataclass
from pyfaidx import Fasta


@dataclass
class HaplotypePair:
    ref_seq: str
    alt_seq: str
    variant_mask: list
    variant_type: str
    ref_len: int
    alt_len: int
    delta_len: int


def normalize_chrom(chrom):
    c = str(chrom)
    return c if c.startswith("chr") else "chr" + c


def classify_variant(ref, alt):
    if len(ref) == len(alt) == 1:
        return "SNV"
    if len(alt) > len(ref):
        return "INS"
    if len(alt) < len(ref):
        return "DEL"
    return "MNV"


class HaplotypeBuilder:
    """Build fixed-length REF/ALT genomic contexts around a 1-based VCF position."""
    def __init__(self, fasta_path, half_window=1000, pad_char="N"):
        self.fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
        self.half_window = int(half_window)
        self.pad_char = pad_char
        self.target_len = 2 * self.half_window + 1

    def _fetch_padded(self, chrom, center_1based):
        chrom = normalize_chrom(chrom)
        center0 = int(center_1based) - 1
        start0 = center0 - self.half_window
        end0 = center0 + self.half_window + 1
        left_pad = max(0, -start0)
        chrom_len = len(self.fasta[chrom])
        right_pad = max(0, end0 - chrom_len)
        s = max(0, start0)
        e = min(chrom_len, end0)
        seq = self.pad_char * left_pad + self.fasta[chrom][s:e] + self.pad_char * right_pad
        return seq[:self.target_len].ljust(self.target_len, self.pad_char), left_pad

    def build(self, chrom, pos, ref, alt, validate=True):
        ref = str(ref).upper()
        alt = str(alt).upper()
        seq, left_pad = self._fetch_padded(chrom, pos)
        anchor = self.half_window
        if validate:
            observed = seq[anchor:anchor+len(ref)]
            if observed != ref:
                raise ValueError(f"GRCh38 REF mismatch at {chrom}:{pos}; expected {ref}, observed {observed}")
        alt_seq = seq[:anchor] + alt + seq[anchor+len(ref):]
        if len(alt_seq) > self.target_len:
            extra = len(alt_seq) - self.target_len
            left_trim = extra // 2
            right_trim = extra - left_trim
            alt_seq = alt_seq[left_trim:len(alt_seq)-right_trim if right_trim else None]
            alt_anchor = anchor - left_trim
        elif len(alt_seq) < self.target_len:
            deficit = self.target_len - len(alt_seq)
            left_pad_n = deficit // 2
            right_pad_n = deficit - left_pad_n
            alt_seq = self.pad_char * left_pad_n + alt_seq + self.pad_char * right_pad_n
            alt_anchor = anchor + left_pad_n
        else:
            alt_anchor = anchor
        alt_seq = alt_seq[:self.target_len].ljust(self.target_len, self.pad_char)
        mask = [0] * self.target_len
        span = max(1, len(ref), len(alt))
        start = max(0, min(self.target_len-1, alt_anchor))
        end = min(self.target_len, start + span)
        for i in range(start, end):
            mask[i] = 1
        return HaplotypePair(
            ref_seq=seq, alt_seq=alt_seq, variant_mask=mask,
            variant_type=classify_variant(ref, alt), ref_len=len(ref), alt_len=len(alt),
            delta_len=len(alt)-len(ref)
        )

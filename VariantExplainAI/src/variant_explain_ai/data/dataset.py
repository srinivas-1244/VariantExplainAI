import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .kmer import KmerTokenizer

VARIANT_TYPE_TO_ID = {"SNV": 0, "INS": 1, "DEL": 2, "MNV": 3}


def nucleotide_mask_to_kmer_mask(mask, k):
    mask = np.asarray(mask, dtype=np.int64)
    if len(mask) < k:
        return np.zeros(0, dtype=np.int64)
    return np.asarray([int(mask[i:i+k].any()) for i in range(len(mask)-k+1)], dtype=np.int64)


class VariantDataset(Dataset):
    def __init__(self, frame_or_csv, k=5):
        self.df = pd.read_csv(frame_or_csv) if isinstance(frame_or_csv, (str, bytes)) else frame_or_csv.reset_index(drop=True)
        self.tokenizer = KmerTokenizer(k)
        self.k = k

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ref_ids = self.tokenizer.encode(r.ref_seq)
        alt_ids = self.tokenizer.encode(r.alt_seq)
        if isinstance(r.variant_mask, str):
            nuc_mask = np.fromiter((int(c) for c in r.variant_mask), dtype=np.int64)
        else:
            nuc_mask = np.asarray(r.variant_mask, dtype=np.int64)
        kmask = nucleotide_mask_to_kmer_mask(nuc_mask, self.k)
        metadata = np.asarray([
            VARIANT_TYPE_TO_ID.get(str(r.variant_type), 3),
            float(r.ref_len), float(r.alt_len), float(r.delta_len)
        ], dtype=np.float32)
        return {
            "ref_ids": torch.tensor(ref_ids, dtype=torch.long),
            "alt_ids": torch.tensor(alt_ids, dtype=torch.long),
            "variant_mask": torch.tensor(kmask, dtype=torch.bool),
            "metadata": torch.tensor(metadata, dtype=torch.float32),
            "label": torch.tensor(int(r.label), dtype=torch.long),
            "variant_id": str(r.variant_id),
        }

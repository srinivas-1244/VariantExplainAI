from itertools import product
import numpy as np


class KmerTokenizer:
    def __init__(self, k=5):
        self.k = int(k)
        kmers = ["".join(p) for p in product("ACGT", repeat=self.k)]
        self.pad_token = "<PAD>"
        self.unk_token = "<N>"
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        for i, kmer in enumerate(kmers, start=2):
            self.token_to_id[kmer] = i
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def tokenize(self, seq):
        seq = str(seq).upper()
        if len(seq) < self.k:
            return []
        return [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]

    def encode(self, seq):
        ids = []
        for tok in self.tokenize(seq):
            ids.append(self.token_to_id.get(tok, self.token_to_id[self.unk_token]))
        return np.asarray(ids, dtype=np.int64)

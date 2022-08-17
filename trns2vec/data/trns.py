import logging
import os
import re
import numpy as np
from Bio import bgzf, SeqIO

_base_to_int = {
        'T': 0, 't': 0,
        'C': 1, 'c': 1,
        'A': 2, 'a': 2,
        'G': 3, 'g': 3,
        'N': 4} # Like T in two bits
_bases = ['T', 'C', 'A', 'G']

OOV_ID = -1

def _standardization(self, sequence):
    return re.sub(r'[^actg]', '', sequence.lower())

def _read(bgz_file):
    with bgzf.open(bgz_file, 'r') as fa:
        for i, feature in enumerate(SeqIO.parse(fa, "fasta")):
            yield i, str(feature.seq)

def _kmer_tokenizer(sequence, k):
    _mask = np.uint64((np.uint64(1) << np.uint64(2*k))-1)
    kmer = np.uint64(0)
    l = 0
    for n in sequence:
        kmer = (kmer << np.uint64(2) | np.uint64(_base_to_int[n])) & _mask
        l += 1
        if (l >= k):
            yield kmer

def decode(code, length):
    ret = ''
    for _ in range(length):
        index = code & np.uint64(3)
        code >>= np.uint64(2)
        ret = _bases[index] + ret
    return ret

def trns_by_id(bgz_file):
    return {i: s for i, s in _read(bgz_file)}

def tokens_by_trns_id(bgz_file, k):
    return {t_id: list(_kmer_tokenizer(seq, k)) for t_id, seq in trns_by_id(bgz_file).items()}

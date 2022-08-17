import logging
import os
import re
import numpy as np
from Bio import bgzf, SeqIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_base_to_int = {
        'T': 0, 't': 0,
        'C': 1, 'c': 1,
        'A': 2, 'a': 2,
        'G': 3, 'g': 3,
        'N': 4} # Like T in two bits
_bases = ['T', 'C', 'A', 'G']

OOV_ID = -1

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

def trns_by_id(bgz_file):
    logger.info('Loading bgzip file from {}'.format(bgz_file))
    return {i: s for i, s in _read(bgz_file)}

def tokens_by_trns_id(bgz_file, k):
    return {t_id: _kmer_tokenizer(seq, k) for t_id, seq in trns_by_id(bgz_file).items()}
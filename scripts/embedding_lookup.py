import io
import argparse
import h5py
import numpy as np
# importing module
import sys
# appending a path
sys.path.append('trns2vec/.')

from data import vocab

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('trns_embeddings', help='Path to load transcript embeddings file')
    parser.add_argument('vocab', help='Path to load vocab file')
    parser.add_argument('-k', '--k', default=17, type=int, help='kmer window')
    parser.add_argument('--vectors_file', default='vectors.tsv', 
            help='Path to save vectors file')
    parser.add_argument('--metadata_file', default='metadata.tsv', 
            help='Path to save metadata file')

    return parser.parse_args()

_bases = ['T', 'C', 'A', 'G']
def decode(code, length):
    ret = ''
    for _ in range(length):
        index = code & np.uint64(3)
        code >>= np.uint64(2)
        ret = _bases[index] + ret
    return ret

def main():
    args = _parse_args()
    with h5py.File(args.trns_embeddings, "r") as hf:
        embeddings = hf.get('trns_embeddings')[()]
    
    v = vocab.Vocabulary()
    v.load(args.vocab)
    
    out_v = io.open(args.vectors_file, 'w', encoding='utf-8')
    out_m = io.open(args.metadata_file, 'w', encoding='utf-8')
    
    for index in range(v.size):
        if vocab._UNKNOWN == v.to_token(index):
            continue
        vec = embeddings[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(decode(v.to_token(index), args.k) + "\n")
    out_v.close()
    out_m.close()

if __name__ == "__main__":
    main()

import argparse
import itertools

from data import batch_dm, batch_dbow, trns
from model import dm, dbow, model
import vocab

MODEL_TYPES = {
    'dm': (dm.DM, batch_dm.data_generator, batch_dm.batch),
    'dbow': (dbow.DBOW, batch_dbow.data_generator, batch_dbow.batch)
}

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='Path to bgzip feature file')
    parser.add_argument('-k', '--k',
                        default=17,
                        type=int,
                        help='kmer window')
    parser.add_argument('--save', help='Path to save model')
    parser.add_argument('--save_period', 
                        type=int,
                        help='Save model every n epochs')
    parser.add_argument('--save_vocab', help='Path to save vocab file')
    parser.add_argument('--save_doc_embeddings',
                        help='Path to save doc embeddings file')
    parser.add_argument('--save_doc_embeddings_period',
                        type=int,
                        help='Save doc embeddings every n epochs')

    parser.add_argument('--load', help='Path to load model')
    parser.add_argument('--load_vocab', help='Path to load vocab file')

    parser.add_argument('--early_stopping_patience',
                        type=int,
                        help='Stop after no loss decrease for n epochs')

    parser.add_argument('--vocab_size', default=vocab.DEFAULT_SIZE,
                        type=int,
                        help='Max vocabulary size; ignored if loading from file')
    parser.add_argument('--vocab_rare_threshold',
                        default=vocab.DEFAULT_RARE_THRESHOLD,
                        type=int,
                        help=('Words less frequent than this threshold '
                              'will be considered unknown'))

    return parser.parse_args()


def main():
    args = _parse_args()

    tokens_by_trns_id = trns.tokens_by_trns_id(args.path, args.k)

    num_trns = len(tokens_by_trns_id)

    v = vocab.Vocabulary()
    if args.load_vocab:
        v.load(args.load_vocab)
    else:
        all_tokens = list(itertools.chain.from_iterable(tokens_by_trns_id.values()))
        v.build(all_tokens, max_size=args.vocab_size)
        if args.save_vocab:
            v.save(args.save_vocab)

    token_ids_by_trns_id = {d: v.to_ids(t) for d, t in tokens_by_trns_id.items()}

    print(tokens_by_trns_id)

if __name__ == "__main__":
    main()

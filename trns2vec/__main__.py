import os
import argparse
import itertools
from datetime import datetime

from utils import logger
from data import vocab, batch_dm, batch_dbow, trns, tfdataset
from model import dm, dbow, model

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

    parser.add_argument('--model', default='dm',
                        choices=list(MODEL_TYPES.keys()),
                        help='Which model to use')
    parser.add_argument('--logs_dir', help='Logger dir')
    parser.add_argument('--save', help='Path to save model')
    parser.add_argument('--save_period', 
                        type=int,
                        help='Save model every n epochs')
    parser.add_argument('--save_vocab', help='Path to save vocab file')
    parser.add_argument('--save_trns_embeddings',
                        help='Path to save transcript embeddings file')
    parser.add_argument('--save_trns_embeddings_period',
                        type=int,
                        help='Save transcript embeddings every n epochs')

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

    parser.add_argument('--window_size',
                        default=model.DEFAULT_WINDOW_SIZE,
                        type=int,
                        help='Context window size')
    parser.add_argument('--embedding_size',
                        default=model.DEFAULT_EMBEDDING_SIZE,
                        type=int,
                        help='Word and transcript embedding size')

    parser.add_argument('--num_epochs',
                        default=model.DEFAULT_NUM_EPOCHS,
                        type=int,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size',
                        default=model.DEFAULT_BATCH_SIZE,
                        type=int,
                        help='Number of epochs to train for')
    parser.add_argument('--steps_per_epoch',
                        default=model.DEFAULT_STEPS_PER_EPOCH,
                        type=int,
                        help='Number of samples per epoch')
    parser.add_argument('--workers',
                        default=model.DEFAULT_WORKERS,
                        type=int,
                        help='Number of samples per epoch')
    parser.add_argument('--mp', dest='mp', action='store_true')
    parser.add_argument('--no-csv-logs', dest='csv_logs', action='store_false')
    parser.set_defaults(csv_logs=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', dest='train', action='store_true')
    group.add_argument('--no-train', dest='train', action='store_false')
    group.set_defaults(train=False)

    return parser.parse_args()


def main():
    args = _parse_args()
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(args.logs_dir, 'logs-{}'.format(now))
    logs = logger.get_logger('trns2vec', log_file)

    logs.info('Loading bgzip file from {}'.format(args.path))
    tokens_by_trns_id = trns.tokens_by_trns_id(args.path, args.k)

    num_trns = len(tokens_by_trns_id)

    v = vocab.Vocabulary(logger=logs)
    if args.load_vocab:
        v.load(args.load_vocab)
    else:
        all_tokens = list(itertools.chain.from_iterable(tokens_by_trns_id.values()))
        v.build(all_tokens, max_size=args.vocab_size, rare_threshold=args.vocab_rare_threshold)
        if args.save_vocab:
            v.save(args.save_vocab)

    token_ids_by_trns_id = {d: v.to_ids(ts) for d, ts in tokens_by_trns_id.items()}

    model_class, data_generator, batcher = MODEL_TYPES[args.model]

    m = model_class(args.window_size, v.size, num_trns,
                    embedding_size=args.embedding_size, logger=logs)

    if args.load:
        m.load(args.load) 
    else:
        m.build()
        m.compile()

    elapsed_epochs = 0

    if args.train:


        log_csv = os.path.join(args.logs_dir, 'logs-{}.csv'.format(now)) if args.csv_logs else None

        dataset = tfdataset.build_dataset(data_generator(
                token_ids_by_trns_id,
                args.window_size,
                v.size), batch_size=args.batch_size)

        history = m.train(
                dataset,
                epochs=args.num_epochs,
                steps_per_epoch=args.steps_per_epoch,
                workers=args.workers,
                use_multiprocessing=args.mp,
                early_stopping_patience=args.early_stopping_patience,
                save_path=args.save,
                save_period=args.save_period,
                save_doc_embeddings_path=args.save_trns_embeddings,
                save_doc_embeddings_period=args.save_trns_embeddings_period,
                csv_logger_path=log_csv)

        elapsed_epochs = len(history.history['loss'])

    if args.save:
        m.save(
            args.save.format(epoch=elapsed_epochs))
        np.save('history.npy', history.history)

    if args.save_trns_embeddings:
        m.save_doc_embeddings(
            args.save_trns_embeddings.format(epoch=elapsed_epochs))

if __name__ == "__main__":
    main()

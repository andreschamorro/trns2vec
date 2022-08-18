from abc import ABCMeta

import logging
import h5py
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import numpy as np

DEFAULT_WINDOW_SIZE = 8
DEFAULT_EMBEDDING_SIZE = 300
DEFAULT_NUM_EPOCHS = 250
DEFAULT_BATCH_SIZE = 1024 
DEFAULT_STEPS_PER_EPOCH = 10000
DEFAULT_WORKERS = 4

DOC_EMBEDDINGS_LAYER_NAME = 'trns_embeddings'


class Doc2VecModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, window_size, vocab_size, num_docs,
                 embedding_size=DEFAULT_EMBEDDING_SIZE, logger=None):
        self._window_size = window_size
        self._vocab_size = vocab_size
        self._num_docs = num_docs

        self._embedding_size = embedding_size

        self._model = None
        self._logger = logger

        if self._logger is None:
            self._logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)

    @property
    def doc_embeddings(self):
        return _doc_embeddings_from_model(self._model)

    def build(self):
        raise NotImplementedError()

    def compile(self, optimizer=None):
        if not optimizer:
            optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)

        self._model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])

    def train(self, dataset,
              steps_per_epoch=DEFAULT_STEPS_PER_EPOCH,
              epochs=DEFAULT_NUM_EPOCHS,
              early_stopping_patience=None,
              workers=1,
              use_multiprocessing=False,
              save_path=None, save_period=None,
              save_doc_embeddings_path=None, save_doc_embeddings_period=None, 
              csv_logger_path=None):

        callbacks=[]
        if early_stopping_patience:
            callbacks.append(EarlyStopping(monitor='loss',
                                           patience=early_stopping_patience))
        if save_path and save_period:
            callbacks.append(ModelCheckpoint(save_path,
                                             period=save_period))
        if save_doc_embeddings_path and save_doc_embeddings_period:
            callbacks.append(_SaveDocEmbeddings(save_doc_embeddings_path,
                                                save_doc_embeddings_period))
        if csv_logger_path:
            callbacks.append(CSVLogger(
                csv_logger_path, separator=',', append=False))

        history = self._model.fit(
                dataset,
                callbacks=callbacks,
	            steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                workers=workers,
                use_multiprocessing=use_multiprocessing)
  
        return history

    def save(self, path):
        self._logger.info('Saving model to %s', path)
        self._model.save(path)

    def save_doc_embeddings(self, path):
        self._logger.info('Saving doc embeddings to %s', path)
        _write_doc_embeddings(self.doc_embeddings, path)

    def load(self, path):
        self._logger.info('Loading model from %s', path)
        self._model = load_model(path)


class _SaveDocEmbeddings(Callback):

    def __init__(self, path, period):
        self.path = path
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return

        path = self.path.format(epoch=epoch)
        embeddings = _doc_embeddings_from_model(self.model)
        _write_doc_embeddings(embeddings, path)


def _doc_embeddings_from_model(keras_model):
    for layer in keras_model.layers:
        if layer.get_config()['name'] == DOC_EMBEDDINGS_LAYER_NAME:
            return layer.get_weights()[0]


def _write_doc_embeddings(doc_embeddings, path):
    with h5py.File(path, 'w') as f:
        f.create_dataset(DOC_EMBEDDINGS_LAYER_NAME, data=doc_embeddings)

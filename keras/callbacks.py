from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings
import time
from collections import deque

from .utils.generic_utils import Progbar

class CallbackList(object):

    def __init__(self, callbacks, queue_length=10):
        self.callbacks = callbacks
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def _set_params(self, params):
        for callback in self.callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self.callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_batch_begin(self, batch):
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch \
            and delta_t_median > 0.1:
            warnings.warn('Method on_batch_begin() is slow compared '
                'to the batch update (%f). Check your callbacks.' % delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch):
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(batch)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch \
            and delta_t_median > 0.1:
            warnings.warn('Method on_batch_end() is slow compared '
                'to the batch update (%f). Check your callbacks.' % delta_t_median)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class Callback(object):

    def __init__(self):
        pass

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class BaseLogger(Callback):

    def on_train_begin(self):
        self.verbose = self.params['verbose']

    def on_epoch_begin(self, epoch):
        if self.verbose:
            print('Epoch %d' % epoch)
            self.progbar = Progbar(target=self.params['nb_sample'], \
                verbose=self.verbose)
        self.current = 0

    def on_batch_begin(self, batch):
        self.log_values = []

    def on_batch_end(self, batch_index):
        self.current += self.model.batch_history['batch_size'][-1]
        # skip progbar update for the last batch; will be handled by on_epoch_end
        if self.current < self.params['nb_sample']:
            loss = self.model.batch_history['loss'][-1]
            self.log_values.append(('loss', loss))
            if self.params['show_accuracy']:
                accuracy = self.model.batch_history['accuracy'][-1]
                self.log_values.append(('acc.', accuracy))
            if self.verbose:
                self.progbar.update(self.current, self.log_values)

    def on_epoch_end(self, epoch):
        loss = self.model.batch_history['loss'][-1]
        self.log_values.append(('loss', loss))

        if self.params['show_accuracy']:
            accuracy = self.model.batch_history['accuracy'][-1]
            self.log_values.append(('acc.', accuracy))
        if self.params['do_validation']:
            val_loss = self.model.epoch_history['val_loss'][-1]
            self.log_values.append(('val. loss', val_loss))
            if self.params['show_accuracy']:
                val_acc = self.model.epoch_history['val_accuracy'][-1]
                self.log_values.append(('val. acc.', val_acc))
        self.progbar.update(self.current, self.log_values)

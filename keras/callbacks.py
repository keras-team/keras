from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
import warnings

from .utils.generic_utils import Progbar

class CallbackList(object):
    """docstring for Callback"""
    def __init__(self, callbacks):
        self.callbacks = callbacks

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

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch, indices, loss, accuracy, val_loss, val_acc):
        for callback in self.callbacks:
            callback.on_batch_end(batch, indices, loss, accuracy, val_loss, val_acc)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class Callback(object):
    """docstring for Callback"""
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

    def on_batch_end(self, batch, indices, loss, accuracy, val_loss, val_acc):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class History(Callback):

    def on_train_begin(self):
        self.epochs = []
        self.losses = []
        if self.params['show_accuracy']:
            self.accuracies = []
        if self.params['do_validation']:
            self.validation_losses = []
            if self.params['show_accuracy']:
                self.validation_accuracies = []

    def on_epoch_begin(self, epoch):
        self.seen = 0
        self.cum_loss = 0.
        self.cum_accuracy = 0.
        self.val_loss = 0.
        self.val_accuracy = 0.

    def on_batch_end(self, batch, indices, loss, accuracy, val_loss, val_acc):
        batch_length = len(indices)
        self.seen += batch_length
        self.cum_loss += loss * batch_length
        if self.params['show_accuracy']:
            self.cum_accuracy += accuracy * batch_length
        if self.params['do_validation']:
            self.val_loss = val_loss
            if self.params['show_accuracy']:
                self.val_accuracy = val_acc

    def on_epoch_end(self, epoch):
        self.epochs.append(epoch)
        self.losses.append(self.cum_loss / self.seen)
        if self.params['show_accuracy']:
            self.accuracies.append(self.cum_accuracy / self.seen)
        if self.params['do_validation']:
            self.validation_losses.append(self.val_loss)
            if self.params['show_accuracy']:
                self.validation_accuracies.append(self.val_accuracy)

class Logger(Callback):

    def on_train_begin(self):
        self.verbose = self.params['verbose']

    def on_epoch_begin(self, epoch):
        if self.verbose:
            print 'Epoch %d' % epoch
            self.progbar = Progbar(target=self.params['nb_sample'], \
                verbose=self.verbose)
        self.current = 0

    def on_batch_begin(self, batch):
        self.log_values = []

    def on_batch_end(self, batch, indices, loss, accuracy, val_loss, val_acc):
        self.log_values.append(('loss', loss))
        self.current += len(indices)
        if self.params['show_accuracy']:
            self.log_values.append(('acc.', acc))
        if self.params['do_validation']:
            self.log_values.append(('val. loss', val_loss))
            if self.params['show_accuracy']:
                self.log_values.append(('val. acc.', val_acc))
        if self.verbose:
            self.progbar.update(self.current, self.log_values)
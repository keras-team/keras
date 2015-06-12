from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings
import time
from collections import deque

from .messages import *
from .utils.generic_utils import Progbar

# base class of all callbacks
class Callback(object):
    def __init__(self):
        # create a channel to notify all observers of it
        self.pub_stream = Subject()

    def _set_params(self, params):
        self.params = params

    # @data: EpochBegin instance
    def on_epoch_begin(self, data):
        pass
    # @data: EpochEnd instance
    def on_epoch_end(self, data):
        pass
    # @data: BatchBegin instance
    def on_batch_begin(self, data):
        pass
    # @data: BatchEnd instance
    def on_batch_end(self, data):
        pass
    # @data: TrainBegin instance
    def on_train_begin(self, data):
        pass
    # @data: TrainEnd instance
    def on_train_end(self, data):
        pass

class BaseLogger(Callback):
    def __init__(self, stream=None):
        super(BaseLogger, self).__init__()
        if stream:
            self.set_event_source(stream)

    def set_event_source(self, stream):
        stream.filter(lambda x: isinstance(x, TrainBegin)).subscribe(lambda x: self.on_train_begin(x))
        stream.filter(lambda x: isinstance(x, TrainEnd)).subscribe(lambda x: self.on_train_end(x))
        stream.filter(lambda x: isinstance(x, EpochEnd)).subscribe(lambda x: self.on_epoch_end(x))
        stream.filter(lambda x: isinstance(x, EpochBegin)).subscribe(lambda x: self.on_epoch_begin(x))
        stream.filter(lambda x: isinstance(x, BatchEnd)).subscribe(lambda x: self.on_batch_end(x))
        stream.filter(lambda x: isinstance(x, BatchBegin)).subscribe(lambda x: self.on_batch_begin(x))

    def on_train_begin(self, data):
        self.verbose = self.params['verbose']

    def on_epoch_begin(self, data):
        if self.verbose:
            print('Epoch %d' % data.epoch)
            self.progbar = Progbar(target=self.params['nb_sample'], \
                verbose=self.verbose)
        self.current = 0
        self.tot_loss = 0.
        self.tot_acc = 0.

    def on_batch_begin(self, data):
        if self.current < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, data):
        batch_size = data.size
        self.current += batch_size

        loss = data.loss
        self.log_values.append(('loss', loss))
        self.tot_loss += loss * batch_size
        if self.params['show_accuracy']:
            accuracy = data.accuracy
            self.log_values.append(('acc.', accuracy))
            self.tot_acc += accuracy * batch_size
        # skip progbar update for the last batch; will be handled by on_epoch_end
        if self.verbose and self.current < self.params['nb_sample']:
            self.progbar.update(self.current, self.log_values)

    def on_epoch_end(self, data):
        self.log_values.append(('loss', self.tot_loss / self.current))
        if self.params['show_accuracy']:
            self.log_values.append(('acc.', self.tot_acc / self.current))
        if self.params['do_validation']:
            val_loss = data.val_loss
            self.log_values.append(('val. loss', val_loss))
            if self.params['show_accuracy']:
                val_acc = data.val_accuracy
                self.log_values.append(('val. acc.', val_acc))
        self.progbar.update(self.current, self.log_values)


class History(Callback):

    def __init__(self, stream=None):
        super(History, self).__init__()
        if stream:
            self.set_event_source(stream)

    def set_event_source(self, stream):
        stream.filter(lambda x: isinstance(x, TrainBegin)).subscribe(lambda x: self.on_train_begin(x))
        stream.filter(lambda x: isinstance(x, TrainEnd)).subscribe(lambda x: self.on_train_end(x))
        stream.filter(lambda x: isinstance(x, EpochEnd)).subscribe(lambda x: self.on_epoch_end(x))
        stream.filter(lambda x: isinstance(x, EpochBegin)).subscribe(lambda x: self.on_epoch_begin(x))
        stream.filter(lambda x: isinstance(x, BatchEnd)).subscribe(lambda x: self.on_batch_end(x))
        stream.filter(lambda x: isinstance(x, BatchBegin)).subscribe(lambda x: self.on_batch_begin(x))

    def on_train_begin(self, data):
        self.epoch = []
        self.loss = []
        if self.params['show_accuracy']:
            self.accuracy = []
        if self.params['do_validation']:
            self.validation_loss = []
            if self.params['show_accuracy']:
                self.validation_accuracy = []

    def on_epoch_begin(self, data):
        self.seen = 0
        self.tot_loss = 0.
        self.tot_accuracy = 0.

    def on_batch_end(self, data):
        batch_size = data.size
        self.seen += batch_size
        self.tot_loss += data.loss * batch_size
        if self.params['show_accuracy']:
            self.tot_accuracy += data.accuracy * batch_size

    def on_epoch_end(self, data):
        val_loss = data.val_loss#logs.get('val_loss')
        val_acc = data.val_accuracy#logs.get('val_accuracy')
        self.epoch.append(data.epoch)
        self.loss.append(self.tot_loss / self.seen)
        if self.params['show_accuracy']:
            self.accuracy.append(self.tot_accuracy / self.seen)
        if self.params['do_validation']:
            self.validation_loss.append(val_loss)
            if self.params['show_accuracy']:
                self.validation_accuracy.append(val_acc)

class ModelCheckpoint(Callback):
    def __init__(self, filepath, verbose=0, save_best_only=False, stream=None):
        super(ModelCheckpoint, self).__init__()
        
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.loss = []
        self.best_loss = np.Inf
        self.val_loss = []
        self.best_val_loss = np.Inf

        if stream:
            self.set_event_source(stream)

    def set_event_source(self, stream):
        stream.filter(lambda x: isinstance(x, EpochEnd)).subscribe(lambda x: self.on_epoch_end(x))

    def on_epoch_end(self, data):
        '''currently, on_epoch_end receives epoch_logs from keras.models.Sequential.fit
        which does only contain, if at all, the validation loss and validation accuracy'''
        if self.save_best_only and self.params['do_validation']:
            cur_val_loss = data.val_loss#logs.get('val_loss')
            self.val_loss.append(cur_val_loss)
            if cur_val_loss < self.best_val_loss:
                if self.verbose > 0:
                    print("Epoch %05d: valdidation loss improved from %0.5f to %0.5f, saving model to %s"
                        % (data.epoch, self.best_val_loss, cur_val_loss, self.filepath))
                self.best_val_loss = cur_val_loss
                self.pub_stream.on_next(SaveModel(filename=self.filepath, overwrite=True))
            else:
                if self.verbose > 0:
                    print("Epoch %05d: validation loss did not improve" % (data.epoch))
        elif self.save_best_only and not self.params['do_validation']:
            import warnings
            warnings.warn("Can save best model only with validation data, skipping", RuntimeWarning)
        elif not self.save_best_only:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (data.epoch, self.filepath))
            self.pub_stream.on_next(SaveModel(filename=self.filepath, overwrite=True))

class EarlyStop(Callback):
    def __init__(self, nb_epoch_lookback=5, verbose=0, stream=None):
        super(EarlyStop, self).__init__()

        self.nb_epoch_lookback = nb_epoch_lookback
        self.verbose = verbose
        self.best_val_loss_epoch = 0
        self.best_val_loss = np.Inf

        if stream:
            self.set_event_source(stream)

    def set_event_source(self, stream):
        stream.filter(lambda x: isinstance(x, TrainBegin)).subscribe(lambda x: self.on_train_begin(x))
        stream.filter(lambda x: isinstance(x, TrainEnd)).subscribe(lambda x: self.on_train_end(x))
        stream.filter(lambda x: isinstance(x, EpochEnd)).subscribe(lambda x: self.on_epoch_end(x))
        stream.filter(lambda x: isinstance(x, EpochBegin)).subscribe(lambda x: self.on_epoch_begin(x))
        stream.filter(lambda x: isinstance(x, BatchEnd)).subscribe(lambda x: self.on_batch_end(x))
        stream.filter(lambda x: isinstance(x, BatchBegin)).subscribe(lambda x: self.on_batch_begin(x))

    def on_epoch_end(self, data):
        '''currently, on_epoch_end receives epoch_logs from keras.models.Sequential.fit
        which does only contain, if at all, the validation loss and validation accuracy'''
        if self.params['do_validation']:
            cur_val_loss = data.val_loss
            if cur_val_loss < self.best_val_loss:
                self.best_val_loss_epoch = data.epoch
                self.best_val_loss = cur_val_loss
            elif (data.epoch - self.best_val_loss_epoch) > self.nb_epoch_lookback:
                if self.verbose > 0:
                    print("EarlyStop: Did not observe an improvement over the last {0} epochs, stopping.".format(self.nb_epoch_lookback))
                self.pub_stream.on_next(StopTraining())
        else:
            import warnings
            warnings.warn("Can run EarlyStop callback only with validation data, skipping", RuntimeWarning)


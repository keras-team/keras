from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings, time, copy

from . import optimizers
from . import objectives
from . import regularizers
from . import constraints
from . import callbacks as cbks
import time, copy
from .utils.generic_utils import Progbar, printv
from .layers import containers
from six.moves import range


def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    return y


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def standardize_X(X):
    if type(X) == list:
        return X
    else:
        return [X]


def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]


def weighted_objective(fn):
    def weighted(y_true, y_pred, weights):
        # it's important that 0 * Inf == 0, not NaN, so I need to mask first
        masked_y_true = y_true[weights.nonzero()[:-1]]
        masked_y_pred = y_pred[weights.nonzero()[:-1]]
        masked_weights = weights[weights.nonzero()]
        obj_output = fn(masked_y_true, masked_y_pred)
        return (masked_weights.flatten() * obj_output.flatten()).mean()
    return weighted


def standardize_weights(y, sample_weight=None, class_weight=None):
    if sample_weight is not None:
        return standardize_y(sample_weight)
    elif isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise Exception('class_weight not supported for 3+ dimensional targets.')
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        return np.expand_dims(np.array(list(map(lambda x: class_weight[x], y_classes))), 1)
    else:
        return np.ones(y.shape[:-1] + (1,))


class Model(object):
    def _fit(self, f, ins, out_labels=[], batch_size=128, nb_epoch=100, verbose=1, callbacks=[], \
        validation_split=0., val_f=None, val_ins=None, shuffle=True):
        '''
            Abstract fit function for f(*ins). Assume that f returns a list, labelled by out_labels.
        '''

        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print("Train on %d samples, validate on %d samples" % (len(ins[0]), len(val_ins[0])))
        else:
            if 0 < validation_split < 1:
                do_validation = True
                split_at = int(len(ins[0]) * (1 - validation_split))
                (ins, ins_val) = (slice_X(ins, 0, split_at), slice_X(ins, split_at))
                if verbose:
                    print("Train on %d samples, validate on %d samples" % (len(ins[0]), len(val_ins[0])))

        nb_train_sample = len(ins[0])
        index_array = np.arange(nb_train_sample)

        history = cbks.History()
        if verbose:
            callbacks = [history, cbks.BaseLogger()] + callbacks
        else:
            callbacks = [history] + callbacks
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(ins, batch_ids)

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(*ins_batch)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                
                if batch_index == len(batches) - 1: # last batch
                    # validation
                    epoch_logs = {}
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = val_f(*val_ins)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return history


class Sequential(Model, containers.Sequential):
    '''
        Inherits from Model the following methods:
            - _fit
            - _predict
            - _evaluate
        Inherits from containers.Sequential the following methods:
            - __init__
            - add 
            - get_output
            - get_input
            - get_weights
            - set_weights
    '''

    def compile(self, optimizer, loss, class_mode="categorical", theano_mode=None):
        self.optimizer = optimizers.get(optimizer)
        self.loss = weighted_objective(objectives.get(loss))

        # input of model 
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        self.y = T.zeros_like(self.y_train)

        self.weights = T.ones_like(self.y_train)

        train_loss = self.loss(self.y, self.y_train, self.weights)
        test_loss = self.loss(self.y, self.y_test, self.weights)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'
        
        if class_mode == "categorical":
            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))

        elif class_mode == "binary":
            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)))
            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)))
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.y, self.weights]
            test_ins = self.X_test + [self.y, self.weights]
            predict_ins = self.X_test
        else:
            train_ins = [self.X_train, self.y, self.weights]
            test_ins = [self.X_test, self.y, self.weights]
            predict_ins = [self.X_test]

        self._train = theano.function(train_ins, train_loss, 
            updates=updates, allow_input_downcast=True, mode=theano_mode)
        self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy], 
            updates=updates, allow_input_downcast=True, mode=theano_mode)
        self._predict = theano.function(predict_ins, self.y_test, 
            allow_input_downcast=True, mode=theano_mode)
        self._test = theano.function(test_ins, test_loss, 
            allow_input_downcast=True, mode=theano_mode)
        self._test_with_acc = theano.function(test_ins, [test_loss, test_accuracy], 
            allow_input_downcast=True, mode=theano_mode)


    def train(self, X, y, accuracy=False, sample_weight=None):
        X = standardize_X(X)
        y = standardize_y(y)

        if sample_weight is None:
            sample_weight = np.ones(list(y.shape[0:-1]) + [1])
        else:
            sample_weight = standardize_y(sample_weight)

        ins = X + [y, sample_weight]
        if accuracy:
            return self._train_with_acc(*ins)
        else:
            return self._train(*ins)
        

    def test(self, X, y, accuracy=False):
        X = standardize_X(X)
        y = standardize_y(y)
        ins = X + [y]
        if accuracy:
            return self._test_with_acc(*ins)
        else:
            return self._test(*ins)

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False, 
            class_weight=None, sample_weight=None):

        X = standardize_X(X)
        y = standardize_y(y)
        sample_weight = standardize_weights(y, class_weight=class_weight, sample_weight=sample_weight)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test
        if validation_data:
            try:
                X_val, y_val = validation_data
            except:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val). \
                    X_val may be a numpy array or a list of numpy arrays depending on your model input.")
            X_val = standardize_X(X_val)
            y_val = standardize_y(y_val)
            val_ins = X_val + [y_val, np.ones(y_val.shape[:-1] + (1,))]

        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
        else:
            f = self._train
            out_labels = ['loss']

        ins = X + [y, sample_weight]

        return self._fit(f, ins, out_labels=out_labels, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=callbacks, \
            validation_split=validation_split, val_f=val_f, val_ins=val_ins, shuffle=shuffle)


    def predict(self, X, batch_size=128, verbose=1):
        X = standardize_X(X)
        batches = make_batches(len(X[0]), batch_size)
        if verbose == 1:
            progbar = Progbar(target=len(X[0]))
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = slice_X(X, batch_start, batch_end)
            batch_preds = self._predict(*X_batch)

            if batch_index == 0:
                shape = (len(X[0]),) + batch_preds.shape[1:]
                preds = np.zeros(shape)
            preds[batch_start:batch_end] = batch_preds

            if verbose == 1:
                progbar.update(batch_end)

        return preds


    def predict_proba(self, X, batch_size=128, verbose=1):
        preds = self.predict(X, batch_size, verbose)
        if preds.min() < 0 or preds.max() > 1:
            warnings.warn("Network returning invalid probability values.")
        return preds


    def predict_classes(self, X, batch_size=128, verbose=1):
        proba = self.predict(X, batch_size=batch_size, verbose=verbose)
        if self.class_mode == "categorical":
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')


    def evaluate(self, X, y, batch_size=128, show_accuracy=False, verbose=1, sample_weight=None):
        X = standardize_X(X)
        y = standardize_y(y)
        sample_weight = standardize_weights(y, sample_weight=sample_weight)

        if show_accuracy:
            tot_acc = 0.
        tot_score = 0.
        seen = 0

        batches = make_batches(len(y), batch_size)
        if verbose:
            progbar = Progbar(target=len(y), verbose=verbose)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = slice_X(X, batch_start, batch_end)
            y_batch = y[batch_start:batch_end]
            weight_batch = sample_weight[batch_start:batch_end]

            ins = X_batch + [y_batch, weight_batch]
            if show_accuracy:
                loss, acc = self._test_with_acc(*ins)
                tot_acc += acc * len(y_batch)
                log_values = [('loss', loss), ('acc.', acc)]
            else:
                loss = self._test(*ins)
                log_values = [('loss', loss)]
            tot_score += loss * len(y_batch)
            seen += len(y_batch)

            # logging
            if verbose:
                progbar.update(batch_end, log_values)

        if show_accuracy:
            return tot_score / seen, tot_acc / seen
        else:
            return tot_score / seen


    def get_config(self, verbose=0):
        layers = []
        for i, l in enumerate(self.layers):
            config = l.get_config()
            layers.append(config)
        if verbose:
            printv(layers)
        return layers


    def save_weights(self, filepath, overwrite=False):
        # Save weights from all layers to HDF5
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        f.attrs['nb_layers'] = len(self.layers)
        for k, l in enumerate(self.layers):
            g = f.create_group('layer_{}'.format(k))
            weights = l.get_weights()
            g.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()


    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.layers[k].set_weights(weights)
        f.close()


class Graph(containers.Graph):
    def compile(self, optimizer, loss, theano_mode='DebugMode'):
        # loss is a dictionary mapping output name to loss functions
        ys = []
        ys_train = []
        ys_test = []
        train_loss = 0.
        test_loss = 0.
        for output_name in self.output_order:
            loss_fn = loss[output_name]
            output = self.outputs[output_name]
            y_train = output.get_output(True)
            y_test = output.get_output(False)
            y = T.zeros_like(y_test)
            ys.append(y)
            ys_train.append(y_train)
            ys_test.append(y_test)
            
            train_loss += objectives.get(loss_fn)(y, y_train).mean()
            test_loss += objectives.get(loss_fn)(y, y_test).mean()

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        ins = [self.inputs[name].input for name in self.input_order]
        train_ins = ins + ys
        test_ins = ins + ys

        for r in self.regularizers:
            train_loss = r(train_loss)
        self.optimizer = optimizers.get(optimizer)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)

        self._train = theano.function(train_ins, train_loss, 
            updates=updates, allow_input_downcast=True, mode=theano_mode)
        self._test = theano.function(test_ins, test_loss, 
            allow_input_downcast=True, mode=theano_mode)
        self._predict = theano.function(inputs=ins, outputs=ys_test, 
            allow_input_downcast=True, mode=theano_mode)

    def train(self, data):
        # data is a dictionary mapping output and input names to arrays
        ins = [data[name] for name in self.input_order] + [data[name] for name in self.output_order]
        return self._train(*ins)

    def test(self, data):
        # data is a dictionary mapping input names to arrays
        ins = [data[name] for name in self.input_order] + [data[name] for name in self.output_order]
        return self._test(*ins)

    def predict(self, data):
        # data is a dictionary mapping input names to arrays
        ins = [data[name] for name in self.input_order]
        return self._predict(*ins)



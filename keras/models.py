from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings

from . import optimizers
from . import objectives
from . import regularizers
from . import constraints
import time, copy
from .utils.generic_utils import Progbar, printv
from six.moves import range

def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.reshape(y, (len(y), 1))
    return y

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def ndim_tensor(ndim):
    if ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()

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


class Model(object):

    def get_output(self, train):
        raise NotImplementedError

    def get_input(self, train):
        raise NotImplementedError

    def compile(self, optimizer, loss, class_mode="categorical", theano_mode=None):
        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)

        # input of model 
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        self.y = T.zeros_like(self.y_train)

        train_loss = self.loss(self.y, self.y_train)
        test_score = self.loss(self.y, self.y_test)

        if class_mode == "categorical":
            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))

        elif class_mode == "binary":
            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)))
            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)))
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode

        updates = self.optimizer.get_updates(self.params, self.regularizers, self.constraints, train_loss)

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.y]
            test_ins = self.X_test + [self.y]
            predict_ins = self.X_test
        else:
            train_ins = [self.X_train, self.y]
            test_ins = [self.X_test, self.y]
            predict_ins = [self.X_test]

        self._train = theano.function(train_ins, train_loss, 
            updates=updates, allow_input_downcast=True, mode=theano_mode)
        self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy], 
            updates=updates, allow_input_downcast=True, mode=theano_mode)
        self._predict = theano.function(predict_ins, self.y_test, 
            allow_input_downcast=True, mode=theano_mode)
        self._test = theano.function(test_ins, test_score, 
            allow_input_downcast=True, mode=theano_mode)
        self._test_with_acc = theano.function(test_ins, [test_score, test_accuracy], 
            allow_input_downcast=True, mode=theano_mode)


class Sequential(Model):
    def __init__(self):
        self.layers = []
        self.params = [] # learnable
        self.regularizers = [] # same size as params
        self.constraints = [] # same size as params


    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])
        self.params += [p for p in layer.params]
        
        if hasattr(layer, 'regularizers') and len(layer.regularizers) == len(layer.params):
            for r in layer.regularizers:
                if r:
                    self.regularizers.append(r)
                else:
                    self.regularizers.append(regularizers.identity)
        elif hasattr(layer, 'regularizer') and layer.regularizer:
            self.regularizers += [layer.regularizer for _ in range(len(layer.params))]
        else:
            self.regularizers += [regularizers.identity for _ in range(len(layer.params))]

        if hasattr(layer, 'constraints') and len(layer.constraints) == len(layer.params):
            for c in layer.constraints:
                if c:
                    self.constraints.append(c)
                else:
                    self.constraints.append(constraints.identity)
        elif hasattr(layer, 'constraint') and layer.constraint:
            self.constraints += [layer.constraint for _ in range(len(layer.params))]
        else:
            self.constraints += [constraints.identity for _ in range(len(layer.params))]


    def get_output(self, train=False):
        return self.layers[-1].get_output(train)


    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            for l in self.layers:
                if hasattr(l, 'input'):
                    break
            ndim = l.input.ndim 
            self.layers[0].input = ndim_tensor(ndim)
        return self.layers[0].get_input(train)


    def train(self, X, y, accuracy=False):
        X = standardize_X(X)
        y = standardize_y(y)
        ins = X + [y]
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


    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1,
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):
        
        X = standardize_X(X)
        y = standardize_y(y)

        do_validation = False
        if validation_data:
            try:
                X_val, y_val = validation_data
            except:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val). \
                    X_val may be a numpy array or a list of numpy arrays depending on your model input.")
            do_validation = True
            X_val = standardize_X(X_val)
            y_val = standardize_y(y_val)
            if verbose:
                print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))
        else:
            if 0 < validation_split < 1:
                # If a validation split size is given (e.g. validation_split=0.2)
                # then split X into smaller X and X_val,
                # and split y into smaller y and y_val.
                do_validation = True
                split_at = int(len(y) * (1 - validation_split))
                (X, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
                (y, y_val) = (y[0:split_at], y[split_at:])
                if verbose:
                    print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))
        
        history = {'epoch':[], 'loss':[]}
        if show_accuracy:
            history['acc'] = []
        if do_validation:
            history['val_loss'] = []
            if show_accuracy:
                history['val_acc'] = []

        index_array = np.arange(len(y))
        for epoch in range(nb_epoch):
            if verbose:
                print('Epoch', epoch)
                progbar = Progbar(target=len(y), verbose=verbose)
            if shuffle:
                np.random.shuffle(index_array)

            av_loss = 0.
            av_acc = 0.
            seen = 0

            batches = make_batches(len(y), batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                seen += len(batch_ids)
                X_batch = slice_X(X, batch_ids)
                y_batch = y[batch_ids]

                ins = X_batch + [y_batch]
                if show_accuracy:
                    loss, acc = self._train_with_acc(*ins)
                    log_values = [('loss', loss), ('acc.', acc)]
                    av_loss += loss * len(batch_ids)
                    av_acc += acc * len(batch_ids)
                else:
                    loss = self._train(*ins)
                    log_values = [('loss', loss)]
                    av_loss += loss * len(batch_ids)

                # validation
                if do_validation and (batch_index == len(batches) - 1):
                    if show_accuracy:
                        val_loss, val_acc = self.evaluate(X_val, y_val, batch_size=batch_size, verbose=0, show_accuracy=True)
                        log_values += [('val. loss', val_loss), ('val. acc.', val_acc)]
                    else:
                        val_loss = self.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
                        log_values += [('val. loss', val_loss)]
                
                # logging
                if verbose:
                    progbar.update(batch_end, log_values)

            history['epoch'].append(epoch)
            history['loss'].append(av_loss/seen)
            if do_validation:
                history['val_loss'].append(float(val_loss))
            if show_accuracy:
                history['acc'].append(av_acc/seen)
                if do_validation:
                    history['val_acc'].append(float(val_acc))
        return history

    def predict(self, X, batch_size=128, verbose=1):
        X = standardize_X(X)
        batches = make_batches(len(X[0]), batch_size)
        if verbose==1:
            progbar = Progbar(target=len(X[0]))
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = slice_X(X, batch_start, batch_end)
            batch_preds = self._predict(*X_batch)

            if batch_index == 0:
                shape = (len(X[0]),) + batch_preds.shape[1:]
                preds = np.zeros(shape)
            preds[batch_start:batch_end] = batch_preds

            if verbose==1:
                progbar.update(batch_end)

        return preds

    def predict_proba(self, X, batch_size=128, verbose=1):
        preds = self.predict(X, batch_size, verbose)
        if preds.min()<0 or preds.max()>1:
            warnings.warn("Network returning invalid probability values.")
        return preds


    def predict_classes(self, X, batch_size=128, verbose=1):
        proba = self.predict(X, batch_size=batch_size, verbose=verbose)
        if self.class_mode == "categorical":
            return proba.argmax(axis=-1)
        else:
            return (proba>0.5).astype('int32')


    def evaluate(self, X, y, batch_size=128, show_accuracy=False, verbose=1):
        X = standardize_X(X)
        y = standardize_y(y)

        if show_accuracy:
            tot_acc = 0.
        tot_score = 0.

        batches = make_batches(len(y), batch_size)
        if verbose:
            progbar = Progbar(target=len(y), verbose=verbose)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = slice_X(X, batch_start, batch_end)
            y_batch = y[batch_start:batch_end]

            ins = X_batch + [y_batch]
            if show_accuracy:
                loss, acc = self._test_with_acc(*ins)
                tot_acc += acc
                log_values = [('loss', loss), ('acc.', acc)]
            else:
                loss = self._test(*ins)
                log_values = [('loss', loss)]
            tot_score += loss

            # logging
            if verbose:
                progbar.update(batch_end, log_values)

        if show_accuracy:
            return tot_score/len(batches), tot_acc/len(batches)
        else:
            return tot_score/len(batches)

    def get_config(self, verbose=0):
        layers = []
        for i, l in enumerate(self.layers):
            config = l.get_config()
            layers.append(config)
        if verbose:
            printv(layers)
        return layers

    def get_weights(self):
        res = []
        for l in self.layers:
            res += l.get_weights()
        return res

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def save_weights(self, filepath):
        # Save weights from all layers to HDF5
        import h5py
        # FIXME: fail if file exists, or add option to overwrite!
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
        
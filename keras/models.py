from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np

from . import optimizers
from . import objectives
import time, copy
from .utils.generic_utils import Progbar
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

class Sequential(object):
    def __init__(self):
        self.layers = []
        self.params = []

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])
        self.params += [p for p in layer.params]

    def compile(self, optimizer, loss, class_mode="categorical"):
        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)

        self.X = self.layers[0].input # input of model 
        # (first layer must have an "input" attribute!)
        self.y_train = self.layers[-1].output(train=True)
        self.y_test = self.layers[-1].output(train=False)

        # output of model
        self.y = T.matrix() # TODO: support for custom output shapes

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

        updates = self.optimizer.get_updates(self.params, train_loss)

        self._train = theano.function([self.X, self.y], train_loss, 
            updates=updates, allow_input_downcast=True)
        self._train_with_acc = theano.function([self.X, self.y], [train_loss, train_accuracy], 
            updates=updates, allow_input_downcast=True)
        self._predict = theano.function([self.X], self.y_test, 
            allow_input_downcast=True)
        self._test = theano.function([self.X, self.y], test_score, 
            allow_input_downcast=True)
        self._test_with_acc = theano.function([self.X, self.y], [test_score, test_accuracy], 
            allow_input_downcast=True)

    def train(self, X, y, accuracy=False):
        y = standardize_y(y)
        if accuracy:
            return self._train_with_acc(X, y)
        else:
            return self._train(X, y)
        

    def test(self, X, y, accuracy=False):
        y = standardize_y(y)
        if accuracy:
            return self._test_with_acc(X, y)
        else:
            return self._test(X, y)


    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1,
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):
        y = standardize_y(y)

        do_validation = False
        if validation_data:
            try:
                X_val, y_val = validation_data
            except:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val).")
            do_validation = True
            y_val = standardize_y(y_val)
            if verbose:
                print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))
        else:
            if 0 < validation_split < 1:
                # If a validation split size is given (e.g. validation_split=0.2)
                # then split X into smaller X and X_val,
                # and split y into smaller y and y_val.
                do_validation = True
                split_at = int(len(X) * (1 - validation_split))
                (X, X_val) = (X[0:split_at], X[split_at:])
                (y, y_val) = (y[0:split_at], y[split_at:])
                if verbose:
                    print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))
        
        index_array = np.arange(len(X))
        for epoch in range(nb_epoch):
            if verbose:
                print('Epoch', epoch)
                progbar = Progbar(target=len(X), verbose=verbose)
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(len(X), batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                if shuffle:
                    batch_ids = index_array[batch_start:batch_end]
                else:
                    batch_ids = slice(batch_start, batch_end)
                X_batch = X[batch_ids]
                y_batch = y[batch_ids]

                if show_accuracy:
                    loss, acc = self._train_with_acc(X_batch, y_batch)
                    log_values = [('loss', loss), ('acc.', acc)]
                else:
                    loss = self._train(X_batch, y_batch)
                    log_values = [('loss', loss)]

                # validation
                if do_validation and (batch_index == len(batches) - 1):
                    if show_accuracy:
                        val_loss, val_acc = self.test(X_val, y_val, accuracy=True)
                        log_values += [('val. loss', val_loss), ('val. acc.', val_acc)]
                    else:
                        val_loss = self.test(X_val, y_val)
                        log_values += [('val. loss', val_loss)]
                
                # logging
                if verbose:
                    progbar.update(batch_end, log_values)

            
    def predict_proba(self, X, batch_size=128, verbose=1):
        batches = make_batches(len(X), batch_size)
        if verbose==1:
            progbar = Progbar(target=len(X))
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = X[batch_start:batch_end]
            batch_preds = self._predict(X_batch)

            if batch_index == 0:
                shape = (len(X),) + batch_preds.shape[1:]
                preds = np.zeros(shape)
            preds[batch_start:batch_end] = batch_preds

            if verbose==1:
                progbar.update(batch_end)

        return preds


    def predict_classes(self, X, batch_size=128, verbose=1):
        proba = self.predict_proba(X, batch_size=batch_size, verbose=verbose)
        if self.class_mode == "categorical":
            return proba.argmax(axis=-1)
        else:
            return (proba>0.5).astype('int32')


    def evaluate(self, X, y, batch_size=128, show_accuracy=False, verbose=1):
        y = standardize_y(y)

        if show_accuracy:
            tot_acc = 0.
        tot_score = 0.

        batches = make_batches(len(X), batch_size)
        if verbose:
            progbar = Progbar(target=len(X), verbose=verbose)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            if show_accuracy:
                loss, acc = self._test_with_acc(X_batch, y_batch)
                tot_acc += acc
                log_values = [('loss', loss), ('acc.', acc)]
            else:
                loss = self._test(X_batch, y_batch)
                log_values = [('loss', loss)]
            tot_score += loss

            # logging
            if verbose:
                progbar.update(batch_end, log_values)

        if show_accuracy:
            return tot_score/len(batches), tot_acc/len(batches)
        else:
            return tot_score/len(batches)

    def save(self, fpath):
        import cPickle
        import types

        weights = []
        for l in self.layers:
            weights.append(l.get_weights())
            
            attributes = []
            for a in dir(l):
                if a[:2] != '__' and a not in ['params', 'previous_layer', 'input']:
                    if type(getattr(l, a)) != types.MethodType:
                        attributes.append((a, getattr(l, a)))

        print('Pickle model...')
        cPickle.dump(self, open(fpath + '.pkl', 'w'))
        print('Pickle weights...')
        cPickle.dump(weights, open(fpath + '.h5', 'w'))

        #save_weights_to_hdf5(fpath + '.h5', weights)


                




import theano
import theano.tensor as T
import numpy as np

import optimizers
import objectives
import time, copy
from utils.generic_utils import Progbar

def standardize_y(y):
    y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.reshape(y, (len(y), 1))
    return y

class Sequential(object):
    def __init__(self):
        self.layers = []
        self.params = []

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])
        self.params += [p for p in layer.params]

    def compile(self, optimizer, loss):
        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)

        self.X = self.layers[0].input # input of model 
        # (first layer must have an "input" attribute!)
        self.y_train = self.layers[-1].output(train=True)
        self.y_test = self.layers[-1].output(train=False)

        # output of model
        self.Y = T.matrix() # TODO: support for custom output shapes

        train_loss = self.loss(self.Y, self.y_train)
        test_score = self.loss(self.Y, self.y_test)
        updates = self.optimizer.get_updates(self.params, train_loss)

        self._train = theano.function([self.X, self.Y], train_loss, 
            updates=updates, allow_input_downcast=True)
        self._predict = theano.function([self.X], self.y_test, 
            allow_input_downcast=True)
        self._test = theano.function([self.X, self.Y], test_score, 
            allow_input_downcast=True)

    def train(self, X, y):
        y = standardize_y(y)
        loss = self._train(X, y)
        return loss

    def test(self, X, y):
        y = standardize_y(y)
        score = self._test(X, y)
        return score

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1,
            validation_split=0., shuffle=True):
        # If a validation split size is given (e.g. validation_split=0.2)
        # then split X into smaller X and X_val,
        # and split y into smaller y and y_val.
        y = standardize_y(y)

        do_validation = False
        if validation_split > 0 and validation_split < 1:
            do_validation = True
            split_at = int(len(X) * (1 - validation_split))
            (X, X_val) = (X[0:split_at], X[split_at:])
            (y, y_val) = (y[0:split_at], y[split_at:])
            if verbose:
                print "Train on %d samples, validate on %d samples" % (len(y), len(y_val))
        
        index_array = np.arange(len(X))
        for epoch in range(nb_epoch):
            if verbose:
                print 'Epoch', epoch
            if shuffle:
                np.random.shuffle(index_array)

            nb_batch = len(X)/batch_size+1
            progbar = Progbar(target=len(X))
            for batch_index in range(0, nb_batch):
                batch_start = batch_index*batch_size
                batch_end = min(len(X), (batch_index+1)*batch_size)
                batch_ids = index_array[batch_start:batch_end]

                X_batch = X[batch_ids]
                y_batch = y[batch_ids]
                loss = self._train(X_batch, y_batch)
                
                if verbose:
                    is_last_batch = (batch_index == nb_batch - 1)
                    if not is_last_batch or not do_validation:
                        progbar.update(batch_end, [('loss', loss)])
                    else:
                        progbar.update(batch_end, [('loss', loss), ('val. loss', self.test(X_val, y_val))])
            
    def predict_proba(self, X, batch_size=128):
        for batch_index in range(0, len(X)/batch_size+1):
            batch = range(batch_index*batch_size, min(len(X), (batch_index+1)*batch_size))
            if not batch:
                break
            batch_preds = self._predict(X[batch])

            if batch_index == 0:
                shape = (len(X),) + batch_preds.shape[1:]
                preds = np.zeros(shape)
            preds[batch] = batch_preds
        return preds

    def predict_classes(self, X, batch_size=128):
        proba = self.predict_proba(X, batch_size=batch_size)
        # The last dimension is the one containing the class probas
        classes_dim = proba.ndim - 1
        if proba.shape[classes_dim] > 1:
            return proba.argmax(axis=classes_dim)
        else:
            return (proba>0.5).astype('int32')

    def evaluate(self, X, y, batch_size=128):
        y = standardize_y(y)
        av_score = 0.
        samples = 0
        for batch_index in range(0, len(X)/batch_size+1):
            batch = range(batch_index*batch_size, min(len(X), (batch_index+1)*batch_size))
            if not batch:
                break
            score = self._test(X[batch], y[batch])
            av_score += len(batch)*score
            samples += len(batch)
        return av_score/samples



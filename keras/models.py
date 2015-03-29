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

        Y = T.matrix() # ouput of model
        self.Y = Y

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

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1):
        y = standardize_y(y)
        for epoch in range(nb_epoch):
            if verbose:
                print 'Epoch', epoch
            
            nb_batch = len(X)/batch_size+1
            progbar = Progbar(target=len(X))
            for batch_index in range(0, nb_batch):
                batch = range(batch_index*batch_size, min(len(X), (batch_index+1)*batch_size))
                if not batch:
                    break
                loss = self._train(X[batch], y[batch])
                if verbose:                
                    progbar.update(batch[-1]+1, [('loss', loss)])
            
    def predict_proba(self, X, batch_size=128):
        for batch_index in range(0, len(X)/batch_size+1):
            batch = range(batch_index*batch_size, min(len(X), (batch_index+1)*batch_size))
            if not batch:
                break
            batch_preds = self._predict(X[batch])

            if batch_index == 0:
                preds = np.zeros((len(X), batch_preds.shape[1]))
            preds[batch] = batch_preds
        return preds

    def predict_classes(self, X, batch_size=128):
        proba = self.predict_proba(X, batch_size=batch_size)
        if proba.shape[1] > 1:
            return proba.argmax(axis=1)
        else:
            return np.array([1 if p > 0.5 else 0 for p in proba])

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



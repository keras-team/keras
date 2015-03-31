import theano
import theano.tensor as T
import numpy as np

import optimizers
import objectives
import time, copy
from utils.generic_utils import Progbar
from math import ceil

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
            validation_split=None, shuffle=True):
        # If a validation split size is given (e.g. validation_split=0.2)
        # then split X into smaller X and X_val,
        # and split y into smaller y and y_val.
        do_validation = False
        if validation_split is not None and validation_split > 0 and validation_split < 1:
            do_validation = True
            split_at = int(len(X) * (1 - validation_split))
            (X, X_val) = (X[0:split_at], X[split_at:])
            (y, y_val) = (y[0:split_at], y[split_at:])
            y_val = standardize_y(y_val)
        
        y = standardize_y(y)
        index_array = np.arange(len(X))
        for epoch in range(nb_epoch):
            if verbose:
                print 'Epoch', epoch
            if shuffle:
                np.random.shuffle(index_array)
            nb_batch = int(ceil(float(len(X))/float(batch_size)))
            progbar = Progbar(target=len(X))
            for batch_index in range(0, nb_batch):
                batch_start = batch_index*batch_size
                batch_end = min(len(X), (batch_index+1)*batch_size)
                batch_ids = index_array[batch_start:batch_end]
                X_batch = [X[idx] for idx in batch_ids]
                y_batch = [y[idx] for idx in batch_ids]
                loss = self._train(X_batch, y_batch)
                
                if verbose:
                    is_last_batch = (batch_index == nb_batch - 1)
                    if not is_last_batch:
                        progbar.update(batch_end, [('loss', loss)])
                    else:
                        progbar.update(batch_end, [('loss', loss), ('val. loss', self.test(X_val, y_val))])
    
    """
    A function to fit on large datasets that are two big for gpu and/or main memory.
    
    It expects a function that creates a new generator which iterates over the dataset.
    It uses that function to create a new generator at the start of every
    epoch. It then loads chunk_size rows from the dataset and fits on these
    like fit() does. It proceeds to load more chunks of rows until the
    end of the dataset is reached. Then it starts the next epoch and creates
    another generator on the same dataset.
    A seperate function to create a new generator is needed, because there
    is no way in Python to reset an existing generator.
    Notice that the dataset generator should be fast as it will be called
    multiple times.
    
    
    Example
    The following example creates a dataset with two inputs per row (0 or 1 each)
    and one output that has a probability of 0.5 of being 1 when
    either of the two inputs is 1.
    The dataset contains 1 Million rows. The fit_chunked() method loads
    chunks of 100000 rows of the dataset, so 10 chunks per epoch.
    
        def Xy_generator(count):
            for _ in range(0, count):
                a = random.randint(0, 1)
                b = random.randint(0, 1)
                c = random.randint(0, 1) if a == 1 or b == 1 else 0
                yield([a, b], [c])
    
        def create_Xy_generator():
            return Xy_generator(1 * 1000 * 1000)
    
        ...
    
        model.fit_chunked(create_Xy_generator, chunk_size=100000)
    """
    def fit_chunked(self, create_Xy_generator_func, chunk_size=12800,
                    batch_size=128, nb_epoch=100,
                    count=None,
                    X_val=None, y_val=None, verbose=1):
        # You may provide a validation dataset X_val and y_val.
        # At the end of every epoch the current net will be run
        # on that dataset and the loss will be printed.
        do_validate = False
        if (X_val is not None and y_val is None) or (X_val is not None and y_val is None):
            raise Exception("Incomplete validation data in fit_chunked(). Both X_val and y_val or none of them must be given.")
        elif X_val is not None and y_val is not None:
            do_validate = True
            y_val = standardize_y(y_val)
        
        # Count how many examples the dataset contains in total.
        # Thats necessary for the progress bar.
        # If the parameter 'count' is provided, that value is used
        # instead and the counting is skipped.
        if count is not None:
            count_all = count
        else:
            Xy_gen = create_Xy_generator_func()
            if verbose:
                print "Counting examples in dataset..."
            count_all = 0
            for _ in Xy_gen:
                count_all += 1
            if verbose:
                print "Counted. %s examples found in dataset." % (str(count_all),)
        
        # Repeat training on the full dataset nb_epoch times
        count_chunks = int(ceil(float(count_all) / float(chunk_size)))
        for epoch in range(nb_epoch):
            if verbose:
                print 'Epoch', epoch
            progbar = Progbar(target=count_all)
            Xmins, Xmaxs, ymins, ymaxs = None, None, None, None
            
            # Create a new generator to iterate over the dataset.
            Xy_gen = create_Xy_generator_func()
            
            
            # Iterate over every chunk of examples in the dataset,
            # collect all examples that belong to that chunk,
            # then split that chunk into (mini-)batches
            # then train on every single batch.
            for chunk_index in range(0, count_chunks):
                # Collect X and y for the current chunk
                chunk_start = chunk_index * chunk_size
                chunk_end = min(count_all, (chunk_index+1)*chunk_size)
                is_last_chunk = (chunk_index == count_chunks - 1)
                count_to_load = chunk_end - chunk_start
                count_loaded = 0
                X = []
                y = []
                
                # Create X and y for this chunk
                #
                # Meanwhile, count the sizes of each X and y row
                # so that we can check (later) whether X and y rows
                # do not change their sizes between rows. Thats not
                # necessary, but helps to spot nasty bugs early.
                for (row_index, (X_row, y_row)) in enumerate(Xy_gen):
                    if epoch == 0:
                        if Xmins is None:
                            Xmins, Xmaxs, ymins, ymaxs = len(X_row), len(X_row), len(y_row), len(y_row)
                        else:
                            Xmins = min(Xmins, len(X_row))
                            Xmaxs = max(Xmaxs, len(X_row))
                            ymins = min(ymins, len(y_row))
                            ymaxs = max(ymaxs, len(y_row))
                    
                    X.append(X_row)
                    y.append(y_row)
                    count_loaded += 1
                    if count_loaded >= count_to_load:
                        break
                
                y = standardize_y(y)
                
                # Print an error if X or y rows change in size between rows.
                # (Otherwise we would get a not very helpful numpy error.)
                if Xmins != Xmaxs:
                    raise Exception("Row size mismatch for X: min %s vs max %s" % (str(Xmins), str(Xmaxs),))
                if ymins != ymaxs:
                    raise Exception("Row size mismatch for y: min %s vs max %s" % (str(ymins), str(ymaxs),))
                
                # Process the current chunk in batches,
                # similar to the normal fit()
                nb_batch = int(ceil(float(len(X)) / float(batch_size)))
                for batch_index in range(0, nb_batch):
                    batch_start = batch_index*batch_size
                    batch_end = min(len(X), (batch_index+1)*batch_size)
                    is_last_batch_in_chunk = (batch_index == nb_batch - 1)
                    loss = self._train(X[batch_start:batch_end], y[batch_start:batch_end])
                    if verbose:
                        if do_validate and (is_last_chunk and is_last_batch_in_chunk):
                            progbar.update(chunk_start + batch_end, [('loss', loss), ('val. loss', self.test(X_val, y_val))])
                        else:
                            progbar.update(chunk_start + batch_end, [('loss', loss)])
    
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
        
        nb_batch = int(ceil(len(X)/batch_size))
        for batch_index in range(0, nb_batch):
            batch_start = batch_index*batch_size
            batch_end = min(len(X), (batch_index+1)*batch_size)
            batch_preds = self._predict(X[batch_start:batch_end])

            if batch_index == 0:
                preds = np.zeros((len(X), batch_preds.shape[1]))
            preds[batch_start:batch_end] = batch_preds
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



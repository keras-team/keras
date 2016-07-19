from __future__ import print_function
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation


def test_multiprocessing_training():

    reached_end = False

    arr_data = np.random.randint(0,256, (500, 200))
    arr_labels = np.random.randint(0, 2, 500)

    def myGenerator():

        batch_size = 32
        n_samples = 500

        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(10, input_shape=(200, )))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adadelta')

    model.fit_generator(myGenerator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        nb_worker=4,
                        pickle_safe=True)

    model.fit_generator(myGenerator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)

    reached_end = True

    assert reached_end


def test_multiprocessing_training_fromfile():

    reached_end = False

    arr_data = np.random.randint(0,256, (500, 200))
    arr_labels = np.random.randint(0, 2, 500)
    np.savez("data.npz", **{"data": arr_data, "labels": arr_labels})

    def myGenerator():

        batch_size = 32
        n_samples = 500

        arr = np.load("data.npz")

        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr["data"][start: end]
            y = arr["labels"][start: end]
            yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(10, input_shape=(200, )))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adadelta')

    model.fit_generator(myGenerator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        nb_worker=2,
                        pickle_safe=True)

    model.fit_generator(myGenerator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)
    reached_end = True

    assert reached_end


def test_multiprocessing_predicting():

    reached_end = False

    arr_data = np.random.randint(0,256, (500, 200))

    def myGenerator():

        batch_size = 32
        n_samples = 500

        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            yield X

    # Build a NN
    model = Sequential()
    model.add(Dense(10, input_shape=(200, )))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adadelta')
    model.predict_generator(myGenerator(),
                            val_samples=320,
                            max_q_size=10,
                            nb_worker=2,
                            pickle_safe=True)
    model.predict_generator(myGenerator(),
                            val_samples=320,
                            max_q_size=10,
                            pickle_safe=False)
    reached_end = True

    assert reached_end


def test_multiprocessing_evaluating():

    reached_end = False

    arr_data = np.random.randint(0,256, (500, 200))
    arr_labels = np.random.randint(0, 2, 500)

    def myGenerator():

        batch_size = 32
        n_samples = 500

        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(10, input_shape=(200, )))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adadelta')

    model.evaluate_generator(myGenerator(),
                             val_samples=320,
                             max_q_size=10,
                             nb_worker=2,
                             pickle_safe=True)
    model.evaluate_generator(myGenerator(),
                             val_samples=320,
                             max_q_size=10,
                             pickle_safe=False)
    reached_end = True

    assert reached_end


if __name__ == '__main__':

    pytest.main([__file__])

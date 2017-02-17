from __future__ import print_function

import numpy as np
import pytest

from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils.test_utils import keras_test


batch_size = 32
n_samples = 500
good_batches = 5


@keras_test
def test_multiprocessing_training():

    arr_data = np.random.randint(0, 256, (n_samples, 2))
    arr_labels = np.random.randint(0, 2, n_samples)

    def my_generator():
        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    model.fit_generator(my_generator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        nb_worker=4,
                        pickle_safe=True)

    model.fit_generator(my_generator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)


@keras_test
def test_multiprocessing_training_fromfile():

    arr_data = np.random.randint(0, 256, (n_samples, 2))
    arr_labels = np.random.randint(0, 2, n_samples)
    np.savez("data.npz", **{"data": arr_data, "labels": arr_labels})

    def my_generator():
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
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    model.fit_generator(my_generator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        nb_worker=2,
                        pickle_safe=True)

    model.fit_generator(my_generator(),
                        samples_per_epoch=320,
                        nb_epoch=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)


@keras_test
def test_multiprocessing_predicting():

    arr_data = np.random.randint(0, 256, (n_samples, 2))

    def my_generator():
        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            yield X

    # Build a NN
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')
    model.predict_generator(my_generator(),
                            val_samples=320,
                            max_q_size=10,
                            nb_worker=2,
                            pickle_safe=True)
    model.predict_generator(my_generator(),
                            val_samples=320,
                            max_q_size=10,
                            pickle_safe=False)


@keras_test
def test_multiprocessing_evaluating():

    arr_data = np.random.randint(0, 256, (n_samples, 2))
    arr_labels = np.random.randint(0, 2, n_samples)

    def my_generator():
        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    model.evaluate_generator(my_generator(),
                             val_samples=320,
                             max_q_size=10,
                             nb_worker=2,
                             pickle_safe=True)
    model.evaluate_generator(my_generator(),
                             val_samples=320,
                             max_q_size=10,
                             pickle_safe=False)


@keras_test
def test_multiprocessing_fit_error():

    def my_generator():
        """Raises an exception after a few good batches"""
        for _ in range(good_batches):
            yield (np.random.randint(0, high=256, size=(batch_size, 2)),
                   np.random.randint(0, high=2, size=batch_size))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.fit_generator(
            my_generator(), samples, 1,
            nb_worker=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.fit_generator(
            my_generator(), samples, 1,
            pickle_safe=False,
        )


@keras_test
def test_multiprocessing_evaluate_error():

    def my_generator():
        """Raises an exception after a few good batches"""
        for _ in range(good_batches):
            yield (np.random.randint(0, high=256, size=(batch_size, 2)),
                   np.random.randint(0, high=2, size=batch_size))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.evaluate_generator(
            my_generator(), samples,
            nb_worker=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.evaluate_generator(
            my_generator(), samples,
            pickle_safe=False,
        )


@keras_test
def test_multiprocessing_predict_error():

    def my_generator():
        """Raises an exception after a few good batches"""
        for _ in range(good_batches):
            yield (np.random.randint(0, high=256, size=(batch_size, 2)),
                   np.random.randint(0, high=2, size=batch_size))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.predict_generator(
            my_generator(), samples,
            nb_worker=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.predict_generator(
            my_generator(), samples,
            pickle_safe=False,
        )


if __name__ == '__main__':
    pytest.main([__file__])

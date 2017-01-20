from __future__ import print_function
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import keras_test


@keras_test
def test_multiprocessing_training():

    reached_end = False

    arr_data = np.random.randint(0, 256, (500, 2))
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
    model.add(Dense(1, input_shape=(2, )))
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


@keras_test
def test_multiprocessing_training_fromfile():

    reached_end = False

    arr_data = np.random.randint(0, 256, (500, 2))
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
    model.add(Dense(1, input_shape=(2, )))
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


@keras_test
def test_multiprocessing_predicting():

    reached_end = False

    arr_data = np.random.randint(0, 256, (500, 2))

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
    model.add(Dense(1, input_shape=(2, )))
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


@keras_test
def test_multiprocessing_evaluating():

    reached_end = False

    arr_data = np.random.randint(0, 256, (500, 2))
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
    model.add(Dense(1, input_shape=(2, )))
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


@keras_test
def test_multiprocessing_fit_error():

    batch_size = 32
    good_batches = 5

    def myGenerator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(batch_size, 256, (500, 2)),
                   np.random.randint(batch_size, 2, 500))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.fit_generator(
            myGenerator(), samples, 1,
            nb_worker=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.fit_generator(
            myGenerator(), samples, 1,
            pickle_safe=False,
        )


@keras_test
def test_multiprocessing_evaluate_error():

    batch_size = 32
    good_batches = 5

    def myGenerator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(batch_size, 256, (500, 2)),
                   np.random.randint(batch_size, 2, 500))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.evaluate_generator(
            myGenerator(), samples, 1,
            nb_worker=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.evaluate_generator(
            myGenerator(), samples, 1,
            pickle_safe=False,
        )


@keras_test
def test_multiprocessing_predict_error():

    batch_size = 32
    good_batches = 5

    def myGenerator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(batch_size, 256, (500, 2)),
                   np.random.randint(batch_size, 2, 500))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.predict_generator(
            myGenerator(), samples, 1,
            nb_worker=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.predict_generator(
            myGenerator(), samples, 1,
            pickle_safe=False,
        )


if __name__ == '__main__':

    pytest.main([__file__])

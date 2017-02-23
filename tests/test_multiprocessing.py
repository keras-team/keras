from __future__ import print_function
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import keras_test


@keras_test
def test_multiprocessing_training():
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)

    def custom_generator():
        batch_size = 10
        n_samples = 50

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

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        workers=4,
                        pickle_safe=True)

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)


@keras_test
def test_multiprocessing_training_fromfile():
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)
    np.savez('data.npz', **{'data': arr_data, 'labels': arr_labels})

    def custom_generator():

        batch_size = 10
        n_samples = 50

        arr = np.load('data.npz')

        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr['data'][start: end]
            y = arr['labels'][start: end]
            yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        workers=2,
                        pickle_safe=True)

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)


@keras_test
def test_multiprocessing_predicting():
    arr_data = np.random.randint(0, 256, (50, 2))

    def custom_generator():
        batch_size = 10
        n_samples = 50

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
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_q_size=10,
                            workers=2,
                            pickle_safe=True)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_q_size=10,
                            pickle_safe=False)


@keras_test
def test_multiprocessing_evaluating():
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)

    def custom_generator():
        batch_size = 10
        n_samples = 50

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

    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_q_size=10,
                             workers=2,
                             pickle_safe=True)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_q_size=10,
                             pickle_safe=False)


@keras_test
def test_multiprocessing_fit_error():
    batch_size = 10
    good_batches = 3

    def custom_generator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(batch_size, 256, (50, 2)),
                   np.random.randint(batch_size, 2, 50))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    with pytest.raises(Exception):
        model.fit_generator(
            custom_generator(), samples, 1,
            workers=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.fit_generator(
            custom_generator(), samples, 1,
            pickle_safe=False,
        )


@keras_test
def test_multiprocessing_evaluate_error():
    batch_size = 10
    good_batches = 3

    def custom_generator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(batch_size, 256, (50, 2)),
                   np.random.randint(batch_size, 2, 50))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    with pytest.raises(Exception):
        model.evaluate_generator(
            custom_generator(), good_batches + 1, 1,
            workers=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.evaluate_generator(
            custom_generator(), good_batches + 1, 1,
            pickle_safe=False,
        )


@keras_test
def test_multiprocessing_predict_error():
    batch_size = 10
    good_batches = 3

    def custom_generator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(batch_size, 256, (50, 2)),
                   np.random.randint(batch_size, 2, 50))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    with pytest.raises(Exception):
        model.predict_generator(
            custom_generator(), good_batches + 1, 1,
            workers=4, pickle_safe=True,
        )

    with pytest.raises(Exception):
        model.predict_generator(
            custom_generator(), good_batches + 1, 1,
            pickle_safe=False,
        )


if __name__ == '__main__':
    pytest.main([__file__])

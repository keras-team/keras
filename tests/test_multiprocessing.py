from __future__ import print_function
import os
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import keras_test


@pytest.fixture
def in_tmpdir(tmpdir):
    """Runs a function in a temporary directory.

    Checks that the directory is empty afterwards.
    """
    with tmpdir.as_cwd():
        yield None
    assert not tmpdir.listdir()


@keras_test
def test_multiprocessing_training():
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)
    arr_weights = np.random.random(50)

    def custom_generator(use_weights=False):
        batch_size = 10
        n_samples = 50

        while True:
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            if use_weights:
                w = arr_weights[start: end]
                yield X, y, w
            else:
                yield X, y

    # Build a NN
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        workers=4,
                        use_multiprocessing=True)

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)

    model.fit_generator(custom_generator(True),
                        steps_per_epoch=5,
                        validation_data=(arr_data[:10],
                                         arr_labels[:10],
                                         arr_weights[:10]),
                        validation_steps=1)

    model.fit_generator(custom_generator(True),
                        steps_per_epoch=5,
                        validation_data=custom_generator(True),
                        validation_steps=1)

    # Test invalid use cases
    def invalid_generator():
        while True:
            yield arr_data[:10], arr_data[:10], arr_labels[:10], arr_labels[:10]

    # not specified `validation_steps`
    with pytest.raises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            validation_data=custom_generator())

    # validation data is neither a tuple nor a triple.
    with pytest.raises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            validation_data=(arr_data[:10],
                                             arr_data[:10],
                                             arr_labels[:10],
                                             arr_weights[:10]),
                            validation_steps=1)

    # validation generator is neither a tuple nor a triple.
    with pytest.raises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            validation_data=invalid_generator(),
                            validation_steps=1)


@keras_test
def test_multiprocessing_training_fromfile(in_tmpdir):
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
                        max_queue_size=10,
                        workers=2,
                        use_multiprocessing=True)

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)

    os.remove('data.npz')


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
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)


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
                             max_queue_size=10,
                             workers=2,
                             use_multiprocessing=True)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)


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

    with pytest.raises(StopIteration):
        model.fit_generator(
            custom_generator(), samples, 1,
            workers=4, use_multiprocessing=True,
        )

    with pytest.raises(StopIteration):
        model.fit_generator(
            custom_generator(), samples, 1,
            use_multiprocessing=False,
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

    with pytest.raises(StopIteration):
        model.evaluate_generator(
            custom_generator(), good_batches + 1, 1,
            workers=4, use_multiprocessing=True,
        )

    with pytest.raises(StopIteration):
        model.evaluate_generator(
            custom_generator(), good_batches + 1, 1,
            use_multiprocessing=False,
        )


@keras_test
def test_multiprocessing_predict_error():
    good_batches = 3
    workers = 4

    def custom_generator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            yield (np.random.randint(1, 256, size=(2, 5)),
                   np.random.randint(1, 256, size=(2, 5)))
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(5,)))
    model.compile(loss='mse', optimizer='adadelta')

    with pytest.raises(StopIteration):
        model.predict_generator(
            custom_generator(), good_batches * workers + 1, 1,
            workers=workers, use_multiprocessing=True,
        )

    with pytest.raises(StopIteration):
        model.predict_generator(
            custom_generator(), good_batches + 1, 1,
            use_multiprocessing=False,
        )


if __name__ == '__main__':
    pytest.main([__file__])

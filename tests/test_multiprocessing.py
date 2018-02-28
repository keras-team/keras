from __future__ import print_function
import os
import threading
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import keras_test
from keras.utils import Sequence

STEPS_PER_EPOCH = 100
STEPS = 100
WORKERS = 4


class DummySequence(Sequence):
    def __getitem__(self, idx):
        return np.zeros([10, 2]), np.ones([10])

    def __len__(self):
        return 10


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

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=STEPS_PER_EPOCH,
                                epochs=1,
                                verbose=1,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=WORKERS,
                                use_multiprocessing=True)
    else:
        model.fit_generator(custom_generator(),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=1,
                            verbose=1,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=WORKERS,
                            use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    model.fit_generator(custom_generator(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=1,
                        verbose=1,
                        validation_steps=None,
                        max_queue_size=10,
                        workers=WORKERS,
                        use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(True),
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_data=(arr_data[:10],
                                                 arr_labels[:10],
                                                 arr_weights[:10]),
                                validation_steps=1,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=True)
    else:
        model.fit_generator(custom_generator(True),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=(arr_data[:10],
                                             arr_labels[:10],
                                             arr_weights[:10]),
                            validation_steps=1,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    model.fit_generator(custom_generator(True),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=(arr_data[:10],
                                         arr_labels[:10],
                                         arr_weights[:10]),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(True),
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_data=custom_generator(True),
                                validation_steps=1,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=True)
    else:
        model.fit_generator(custom_generator(True),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=custom_generator(True),
                            validation_steps=1,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=True)

    # - Produce data on 1 worker thread AT A TIME, consume on main thread:
    #   - Worker threads for training and validation run generator SEQUENTIALLY
    model.fit_generator(custom_generator(True),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=custom_generator(True),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False)

    # - Produce and consume data without a queue on main thread
    #   - Make sure the value of `use_multiprocessing` is ignored
    model.fit_generator(custom_generator(True),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=custom_generator(True),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=True)
    model.fit_generator(custom_generator(True),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=custom_generator(True),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=False)

    # - For Sequence
    model.fit_generator(DummySequence(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=custom_generator(True),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=True)
    model.fit_generator(DummySequence(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=custom_generator(True),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=False)

    # Test invalid use cases
    def invalid_generator():
        while True:
            yield arr_data[:10], arr_data[:10], arr_labels[:10], arr_labels[:10]

    # not specified `validation_steps`
    with pytest.raises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=custom_generator(),
                            validation_steps=None,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)

    # validation data is neither a tuple nor a triple.
    with pytest.raises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=(arr_data[:10],
                                             arr_data[:10],
                                             arr_labels[:10],
                                             arr_weights[:10]),
                            validation_steps=1,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)

    # validation generator is neither a tuple nor a triple.
    with pytest.raises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=invalid_generator(),
                            validation_steps=1,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)


@keras_test
def test_multiprocessing_training_from_file(in_tmpdir):
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

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=STEPS_PER_EPOCH,
                                epochs=1,
                                verbose=1,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=WORKERS,
                                use_multiprocessing=True)
    else:
        model.fit_generator(custom_generator(),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=1,
                            verbose=1,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=WORKERS,
                            use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    model.fit_generator(custom_generator(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=1,
                        verbose=1,
                        validation_steps=None,
                        max_queue_size=10,
                        workers=WORKERS,
                        use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=STEPS_PER_EPOCH,
                                epochs=1,
                                verbose=1,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=True)
    else:
        model.fit_generator(custom_generator(),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=1,
                            verbose=1,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    model.fit_generator(custom_generator(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=1,
                        verbose=1,
                        validation_steps=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False)

    # - Produce and consume data without a queue on main thread
    #   - Make sure the value of `use_multiprocessing` is ignored
    model.fit_generator(custom_generator(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=1,
                        verbose=1,
                        validation_steps=None,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=True)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=1,
                        verbose=1,
                        validation_steps=None,
                        max_queue_size=10,
                        workers=0,
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

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `predict_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.predict_generator(custom_generator(),
                                    steps=STEPS,
                                    max_queue_size=10,
                                    workers=WORKERS,
                                    use_multiprocessing=True)
    else:
        model.predict_generator(custom_generator(),
                                steps=STEPS,
                                max_queue_size=10,
                                workers=WORKERS,
                                use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    model.predict_generator(custom_generator(),
                            steps=STEPS,
                            max_queue_size=10,
                            workers=WORKERS,
                            use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `predict_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.predict_generator(custom_generator(),
                                    steps=STEPS,
                                    max_queue_size=10,
                                    workers=1,
                                    use_multiprocessing=True)
    else:
        model.predict_generator(custom_generator(),
                                steps=STEPS,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    model.predict_generator(custom_generator(),
                            steps=STEPS,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)

    # - Main thread runs the generator without a queue
    #   - Make sure the value of `use_multiprocessing` is ignored
    model.predict_generator(custom_generator(),
                            steps=STEPS,
                            max_queue_size=10,
                            workers=0,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(),
                            steps=STEPS,
                            max_queue_size=10,
                            workers=0,
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

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries
    #       -> make sure `evaluate_generator()` raises raises ValueError
    #          exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.evaluate_generator(custom_generator(),
                                     steps=STEPS,
                                     max_queue_size=10,
                                     workers=WORKERS,
                                     use_multiprocessing=True)
    else:
        model.evaluate_generator(custom_generator(),
                                 steps=STEPS,
                                 max_queue_size=10,
                                 workers=WORKERS,
                                 use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    model.evaluate_generator(custom_generator(),
                             steps=STEPS,
                             max_queue_size=10,
                             workers=WORKERS,
                             use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `evaluate_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.evaluate_generator(custom_generator(),
                                     steps=STEPS,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=True)
    else:
        model.evaluate_generator(custom_generator(),
                                 steps=STEPS,
                                 max_queue_size=10,
                                 workers=1,
                                 use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    model.evaluate_generator(custom_generator(),
                             steps=STEPS,
                             max_queue_size=10,
                             workers=1,
                             use_multiprocessing=False)

    # - Produce and consume data without a queue on main thread
    #   - Make sure the value of `use_multiprocessing` is ignored
    model.evaluate_generator(custom_generator(),
                             steps=STEPS,
                             max_queue_size=10,
                             workers=0,
                             use_multiprocessing=True)
    model.evaluate_generator(custom_generator(),
                             steps=STEPS,
                             max_queue_size=10,
                             workers=0,
                             use_multiprocessing=False)


@keras_test
def test_multiprocessing_fit_error():
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)
    batch_size = 10
    n_samples = 50
    good_batches = 3

    def custom_generator(use_weights=False):
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            yield X, y
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    samples = batch_size * (good_batches + 1)

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    #   - On other platforms, make sure `RuntimeError` exception bubbles up
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=samples,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=WORKERS,
                                use_multiprocessing=True)
    else:
        with pytest.raises(RuntimeError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=samples,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=WORKERS,
                                use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=samples,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=WORKERS,
                            use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `fit_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    #   - On other platforms, make sure `RuntimeError` exception bubbles up
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=samples,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=True)
    else:
        with pytest.raises(RuntimeError):
            model.fit_generator(custom_generator(),
                                steps_per_epoch=samples,
                                validation_steps=None,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=samples,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)

    # - Produce and consume data without a queue on main thread
    #   - Make sure the value of `use_multiprocessing` is ignored
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=samples,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=0,
                            use_multiprocessing=True)
    with pytest.raises(RuntimeError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=samples,
                            validation_steps=None,
                            max_queue_size=10,
                            workers=0,
                            use_multiprocessing=False)


@keras_test
def test_multiprocessing_evaluate_error():
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)
    batch_size = 10
    n_samples = 50
    good_batches = 3

    def custom_generator():
        """Raises an exception after a few good batches"""
        for i in range(good_batches):
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            y = arr_labels[start: end]
            yield X, y
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `evaluate_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    #   - On other platforms, make sure `RuntimeError` exception bubbles up
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.evaluate_generator(custom_generator(),
                                     steps=good_batches * WORKERS + 1,
                                     max_queue_size=10,
                                     workers=WORKERS,
                                     use_multiprocessing=True)
    else:
        with pytest.raises(RuntimeError):
            model.evaluate_generator(custom_generator(),
                                     steps=good_batches * WORKERS + 1,
                                     max_queue_size=10,
                                     workers=WORKERS,
                                     use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.evaluate_generator(custom_generator(),
                                 steps=good_batches * WORKERS + 1,
                                 max_queue_size=10,
                                 workers=WORKERS,
                                 use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `evaluate_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    #   - On other platforms, make sure `RuntimeError` exception bubbles up
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.evaluate_generator(custom_generator(),
                                     steps=good_batches + 1,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=True)
    else:
        with pytest.raises(RuntimeError):
            model.evaluate_generator(custom_generator(),
                                     steps=good_batches + 1,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.evaluate_generator(custom_generator(),
                                 steps=good_batches + 1,
                                 max_queue_size=10,
                                 workers=1,
                                 use_multiprocessing=False)

    # - Produce and consume data without a queue on main thread
    #   - Make sure the value of `use_multiprocessing` is ignored
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.evaluate_generator(custom_generator(),
                                 steps=good_batches + 1,
                                 max_queue_size=10,
                                 workers=0,
                                 use_multiprocessing=True)
    with pytest.raises(RuntimeError):
        model.evaluate_generator(custom_generator(),
                                 steps=good_batches + 1,
                                 max_queue_size=10,
                                 workers=0,
                                 use_multiprocessing=False)


@keras_test
def test_multiprocessing_predict_error():
    arr_data = np.random.randint(0, 256, (50, 2))
    good_batches = 3

    def custom_generator():
        """Raises an exception after a few good batches"""
        batch_size = 10
        n_samples = 50

        for i in range(good_batches):
            batch_index = np.random.randint(0, n_samples - batch_size)
            start = batch_index
            end = start + batch_size
            X = arr_data[start: end]
            yield X
        raise RuntimeError

    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.compile(loss='mse', optimizer='adadelta')

    # - Produce data on 4 worker processes, consume on main process:
    #   - Each worker process runs OWN copy of generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `predict_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    #   - On other platforms, make sure `RuntimeError` exception bubbles up
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.predict_generator(custom_generator(),
                                    steps=good_batches * WORKERS + 1,
                                    max_queue_size=10,
                                    workers=WORKERS,
                                    use_multiprocessing=True)
    else:
        with pytest.raises(RuntimeError):
            model.predict_generator(custom_generator(),
                                    steps=good_batches * WORKERS + 1,
                                    max_queue_size=10,
                                    workers=WORKERS,
                                    use_multiprocessing=True)

    # - Produce data on 4 worker threads, consume on main thread:
    #   - All worker threads share the SAME generator
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.predict_generator(custom_generator(),
                                steps=good_batches * WORKERS + 1,
                                max_queue_size=10,
                                workers=WORKERS,
                                use_multiprocessing=False)

    # - Produce data on 1 worker process, consume on main process:
    #   - Worker process runs generator
    #   - BUT on Windows, `multiprocessing` won't marshall generators across
    #     process boundaries -> make sure `predict_generator()` raises ValueError
    #     exception and does not attempt to run the generator.
    #   - On other platforms, make sure `RuntimeError` exception bubbles up
    if os.name is 'nt':
        with pytest.raises(ValueError):
            model.predict_generator(custom_generator(),
                                    steps=good_batches + 1,
                                    max_queue_size=10,
                                    workers=1,
                                    use_multiprocessing=True)
    else:
        with pytest.raises(RuntimeError):
            model.predict_generator(custom_generator(),
                                    steps=good_batches + 1,
                                    max_queue_size=10,
                                    workers=1,
                                    use_multiprocessing=True)

    # - Produce data on 1 worker thread, consume on main thread:
    #   - Worker thread is the only thread running the generator
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.predict_generator(custom_generator(),
                                steps=good_batches + 1,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=False)

    # - Produce and consume data without a queue on main thread
    #   - Make sure the value of `use_multiprocessing` is ignored
    #   - Make sure `RuntimeError` exception bubbles up
    with pytest.raises(RuntimeError):
        model.predict_generator(custom_generator(),
                                steps=good_batches + 1,
                                max_queue_size=10,
                                workers=0,
                                use_multiprocessing=True)
    with pytest.raises(RuntimeError):
        model.predict_generator(custom_generator(),
                                steps=good_batches + 1,
                                max_queue_size=10,
                                workers=0,
                                use_multiprocessing=False)

if __name__ == '__main__':
    pytest.main([__file__])

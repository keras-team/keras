import itertools
import multiprocessing.dummy
import queue
import random
import threading
import warnings
import weakref
from contextlib import closing

import numpy as np

from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


@keras_export(["keras.utils.PyDataset", "keras.utils.Sequence"])
class PyDataset:
    """Base class for defining a parallel dataset using Python code.

    Every `PyDataset` must implement the `__getitem__()` and the `__len__()`
    methods. If you want to modify your dataset between epochs,
    you may additionally implement `on_epoch_end()`,
    or `on_epoch_begin` to be called at the start of each epoch.
    The `__getitem__()` method should return a complete batch
    (not a single sample), and the `__len__` method should return
    the number of batches in the dataset (rather than the number of samples).

    Args:
        workers: Number of workers to use in multithreading or
            multiprocessing.
        use_multiprocessing: Whether to use Python multiprocessing for
            parallelism. Setting this to `True` means that your
            dataset will be replicated in multiple forked processes.
            This is necessary to gain compute-level (rather than I/O level)
            benefits from parallelism. However it can only be set to
            `True` if your dataset can be safely pickled.
        max_queue_size: Maximum number of batches to keep in the queue
            when iterating over the dataset in a multithreaded or
            multiprocessed setting.
            Reduce this value to reduce the CPU memory consumption of
            your dataset. Defaults to 10.

    Notes:

    - `PyDataset` is a safer way to do multiprocessing.
        This structure guarantees that the model will only train
        once on each sample per epoch, which is not the case
        with Python generators.
    - The arguments `workers`, `use_multiprocessing`, and `max_queue_size`
        exist to configure how `fit()` uses parallelism to iterate
        over the dataset. They are not being used by the `PyDataset` class
        directly. When you are manually iterating over a `PyDataset`,
        no parallelism is applied.

    Example:

    ```python
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    import math

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    class CIFAR10PyDataset(keras.utils.PyDataset):

        def __init__(self, x_set, y_set, batch_size, **kwargs):
            super().__init__(**kwargs)
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            # Return number of batches.
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            # Return x, y for batch idx.
            low = idx * self.batch_size
            # Cap upper bound at array length; the last batch may be smaller
            # if the total number of items is not a multiple of batch size.
            high = min(low + self.batch_size, len(self.x))
            batch_x = self.x[low:high]
            batch_y = self.y[low:high]

            return np.array([
                resize(imread(file_name), (200, 200))
                   for file_name in batch_x]), np.array(batch_y)
    ```
    """

    def __init__(self, workers=1, use_multiprocessing=False, max_queue_size=10):
        self._workers = workers
        self._use_multiprocessing = use_multiprocessing
        self._max_queue_size = max_queue_size

    def _warn_if_super_not_called(self):
        warn = False
        if not hasattr(self, "_workers"):
            self._workers = 1
            warn = True
        if not hasattr(self, "_use_multiprocessing"):
            self._use_multiprocessing = False
            warn = True
        if not hasattr(self, "_max_queue_size"):
            self._max_queue_size = 10
            warn = True
        if warn:
            warnings.warn(
                "Your `PyDataset` class should call "
                "`super().__init__(**kwargs)` in its constructor. "
                "`**kwargs` can include `workers`, "
                "`use_multiprocessing`, `max_queue_size`. Do not pass "
                "these arguments to `fit()`, as they will be ignored.",
                stacklevel=2,
            )

    @property
    def workers(self):
        self._warn_if_super_not_called()
        return self._workers

    @workers.setter
    def workers(self, value):
        self._workers = value

    @property
    def use_multiprocessing(self):
        self._warn_if_super_not_called()
        return self._use_multiprocessing

    @use_multiprocessing.setter
    def use_multiprocessing(self, value):
        self._use_multiprocessing = value

    @property
    def max_queue_size(self):
        self._warn_if_super_not_called()
        return self._max_queue_size

    @max_queue_size.setter
    def max_queue_size(self, value):
        self._max_queue_size = value

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Args:
            index: position of the batch in the PyDataset.

        Returns:
            A batch
        """
        del index
        raise NotImplementedError

    def __iter__(self):
        index_range = None
        try:
            num_batches = self.num_batches
            if num_batches is not None:
                index_range = range(num_batches)
        except NotImplementedError:
            pass

        if index_range is None:
            index_range = itertools.count()

        for index in index_range:
            yield self[index]

    @property
    def num_batches(self):
        """Number of batches in the PyDataset.

        Returns:
            The number of batches in the PyDataset or `None` to indicate that
            the dataset is infinite.
        """
        # For backwards compatibility, support `__len__`.
        if hasattr(self, "__len__"):
            return len(self)
        raise NotImplementedError(
            "You need to implement the `num_batches` property:\n\n"
            "@property\ndef num_batches(self):\n  return ..."
        )

    def on_epoch_begin(self):
        """Method called at the beginning of every epoch."""
        pass

    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        pass


class PyDatasetAdapter(DataAdapter):
    """Adapter for `keras.utils.PyDataset` instances."""

    def __init__(
        self,
        x,
        class_weight=None,
        shuffle=False,
    ):
        self.py_dataset = x
        self.class_weight = class_weight
        self.enqueuer = None
        self.shuffle = shuffle
        self._output_signature = None
        self._within_epoch = False

        workers = self.py_dataset.workers
        use_multiprocessing = self.py_dataset.use_multiprocessing
        if workers > 1 or (workers > 0 and use_multiprocessing):
            self.enqueuer = OrderedEnqueuer(
                self.py_dataset,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                max_queue_size=self.py_dataset.max_queue_size,
                shuffle=self.shuffle,
            )

    def _standardize_batch(self, batch):
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, np.ndarray):
            batch = (batch,)
        if isinstance(batch, list):
            batch = tuple(batch)
        if not isinstance(batch, tuple) or len(batch) not in {1, 2, 3}:
            raise ValueError(
                "PyDataset.__getitem__() must return a tuple or a dict. "
                "If a tuple, it must be ordered either "
                "(input,) or (inputs, targets) or "
                "(inputs, targets, sample_weights). "
                f"Received: {str(batch)[:100]}... of type {type(batch)}"
            )
        if self.class_weight is not None:
            if len(batch) == 3:
                raise ValueError(
                    "You cannot specify `class_weight` "
                    "and `sample_weight` at the same time."
                )
            if len(batch) == 2:
                sw = data_adapter_utils.class_weight_to_sample_weights(
                    batch[1], self.class_weight
                )
                batch = batch + (sw,)
        return batch

    def _infinite_generator(self):
        for i in itertools.count():
            yield self._standardize_batch(self.py_dataset[i])

    def _finite_generator(self):
        indices = range(self.py_dataset.num_batches)
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)

        for i in indices:
            yield self._standardize_batch(self.py_dataset[i])

    def _infinite_enqueuer_generator(self):
        self.enqueuer.start()
        for batch in self.enqueuer.get():
            yield self._standardize_batch(batch)

    def _finite_enqueuer_generator(self):
        self.enqueuer.start()
        num_batches = self.py_dataset.num_batches
        for i, batch in enumerate(self.enqueuer.get()):
            yield self._standardize_batch(batch)
            if i >= num_batches - 1:
                self.enqueuer.stop()
                return

    def _get_iterator(self):
        if self.enqueuer is None:
            if self.py_dataset.num_batches is None:
                return self._infinite_generator()
            else:
                return self._finite_generator()
        else:
            if self.py_dataset.num_batches is None:
                return self._infinite_enqueuer_generator()
            else:
                return self._finite_enqueuer_generator()

    def get_numpy_iterator(self):
        return data_adapter_utils.get_numpy_iterator(self._get_iterator())

    def get_jax_iterator(self):
        return data_adapter_utils.get_jax_iterator(self._get_iterator())

    def get_tf_dataset(self):
        from keras.src.utils.module_utils import tensorflow as tf

        num_batches = self.py_dataset.num_batches
        if self._output_signature is None:
            num_samples = data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC
            if num_batches is not None:
                num_samples = min(num_samples, num_batches)
            batches = [
                self._standardize_batch(self.py_dataset[i])
                for i in range(num_samples)
            ]
            if len(batches) == 0:
                raise ValueError("The PyDataset has length 0")
            self._output_signature = data_adapter_utils.get_tensor_spec(batches)

        ds = tf.data.Dataset.from_generator(
            self._get_iterator,
            output_signature=self._output_signature,
        )
        if self.enqueuer is not None:
            # The enqueuer does its own multithreading / multiprocesssing to
            # prefetch items. Disable the tf.data.Dataset prefetching and
            # threading as it interferes.
            options = tf.data.Options()
            options.autotune.enabled = False
            options.threading.private_threadpool_size = 1
            ds = ds.with_options(options)
        else:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_torch_dataloader(self):
        return data_adapter_utils.get_torch_dataloader(self._get_iterator())

    def on_epoch_begin(self):
        if self._within_epoch:
            raise ValueError(
                "`on_epoch_begin` was called twice without `on_epoch_end` "
                "having been called."
            )
        self._within_epoch = True
        if self.enqueuer:
            self.enqueuer.start()
        self.py_dataset.on_epoch_begin()

    def on_epoch_end(self):
        if self.enqueuer:
            self.enqueuer.stop()
        self.py_dataset.on_epoch_end()
        self._within_epoch = False

    @property
    def num_batches(self):
        return self.py_dataset.num_batches

    @property
    def batch_size(self):
        return None


# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None


# Because multiprocessing pools are inherently unsafe, starting from a clean
# state can be essential to avoiding deadlocks. In order to accomplish this, we
# need to be able to check on the status of Pools that we create.
_DATA_POOLS = weakref.WeakSet()
_WORKER_ID_QUEUE = None  # Only created if needed.
_FORCE_THREADPOOL = False


def get_pool_class(use_multiprocessing):
    global _FORCE_THREADPOOL
    if not use_multiprocessing or _FORCE_THREADPOOL:
        return multiprocessing.dummy.Pool  # ThreadPool
    return multiprocessing.Pool


def get_worker_id_queue():
    """Lazily create the queue to track worker ids."""
    global _WORKER_ID_QUEUE
    if _WORKER_ID_QUEUE is None:
        _WORKER_ID_QUEUE = multiprocessing.Queue()
    return _WORKER_ID_QUEUE


def get_index(uid, i):
    """Get the value from the PyDataset `uid` at index `i`.

    To allow multiple PyDatasets to be used at the same time, we use `uid` to
    get a specific one. A single PyDataset would cause the validation to
    overwrite the training PyDataset.

    This methods is called from worker threads.

    Args:
        uid: int, PyDataset identifier
        i: index

    Returns:
        The value at index `i`.
    """
    return _SHARED_SEQUENCES[uid][i]


class PyDatasetEnqueuer:
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    Example:

    ```python
        enqueuer = PyDatasetEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.stop()
    ```

    The `enqueuer.get()` should be an infinite stream of data.
    """

    def __init__(
        self,
        py_dataset,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    ):
        self.py_dataset = py_dataset

        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value("i", 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.ready_queue = queue.Queue()
        self.future_queue = queue.Queue(max_queue_size)
        self.running = False
        self.start_stop_lock = threading.Lock()
        self.run_thread = None
        if use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            # We do not need the init since it's threads.
            self.executor_fn = lambda _: get_pool_class(False)(workers)

    def is_running(self):
        """Whether the enqueuer is running.

        This method is thread safe and called from many threads.

        Returns: boolean indicating whether this enqueuer is running.
        """
        return self.running

    def start(self):
        """Starts the handler's workers.

        This method is thread safe but is called from the main thread.
        It is safe to call this method multiple times, extra calls are ignored.
        """
        with self.start_stop_lock:
            if self.running:
                return
            self.running = True
            self.run_thread = threading.Thread(target=self._run)
            self.run_thread.name = f"Worker_{self.uid}"
            self.run_thread.daemon = True
            self.run_thread.start()

    def stop(self, drain_queue_and_join=True):
        """Stops running threads and wait for them to exit, if necessary.

        This method is thread safe and is called from various threads. Note that
        the `drain_queue_and_join` argument must be set correctly.
        It is safe to call this method multiple times, extra calls are ignored.

        Args:
            drain_queue_and_join: set to True to drain the queue of pending
                items and wait for the worker thread to complete. Set to False
                if invoked from a worker thread to avoid deadlocks. Note that
                setting this to False means this enqueuer won't be reused.
        """
        with self.start_stop_lock:
            if not self.running:
                return
            self.running = False

            if drain_queue_and_join:
                # Drain the `future_queue` and put items in `ready_queue` for
                # the next run.
                while True:
                    try:
                        value = self.future_queue.get(block=True, timeout=0.1)
                        if isinstance(value, Exception):
                            raise value  # Propagate exception from other thread
                        inputs = value.get()
                        self.future_queue.task_done()
                        if inputs is not None:
                            self.ready_queue.put(inputs)
                    except queue.Empty:
                        break
                self.run_thread.join()

            self.run_thread = None
            _SHARED_SEQUENCES[self.uid] = None

    def _send_py_dataset(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.py_dataset

    def __del__(self):
        self.stop(drain_queue_and_join=False)

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        raise NotImplementedError

    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.

        Args:
            workers: Number of workers.

        Returns:
            Function, a Function to initialize the pool
        """
        raise NotImplementedError

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        This method is called from the main thread.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError


class OrderedEnqueuer(PyDatasetEnqueuer):
    """Builds a Enqueuer from a PyDataset.

    Args:
        py_dataset: A `keras.utils.PyDataset` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(
        self,
        py_dataset,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        shuffle=False,
    ):
        super().__init__(
            py_dataset, workers, use_multiprocessing, max_queue_size
        )
        self.shuffle = shuffle
        if self.py_dataset.num_batches is None:
            # For infinite datasets, `self.indices` is created here once for all
            # so that subsequent runs resume from where they stopped.
            self.indices = itertools.count()

    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.

        Args:
            workers: Number of workers.

        Returns:
            Function, a Function to initialize the pool
        """

        def pool_fn(seqs):
            pool = get_pool_class(True)(
                workers,
                initializer=init_pool_generator,
                initargs=(seqs, None, get_worker_id_queue()),
            )
            _DATA_POOLS.add(pool)
            return pool

        return pool_fn

    def _run(self):
        """Submits request to the executor and queue the `Future` objects.

        This method is the run method of worker threads.
        """
        try:
            if self.py_dataset.num_batches is not None:
                # For finite datasets, `self.indices` is created here so that
                # shuffling creates different a order each time.
                indices = range(self.py_dataset.num_batches)
                if self.shuffle:
                    indices = list(indices)
                    random.shuffle(indices)
                self.indices = iter(indices)
            self._send_py_dataset()  # Share the initial py_dataset

            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                while self.is_running():
                    try:
                        i = next(self.indices)
                        self.future_queue.put(
                            executor.apply_async(get_index, (self.uid, i)),
                            block=True,
                        )
                    except StopIteration:
                        break
        except Exception as e:
            self.future_queue.put(e)  # Report exception

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        This method is called from the main thread.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            try:
                inputs = self.ready_queue.get(block=False)
                yield inputs
                continue  # Retry the ready_queue
            except queue.Empty:
                pass

            try:
                value = self.future_queue.get(block=True, timeout=5)
                self.future_queue.task_done()
                if isinstance(value, Exception):
                    raise value  # Propagate exception from other thread
                inputs = value.get()
                if inputs is not None:
                    yield inputs
            except queue.Empty:
                pass
            except Exception as e:
                self.stop(drain_queue_and_join=True)
                raise e

        # Note that it is ok to poll the iterator after the initial `start`,
        # which may happen before the first `on_epoch_begin`. But it's not ok to
        # poll after `on_epoch_end`.
        raise ValueError(
            "Iterator called after `on_epoch_end` or before `on_epoch_begin`."
        )


def init_pool_generator(gens, random_seed=None, id_queue=None):
    """Initializer function for pool workers.

    Args:
        gens: State which should be made available to worker processes.
        random_seed: An optional value with which to seed child processes.
        id_queue: A multiprocessing Queue of worker ids.
            This is used to indicate that a worker process
            was created by Keras.
    """
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = gens

    worker_proc = multiprocessing.current_process()

    # name isn't used for anything, but setting a more descriptive name is
    # helpful when diagnosing orphaned processes.
    worker_proc.name = f"Keras_worker_{worker_proc.name}"

    if random_seed is not None:
        np.random.seed(random_seed + worker_proc.ident)

    if id_queue is not None:
        # If a worker dies during init, the pool will just create a replacement.
        id_queue.put(worker_proc.ident, block=True, timeout=0.1)

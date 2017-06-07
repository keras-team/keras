import queue
import random
import threading
from abc import abstractmethod
import concurrent.futures
import itertools
import time
import numpy as np

import multiprocessing

"""Get the uid for the default graph.

# Arguments
    prefix: An optional prefix of the graph.

# Returns
    A unique identifier for the graph.
"""


class DatasetHandler():
    """Base class to enqueue datasets."""
    @abstractmethod
    def is_running(self):
        raise NotImplemented

    @abstractmethod
    def start(self, workers=1, max_q_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
        """
        raise NotImplemented

    @abstractmethod
    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        raise NotImplemented

    @abstractmethod
    def get(self):
        """Create a generator to extract data from the queue.

        #Returns
            A generator
        """
        raise NotImplemented


class OrderedEnqueuer(DatasetHandler):
    """Builds a Enqueuer from a Dataset.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        dataset: A `keras.data.dataset.Dataset` object.
        pickle_safe: use multiprocessing if True, otherwise threading
        scheduling: Sequential querying of datas if 'sequential', random otherwise.
    """

    def __init__(self, dataset, pickle_safe=False, scheduling='sequential'):
        self.dataset = dataset
        self.pickle_safe = pickle_safe
        self.scheduling = scheduling
        self.workers = 0
        self.executor = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_q_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, workers could block on put())
        """
        if self.pickle_safe:
            self.executor = concurrent.futures.ProcessPoolExecutor(workers)
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(workers)
        self.queue = queue.Queue(max_q_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _run(self):
        """ Function to submit request to the executor and queue the `Future` objects."""
        indexes = range(len(self.dataset))
        if self.scheduling is not 'sequential':
            random.shuffle(indexes)
        indexes = itertools.cycle(indexes)
        for i in indexes:
            if self.stop_signal.is_set():
                return
            self.queue.put(self.executor.submit(self.dataset.__getitem__, [i]), block=True)

    def get(self):
        """Create a generator to extract data from the queue.

        #Returns
            A generator
        """
        try:
            while self.is_running():
                yield self.queue.get(block=True).result()
        except Exception as e:
            self.stop()
            raise StopIteration(e)

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.executor.shutdown(False)
        self.run_thread.join(timeout)


class GeneratorEnqueuer(DatasetHandler):
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to put()
    """

    def __init__(self, generator, pickle_safe=False, wait_time=0.05):
        self.wait_time = wait_time
        self._generator = generator
        self._pickle_safe = pickle_safe
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.wait_time = 0.0

    def start(self, workers=1, max_q_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._pickle_safe or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._pickle_safe:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._pickle_safe:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed()
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._pickle_safe:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._pickle_safe:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        while self.is_running():
            if not self.queue.empty():
                yield self.queue.get()
            else:
                time.sleep(self.wait_time)

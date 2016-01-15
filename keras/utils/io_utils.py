from __future__ import absolute_import
import h5py
import itertools
import multiprocessing
import threading
import numpy as np
from collections import defaultdict
from multiprocessing import Queue
from Queue import Empty as QueueEmpty



class HDF5Matrix():
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start, end, normalizer=None):
        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.start = start
        self.end = end
        self.data = f[dataset]
        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start+self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key+self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        if self.normalizer is not None:
            return self.normalizer(self.data[idx])
        else:
            return self.data[idx]

    @property
    def shape(self):
        return tuple([self.end - self.start, self.data.shape[1]])


def save_array(array, name):
    import tables
    f = tables.open_file(name, 'w')
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.createCArray(f.root, 'data', atom, array.shape)
    ds[:] = array
    f.close()


def load_array(name):
    import tables
    f = tables.open_file(name)
    array = f.root.data
    a = np.empty(shape=array.shape, dtype=array.dtype)
    a[:] = array[:]
    f.close()
    return a

def generate_infinit_iterators(iterator_constructor, *args, **kwargs):
    """
    Create generator of iterators
    Useful for NN training with fit_on_generator and ParallelizeIt 
    # Arguments
      iterator_constructor: constructor of iterator
      *args: arguments to be passed to constructor
      **kwargs: keyword arguments to be passed to constructor
    # Returns
      generator of new instances of iterators created by calling iterator_constructor

    # Example:
    ```python
    #this would create infinite iterator of iterators on list [0,1,2,3]
    keras.utils.io_utils.GenerateInfinitIterators(iter, [0,1,2,3])
    ```
    """
    return (iterator_constructor(*args, **kwargs) for _ in itertools.count())

def isplit_every(n, it):
    """
    Lazy split of iterator to subset of n items
    # Arguments
      n: number of items in split
      it: iterator to split
    # Returns
      generator of iterators
    # Example
    ```python
    #create iterator over infinit iterator, splitted by 10 items
    it = keras.utils.io_utils.isplit_every(10, itertools.count())
    ```
    """
    return itertools.takewhile(bool, (itertools.islice(it, n) for _ in itertools.count(0)))


class ParallelizeIt(object):
    """
    Parallelize iterable/iterator reading via threading/multiprocessing
    Synchronization is done via Queue object.
    # Arguments
      iterator: iterable or iterator to parallelize.
      max_records_to_read: int, maximum number of items to read from iterator.
      queue_size: int, default 1000, size of a queue buffer for preloading of items.
      threaded: boolean, default True. If True, use threading module, 
        otherwise use multiprocessing
    # Returns
      Iterator over `iterator` argument
    # Example  
    ```python
    import numpy as np

    l = np.arange(10000)
    ll = keras.utils.io_utils.ParallelizeIt(l, queue_size=2)
    for a in ll:
      print "Next is:", a

    #for usage with model.fit_on_generator create infinit generator of ParallelizeIt objects
    import numpy as np

    l = np.arange(10000)
    ll = keras.utils.io_utils.generate_infinit_iterators(keras.utils.io_utils.ParallelizeIt, l, queue_size=2)
    for lll in ll:
      for a in lll:
        print "Next is:", a
    ```
    """
    def read_worker(self):
        for i, r in enumerate(self._iterator):      
            if self._closed.is_set() or self._max_records_to_read == i:
                break
            self._queue.put(r)
        #put sentinel to signalize end of queue
        self._queue.put(self._sentinel)

    def __init__(self, iterator, max_records_to_read=None, queue_size=1000, threaded=True):
        self._iterator = iter(iterator)
        self._queue = multiprocessing.Queue(maxsize=queue_size)
        self._max_records_to_read = max_records_to_read
        self._threaded = threaded
        self._sentinel = 'QueueSentinel'
        if self._threaded:
            self._w = threading.Thread(target=self.read_worker, args=())
            self._w.daemon = True
            self._closed = threading.Event()
        else:
            self._w = multiprocessing.Process(target=self.read_worker, args=())
            self._closed = multiprocessing.Event()
        self._initialized = False

    def __iter__(self):
        self._open()
        return self

    def __enter__(self):
        self._open()
        return self

    def next(self):
        try:
            i = self._queue.get()
            if i == self._sentinel:
                self._close()
            else:
                return i
        except KeyboardInterrupt:
            self._close()

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _open(self):
        if self._closed.is_set():
            raise StopIteration
        if self._initialized == False:
            self._w.start()
        self._initialized = True


    def _close(self):
        if not self._closed.is_set():
            self._closed.set()
            try:
            #for gratefull shutdown on interuption
            # kick the worker to free up one space in queue and after another 
            # call to worker shutdown its thread
                self._queue.get_nowait()
            except QueueEmpty as e:
                pass
        raise StopIteration

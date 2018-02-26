"""Tests for functions in data_utils.py.
"""
import os
import sys
import tarfile
import threading
import zipfile
from itertools import cycle
import multiprocessing as mp
import numpy as np
import pytest
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.request import pathname2url

from keras.utils import GeneratorEnqueuer
from keras.utils import OrderedEnqueuer
from keras.utils import Sequence
from keras.utils.data_utils import _hash_file
from keras.utils.data_utils import get_file
from keras.utils.data_utils import validate_file

if sys.version_info < (3,):
    def next(x):
        return x.next()


def use_spawn(func):
    """Decorator to test both Unix (fork) and Windows (spawn)"""
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        if sys.version_info > (3, 4):
            mp.set_start_method('spawn', force=True)
            func(*args, **kwargs)
            mp.set_start_method('fork', force=True)
        return out
    return wrapper


@pytest.fixture
def in_tmpdir(tmpdir):
    """Runs a function in a temporary directory.

    Checks that the directory is empty afterwards.
    """
    with tmpdir.as_cwd():
        yield None
    assert not tmpdir.listdir()


def test_data_utils(in_tmpdir):
    """Tests get_file from a url, plus extraction and validation.
    """
    dirname = 'data_utils'

    with open('test.txt', 'w') as text_file:
        text_file.write('Float like a butterfly, sting like a bee.')

    with tarfile.open('test.tar.gz', 'w:gz') as tar_file:
        tar_file.add('test.txt')

    with zipfile.ZipFile('test.zip', 'w') as zip_file:
        zip_file.write('test.txt')

    origin = urljoin('file://', pathname2url(os.path.abspath('test.tar.gz')))

    path = get_file(dirname, origin, untar=True)
    filepath = path + '.tar.gz'
    hashval_sha256 = _hash_file(filepath)
    hashval_md5 = _hash_file(filepath, algorithm='md5')
    path = get_file(dirname, origin, md5_hash=hashval_md5, untar=True)
    path = get_file(filepath, origin, file_hash=hashval_sha256, extract=True)
    assert os.path.exists(filepath)
    assert validate_file(filepath, hashval_sha256)
    assert validate_file(filepath, hashval_md5)
    os.remove(filepath)
    os.remove('test.tar.gz')

    origin = urljoin('file://', pathname2url(os.path.abspath('test.zip')))

    hashval_sha256 = _hash_file('test.zip')
    hashval_md5 = _hash_file('test.zip', algorithm='md5')
    path = get_file(dirname, origin, md5_hash=hashval_md5, extract=True)
    path = get_file(dirname, origin, file_hash=hashval_sha256, extract=True)
    assert os.path.exists(path)
    assert validate_file(path, hashval_sha256)
    assert validate_file(path, hashval_md5)

    os.remove(path)
    os.remove('test.txt')
    os.remove('test.zip')


"""Enqueuers Tests"""


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class DummySequence(Sequence):
    def __init__(self, shape, value=1.0):
        self.shape = shape
        self.inner = value

    def __getitem__(self, item):
        return np.ones(self.shape, dtype=np.uint32) * item * self.inner

    def __len__(self):
        return 100

    def on_epoch_end(self):
        self.inner *= 5.0


class FaultSequence(Sequence):
    def __getitem__(self, item):
        raise IndexError(item, 'is not present')

    def __len__(self):
        return 100

    def on_epoch_end(self):
        pass


@threadsafe_generator
def create_generator_from_sequence_threads(ds):
    for i in cycle(range(len(ds))):
        yield ds[i]


def create_generator_from_sequence_pcs(ds):
    for i in cycle(range(len(ds))):
        yield ds[i]


def test_generator_enqueuer_threads():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_threads(
        DummySequence([3, 200, 200, 3])), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(int(next(gen_output)[0, 0, 0, 0]))

    """
     Not comparing the order since it is not guaranteed.
     It may get ordered, but not a lot, one thread can take the GIL before he was supposed to.
    """
    assert len(set(acc) - set(range(100))) == 0, "Output is not the same"
    enqueuer.stop()


def test_generator_enqueuer_processes():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_pcs(
        DummySequence([3, 200, 200, 3])), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(int(next(gen_output)[0, 0, 0, 0]))
    assert acc != list(range(100)), "Order was keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


def test_generator_enqueuer_fail_threads():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_threads(
        FaultSequence()), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(IndexError):
        next(gen_output)


def test_generator_enqueuer_fail_processes():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_pcs(
        FaultSequence()), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(IndexError):
        next(gen_output)


def test_ordered_enqueuer_threads():
    enqueuer = OrderedEnqueuer(DummySequence([3, 200, 200, 3]), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc == list(range(100)), "Order was not keep in GeneratorEnqueuer with threads"
    enqueuer.stop()


def test_ordered_enqueuer_threads_not_ordered():
    enqueuer = OrderedEnqueuer(DummySequence([3, 200, 200, 3]),
                               use_multiprocessing=False,
                               shuffle=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc != list(range(100)), "Order was not keep in GeneratorEnqueuer with threads"
    enqueuer.stop()


@use_spawn
def test_ordered_enqueuer_processes():
    enqueuer = OrderedEnqueuer(DummySequence([3, 200, 200, 3]), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc == list(range(100)), "Order was not keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


def test_ordered_enqueuer_fail_threads():
    enqueuer = OrderedEnqueuer(FaultSequence(), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


@use_spawn
def test_on_epoch_end_processes():
    enqueuer = OrderedEnqueuer(DummySequence([3, 200, 200, 3]), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(200):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc[100:] == list([k * 5 for k in range(100)]), "Order was not keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


@use_spawn
def test_context_switch():
    enqueuer = OrderedEnqueuer(DummySequence([3, 200, 200, 3]), use_multiprocessing=True)
    enqueuer2 = OrderedEnqueuer(DummySequence([3, 200, 200, 3], value=15), use_multiprocessing=True)
    enqueuer.start(3, 10)
    enqueuer2.start(3, 10)
    gen_output = enqueuer.get()
    gen_output2 = enqueuer2.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc[-1] == 99
    # One epoch is completed so enqueuer will switch the Sequence

    acc = []
    for i in range(100):
        acc.append(next(gen_output2)[0, 0, 0, 0])
    assert acc[-1] == 99 * 15
    # One epoch has been completed so enqueuer2 will switch

    # Be sure that both Sequence were updated
    assert next(gen_output)[0, 0, 0, 0] == 0
    assert next(gen_output)[0, 0, 0, 0] == 5
    assert next(gen_output2)[0, 0, 0, 0] == 0
    assert next(gen_output2)[0, 0, 0, 0] == 15 * 5

    # Tear down everything
    enqueuer.stop()
    enqueuer2.stop()


def test_on_epoch_end_threads():
    enqueuer = OrderedEnqueuer(DummySequence([3, 200, 200, 3]), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc == list([k * 5 for k in range(100)]), "Order was not keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


@use_spawn
def test_ordered_enqueuer_fail_processes():
    enqueuer = OrderedEnqueuer(FaultSequence(), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


@threadsafe_generator
def create_finite_generator_from_sequence_threads(ds):
    for i in range(len(ds)):
        yield ds[i]


def create_finite_generator_from_sequence_pcs(ds):
    for i in range(len(ds)):
        yield ds[i]


def test_finite_generator_enqueuer_threads():
    enqueuer = GeneratorEnqueuer(create_finite_generator_from_sequence_threads(
        DummySequence([3, 200, 200, 3])), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for output in gen_output:
        acc.append(int(output[0, 0, 0, 0]))
    assert set(acc) == set(range(100)), "Output is not the same"
    enqueuer.stop()


def test_finite_generator_enqueuer_processes():
    enqueuer = GeneratorEnqueuer(create_finite_generator_from_sequence_pcs(
        DummySequence([3, 200, 200, 3])), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for output in gen_output:
        acc.append(int(output[0, 0, 0, 0]))
    assert acc != list(range(100)), "Order was keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


if __name__ == '__main__':
    pytest.main([__file__])

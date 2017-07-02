"""Tests for functions in data_utils.py.
"""
import os
import sys
import tarfile
import threading
import multiprocessing
import random
import zipfile
from itertools import cycle

import numpy as np
import pytest
from six.moves.urllib.parse import urljoin
from six.moves.urllib.request import pathname2url

from keras.utils import Sequence
from keras.utils import GeneratorEnqueuer
from keras.utils import OrderedEnqueuer
from keras.utils.data_utils import ValueStruct
from keras.utils.data_utils import _hash_file
from keras.utils.data_utils import get_file
from keras.utils.data_utils import validate_file

if sys.version_info < (3,):
    def next(x):
        return x.next()


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


class TestSequence(Sequence):
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, item):
        return np.ones(self.shape, dtype=np.uint8) * item

    def __len__(self):
        return 100


class FaultSequence(Sequence):
    def __getitem__(self, item):
        raise IndexError(item, 'is not present')

    def __len__(self):
        return 100


@threadsafe_generator
def create_generator_from_sequence_threads(ds):
    for i in cycle(range(len(ds))):
        yield ds[i]


def create_generator_from_sequence_pcs(ds):
    for i in cycle(range(len(ds))):
        yield ds[i]


class SimpleGenerator(object):
    def __init__(self, use_multiprocessing=False):

        if use_multiprocessing:
            self.nums = multiprocessing.RawArray('i', 50)
            self.shuffled_nums = [i for i in range(50)]
            random.shuffle(self.shuffled_nums)
            self.lock = multiprocessing.Lock()
            self.ind = multiprocessing.RawValue('i', 0)

        else:
            self.nums = np.zeros((50))
            self.shuffled_nums = [i for i in range(50)]
            random.shuffle(self.shuffled_nums)
            self.lock = threading.Lock()
            self.ind = ValueStruct(0)

        for i in range(50):
            self.nums[i] = self.shuffled_nums[i]

    def reset(self):
        pass

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next()

    def next(self):
        with self.lock:
            if self.ind.value < 50:
                ret = self.nums[self.ind.value]
            else:
                ret = self.ind.value
            self.ind.value += 1
        return ret


def test_generator_enqueuer_threads():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_threads(
        TestSequence([3, 200, 200, 3])), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(int(next(gen_output)[0, 0, 0, 0]))

    assert acc == list(
        range(100)), "Order was not kept in GeneratorEnqueuer with processes"
    assert len(set(acc) - set(range(100))) == 0, "Output is not the same"
    enqueuer.stop()


def test_complex_generator_enqueuer_threads():
    test_generator = SimpleGenerator(use_multiprocessing=False)

    enq = GeneratorEnqueuer(test_generator, use_multiprocessing=False)
    enq.start(workers=8, max_queue_size=10)
    out_gen = enq.get()

    outs = []
    for i in range(50):
        outs.append(next(out_gen))

    assert outs == test_generator.shuffled_nums, 'Ordering is not kept ' \
                                                 'with ' \
                                                 '`use_multiprocessing={' \
                                                 '}`'.format(False)


def test_generator_enqueuer_processes():
    test_generator = SimpleGenerator(use_multiprocessing=True)
    enq = GeneratorEnqueuer(test_generator, use_multiprocessing=True)
    enq.start(workers=8, max_queue_size=10)
    out_gen = enq.get()

    outs = []
    for i in range(50):
        outs.append(next(out_gen))

    assert outs == test_generator.shuffled_nums, 'Ordering is not kept ' \
                                                 'with ' \
                                                 '`use_multiprocessing={' \
                                                 '}`'.format(True)


def test_generator_enqueuer_fail_threads():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_threads(
        FaultSequence()), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


def test_generator_enqueuer_fail_processes():
    enqueuer = GeneratorEnqueuer(create_generator_from_sequence_pcs(
        FaultSequence()), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


def test_ordered_enqueuer_threads():
    enqueuer = OrderedEnqueuer(TestSequence([3, 200, 200, 3]), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc == list(range(100)), "Order was not keep in GeneratorEnqueuer with threads"
    enqueuer.stop()


def test_ordered_enqueuer_processes():
    enqueuer = OrderedEnqueuer(TestSequence([3, 200, 200, 3]), use_multiprocessing=True)
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


def test_ordered_enqueuer_fail_processes():
    enqueuer = OrderedEnqueuer(FaultSequence(), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


if __name__ == '__main__':
    pytest.main([__file__])

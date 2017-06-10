"""Tests for functions in data_utils.py.
"""
import os
import sys
import tarfile
import threading
import zipfile
from itertools import cycle

import numpy as np
import pytest
from six.moves.urllib.parse import urljoin
from six.moves.urllib.request import pathname2url

from keras.utils.data_utils import Dataset, GeneratorEnqueuer, OrderedEnqueuer
from keras.utils.data_utils import _hash_file
from keras.utils.data_utils import get_file
from keras.utils.data_utils import validate_file

if sys.version_info < (3,):
    def next(x):
        return x.next()


def test_data_utils():
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


class TestDataset(Dataset):
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, item):
        return np.ones(self.shape) * item

    def __len__(self):
        return 100


class FaultDataset(Dataset):
    def __getitem__(self, item):
        raise IndexError(item, 'is not present')

    def __len__(self):
        return 100


@threadsafe_generator
def create_generator_from_dataset_threads(ds):
    for i in cycle(range(len(ds))):
        yield ds[i]


def create_generator_from_dataset_pcs(ds):
    for i in cycle(range(len(ds))):
        yield ds[i]


def test_generator_enqueuer_threads():
    enqueuer = GeneratorEnqueuer(create_generator_from_dataset_threads(TestDataset([3, 200, 200, 3])),
                                 pickle_safe=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])

    """
     Not comparing the order since it is not guarantee.
     It may get ordered, but not a lot, one thread can take the GIL before he was supposed to.
    """
    assert set(acc) == set(range(100)), "Output is not the same"
    enqueuer.stop()


def test_generator_enqueuer_processes():
    enqueuer = GeneratorEnqueuer(create_generator_from_dataset_pcs(TestDataset([3, 200, 200, 3])), pickle_safe=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc != list(range(100)), "Order was keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


def test_generator_enqueuer_fail_threads():
    enqueuer = GeneratorEnqueuer(create_generator_from_dataset_threads(FaultDataset()), pickle_safe=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


def test_generator_enqueuer_fail_processes():
    enqueuer = GeneratorEnqueuer(create_generator_from_dataset_pcs(FaultDataset()), pickle_safe=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


def test_ordered_enqueuer_threads():
    enqueuer = OrderedEnqueuer(TestDataset([3, 200, 200, 3]), pickle_safe=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc == list(range(100)), "Order was not keep in GeneratorEnqueuer with threads"
    enqueuer.stop()


def test_ordered_enqueuer_processes():
    enqueuer = OrderedEnqueuer(TestDataset([3, 200, 200, 3]), pickle_safe=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for i in range(100):
        acc.append(next(gen_output)[0, 0, 0, 0])
    assert acc == list(range(100)), "Order was not keep in GeneratorEnqueuer with processes"
    enqueuer.stop()


def test_ordered_enqueuer_fail_threads():
    enqueuer = OrderedEnqueuer(FaultDataset(), pickle_safe=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


def test_ordered_enqueuer_fail_processes():
    enqueuer = OrderedEnqueuer(FaultDataset(), pickle_safe=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with pytest.raises(StopIteration):
        next(gen_output)


if __name__ == '__main__':
    pytest.main([__file__])

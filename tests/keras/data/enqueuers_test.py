import sys
from itertools import cycle

import numpy as np
import pytest

from keras.data.dataset import Dataset
from keras.data.enqueuers import GeneratorEnqueuer, OrderedEnqueuer
from keras.utils.threading_utils import threadsafe_generator

if sys.version_info < (3,):
    def next(x):
        return x.next()


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

    # May happen, but not a lot, one thread can take the GIL before he was supposed to.
    assert len([i for i, j in zip(acc, range(100)) if i != j]) < 3, "Threads are not behaving"
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

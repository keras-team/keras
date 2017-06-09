import sys
import time
from multiprocessing.pool import ThreadPool

import pytest

from keras.utils.threading_utils import threadsafe_generator

if sys.version_info < (3,):
    def next(x):
        return x.next()


def test_failure_case():
    def dummy_gen():
        while True:
            time.sleep(0.5)
            yield 1

    def fun(gen):
        return next(gen)

    gen = dummy_gen()
    gen = dummy_gen()
    pool = ThreadPool(10)
    results = [pool.apply_async(fun, (gen,)) for i in range(10)]
    with pytest.raises(ValueError):
        res = [fut.get() for fut in results]
    pool.close()


def test_success_case():
    @threadsafe_generator
    def dummy_gen():
        while True:
            time.sleep(0.1)
            yield 1

    def fun(gen):
        return next(gen)

    gen = dummy_gen()
    pool = ThreadPool(10)
    futures = [pool.apply_async(fun, (gen,)) for i in range(10)]
    [i.get() for i in futures]
    pool.close()


if __name__ == '__main__':
    pytest.main([__file__])

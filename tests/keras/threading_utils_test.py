import sys
import time
import pytest
import concurrent.futures
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fun, gen) for i in range(10)]
        with pytest.raises(ValueError):
            res = [fut.result() for fut in futures]


def test_success_case():
    @threadsafe_generator
    def dummy_gen():
        while True:
            time.sleep(0.5)
            yield 1

    def fun(gen):
        return next(gen)

    gen = dummy_gen()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fun, gen) for i in range(10)]
        res = [fut.result() for fut in futures]


if __name__ == '__main__':
    pytest.main([__file__])

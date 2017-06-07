import threading
import sys

"""
Purpose of this file is to have multiples workers working on the same generator to speed up training/testing
Thanks to http://anandology.com for the base code. Handle Python 2 and 3.
"""

if sys.version_info < (3,):
    def next(x):
        return x.next()


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

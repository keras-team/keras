from __future__ import print_function

import pytest
from keras import backend as K
import os
from psutil import Process
import warnings

_proc = Process(os.getpid())

history_of_memory_used = []


def get_consumed_ram():
    return _proc.memory_info().rss / (1024 * 1024)


@pytest.fixture(autouse=True)
def clear_session_after_test():
    """Test wrapper to clean up after TensorFlow and CNTK tests.

    This wrapper runs for all the tests in the keras test suite.
    """
    yield
    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()

    mem_used = get_consumed_ram()
    history_of_memory_used.append(mem_used)
    warnings.warn('Memory used by ' + str(os.getpid()) + ' is ' + str(mem_used))
    warnings.warn('History of ' + str(os.getpid()) + ' is ' + str(history_of_memory_used))

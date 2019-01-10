from __future__ import print_function

import pytest
from keras import backend as K
import os
from psutil import Process
import warnings
import gc

_proc = Process(os.getpid())

mem_usage = [0]


def get_consumed_ram():
    return _proc.memory_info().rss / (1024 * 1024)


@pytest.fixture(autouse=True)
def clear_session_after_test(request):
    """Test wrapper to clean up after TensorFlow and CNTK tests.

    This wrapper runs for all the tests in the keras test suite.
    """
    yield
    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()
    gc.collect()

    current_memory = get_consumed_ram()

    test_name = request.node.name

    warnings.warn('GABYMEMINFO: ' + str(os.getpid()) + ' '
                  + str(current_memory - mem_usage[-1]) + ' '
                  + test_name)

    mem_usage.append(current_memory)

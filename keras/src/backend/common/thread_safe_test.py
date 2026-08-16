import concurrent

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing


class TestThreadSafe(testing.TestCase):
    def test_is_thread_safe(self):
        if backend.IS_THREAD_SAFE:
            executor = concurrent.futures.ThreadPoolExecutor()

            def sum(x, axis):
                return ops.sum(x, axis=axis)

            futures = []

            for i in range(10000):
                futures.clear()
                x = ops.convert_to_tensor(np.random.rand(100, 100))
                futures.append(executor.submit(sum, x, 1))
                x = ops.convert_to_tensor(np.random.rand(100))
                futures.append(executor.submit(sum, x, 0))
                concurrent.futures.wait(
                    futures, return_when=concurrent.futures.ALL_COMPLETED
                )
                [future.result() for future in futures]

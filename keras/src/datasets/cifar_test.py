import os
import pickle

import numpy as np

from keras.src import testing
from keras.src.datasets import cifar


class LoadBatchTest(testing.TestCase):
    def _write_batch(self, obj):
        path = os.path.join(self.get_temp_dir(), "data_batch")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        return path

    def test_loads_valid_batch(self):
        data = np.arange(2 * 3072, dtype="uint8").reshape(2, 3072)
        path = self._write_batch({b"data": data, b"labels": [3, 7]})

        images, labels = cifar.load_batch(path)

        self.assertEqual(images.shape, (2, 3, 32, 32))
        self.assertEqual(list(labels), [3, 7])

    def test_rejects_pickle_gadget(self):
        # A crafted batch whose unpickling would run code must be refused,
        # without executing the payload.
        marker = os.path.join(self.get_temp_dir(), "marker")

        class Exploit:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        path = self._write_batch({b"data": Exploit(), b"labels": [0]})

        with self.assertRaises(pickle.UnpicklingError):
            cifar.load_batch(path)
        self.assertFalse(os.path.exists(marker))

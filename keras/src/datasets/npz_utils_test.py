import os
import pickle

import numpy as np

from keras.src import testing
from keras.src.datasets import npz_utils


class LoadNpzTest(testing.TestCase):
    def test_loads_ragged_object_arrays(self):
        # IMDB/Reuters store ragged sequences as `dtype=object` arrays.
        xs = np.array([[1, 14, 22], [1, 194], [1, 14, 47, 8]], dtype=object)
        ys = np.array([1, 0, 1])
        path = os.path.join(self.get_temp_dir(), "ragged.npz")
        np.savez(path, x=xs, y=ys)

        loaded = npz_utils.load_npz(path)

        self.assertEqual(
            [list(seq) for seq in loaded["x"]],
            [[1, 14, 22], [1, 194], [1, 14, 47, 8]],
        )
        self.assertAllEqual(loaded["y"], ys)

    def test_loads_numeric_arrays(self):
        path = os.path.join(self.get_temp_dir(), "numeric.npz")
        np.savez(path, a=np.arange(10), b=np.ones((3, 4), dtype="float32"))

        loaded = npz_utils.load_npz(path)

        self.assertAllEqual(loaded["a"], np.arange(10))
        self.assertEqual(loaded["b"].shape, (3, 4))
        self.assertEqual(loaded["b"].dtype, np.float32)

    def test_rejects_pickle_gadget(self):
        # A crafted member whose unpickling would run code must be refused,
        # without executing the payload.
        marker = os.path.join(self.get_temp_dir(), "marker")

        class Exploit:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        payload = np.empty(1, dtype=object)
        payload[0] = Exploit()
        path = os.path.join(self.get_temp_dir(), "evil.npz")
        np.savez(path, x=payload)

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)
        self.assertFalse(os.path.exists(marker))

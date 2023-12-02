import math
import time

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras import testing
from keras.trainers.data_adapters import py_dataset_adapter
from keras.utils.rng_utils import set_random_seed


class ExamplePyDataset(py_dataset_adapter.PyDataset):
    def __init__(
        self, x_set, y_set, sample_weight=None, batch_size=32, delay=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.delay = delay

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # Create artificial delay to test multiprocessing
        time.sleep(self.delay)

        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        if self.sample_weight is not None:
            return batch_x, batch_y, self.sample_weight[low:high]
        return batch_x, batch_y


class DictPyDataset(py_dataset_adapter.PyDataset):
    def __init__(self, inputs, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.inputs["x"]) / self.batch_size)

    def __getitem__(self, idx):
        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.inputs["x"]))
        batch_x = self.inputs["x"][low:high]
        batch_y = self.inputs["y"][low:high]
        batch = {"x": batch_x, "y": batch_y}
        return batch


class PyDatasetAdapterTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            (True, 2, True, 10),
            (False, 2, True, 10),
            (True, 2, False, 10),
            (False, 2, False, 10),
            (True, 0, False, 0),
            (False, 0, False, 0),
        ]
    )
    def test_basic_flow(
        self, shuffle, workers, use_multiprocessing, max_queue_size
    ):
        set_random_seed(1337)
        x = np.random.random((64, 4))
        y = np.array([[i, i] for i in range(64)], dtype="float64")
        py_dataset = ExamplePyDataset(
            x,
            y,
            batch_size=16,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(
            py_dataset, shuffle=shuffle
        )

        gen = adapter.get_numpy_iterator()
        sample_order = []
        for batch in gen:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.shape, (16, 4))
            self.assertEqual(by.shape, (16, 2))
            for i in range(by.shape[0]):
                sample_order.append(by[i, 0])
        if shuffle:
            self.assertFalse(sample_order == list(range(64)))
        else:
            self.assertAllClose(sample_order, list(range(64)))

        ds = adapter.get_tf_dataset()
        sample_order = []
        for batch in ds:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(tuple(bx.shape), (16, 4))
            self.assertEqual(tuple(by.shape), (16, 2))
            for i in range(by.shape[0]):
                sample_order.append(by[i, 0])
        if shuffle:
            self.assertFalse(sample_order == list(range(64)))
        else:
            self.assertAllClose(sample_order, list(range(64)))

    # TODO: test class_weight
    # TODO: test sample weights
    # TODO: test inference mode (single output)

    def test_speedup(self):
        x = np.random.random((40, 4))
        y = np.random.random((40, 2))

        no_speedup_py_dataset = ExamplePyDataset(
            x,
            y,
            batch_size=4,
            delay=0.5,
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(
            no_speedup_py_dataset, shuffle=False
        )
        gen = adapter.get_numpy_iterator()
        t0 = time.time()
        for batch in gen:
            pass
        no_speedup_time = time.time() - t0

        speedup_py_dataset = ExamplePyDataset(
            x,
            y,
            batch_size=4,
            workers=4,
            # TODO: the github actions runner may have performance issue with
            # multiprocessing
            # use_multiprocessing=True,
            max_queue_size=8,
            delay=0.5,
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(
            speedup_py_dataset, shuffle=False
        )
        gen = adapter.get_numpy_iterator()
        t0 = time.time()
        for batch in gen:
            pass
        speedup_time = time.time() - t0

        self.assertLess(speedup_time, no_speedup_time)

    def test_dict_inputs(self):
        inputs = {
            "x": np.random.random((40, 4)),
            "y": np.random.random((40, 2)),
        }
        py_dataset = DictPyDataset(inputs, batch_size=4)
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset, shuffle=False)
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 2)
            bx, by = batch["x"], batch["y"]
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.shape, (4, 4))
            self.assertEqual(by.shape, (4, 2))

        ds = adapter.get_tf_dataset()
        for batch in ds:
            self.assertEqual(len(batch), 2)
            bx, by = batch["x"], batch["y"]
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(tuple(bx.shape), (4, 4))
            self.assertEqual(tuple(by.shape), (4, 2))

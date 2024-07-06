import math
import time

import jax
import numpy as np
import pytest
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters import py_dataset_adapter
from keras.src.utils.rng_utils import set_random_seed


class ExamplePyDataset(py_dataset_adapter.PyDataset):
    def __init__(
        self,
        x_set,
        y_set,
        sample_weight=None,
        batch_size=32,
        delay=0,
        infinite=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.delay = delay
        self.infinite = infinite

    @property
    def num_batches(self):
        if self.infinite:
            return None
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # Create artificial delay to test multiprocessing
        time.sleep(self.delay)

        if self.infinite:
            idx = idx % math.ceil(len(self.x) / self.batch_size)
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

    @property
    def num_batches(self):
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


class ExceptionPyDataset(py_dataset_adapter.PyDataset):

    @property
    def num_batches(self):
        return 4

    def __getitem__(self, index):
        if index < 2:
            return (
                np.random.random((64, 4)).astype("float32"),
                np.random.random((64, 2)).astype("float32"),
            )
        raise ValueError("Expected exception")


class PyDatasetAdapterTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "multiprocessing",
                    "workers": 2,
                    "use_multiprocessing": True,
                    "max_queue_size": 10,
                    "dataset_type": "np",
                },
                {
                    "testcase_name": "multithreading",
                    "workers": 2,
                    "use_multiprocessing": False,
                    "max_queue_size": 10,
                    "dataset_type": "np",
                },
                {
                    "testcase_name": "single_np",
                    "dataset_type": "np",
                },
                {
                    "testcase_name": "single_tf",
                    "dataset_type": "tf",
                },
                {
                    "testcase_name": "single_jax",
                    "dataset_type": "jax",
                },
                {
                    "testcase_name": "single_torch",
                    "dataset_type": "torch",
                },
            ],
            infinite=[True, False],
            shuffle=[True, False],
        )
    )
    def test_basic_flow(
        self,
        shuffle,
        dataset_type,
        infinite,
        workers=0,
        use_multiprocessing=False,
        max_queue_size=0,
    ):
        if use_multiprocessing and (infinite or shuffle):
            pytest.skip("Starting processes is slow, only test one variant")
        if testing.tensorflow_uses_gpu():
            pytest.skip("This test is flaky with TF on GPU")

        set_random_seed(1337)
        x = np.random.random((64, 4)).astype("float32")
        y = np.array([[i, i] for i in range(64)], dtype="float32")
        if dataset_type == "tf":
            x, y = tf.constant(x), tf.constant(y)
        elif dataset_type == "jax":
            x, y = jax.numpy.array(x), jax.numpy.array(y)
        elif dataset_type == "torch":
            x, y = torch.as_tensor(x), torch.as_tensor(y)
        py_dataset = ExamplePyDataset(
            x,
            y,
            batch_size=16,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
            infinite=infinite,
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(
            py_dataset, shuffle=shuffle
        )

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = jax.Array
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor

        sample_order = []
        for batch in it:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            self.assertEqual(bx.shape, (16, 4))
            self.assertEqual(by.shape, (16, 2))
            for i in range(by.shape[0]):
                sample_order.append(by[i, 0])
            if infinite and len(sample_order) >= 128:
                break
        expected_order = list(range(64))
        if infinite:
            # When the dataset is infinite, we cycle through the data twice.
            expected_order = expected_order + expected_order
        if shuffle and not infinite:
            self.assertNotAllClose(sample_order, expected_order)
        else:
            self.assertAllClose(sample_order, expected_order)

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

    def test_with_different_shapes(self):

        class TestPyDataset(py_dataset_adapter.PyDataset):
            @property
            def num_batches(self):
                return 3

            def __getitem__(self, idx):
                if idx == 0:
                    return np.ones([16, 4], "float32"), np.ones(
                        [16, 2], "float32"
                    )
                if idx == 1:
                    return np.ones([16, 5], "float32"), np.ones(
                        [16, 2], "float32"
                    )
                else:
                    return np.ones([2, 6], "float32"), np.ones(
                        [2, 2], "float32"
                    )

        adapter = py_dataset_adapter.PyDatasetAdapter(
            TestPyDataset(), shuffle=False
        )

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i == 0:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            elif i == 1:
                self.assertEqual(bx.shape, (16, 5))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 6))
                self.assertEqual(by.shape, (2, 2))

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "multiprocessing",
                "workers": 2,
                "use_multiprocessing": True,
                "max_queue_size": 10,
            },
            {
                "testcase_name": "multithreading",
                "workers": 2,
                "max_queue_size": 10,
            },
            {
                "testcase_name": "single",
            },
        ]
    )
    def test_exception_reported(
        self,
        workers=0,
        use_multiprocessing=False,
        max_queue_size=0,
    ):
        dataset = ExceptionPyDataset(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(dataset, shuffle=False)

        expected_exception_class = ValueError
        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            # tf.data wraps the exception
            expected_exception_class = tf.errors.InvalidArgumentError
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()

        it = iter(it)
        next(it)
        next(it)
        with self.assertRaisesRegex(
            expected_exception_class, "Expected exception"
        ):
            next(it)

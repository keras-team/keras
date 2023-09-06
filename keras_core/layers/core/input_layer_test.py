import numpy as np
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core.backend import KerasTensor
from keras_core.layers import InputLayer


class InputLayerTest(testing.TestCase, parameterized.TestCase):
    # Testing happy path for layer without input tensor
    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    def test_input_basic(self, sparse):
        input_shape = (2, 3)
        batch_size = 4
        dtype = "float32"
        ndim = len(tuple((batch_size,) + input_shape))

        init_kwargs = {
            "shape": input_shape,
            "batch_size": batch_size,
            "dtype": dtype,
            "sparse": sparse,
        }

        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            with self.assertRaisesRegex(
                ValueError, "`sparse=True` is not supported"
            ):
                InputLayer(**init_kwargs)
            return

        values = InputLayer(**init_kwargs)

        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.batch_shape[0], batch_size)
        self.assertEqual(values.batch_shape[1:], input_shape)
        self.assertEqual(values.sparse, sparse)
        self.assertEqual(values.trainable, True)
        self.assertIsInstance(values.output, KerasTensor)
        self.assertEqual(values.output.ndim, ndim)
        self.assertEqual(values.output.dtype, dtype)
        self.assertEqual(values.output.sparse, sparse)

    # Testing shape is not None and batch_shape is not None condition
    def test_input_error1(self):
        input_shape = (2, 3)

        with self.assertRaisesRegex(
            ValueError, "cannot pass both `shape` and `batch_shape`"
        ):
            InputLayer(shape=input_shape, batch_shape=input_shape)

    # Testing batch_size is not None and batch_shape is not None
    def test_input_error2(self):
        input_shape = (2, 3)
        batch_size = 4

        with self.assertRaisesRegex(
            ValueError, "cannot pass both `batch_size` and `batch_shape`"
        ):
            InputLayer(batch_size=batch_size, batch_shape=input_shape)

    # Testing shape is None and batch_shape is None
    def test_input_error3(self):
        with self.assertRaisesRegex(ValueError, "pass a `shape` argument."):
            InputLayer(shape=None, batch_shape=None)

    # Testing Input tensor is not Keras tensor
    def test_input_tensor_error(self):
        input_shape = (2, 3)
        batch_size = 4
        input_tensor = np.zeros(input_shape)

        with self.assertRaisesRegex(
            ValueError, "Argument `input_tensor` must be a KerasTensor"
        ):
            InputLayer(
                shape=input_shape,
                batch_size=batch_size,
                input_tensor=input_tensor,
            )

    # Testing happy path for layer with input tensor
    def testing_input_tensor(self):
        input_shape = (2, 3)
        batch_size = 4
        dtype = "float32"
        input_tensor = KerasTensor(shape=input_shape, dtype=dtype)

        values = InputLayer(
            shape=input_shape,
            batch_size=batch_size,
            input_tensor=input_tensor,
            dtype=dtype,
        )

        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.batch_shape[0], batch_size)
        self.assertEqual(values.batch_shape[1:], input_shape)
        self.assertEqual(values.trainable, True)
        self.assertIsInstance(values.output, KerasTensor)
        self.assertEqual(values.output, input_tensor)
        self.assertEqual(values.output.ndim, input_tensor.ndim)
        self.assertEqual(values.output.dtype, dtype)

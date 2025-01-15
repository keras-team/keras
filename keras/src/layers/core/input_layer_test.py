import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend import KerasTensor
from keras.src.layers import InputLayer


class InputLayerTest(testing.TestCase):
    # Testing happy path for layer without input tensor
    @parameterized.named_parameters(
        [
            {"testcase_name": "dense"},
            {"testcase_name": "sparse", "sparse": True},
            {"testcase_name": "ragged", "ragged": True},
        ]
    )
    def test_input_basic(self, sparse=False, ragged=False):
        input_shape = (2, 3)
        batch_size = 4
        dtype = "float32"
        ndim = len(tuple((batch_size,) + input_shape))

        init_kwargs = {
            "shape": input_shape,
            "batch_size": batch_size,
            "dtype": dtype,
            "sparse": sparse,
            "ragged": ragged,
        }

        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            with self.assertRaisesRegex(
                ValueError, "`sparse=True` is not supported"
            ):
                InputLayer(**init_kwargs)
            return
        if ragged and not backend.SUPPORTS_RAGGED_TENSORS:
            with self.assertRaisesRegex(
                ValueError, "`ragged=True` is not supported"
            ):
                InputLayer(**init_kwargs)
            return

        values = InputLayer(**init_kwargs)

        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.batch_shape[0], batch_size)
        self.assertEqual(values.batch_shape[1:], input_shape)
        self.assertEqual(values.sparse, sparse)
        self.assertEqual(values.ragged, ragged)
        self.assertEqual(values.trainable, True)
        self.assertIsInstance(values.output, KerasTensor)
        self.assertEqual(values.output.ndim, ndim)
        self.assertEqual(values.output.dtype, dtype)
        self.assertEqual(values.output.sparse, sparse)
        self.assertEqual(values.output.ragged, ragged)

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
        dtype = "float32"
        input_tensor = KerasTensor(shape=input_shape, dtype=dtype)

        layer = InputLayer(
            input_tensor=input_tensor,
        )

        self.assertEqual(layer.dtype, dtype)
        self.assertEqual(layer.batch_shape, (2, 3))
        self.assertEqual(layer.trainable, True)
        self.assertIsInstance(layer.output, KerasTensor)
        self.assertEqual(layer.output, input_tensor)
        self.assertEqual(layer.output.ndim, input_tensor.ndim)
        self.assertEqual(layer.output.dtype, dtype)

    def test_input_shape_deprecated(self):
        input_shape = (2, 3)
        batch_size = 4
        dtype = "float32"

        with self.assertWarnsRegex(
            UserWarning,
            "Argument `input_shape` is deprecated. Use `shape` instead.",
        ):
            layer = InputLayer(
                input_shape=input_shape, batch_size=batch_size, dtype=dtype
            )

        self.assertEqual(layer.batch_shape[0], batch_size)
        self.assertEqual(layer.batch_shape[1:], input_shape)
        self.assertEqual(layer.dtype, dtype)
        self.assertIsInstance(layer.output, KerasTensor)

    def test_call_method(self):
        layer = InputLayer(shape=(32,))
        output = layer.call()
        self.assertIsNone(output)

    def test_numpy_shape(self):
        # non-python int type shapes should be ok
        InputLayer(shape=(np.int64(32),))

    def test_invalid_arg_combinations(self):
        input_tensor = KerasTensor(shape=(2, 3), dtype="float32")

        with self.assertRaisesRegex(
            ValueError, "cannot provide an incompatible `shape`"
        ):
            _ = InputLayer(
                shape=(2, 4),
                input_tensor=input_tensor,
            )
        with self.assertRaisesRegex(
            ValueError, "cannot provide an incompatible `batch_shape`"
        ):
            _ = InputLayer(
                batch_shape=(2, 4),
                input_tensor=input_tensor,
            )
        with self.assertRaisesRegex(
            ValueError, "cannot provide an incompatible `batch_size`"
        ):
            _ = InputLayer(
                batch_size=5,
                input_tensor=input_tensor,
            )
        with self.assertRaisesRegex(
            ValueError, "cannot provide an incompatible `dtype`"
        ):
            _ = InputLayer(
                dtype="float16",
                input_tensor=input_tensor,
            )
        with self.assertRaisesRegex(
            ValueError, "cannot provide an incompatible `sparse`"
        ):
            _ = InputLayer(
                sparse=True,
                input_tensor=input_tensor,
            )

        # This works
        _ = InputLayer(
            shape=(3,),
            batch_size=2,
            sparse=False,
            dtype="float32",
            input_tensor=input_tensor,
        )

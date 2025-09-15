from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.layers.core import input_layer
from keras.src.ops import operation_utils


class OperationUtilsTest(testing.TestCase):
    def test_get_source_inputs(self):
        x1 = backend.KerasTensor(shape=(2,))
        x2 = backend.KerasTensor(shape=(2,))
        x = x1 + x2
        x += 2
        x = ops.square(x)
        self.assertEqual(operation_utils.get_source_inputs(x), [x1, x2])

    def test_get_source_inputs_return_input_tensor(self):
        inputs = input_layer.Input(shape=(10,))
        self.assertIs(operation_utils.get_source_inputs(inputs)[0], inputs)

    def test_compute_expand_dims_output_shape(self):
        input_shape = (2, 3, 4)
        axis = -1
        output_shape = operation_utils.compute_expand_dims_output_shape(
            input_shape, axis
        )
        expected_output_shape = (2, 3, 4, 1)
        self.assertEqual(output_shape, expected_output_shape)

        input_shape = (2, 3, 4)
        axis = (1, -1)
        output_shape = operation_utils.compute_expand_dims_output_shape(
            input_shape, axis
        )
        expected_output_shape = (2, 1, 3, 4, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_compute_pooling_output_shape(self):
        input_shape = (1, 4, 4, 1)
        pool_size = (2, 2)
        strides = (2, 2)
        output_shape = operation_utils.compute_pooling_output_shape(
            input_shape, pool_size, strides
        )
        expected_output_shape = (1, 2, 2, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_compute_pooling_output_shape_with_none(self):
        input_shape = (None, 4, 4, 1)
        pool_size = (2, 2)
        strides = (2, 2)
        output_shape = operation_utils.compute_pooling_output_shape(
            input_shape, pool_size, strides
        )
        expected_output_shape = (None, 2, 2, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_compute_pooling_output_shape_valid_padding(self):
        input_shape = (1, 4, 4, 1)
        pool_size = (2, 2)
        strides = (2, 2)
        output_shape = operation_utils.compute_pooling_output_shape(
            input_shape, pool_size, strides, padding="valid"
        )
        self.assertEqual(output_shape, (1, 2, 2, 1))

    def test_compute_pooling_output_shape_channels_last(self):
        input_shape = (1, 4, 4, 3)
        pool_size = (2, 2)
        strides = (2, 2)
        output_shape = operation_utils.compute_pooling_output_shape(
            input_shape,
            pool_size,
            strides,
            padding="valid",
            data_format="channels_last",
        )
        self.assertEqual(output_shape, (1, 2, 2, 3))

    def test_compute_pooling_output_shape_same_padding_stride1(self):
        input_shape = (1, 4, 4, 3)
        pool_size = (2, 2)
        strides = (1, 1)
        output_shape = operation_utils.compute_pooling_output_shape(
            input_shape,
            pool_size,
            strides,
            padding="same",
            data_format="channels_last",
        )
        self.assertEqual(output_shape, (1, 4, 4, 3))

    def test_compute_conv_output_shape(self):
        input_shape = (1, 4, 4, 1)
        filters = 1
        kernel_size = (3, 3)
        strides = (1, 1)
        output_shape = operation_utils.compute_conv_output_shape(
            input_shape, filters, kernel_size, strides
        )
        expected_output_shape = (1, 2, 2, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_compute_conv_output_shape_with_none(self):
        input_shape = (None, 4, 4, 1)
        kernel_size = (3, 3)
        filters = 1
        strides = (1, 1)
        output_shape = operation_utils.compute_conv_output_shape(
            input_shape, filters, kernel_size, strides
        )
        expected_output_shape = (None, 2, 2, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_compute_conv_output_shape_valid_padding(self):
        input_shape = (1, 4, 4, 1)
        kernel_size = (3, 3)
        filters = 1
        strides = (2, 2)
        output_shape = operation_utils.compute_conv_output_shape(
            input_shape, filters, kernel_size, strides, padding="valid"
        )
        self.assertEqual(output_shape, (1, 1, 1, 1))

    def test_compute_conv_output_shape_channels_last(self):
        input_shape = (1, 4, 4, 3)
        kernel_size = (3, 3)
        filters = 3
        strides = (2, 2)
        output_shape = operation_utils.compute_conv_output_shape(
            input_shape,
            filters,
            kernel_size,
            strides,
            padding="valid",
            data_format="channels_last",
        )
        self.assertEqual(output_shape, (1, 1, 1, 3))

    def test_compute_conv_output_shape_same_padding_stride1(self):
        input_shape = (1, 4, 4, 3)
        kernel_size = (3, 3)
        filters = 3
        strides = (1, 1)
        output_shape = operation_utils.compute_conv_output_shape(
            input_shape,
            filters,
            kernel_size,
            strides,
            padding="same",
            data_format="channels_last",
        )
        self.assertEqual(output_shape, (1, 4, 4, 3))

    def test_compute_reshape_output_shape(self):
        input_shape = (1, 4, 4, 1)
        target_shape = (16, 1)
        output_shape = operation_utils.compute_reshape_output_shape(
            input_shape, newshape=target_shape, newshape_arg_name="New shape"
        )
        self.assertEqual(output_shape, target_shape)

    def test_reduce_shape_no_axes_no_keepdims(self):
        input_shape = (1, 4, 4, 1)
        output_shape = operation_utils.reduce_shape(input_shape)
        expected_output_shape = ()
        self.assertEqual(output_shape, expected_output_shape)

    def test_reduce_shape_no_axes_with_keepdims(self):
        input_shape = (1, 4, 4, 1)
        output_shape = operation_utils.reduce_shape(input_shape, keepdims=True)
        expected_output_shape = (1, 1, 1, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_reduce_shape_single_axis_no_keepdims(self):
        input_shape = (1, 4, 4, 1)
        axes = [1]
        output_shape = operation_utils.reduce_shape(input_shape, axes)
        expected_output_shape = (1, 4, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_reduce_shape_single_axis_with_keepdims(self):
        input_shape = (1, 4, 4, 1)
        axes = [1]
        output_shape = operation_utils.reduce_shape(
            input_shape, axes, keepdims=True
        )
        expected_output_shape = (1, 1, 4, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_reduce_shape_multiple_axes_no_keepdims(self):
        input_shape = (1, 4, 4, 1)
        axes = [1, 2]
        output_shape = operation_utils.reduce_shape(input_shape, axes)
        expected_output_shape = (1, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_reduce_shape_out_of_order_axes_no_keepdims(self):
        input_shape = (1, 4, 4, 1)
        axes = [2, 1]
        output_shape = operation_utils.reduce_shape(input_shape, axes)
        expected_output_shape = (1, 1)
        self.assertEqual(output_shape, expected_output_shape)

    def test_reduce_shape_negative_axes_no_keepdims(self):
        input_shape = (1, 4, 4, 1)
        axes = [-2, -3]
        output_shape = operation_utils.reduce_shape(input_shape, axes)
        expected_output_shape = (1, 1)
        self.assertEqual(output_shape, expected_output_shape)

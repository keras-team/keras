from keras.backend.common.backend_utils import (
    _convert_conv_tranpose_padding_args_from_keras_to_jax,
)
from keras.backend.common.backend_utils import (
    _convert_conv_tranpose_padding_args_from_keras_to_torch,
)
from keras.backend.common.backend_utils import (
    _get_output_shape_given_tf_padding,
)
from keras.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_torch,
)
from keras.testing import test_case


class ConvertConvTransposePaddingArgsJAXTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test conversion with 'valid' padding and no output padding"""
        (
            left_pad,
            right_pad,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_jax(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="valid",
            output_padding=None,
        )
        self.assertEqual(left_pad, 2)
        self.assertEqual(right_pad, 2)

    def test_same_padding_without_output_padding(self):
        """Test conversion with 'same' padding and no output padding."""
        (
            left_pad,
            right_pad,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_jax(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="same",
            output_padding=None,
        )
        self.assertEqual(left_pad, 2)
        self.assertEqual(right_pad, 1)


class ConvertConvTransposePaddingArgsTorchTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test conversion with 'valid' padding and no output padding"""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="valid",
            output_padding=None,
        )
        self.assertEqual(torch_padding, 0)
        self.assertEqual(torch_output_padding, 0)

    def test_same_padding_without_output_padding(self):
        """Test conversion with 'same' padding and no output padding"""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="same",
            output_padding=None,
        )
        self.assertEqual(torch_padding, 1)
        self.assertEqual(torch_output_padding, 1)


class ComputeConvTransposePaddingArgsForJAXTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test computation with 'valid' padding and no output padding"""
        jax_padding = compute_conv_transpose_padding_args_for_jax(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="valid",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(jax_padding, [(2, 2), (2, 2)])

    def test_same_padding_without_output_padding(self):
        """Test computation with 'same' padding and no output padding"""
        jax_padding = compute_conv_transpose_padding_args_for_jax(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="same",
            output_padding=None,
            dilation_rate=1,
        )

        self.assertEqual(jax_padding, [(2, 1), (2, 1)])


class ComputeConvTransposePaddingArgsForTorchTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test computation with 'valid' padding and no output padding"""
        (
            torch_paddings,
            torch_output_paddings,
        ) = compute_conv_transpose_padding_args_for_torch(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="valid",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(torch_paddings, [0, 0])
        self.assertEqual(torch_output_paddings, [0, 0])

    def test_same_padding_without_output_padding(self):
        """Test computation with 'same' padding and no output padding"""
        (
            torch_paddings,
            torch_output_paddings,
        ) = compute_conv_transpose_padding_args_for_torch(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="same",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(torch_paddings, [1, 1])
        self.assertEqual(torch_output_paddings, [1, 1])

    def test_valid_padding_with_none_output_padding(self):
        """Test conversion with 'valid' padding and no output padding"""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="valid",
            output_padding=None,
        )
        self.assertEqual(torch_padding, 0)
        self.assertEqual(torch_output_padding, 0)

    def test_valid_padding_with_output_padding(self):
        """Test conversion with 'valid' padding and output padding for Torch."""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="valid",
            output_padding=1,
        )
        self.assertEqual(torch_padding, 0)
        self.assertEqual(torch_output_padding, 1)


class GetOutputShapeGivenTFPaddingTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test computation with 'valid' padding and no output padding."""
        output_shape = _get_output_shape_given_tf_padding(
            input_size=5,
            kernel_size=3,
            strides=2,
            padding="valid",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(output_shape, 11)

    def test_same_padding_without_output_padding(self):
        """Test computation with 'same' padding and no output padding."""
        output_shape = _get_output_shape_given_tf_padding(
            input_size=5,
            kernel_size=3,
            strides=2,
            padding="same",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(output_shape, 10)

    def test_valid_padding_with_output_padding(self):
        """Test computation with 'valid' padding and output padding."""
        output_shape = _get_output_shape_given_tf_padding(
            input_size=5,
            kernel_size=3,
            strides=2,
            padding="valid",
            output_padding=1,
            dilation_rate=1,
        )
        self.assertEqual(output_shape, 12)

    def test_warning_for_inconsistencies(self):
        """Test that a warning is raised for potential inconsistencies"""
        with self.assertWarns(Warning):
            _convert_conv_tranpose_padding_args_from_keras_to_torch(
                kernel_size=3,
                stride=2,
                dilation_rate=1,
                padding="same",
                output_padding=1,
            )

    def test_same_padding_without_output_padding_for_torch_(self):
        """Test conversion with 'same' padding and no output padding."""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="same",
            output_padding=None,
        )
        self.assertEqual(torch_padding, max(-((3 % 2 - 3) // 2), 0))
        self.assertEqual(torch_output_padding, 1)

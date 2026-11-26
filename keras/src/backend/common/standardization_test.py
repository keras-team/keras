from unittest.mock import create_autospec

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import testing
from keras.src.backend import config
from keras.src.backend.common.standardization import standardize_data_format
from keras.src.backend.common.standardization import standardize_shape


class StandardizeDataFormatTest(testing.TestCase):
    @parameterized.named_parameters(
        ("channels_first", "channels_first", "channels_first"),
        ("channels_last", "channels_last", "channels_last"),
        ("channels_first_upper", "CHANNELS_FIRST", "channels_first"),
        ("channels_last_mixed", "Channels_Last", "channels_last"),
    )
    def test_valid_data_format(self, data_format, expected):
        self.assertEqual(standardize_data_format(data_format), expected)

    @parameterized.named_parameters(
        ("channels_first", "channels_first"),
        ("channels_last", "channels_last"),
    )
    def test_none_uses_config_default(self, default_format):
        self.addCleanup(
            config.set_image_data_format, config.image_data_format()
        )
        config.set_image_data_format(default_format)
        self.assertEqual(standardize_data_format(None), default_format)

    def test_invalid_data_format_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"must be one of \{'channels_first', 'channels_last'\}"
        ):
            standardize_data_format("last_channels")


class StandardizeShapeTest(testing.TestCase):
    def test_standardize_shape_with_none(self):
        with self.assertRaisesRegex(
            ValueError, "Undefined shapes are not supported."
        ):
            standardize_shape(None)

    def test_standardize_shape_with_non_iterable(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot convert '42' to a shape."
        ):
            standardize_shape(42)

    def test_standardize_shape_with_valid_input(self):
        shape = (3, 4, 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_valid_input_with_none(self):
        shape = (3, None, 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, None, 5))

    def test_standardize_shape_with_valid_not_tuple_input(self):
        shape = [3, 4, 5]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_numpy(self):
        shape = [3, np.int32(4), np.int64(5)]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))
        for d in standardized_shape:
            self.assertIsInstance(d, int)

    def test_standardize_shape_with_string(self):
        shape_with_string = (3, 4, "5")
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid dimension '5'.",
        ):
            standardize_shape(shape_with_string)

    def test_standardize_shape_with_float(self):
        shape_with_float = (3, 4, 5.0)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid dimension '5.0'.",
        ):
            standardize_shape(shape_with_float)

    def test_standardize_shape_with_object(self):
        shape_with_object = (3, 4, object())
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid dimension .*object",
        ):
            standardize_shape(shape_with_object)

    def test_standardize_shape_with_negative_dimension(self):
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Negative dimensions",
        ):
            standardize_shape((3, 4, -5))

    @parameterized.named_parameters(
        ("all_dynamic", (None, None, None, 64), (None, None, None, 64)),
        ("mixed", (None, 224, 224, 3), (None, 224, 224, 3)),
        ("all_static", (1, 224, 224, 3), (1, 224, 224, 3)),
    )
    def test_standardize_shape_preserves_none(self, input_shape, expected):
        """Test that None dimensions are preserved correctly."""
        result = standardize_shape(input_shape)
        self.assertEqual(result, expected)


@pytest.mark.skipif(
    config.backend() != "torch",
    reason="Tests for standardize_shape with Torch backend",
)
class TestStandardizeShapeWithTorch(testing.TestCase):
    def test_standardize_shape_with_torch_size(self):
        import torch

        tensor = torch.randn(3, 4, 5)
        shape = tensor.size()
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))
        self.assertIs(type(standardized_shape), tuple)
        for d in standardized_shape:
            self.assertIsInstance(d, int)

    def test_standardize_shape_with_torch_symint(self):
        """Test that torch.SymInt dimensions are converted to None.

        This validates the fix for GitHub issue #22102 where torch.SymInt
        objects from torch.export were causing "Constraints violated" errors.
        """
        import torch

        # Create a mock SymInt object
        sym_int = create_autospec(torch.SymInt, instance=True)
        shape_with_sym_int = (sym_int, 224, 224, 64)

        # SymInt should be converted to None
        result = standardize_shape(shape_with_sym_int)
        self.assertEqual(result, (None, 224, 224, 64))

        # Test with multiple SymInts
        sym_int2 = create_autospec(torch.SymInt, instance=True)
        shape_with_multiple_sym_ints = (sym_int, sym_int2, 64)
        result = standardize_shape(shape_with_multiple_sym_ints)
        self.assertEqual(result, (None, None, 64))


@pytest.mark.skipif(
    config.backend() != "tensorflow",
    reason="Tests for standardize_shape with TensorFlow backend",
)
class TestStandardizeShapeWithTensorflow(testing.TestCase):
    def test_standardize_shape_with_tensor_size(self):
        import tensorflow as tf

        shape = (3, tf.constant(4, dtype=tf.int64), 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))
        self.assertIs(type(standardized_shape), tuple)
        for d in standardized_shape:
            self.assertIsInstance(d, int)

    def test_standardize_shape_with_unknown_rank(self):
        import tensorflow as tf

        with self.assertRaisesRegex(
            ValueError, "unknown rank.*tf.ensure_shape"
        ):
            standardize_shape(tf.TensorShape(None))

    def test_standardize_shape_with_partially_known_shape(self):
        import tensorflow as tf

        self.assertEqual(
            standardize_shape(tf.TensorShape([None, 3])), (None, 3)
        )

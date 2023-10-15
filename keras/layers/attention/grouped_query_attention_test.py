import numpy as np

# import pytest
from absl.testing import parameterized

# from keras import backend
from keras import initializers
from keras import layers
from keras import testing


class GroupedQueryAttentionTest(testing.TestCase, parameterized.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=8,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "use_bias": False,
                "dropout": 0.5,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    @parameterized.named_parameters(
        ("without_key_proj", (4, 8), (2, 8), None),
        ("with_key_proj", (4, 8), (2, 8), (2, 3)),
    )
    def test_compute_output_shape(self, query_dims, value_dims, key_dims):
        """Test computed shape is equal to the layer output's shape."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=2,
            num_key_value_heads=2,
            head_dim=2,
        )
        batch_size = 7
        query_shape = (batch_size,) + query_dims
        value_shape = (batch_size,) + value_dims
        key_shape = (batch_size,) + key_dims if key_dims else None

        query = np.ones(query_shape)
        value = np.ones(value_shape)
        key = np.ones(key_shape) if key_shape else None
        output = layer(query=query, value=value, key=key)
        comp_output_shape = layer.compute_output_shape(
            query_shape, value_shape, key_shape
        )
        self.assertEqual(output.shape, comp_output_shape)

    @parameterized.named_parameters(
        ("query_value_dim_mismatch", (2, 4, 8), (2, 2, 7), 2),
        ("key_value_dim_mismatch", (2, 4, 8), (2, 2, 8), (2, 1, 7)),
    )
    def test_shape_mismatch_error(self, query_shape, value_shape, key_shape):
        """Test dimension mismatches"""
        layer = layers.GroupedQueryAttention(
            num_query_heads=4,
            num_key_value_heads=4,
            head_dim=2,
        )
        with self.assertRaisesRegex(ValueError, r"must be equal"):
            layer.compute_output_shape(query_shape, value_shape, key_shape)

    def test_initializer(self):
        # Test with a specified initializer.
        layer = layers.GroupedQueryAttention(
            num_query_heads=16,
            num_key_value_heads=16,
            head_dim=64,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        layer.build((2, 4, 8), (2, 4, 8))

        # Make sure the sub layers have different kernel init value.
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._key_dense.kernel,
        )
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._value_dense.kernel,
        )
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._output_dense.kernel,
        )

import numpy as np
# import pytest
from absl.testing import parameterized

# from keras import backend
# from keras import initializers
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
        ("4d_inputs_2d_attention", (3, 4), (3, 2)),
        ("5d_inputs_2d_attention", (5, 3, 4), (5, 3, 2)),
    )
    def test_high_dim_attention(
        self, q_dims, v_dims,
    ):
        batch_size, hidden_size = 3, 8
        query_shape = (batch_size,) + q_dims + (hidden_size,)
        value_shape = (batch_size,) + v_dims + (hidden_size,)
        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
            },
            input_shape={
                "query_shape": query_shape,
                "value_shape": value_shape,
            },
            expected_output_shape=query_shape,
            expected_num_trainable_weights=8,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    @parameterized.named_parameters(
        ("without_key_proj", (4, 8), (2, 8), None),
        ("with_key_proj", (4, 8), (2, 8), (2, 3)),
        ("high_dim_proj", (4, 2, 3, 8), (1, 1, 5, 8), (1, 1, 5, 2)),
    )
    def test_compute_output_shape(
        self, query_dims, value_dims, key_dims
    ):
        """Test computed shape is equal to the layer output's shape."""
        layer = layers.GroupedQueryAttention(
            num_heads=2,
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

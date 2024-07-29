import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import testing


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
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    @parameterized.named_parameters(
        ("without_key_proj_mha", (4, 8), (2, 8), None, 2, 2),
        ("with_key_proj_mha", (4, 8), (2, 8), (2, 3), 2, 2),
        ("without_key_proj_gqa", (4, 8), (2, 8), None, 4, 2),
        ("with_key_proj_gqa", (4, 8), (2, 8), (2, 3), 4, 2),
        ("without_key_value_proj_mqa", (4, 8), (2, 8), None, 4, 1),
        ("with_key_value_proj_mqa", (4, 8), (2, 8), (2, 3), 4, 1),
    )
    def test_compute_output_shape(
        self,
        query_dims,
        value_dims,
        key_dims,
        num_query_heads,
        num_key_value_heads,
    ):
        """Test computed shape is equal to the layer output's shape."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
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

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_query_mask_progagation(self):
        """Test automatic propagation of the query's mask."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2
        )
        self.assertTrue(layer.supports_masking)
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.random.normal(size=(3, 3, 8))
        output = layer(query=masked_query, value=value)
        self.assertAllClose(masked_query._keras_mask, output._keras_mask)

    @parameterized.named_parameters(("causal", True), ("not_causal", 0))
    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_masking(self, use_causal_mask):
        """Test that the value and causal masks are taken into account."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2
        )
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.array([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
        masked_value = layers.Embedding(6, 8, mask_zero=True)(value)
        output = layer(
            query=masked_query,
            value=masked_value,
            use_causal_mask=use_causal_mask,
        )
        mask = np.array(
            [[[1, 1, 0]] * 3 + [[0, 0, 0]] * 2]
            + [[[1, 0, 0]] * 5]
            + [[[1, 1, 1]] + [[0, 0, 0]] * 4]
        ).astype(bool)
        if use_causal_mask:
            mask = mask & np.array(
                [[[1, 0, 0], [1, 1, 0]] + [[1, 1, 1]] * 3]
            ).astype(bool)
        del masked_query._keras_mask
        del masked_value._keras_mask
        output_with_manual_mask = layer(
            query=masked_query, value=masked_value, attention_mask=mask
        )
        self.assertAllClose(output, output_with_manual_mask)

    def test_correctness(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        # Setup layer.
        num_heads = 2
        key_dim = 2
        layer = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        layer.build(query.shape, key.shape, value.shape)

        # Set layer weights.
        kernel = np.identity(key_dim)
        # To get an identity kernel we need to add a head dim and repeat on it.
        kernel = np.repeat(kernel[:, np.newaxis, :], num_heads, axis=1)
        # Zeros for all biases.
        bias = np.zeros((2, 2))
        output_bias = np.zeros((2,))
        layer.set_weights([kernel, bias] * 3 + [kernel, output_bias])

        # Call layer and assert output.
        output, scores = layer(
            query=query,
            value=value,
            key=key,
            return_attention_scores=True,
        )
        self.assertAllClose(output, [[[5.679, 5.679], [4.32, 4.32]]], atol=1e-3)
        self.assertAllClose(
            scores,
            [[[[0.33, 0.67], [0.67, 0.33]], [[0.33, 0.67], [0.67, 0.33]]]],
            atol=1e-3,
        )

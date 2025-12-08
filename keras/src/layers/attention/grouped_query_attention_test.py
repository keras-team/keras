import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import testing
from keras.src.backend.config import disable_flash_attention
from keras.src.backend.config import enable_flash_attention
from keras.src.backend.config import is_flash_attention_enabled


class GroupedQueryAttentionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Flash attention is a newly introduced feature. We need to disable it
        # for testing purposes.
        disable_flash_attention()

    def tearDown(self):
        enable_flash_attention()
        return super().tearDown()

    def test_basics(self):
        self.assertFalse(is_flash_attention_enabled())
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

    def test_basics_with_flash_attention(self):
        enable_flash_attention()
        init_kwargs = {
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "dtype": "float16",
        }
        input_shape = {
            "query_shape": (2, 8, 16),
            "value_shape": (2, 4, 16),
        }
        expected_output_shape = (2, 8, 16)
        if backend.backend() in ("tensorflow", "numpy"):
            self.skipTest(
                "Flash attention is not supported in tensorflow and numpy "
                "backends."
            )
        elif backend.backend() == "torch":
            try:
                self.run_layer_test(
                    layers.GroupedQueryAttention,
                    init_kwargs=init_kwargs,
                    input_shape=input_shape,
                    expected_output_shape=expected_output_shape,
                    expected_num_trainable_weights=8,
                    expected_num_non_trainable_weights=0,
                    expected_num_seed_generators=0,
                    expected_num_losses=0,
                    supports_masking=True,
                    run_training_check=False,
                )
            except ImportError as e:
                if "Flash attention is not supported" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "PyTorch version."
                        )
                        in str(e.args[0])
                    )
            except RuntimeError as e:
                if (
                    "Flash attention is not supported with the provided inputs"
                    in str(e.args[0])
                ):
                    self.assertTrue(
                        (
                            "Flash attention is not supported with the "
                            "provided inputs"
                        )
                        in str(e.args[0])
                    )
        elif backend.backend() == "jax":
            try:
                self.run_layer_test(
                    layers.GroupedQueryAttention,
                    init_kwargs=init_kwargs,
                    input_shape=input_shape,
                    expected_output_shape=expected_output_shape,
                    expected_num_trainable_weights=8,
                    expected_num_non_trainable_weights=0,
                    expected_num_seed_generators=0,
                    expected_num_losses=0,
                    supports_masking=True,
                    run_training_check=False,
                )
            except ImportError as e:
                if "Flash attention is not supported" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "JAX version."
                        )
                        in str(e.args[0])
                    )
            except RuntimeError as e:
                if "cuDNN" in str(e.args[0]):
                    self.assertTrue("cuDNN is not detected." in str(e.args[0]))
                elif "Require at least" in str(e.args[0]):
                    self.assertTrue(
                        "Require at least Ampere arch to run" in str(e.args[0])
                    )
                elif "Flash attention" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "JAX version."
                        )
                        in str(e.args[0])
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
    def test_query_mask_propagation(self):
        """Test automatic propagation of the query's mask."""
        try:
            layer = layers.GroupedQueryAttention(
                num_query_heads=2, num_key_value_heads=2, head_dim=2
            )
            self.assertTrue(layer.supports_masking)
            query = np.array(
                [[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]]
            )
            masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
            value = np.random.normal(size=(3, 3, 8))
            output = layer(query=masked_query, value=value)
        except RuntimeError as e:
            if e.args[0].startswith(
                "(*bias): last dimension must be contiguous"
            ):
                self.skipTest(
                    "PyTorch errors out on GPU: issue to track bug is here "
                    "https://github.com/keras-team/keras/issues/20459"
                )
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

    @parameterized.named_parameters(
        ("disable_flash_attention", False), ("enable_flash_attention", True)
    )
    def test_correctness(self, flash_attention):
        if flash_attention:
            # Let the backend decide whether to use flash attention
            enable_flash_attention()
        dtype = "float16"  # Flash attention only accepts float16/bfloat16
        head_dim = 8  # key_dim % 8 == 0 to enable flash attention
        num_query_heads = num_key_value_heads = 8

        query = np.identity(head_dim)[np.newaxis, ...]
        key = np.identity(head_dim)[np.newaxis, ...]
        value = (
            np.reshape(np.arange(head_dim * head_dim), (1, head_dim, head_dim))
            / 100.0  # Prevent overflow/underflow
        )

        # Setup layer.
        layer = layers.GroupedQueryAttention(
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            dtype=dtype,
        )
        layer.build(query.shape, key.shape, value.shape)

        # Set layer weights.
        kernel = np.identity(head_dim)
        # To get an identity kernel we need to add a head dim and repeat on it.
        kernel = np.repeat(kernel[:, np.newaxis, :], num_query_heads, axis=1)
        # Zeros for all biases.
        bias = np.zeros((num_query_heads, head_dim))
        output_bias = np.zeros((head_dim,))
        layer.set_weights([kernel, bias] * 3 + [kernel, output_bias])

        # Call layer and assert output.
        expected_output = np.array(
            [2.406, 2.440, 2.473, 2.504, 2.535, 2.568, 2.602, 2.633]
        )
        expected_output = np.tile(
            expected_output[np.newaxis, :, np.newaxis], (1, 1, head_dim)
        )
        expected_score = np.array(
            [
                [0.1187] * 0 + [0.1691] + [0.1187] * 7,
                [0.1187] * 1 + [0.1691] + [0.1187] * 6,
                [0.1187] * 2 + [0.1691] + [0.1187] * 5,
                [0.1187] * 3 + [0.1691] + [0.1187] * 4,
                [0.1187] * 4 + [0.1691] + [0.1187] * 3,
                [0.1187] * 5 + [0.1691] + [0.1187] * 2,
                [0.1187] * 6 + [0.1691] + [0.1187] * 1,
                [0.1187] * 7 + [0.1691] + [0.1187] * 0,
            ]
        )
        expected_score = np.tile(
            expected_score[np.newaxis, np.newaxis, ...], (1, head_dim, 1, 1)
        )
        if flash_attention:
            output = layer(query=query, value=value, key=key)
            self.assertAllClose(output, expected_output, atol=1e-2)
        else:
            output, scores = layer(
                query=query,
                value=value,
                key=key,
                return_attention_scores=True,
            )
            self.assertAllClose(output, expected_output, atol=1e-2)
            self.assertAllClose(scores, expected_score, atol=1e-2)

    def test_flash_attention_with_errors(self):
        if backend.backend() in ("numpy", "tensorflow"):
            pytest.skip(
                reason=(
                    "Flash attention is not supported on tensorflow and numpy."
                )
            )
        # Check `flash_attention=True` and `dropout=0.1`
        with self.assertRaisesRegex(
            ValueError,
            "Dropout is not supported when flash attention is enabled.",
        ):
            layer = layers.GroupedQueryAttention(
                head_dim=2,
                num_query_heads=2,
                num_key_value_heads=2,
                flash_attention=True,
                dropout=0.1,
            )

        # Check `flash_attention=True` and `return_attention_scores=True`
        layer = layers.GroupedQueryAttention(
            head_dim=2,
            num_query_heads=2,
            num_key_value_heads=2,
            flash_attention=True,
        )
        self.assertTrue(layer._flash_attention)
        query = np.random.random((2, 4, 8))
        value = np.random.random((2, 4, 8))
        with self.assertRaisesRegex(
            ValueError,
            "Returning attention scores is not supported when flash "
            "attention is enabled. Please disable flash attention to access"
            " attention scores.",
        ):
            layer(query=query, value=value, return_attention_scores=True)

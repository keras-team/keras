import os
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src.backend.config import disable_flash_attention
from keras.src.backend.config import enable_flash_attention
from keras.src.backend.config import is_flash_attention_enabled


class MultiHeadAttentionTest(testing.TestCase):
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
            layers.MultiHeadAttention,
            init_kwargs={
                "num_heads": 2,
                "key_dim": 2,
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
            layers.MultiHeadAttention,
            init_kwargs={
                "num_heads": 2,
                "key_dim": 2,
                "value_dim": 4,
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
        if backend.backend() in ("tensorflow", "numpy"):
            self.skipTest(
                "Flash attention is not supported in tensorflow and numpy "
                "backends."
            )
        elif backend.backend() == "torch":
            try:
                self.run_layer_test(
                    layers.MultiHeadAttention,
                    init_kwargs={
                        "num_heads": 2,
                        "key_dim": 8,
                        "dtype": "float16",
                    },
                    input_shape={
                        "query_shape": (2, 8, 16),
                        "value_shape": (2, 4, 16),
                    },
                    expected_output_shape=(2, 8, 16),
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
                    layers.MultiHeadAttention,
                    init_kwargs={
                        "num_heads": 2,
                        "key_dim": 8,
                        "dtype": "float16",
                    },
                    input_shape={
                        "query_shape": (2, 8, 16),
                        "value_shape": (2, 4, 16),
                    },
                    expected_output_shape=(2, 8, 16),
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
        ("4d_inputs_1freebatch_mask2", (3, 4), (3, 2), (4, 2), (2,)),
        ("4d_inputs_1freebatch_mask3", (3, 4), (3, 2), (3, 4, 2), (2,)),
        ("4d_inputs_1freebatch_mask4", (3, 4), (3, 2), (3, 2, 4, 2), (2,)),
        ("4d_inputs_2d_attention", (3, 4), (3, 2), (3, 4, 3, 2), (1, 2)),
        ("5d_inputs_2d_attention", (5, 3, 4), (5, 3, 2), (3, 4, 3, 2), (2, 3)),
        (
            "5d_inputs_2d_attention_fullmask",
            (5, 3, 4),
            (5, 3, 2),
            (5, 3, 4, 3, 2),
            (2, 3),
        ),
    )
    def test_high_dim_attention(
        self, q_dims, v_dims, mask_dims, attention_axes
    ):
        batch_size, hidden_size = 3, 8
        query_shape = (batch_size,) + q_dims + (hidden_size,)
        value_shape = (batch_size,) + v_dims + (hidden_size,)
        self.run_layer_test(
            layers.MultiHeadAttention,
            init_kwargs={
                "num_heads": 2,
                "key_dim": 2,
                "attention_axes": attention_axes,
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

    def test_attention_axes_negative_indexing(self):
        x = np.random.normal(size=(2, 3, 8, 4))

        # Create two layers with equivalent positive and negative indices
        mha_pos = layers.MultiHeadAttention(
            num_heads=2, key_dim=4, attention_axes=2
        )
        mha_neg = layers.MultiHeadAttention(
            num_heads=2, key_dim=4, attention_axes=-2
        )

        # Initialize both layers
        _ = mha_pos(x, x)
        _ = mha_neg(x, x)

        # Set same weights for fair comparison
        mha_neg.set_weights(mha_pos.get_weights())

        # Get outputs and attention scores
        z_pos, a_pos = mha_pos(x, x, return_attention_scores=True)
        z_neg, a_neg = mha_neg(x, x, return_attention_scores=True)

        # Verify shapes match
        self.assertEqual(z_pos.shape, z_neg.shape)
        self.assertEqual(a_pos.shape, a_neg.shape)

        # Verify outputs are identical
        self.assertAllClose(z_pos, z_neg, rtol=1e-5, atol=1e-5)
        self.assertAllClose(a_pos, a_neg, rtol=1e-5, atol=1e-5)

    @parameterized.named_parameters(
        ("without_key_same_proj", (4, 8), (2, 8), None, None),
        ("with_key_same_proj", (4, 8), (2, 8), (2, 3), None),
        ("without_key_different_proj", (4, 8), (2, 8), None, (3, 4)),
        ("with_key_different_proj", (4, 8), (2, 8), (2, 3), (1, 5)),
        ("high_dim_same_proj", (4, 2, 3, 8), (1, 1, 5, 8), (1, 1, 5, 2), None),
        (
            "high_dim_different_proj",
            (4, 2, 3, 8),
            (1, 1, 5, 8),
            (1, 1, 5, 2),
            (3, 2),
        ),
        (
            "different_qv_last_dims",
            (4, 2, 3, 8),
            (4, 2, 3, 7),
            (4, 2, 3, 8),
            None,
        ),
    )
    def test_compute_output_shape(
        self, query_dims, value_dims, key_dims, output_shape
    ):
        """Test computed shape is equal to the layer output's shape."""
        layer = layers.MultiHeadAttention(
            num_heads=2,
            key_dim=2,
            value_dim=2,
            output_shape=output_shape,
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

        # Test shapes as lists.
        comp_output_shape = layer.compute_output_shape(
            list(query_shape),
            list(value_shape),
            list(key_shape) if key_shape is not None else None,
        )
        self.assertEqual(output.shape, comp_output_shape)

    @parameterized.named_parameters(
        ("query_value_dim_mismatch", (2, 4, 8), (2, 2, 7), (2,)),
        ("key_value_dim_mismatch", (2, 4, 8), (2, 2, 8), (2, 1, 7)),
        (
            "key_value_dim_mismatch_high_dim",
            (2, 4, 2, 3, 8),
            (2, 1, 1, 5, 8),
            (2, 1, 15, 5, 2),
        ),
    )
    def test_shape_mismatch_error(self, query_shape, value_shape, key_shape):
        """Test dimension mismatches"""
        layer = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=2,
            value_dim=2,
        )
        with self.assertRaisesRegex(ValueError, r"must be equal"):
            layer.compute_output_shape(query_shape, value_shape, key_shape)
        with self.assertRaisesRegex(ValueError, r"must be equal"):
            layer(
                np.ones(query_shape), np.ones(value_shape), np.ones(key_shape)
            )

    def test_initializer(self):
        # Test with a specified initializer.
        layer = layers.MultiHeadAttention(
            num_heads=12,
            key_dim=64,
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
            layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)
            self.assertTrue(layer.supports_masking)
            query = np.array(
                [[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]]
            )
            masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
            query_mask = backend.get_keras_mask(masked_query)
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
        self.assertAllClose(query_mask, output._keras_mask)

    @parameterized.named_parameters(("causal", True), ("not_causal", 0))
    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_masking(self, use_causal_mask):
        """Test that the value and causal masks are taken into account."""
        layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)
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
        )
        if use_causal_mask:
            mask = mask & np.array([[[1, 0, 0], [1, 1, 0]] + [[1, 1, 1]] * 3])
        del masked_query._keras_mask
        del masked_value._keras_mask
        output_with_manual_mask = layer(
            query=masked_query, value=masked_value, attention_mask=mask
        )
        self.assertAllClose(output, output_with_manual_mask)

    def test_masking_with_different_shapes(self):
        x = random.uniform(shape=(2, 5, 8))
        mask = ops.tril(ops.ones((5, 5)))  # (5, 5)
        layer = layers.MultiHeadAttention(num_heads=2, key_dim=4)
        output_1 = layer(query=x, value=x, attention_mask=mask)

        mask = ops.tile(mask[None, ...], (2, 1, 1))  # (2, 5, 5)
        output_2 = layer(query=x, value=x, attention_mask=mask)

        mask = ops.tile(mask[:, None, ...], (1, 2, 1, 1))  # (2, 2, 5, 5)
        output_3 = layer(query=x, value=x, attention_mask=mask)

        self.assertAllClose(output_1, output_2)
        self.assertAllClose(output_1, output_3)

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_no_warning_with_keras_mask(self):
        layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.array([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
        masked_value = layers.Embedding(6, 8, mask_zero=True)(value)

        with warnings.catch_warnings(record=True) as warning_logs:
            _ = layer(query=masked_query, value=masked_value)
            self.assertLen(warning_logs, 0)

    @parameterized.named_parameters(
        ("disable_flash_attention", False), ("enable_flash_attention", True)
    )
    def test_correctness(self, flash_attention):
        if flash_attention:
            # Let the backend decide whether to use flash attention
            enable_flash_attention()
        dtype = "float16"  # Flash attention only accepts float16/bfloat16

        num_heads = 8
        key_dim = 8  # key_dim % 8 == 0 to enable flash attention

        query = np.identity(key_dim)[np.newaxis, ...]
        key = np.identity(key_dim)[np.newaxis, ...]
        value = (
            np.reshape(np.arange(key_dim * key_dim), (1, key_dim, key_dim))
            / 100.0  # Prevent overflow/underflow
        )

        # Setup layer.
        layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dtype=dtype
        )
        layer.build(query.shape, key.shape, value.shape)

        # Set layer weights.
        kernel = np.identity(key_dim)
        # To get an identity kernel we need to add a head dim and repeat on it.
        kernel = np.repeat(kernel[:, np.newaxis, :], num_heads, axis=1)
        # Zeros for all biases.
        bias = np.zeros((num_heads, key_dim))
        output_bias = np.zeros((key_dim,))
        layer.set_weights([kernel, bias] * 3 + [kernel, output_bias])
        # Call layer and assert output.
        expected_output = np.array(
            [2.406, 2.440, 2.473, 2.504, 2.535, 2.568, 2.602, 2.633]
        )
        expected_output = np.tile(
            expected_output[np.newaxis, :, np.newaxis], (1, 1, key_dim)
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
            expected_score[np.newaxis, np.newaxis, ...], (1, key_dim, 1, 1)
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

    def test_mha_constraints(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        num_heads = 2
        key_dim = 2
        layer = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            kernel_constraint="non_neg",
        )
        layer.build(query.shape, key.shape, value.shape)
        self.assertIsInstance(
            layer._query_dense.kernel.constraint, constraints.NonNeg
        )
        self.assertIsInstance(
            layer._value_dense.kernel.constraint, constraints.NonNeg
        )
        self.assertIsInstance(
            layer._key_dense.kernel.constraint, constraints.NonNeg
        )
        layer = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            bias_constraint="non_neg",
        )
        layer.build(query.shape, key.shape, value.shape)
        self.assertIsInstance(
            layer._query_dense.bias.constraint, constraints.NonNeg
        )
        self.assertIsInstance(
            layer._value_dense.bias.constraint, constraints.NonNeg
        )
        self.assertIsInstance(
            layer._key_dense.bias.constraint, constraints.NonNeg
        )

    @pytest.mark.requires_trainable_backend
    def test_lora(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        layer = layers.MultiHeadAttention(
            num_heads=3,
            key_dim=8,
            use_bias=False,
        )
        layer.build(query.shape, key.shape, value.shape)
        layer.query_dense.enable_lora(2)
        layer.key_dense.enable_lora(2)
        layer.value_dense.enable_lora(2)

        self.assertLen(layer.trainable_variables, 7)
        self.assertLen(layer.non_trainable_variables, 3)

        # Try eager call
        x = {
            "query": query,
            "key": key,
            "value": value,
        }
        y = np.random.random((1, 2, 2))
        _ = layer(**x)

        # Try calling fit()
        inputs = {
            "query": layers.Input((2, 2)),
            "key": layers.Input((2, 2)),
            "value": layers.Input((2, 2)),
        }
        outputs = layer(inputs["query"], inputs["key"], inputs["value"])
        model = models.Model(inputs, outputs)
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        inputs = {
            "query": layers.Input((2, 2)),
            "key": layers.Input((2, 2)),
            "value": layers.Input((2, 2)),
        }
        outputs = layers.MultiHeadAttention(
            num_heads=3,
            key_dim=8,
            use_bias=False,
        )(inputs["query"], inputs["key"], inputs["value"])
        new_model = models.Model(inputs, outputs)

        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.parameters([((1, 2, 3),), ((2, 3, 5),)])
    def test_symbolic_return_attention_scores(self, shape):
        mha = layers.MultiHeadAttention(num_heads=4, key_dim=2)
        x = layers.Input(batch_shape=shape)
        y = layers.Input(batch_shape=shape)
        symbolic_out = mha(x, y, return_attention_scores=True)
        self.assertLen(symbolic_out, 2)

        x = np.random.random(shape)
        y = np.random.random(shape)
        out = mha(x, y, return_attention_scores=True)
        self.assertLen(out, 2)
        self.assertEqual(symbolic_out[0].shape, out[0].shape)
        self.assertEqual(symbolic_out[1].shape, out[1].shape)

    def test_dtype_policy_map(self):
        quantized_policy = dtype_policies.QuantizedDTypePolicy(
            "int8", "float32"
        )
        policy_map = dtype_policies.DTypePolicyMap()

        # Preset the quantized policy
        policy_map["mha/query"] = quantized_policy
        policy_map["mha/key"] = quantized_policy
        policy_map["mha/value"] = quantized_policy
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        layer = layers.MultiHeadAttention(
            num_heads=3, key_dim=8, use_bias=False, dtype=policy_map, name="mha"
        )
        layer.build(query.shape, key.shape, value.shape)

        # Sublayers should be quantized
        self.assertDType(layer._query_dense._kernel, "int8")
        self.assertDType(layer._key_dense._kernel, "int8")
        self.assertDType(layer._value_dense._kernel, "int8")

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
            layer = layers.MultiHeadAttention(
                num_heads=2, key_dim=2, flash_attention=True, dropout=0.1
            )

        # Check `flash_attention=True` and `return_attention_scores=True`
        layer = layers.MultiHeadAttention(
            num_heads=2, key_dim=2, flash_attention=True
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

    def test_multi_head_attention_output_shape_as_int(self):
        """Test MultiHeadAttention with output_shape as an int."""
        mha = layers.MultiHeadAttention(num_heads=2, key_dim=16, output_shape=8)
        query = random.uniform((2, 4, 16))
        value = random.uniform((2, 4, 16))
        output = mha(query=query, value=value)

        assert output.shape == (
            2,
            4,
            8,
        ), f"Expected shape (2, 4, 8), got {output.shape}"

    def test_multi_head_attention_output_shape_as_tuple(self):
        """Test MultiHeadAttention with output_shape as a tuple."""
        mha = layers.MultiHeadAttention(
            num_heads=2, key_dim=16, output_shape=(8, 8)
        )
        query = random.uniform((2, 4, 16))
        value = random.uniform((2, 4, 16))
        output = mha(query=query, value=value)

        assert output.shape == (
            2,
            4,
            8,
            8,
        ), f"Expected shape (2, 4, 8, 8), got {output.shape}"

    def test_multi_head_attention_output_shape_error(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `output_shape`"):
            layers.MultiHeadAttention(num_heads=2, key_dim=16, output_shape=8.0)

    def test_quantize_int8(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        layer = layers.MultiHeadAttention(
            num_heads=3,
            key_dim=8,
            use_bias=False,
        )
        layer.build(query.shape, value.shape, key.shape)
        output_float = layer(query, key, value)
        for sublayer in layer._flatten_layers():
            try:
                sublayer.quantize("int8")
            except:
                pass

        # Verify weights dtype
        self.assertDType(layer._query_dense._kernel, "int8")
        self.assertDType(layer._key_dense._kernel, "int8")
        self.assertDType(layer._value_dense._kernel, "int8")
        self.assertDType(layer._output_dense._kernel, "int8")

        # Try eager call and verify output correctness
        output_quantized = layer(query, key, value)
        mse = ops.mean(ops.square(output_float - output_quantized))
        self.assertLess(mse, 1e-3)  # A weak correctness test

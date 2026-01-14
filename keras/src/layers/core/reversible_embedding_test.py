import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product


class ReversibleEmbeddingTest(test_case.TestCase):
    @parameterized.named_parameters(
        ("int8", "int8", {"axis": -1}, {"axis": -1}),
        (
            "int4",
            "int4",
            {"axis": -1, "value_range": (-8, 7), "output_dtype": "int8"},
            {"axis": -1},
        ),
        ("int8_weight_only", "int8", {"axis": -1}, None),
    )
    def test_reversible_embedding_quantize(
        self, mode, weight_quantizer_args, activation_quantizer_args
    ):
        """Test ReversibleEmbedding quantization with QuantizationConfig."""
        layer = layers.ReversibleEmbedding(
            input_dim=10, output_dim=6, tie_weights=True
        )
        layer.build((None,))

        weight_quantizer = AbsMaxQuantizer(**weight_quantizer_args)
        if activation_quantizer_args is not None:
            activation_quantizer = AbsMaxQuantizer(**activation_quantizer_args)
        else:
            activation_quantizer = None

        if mode == "int8":
            config = Int8QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
            )
        elif mode == "int4":
            config = Int4QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
            )

        layer.quantize(mode, config=config)

        if activation_quantizer_args is not None:
            # Verify inputs_quantizer is set correctly
            self.assertIsInstance(layer.inputs_quantizer, AbsMaxQuantizer)
        else:
            # Verify inputs_quantizer is None
            self.assertIsNone(layer.inputs_quantizer)

        # Verify reverse call works
        x = np.random.random((2, 6)).astype("float32")
        y = layer(x, reverse=True)
        self.assertEqual(y.shape, (2, 10))

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    @pytest.mark.requires_trainable_backend
    def test_reversible_embedding_basics(self, tie_weights):
        self.run_layer_test(
            layers.ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "logit_soft_cap": 50,
            },
            input_data=np.random.randint(low=0, high=100, size=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=1 if tie_weights else 2,
        )

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_saving(self, tie_weights):
        input_data = np.random.randint(low=0, high=100, size=(4, 10))
        model = models.Sequential(
            [
                layers.ReversibleEmbedding(
                    input_dim=100,
                    output_dim=32,
                    tie_weights=tie_weights,
                )
            ]
        )
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model_output = model(input_data)
        model.save(path)
        restored_model = saving.load_model(path)
        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output)

    def test_correctness(self):
        layer = layers.ReversibleEmbedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]))

        layer = layers.ReversibleEmbedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([[1.0, 1.0]])), reverse=True)
        self.assertAllClose(out, np.array([[0.0, 4.0, 6.0]]))

        layer = layers.ReversibleEmbedding(
            input_dim=3, output_dim=2, logit_soft_cap=5
        )
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([[1.0, 1.0]])), reverse=True)
        self.assertAllClose(out, np.array([[0.0, 3.320184, 4.168273]]))

    def test_reverse_dtype(self):
        embedding = layers.ReversibleEmbedding(100, 16, reverse_dtype="float32")
        input_data = ops.ones(shape=(4, 10, 16))
        output_data = embedding(input_data, reverse=True)
        self.assertEqual(output_data.shape, (4, 10, 100))
        self.assertDType(output_data, "float32")

        if backend.backend() == "torch":
            import torch

            if not torch.cuda.is_available():
                self.skipTest("Torch CPU does not support float16")

        embedding = layers.ReversibleEmbedding(100, 16, reverse_dtype="float16")
        input_data = ops.ones(shape=(4, 10, 16))
        output_data = embedding(input_data, reverse=True)
        self.assertEqual(output_data.shape, (4, 10, 100))
        self.assertDType(output_data, "float16")

    @parameterized.named_parameters(
        named_product(mode=("int4", "int8"), tie_weights=(False, True))
    )
    def test_quantize_int(self, mode, tie_weights):
        layer = layers.ReversibleEmbedding(10, 16, tie_weights=tie_weights)
        layer.build()
        x = np.random.randint(0, 9, size=(64, 3))
        x_reverse = np.random.uniform(size=(64, 16)).astype("float32")
        y_float = layer(x)
        y_reverse_float = layer(x_reverse, reverse=True)
        layer.quantize(mode)

        # Verify the dtype of the weights.
        if not tie_weights:
            # The reverse_embeddings's dtype is int8, despite the int4
            # quantization, because we pack the int4 values into int8.
            self.assertDType(layer.reverse_embeddings, "int8")
            self.assertDType(
                layer.reverse_embeddings_scale, layer.variable_dtype
            )

        # Verify the correctness of the outputs.
        y_quantized = layer(x)
        y_reverse_quantized = layer(x_reverse, reverse=True)
        mse = ops.mean(ops.square(y_float - y_quantized))
        mse_reverse = ops.mean(
            ops.square(y_reverse_float - y_reverse_quantized)
        )
        self.assertLess(mse, 1e-3)  # A weak correctness test
        self.assertLess(mse_reverse, 1e-3)  # A weak correctness test

        # Check model save / load round-trip.
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Check weights-only save / load round-trip.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential(
            [layers.ReversibleEmbedding(10, 16, tie_weights=tie_weights)]
        )
        new_model.build((None, 3))
        new_model.quantize(mode)
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        ("int8_tie_weights", "int8_from_mixed_bfloat16", True, 0, 2),
        ("int8_untie_weights", "int8_from_mixed_bfloat16", False, 0, 4),
        ("int4_tie_weights", "int4_from_mixed_bfloat16", True, 0, 2),
        ("int4_untie_weights", "int4_from_mixed_bfloat16", False, 0, 4),
    )
    @pytest.mark.requires_trainable_backend
    def test_quantize_dtype_argument(
        self,
        dtype,
        tie_weights,
        num_trainable_weights,
        num_non_trainable_weights,
    ):
        self.run_layer_test(
            layers.ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "dtype": dtype,
            },
            input_data=np.random.randint(low=0, high=100, size=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_non_trainable_variables=num_non_trainable_weights,
        )

    def test_reversible_embedding_int8_custom_quantizer(self):
        """
        Test custom quantizer serialization for reversible embedding layer with
        int8 quantization.
        """
        # Setup
        weight_range = (-20, 20)
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=-1, value_range=weight_range),
        )

        # Build & Quantize
        layer = layers.ReversibleEmbedding(input_dim=100, output_dim=16)
        layer.build(None)
        layer.quantize("int8", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.ReversibleEmbedding.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int8QuantizationConfig
        )
        quantizer = new_layer.quantization_config.weight_quantizer
        self.assertIsInstance(quantizer, AbsMaxQuantizer)
        self.assertAllEqual(quantizer.value_range, weight_range)

    def test_masking(self):
        layer = layers.ReversibleEmbedding(3, 2, mask_zero=True)
        layer.build()

        out = layer(np.array(([2, 1, 0])))
        mask = backend.get_keras_mask(out)
        self.assertAllClose(mask, np.array([True, True, False]))

        out = layer(np.array(([[1.0, 2.0], [0.0, 0.0]])), reverse=True)
        mask = backend.get_keras_mask(out)
        self.assertIsNone(mask)

    @parameterized.named_parameters(
        named_product(
            block_size=(64, 128, None, -1),
            tie_weights=(True, False),
        )
    )
    def test_int4_quantization_block_size(self, block_size, tie_weights):
        """Test int4 quantization with different block_size configurations."""
        import math

        input_dim, output_dim = 100, 256
        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim, tie_weights=tie_weights
        )
        layer.build()

        x = np.random.randint(0, input_dim, size=(4, 8))
        x_reverse = np.random.random((4, output_dim)).astype("float32")
        y_float = layer(x)
        y_reverse_float = layer(x_reverse, reverse=True)

        # Create config with specified block_size
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify block_size is stored
        self.assertEqual(layer._int4_block_size, block_size)

        # Verify embeddings_scale shape
        if block_size is None or block_size == -1:
            expected_scale_shape = (input_dim,)
        else:
            n_groups = math.ceil(output_dim / block_size)
            expected_scale_shape = (input_dim, n_groups)

        self.assertEqual(layer.embeddings_scale.shape, expected_scale_shape)

        # Verify reverse_embeddings_scale shape if not tied
        if not tie_weights:
            if block_size is None or block_size == -1:
                expected_reverse_scale_shape = (input_dim,)
            else:
                n_groups = math.ceil(output_dim / block_size)
                expected_reverse_scale_shape = (n_groups, input_dim)

            self.assertEqual(
                layer.reverse_embeddings_scale.shape,
                expected_reverse_scale_shape,
            )

        # Verify outputs are reasonable
        y_quantized = layer(x)
        y_reverse_quantized = layer(x_reverse, reverse=True)
        mse = ops.mean(ops.square(y_float - y_quantized))
        mse_reverse = ops.mean(
            ops.square(y_reverse_float - y_reverse_quantized)
        )
        self.assertLess(mse, 1e-3)
        self.assertLess(mse_reverse, 1e-2)

    @parameterized.named_parameters(
        named_product(
            block_size=(64, 128, None),
            tie_weights=(True, False),
        )
    )
    def test_int4_block_size_serialization(self, block_size, tie_weights):
        """Test that block_size is preserved through serialization."""
        input_dim, output_dim = 50, 128
        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim, tie_weights=tie_weights
        )
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Get output before serialization
        x = np.random.randint(0, input_dim, size=(2, 8))
        y_before = layer(x)

        # Save and load model to test full serialization roundtrip
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(),
            f"int4_block_size_rev_emb_model_{tie_weights}.keras",
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # Verify block_size is preserved
        loaded_layer = loaded_model.layers[0]
        self.assertIsInstance(
            loaded_layer.quantization_config, Int4QuantizationConfig
        )
        self.assertEqual(
            loaded_layer.quantization_config.block_size, block_size
        )

        # Verify reverse_embeddings_zero is preserved for untied grouped
        if not tie_weights and block_size is not None:
            self.assertTrue(hasattr(loaded_layer, "reverse_embeddings_zero"))
            self.assertAllClose(
                loaded_layer.reverse_embeddings_zero,
                layer.reverse_embeddings_zero,
            )

        # Verify outputs match after deserialization
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)

    @parameterized.named_parameters(
        ("tie_grouped", True, 64),
        ("tie_perchannel", True, None),
        ("untie_grouped", False, 64),
        ("untie_perchannel", False, None),
    )
    def test_int4_grouped_vs_perchannel_scale_shapes(
        self, tie_weights, block_size
    ):
        """Test that grouped and per-channel have different scale shapes."""
        import math

        input_dim, output_dim = 100, 256

        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim, tie_weights=tie_weights
        )
        layer.build()
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        if block_size is None or block_size == -1:
            # Per-channel
            expected_scale_shape = (input_dim,)
            expected_reverse_scale_shape = (input_dim,)
        else:
            # Grouped
            n_groups = math.ceil(output_dim / block_size)
            expected_scale_shape = (input_dim, n_groups)
            expected_reverse_scale_shape = (n_groups, input_dim)

        self.assertEqual(layer.embeddings_scale.shape, expected_scale_shape)

        if not tie_weights:
            self.assertEqual(
                layer.reverse_embeddings_scale.shape,
                expected_reverse_scale_shape,
            )
            # Check reverse_embeddings_zero shape for grouped quantization
            if block_size is not None and block_size != -1:
                self.assertTrue(hasattr(layer, "reverse_embeddings_zero"))
                self.assertEqual(
                    layer.reverse_embeddings_zero.shape,
                    expected_reverse_scale_shape,
                )
            else:
                self.assertFalse(hasattr(layer, "reverse_embeddings_zero"))

    @parameterized.named_parameters(
        ("grouped_block_4", 4),
        ("grouped_block_8", 8),
    )
    def test_int4_subchannel_g_idx_created(self, block_size):
        """Test that g_idx is created for sub-channel int4 quantization."""
        if testing.tensorflow_uses_gpu():
            self.skipTest("Segfault on TF GPU")

        input_dim, output_dim = 10, 16
        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim
        )
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify g_idx is created
        self.assertTrue(hasattr(layer, "g_idx"))

        # Verify g_idx shape (output_dim for embedding)
        self.assertEqual(layer.g_idx.shape, (output_dim,))

        # Verify g_idx values (should map each column to its group)
        expected_g_idx = np.arange(output_dim) // block_size
        self.assertAllClose(layer.g_idx, expected_g_idx)

    def test_int4_perchannel_no_g_idx(self):
        """Test that per-channel int4 does NOT create g_idx."""
        if testing.tensorflow_uses_gpu():
            self.skipTest("Segfault on TF GPU")

        layer = layers.ReversibleEmbedding(input_dim=10, output_dim=16)
        layer.build()

        config = Int4QuantizationConfig(block_size=None)  # Per-channel
        layer.quantize("int4", config=config)

        # Verify g_idx is NOT created for per-channel
        self.assertFalse(hasattr(layer, "g_idx"))

    def test_int4_subchannel_g_idx_serialization(self):
        """Test that g_idx is properly serialized and deserialized."""
        if testing.tensorflow_uses_gpu():
            self.skipTest("Segfault on TF GPU")

        input_dim, output_dim = 10, 16
        block_size = 8

        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim
        )
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        y_before = layer(x)
        g_idx_before = ops.convert_to_numpy(layer.g_idx)

        # Save and load
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "rev_embedding_int4_g_idx_model.keras"
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # Verify g_idx is preserved
        loaded_layer = loaded_model.layers[0]
        self.assertTrue(hasattr(loaded_layer, "g_idx"))
        self.assertAllClose(loaded_layer.g_idx, g_idx_before)

        # Verify outputs match
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)

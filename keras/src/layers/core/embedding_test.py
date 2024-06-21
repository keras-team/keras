import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src.export import export_lib
from keras.src.testing import test_case


class EmbeddingTest(test_case.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_embedding_basics(self):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 4, "output_dim": 3},
            input_shape=(2,),
            input_dtype="int32",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 5, "output_dim": 4, "mask_zero": True},
            input_shape=(2, 3),
            input_dtype="int64",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_sparse(self):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 5, "output_dim": 4},
            input_shape=(2, 3),
            input_dtype="int32",
            input_sparse=True,
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_correctness(self):
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array([2, 1, 0]))
        self.assertAllClose(out, np.array([[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_correctness_sparse(self):
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x = tf.SparseTensor([[0, 0], [1, 2]], [2, 1], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            x = jax_sparse.BCOO(([2, 1], [[0, 0], [1, 2]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        self.assertAllClose(
            layer(x),
            np.array(
                [
                    [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]],
                ]
            ),
        )

    def test_masking(self):
        layer = layers.Embedding(input_dim=3, output_dim=2, mask_zero=True)
        layer.build()
        out = layer.compute_mask(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([True, True, False]))

    def test_compute_mask_no_masking(self):
        layer = layers.Embedding(input_dim=3, output_dim=2, mask_zero=False)
        input_data = np.array([2, 1, 0])
        mask = layer.compute_mask(input_data)
        self.assertIsNone(mask)

    def test_embedding_constraints(self):
        layer = layers.Embedding(3, 2, embeddings_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.embeddings.constraint, constraints.NonNeg)

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.enable_lora(4)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 3)
        # Try eager call
        x = np.random.randint(0, 9, size=(64, 3))
        y = np.random.random((64, 3, 16))
        _ = layer(x[:2])

        init_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        init_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()

        # Try calling fit()
        model = models.Sequential(
            [
                layer,
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        final_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_embeddings_value - final_lora_a_embeddings_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_embeddings_value - final_lora_b_embeddings_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        new_model = models.Sequential(
            [
                layers.Input((3,), dtype="int32"),
                layers.Embedding(10, 16),
            ]
        )
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @pytest.mark.requires_trainable_backend
    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.Embedding,
            init_kwargs={"input_dim": 5, "output_dim": 4, "lora_rank": 2},
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_enable_lora_with_embeddings_constraint(self):
        layer = layers.Embedding(
            input_dim=10, output_dim=16, embeddings_constraint="max_norm"
        )
        with self.assertRaisesRegex(
            ValueError, "incompatible with embedding constraints"
        ):
            layer.enable_lora(rank=2)

    def test_enable_lora_when_already_enabled(self):
        layer = layers.Embedding(input_dim=10, output_dim=16)
        layer.build()
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(ValueError, "lora is already enabled"):
            layer.enable_lora(rank=2)

    # Test quantization-related (int8) methods

    def test_quantize_int8(self):
        layer = layers.Embedding(10, 16)
        layer.build()
        x = np.random.randint(0, 9, size=(64, 3))
        y_float = layer(x)
        layer.quantize("int8")

        # Verify weights dtype
        self.assertEqual(
            backend.standardize_dtype(layer._embeddings.dtype), "int8"
        )
        self.assertEqual(
            backend.standardize_dtype(layer.embeddings_scale.dtype),
            layer.variable_dtype,
        )

        # Try eager call and verify output correctness
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, 1e-3)  # A weak correctness test

        # Try saving and reloading the model
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Try lora
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.enable_lora(4)
        layer.quantize("int8")
        _ = layer(x)

        # Try building with quantized dtype policy
        layer = layers.Embedding(10, 16, dtype="int8_from_mixed_bfloat16")
        layer.build()
        self.assertEqual(
            backend.standardize_dtype(layer._embeddings.dtype), "int8"
        )
        self.assertEqual(
            backend.standardize_dtype(layer.embeddings_scale.dtype), "float32"
        )

    @pytest.mark.requires_trainable_backend
    def test_quantize_dtype_argument(self):
        self.run_layer_test(
            layers.Embedding,
            {
                "input_dim": 4,
                "output_dim": 3,
                "dtype": "int8_from_mixed_bfloat16",
            },
            input_shape=(2,),
            input_dtype="int32",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Embedding,
            {
                "input_dim": 5,
                "output_dim": 4,
                "mask_zero": True,
                "dtype": "int8_from_float32",
            },
            input_shape=(2, 3),
            input_dtype="int64",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_quantize_on_subclass(self):
        class MyEmbedding(layers.Embedding):
            pass

        layer = MyEmbedding(10, 16)
        with self.assertRaises(NotImplementedError):
            layer.quantize("int8")

    def test_quantize_when_already_quantized(self):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.quantize("int8")
        with self.assertRaisesRegex(
            ValueError, "is already quantized with dtype_policy="
        ):
            layer.quantize("int8")

    def test_quantize_when_already_quantized_using_dtype_argument(self):
        layer = layers.Embedding(10, 16, dtype="int8_from_float32")
        layer.build()
        with self.assertRaisesRegex(
            ValueError, "is already quantized with dtype_policy="
        ):
            layer.quantize("int8")

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", 2),
    )
    def test_quantize_by_setting_dtype_policy(
        self, policy, expected_num_variables
    ):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.dtype_policy = policy
        self.assertLen(layer.variables, expected_num_variables)

    @parameterized.named_parameters(
        ("int7", "int7"),
        ("float7", "float7"),
    )
    def test_quantize_invalid_mode(self, mode):
        layer = layers.Embedding(10, 16)
        layer.build()
        x = np.random.randint(0, 9, size=(1, 3))
        # dtype_policy should not be altered by failed quantization
        original_dtype_policy = layer.dtype_policy

        # Test quantize
        with self.assertRaisesRegex(ValueError, "Invalid quantization mode."):
            layer.quantize(mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_build
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            layer.quantized_build((None, 2), mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_call
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            # Explicitly set quantization_mode
            layer._dtype_policy._quantization_mode = mode
            layer.quantized_call(x)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

    @pytest.mark.requires_trainable_backend
    def test_quantize_when_lora_enabled(self):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.enable_lora(4)
        layer.quantize("int8")
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, 2)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)

        # Try calling fit()
        init_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        init_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()
        x = np.random.randint(0, 9, size=(64, 3))
        y = np.random.random((64, 3, 16))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        final_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_embeddings_value - final_lora_a_embeddings_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_embeddings_value - final_lora_b_embeddings_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential(
            [layers.Input((3,), dtype="int32"), layers.Embedding(10, 16)]
        )
        new_model.quantize("int8")
        new_model.load_weights(temp_filepath)
        self.assertFalse(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal((32, 3))
            ref_output = model(ref_input)
            export_lib.export_model(model, temp_filepath)
            reloaded_layer = export_lib.TFSMLayer(temp_filepath)
            self.assertAllClose(
                reloaded_layer(ref_input), ref_output, atol=1e-7
            )
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    def test_weights_constructor_arg(self):
        layer = layers.Embedding(3, 4, weights=np.ones((3, 4)))
        self.assertAllClose(layer.embeddings.numpy(), np.ones((3, 4)))
        layer = layers.Embedding(3, 4, weights=[np.ones((3, 4))])
        self.assertAllClose(layer.embeddings.numpy(), np.ones((3, 4)))

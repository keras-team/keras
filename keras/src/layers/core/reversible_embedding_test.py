import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product


class ReversibleEmbeddingTest(test_case.TestCase):
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
        x_reverse = np.random.uniform(size=(64, 16))
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

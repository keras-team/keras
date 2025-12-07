import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import export
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import quantizers
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.quantizers.gptq_config import GPTQConfig


class DenseTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_dense_basics(self):
        # 2D case, no bias.
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 4,
                "activation": "relu",
                "kernel_initializer": "random_uniform",
                "bias_initializer": "ones",
                "use_bias": False,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # 3D case, some regularizers.
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "bias_regularizer": "l2",
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    @parameterized.named_parameters(
        ("zero", 0),
        ("negative", -3),
        ("float", 2.5),
        ("none", None),
        ("string", "64"),
    )
    def test_dense_invalid_units_raises(self, units):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            layers.Dense(units)

    def test_dense_correctness(self):
        # With bias and activation.
        layer = layers.Dense(units=2, activation="relu")
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
                np.array([5.0, -6.0]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertAllClose(layer(inputs), [[10.0, 0.0]])

        # Just a kernel matmul.
        layer = layers.Dense(units=2, use_bias=False)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_errors(self):
        with self.assertRaisesRegex(ValueError, "incompatible with the layer"):
            layer = layers.Dense(units=2, activation="relu")
            layer(keras_tensor.KerasTensor((1, 2)))
            layer(keras_tensor.KerasTensor((1, 3)))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_dense_sparse(self):
        import tensorflow as tf

        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 4,
            },
            input_shape=(2, 3),
            input_sparse=True,
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
        )

        inputs = 4 * backend.random.uniform((10, 10))
        inputs = tf.sparse.from_dense(tf.nn.dropout(inputs, 0.8))

        inputs = np.random.random((10, 10)).astype("float32")
        inputs = np.multiply(inputs, inputs >= 0.8)

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            inputs = tf.sparse.from_dense(inputs)
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            inputs = jax_sparse.BCOO.fromdense(inputs)
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        layer = layers.Dense(units=10)
        outputs = layer(inputs)

        # Verify the computation is the same as if it had been a dense tensor
        expected_outputs = ops.add(
            ops.matmul(
                backend.convert_to_tensor(inputs, sparse=False), layer.kernel
            ),
            layer.bias,
        )
        self.assertAllClose(
            outputs, expected_outputs, tpu_atol=1e-2, tpu_rtol=1e-2
        )

        # Verify the gradient is sparse
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            with tf.GradientTape() as g:
                outputs = layer(inputs)

            self.assertIsInstance(
                g.gradient(outputs, layer.kernel), tf.IndexedSlices
            )

    def test_dense_no_activation(self):
        layer = layers.Dense(units=2, use_bias=False, activation=None)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_without_activation_set(self):
        layer = layers.Dense(units=2, use_bias=False)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        layer.activation = None
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_with_activation(self):
        layer = layers.Dense(units=2, use_bias=False, activation="relu")
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )

        inputs = np.array(
            [[-1.0, 2.0]],
        )
        output = layer(inputs)
        expected_output = np.array([[5.0, 0.0]])
        self.assertAllClose(output, expected_output)

    def test_dense_constraints(self):
        layer = layers.Dense(units=2, kernel_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.kernel.constraint, constraints.NonNeg)
        layer = layers.Dense(units=2, bias_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.bias.constraint, constraints.NonNeg)

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.enable_lora(4)
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)
        # Try eager call
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
        _ = layer(x[:2])

        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()

        # Try calling fit()
        model = models.Sequential(
            [
                layer,
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
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
                layers.Dense(units=16),
            ]
        )
        new_model.build((None, 8))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @pytest.mark.requires_trainable_backend
    def test_enable_lora_with_alpha(self):
        # Create a `Dense` layer and build it.
        layer = layers.Dense(units=8)
        layer.build((None, 4))

        # Enable LoRA with `rank`=2 and `lora_alpha`=3.0.
        layer.enable_lora(2, lora_alpha=3.0)
        self.assertEqual(layer.lora_rank, 2)
        self.assertEqual(layer.lora_alpha, 3.0)

        # Manually compute the expected effective kernel:
        # `effective_kernel_expected` = `base_kernel` +
        # `lora_alpha / lora_rank` * `lora_kernel_a @ lora_kernel_b`
        base_kernel = ops.convert_to_numpy(layer._kernel)
        lora_update = np.matmul(
            ops.convert_to_numpy(layer.lora_kernel_a),
            ops.convert_to_numpy(layer.lora_kernel_b),
        )
        effective_kernel_expected = base_kernel + (3.0 / 2) * lora_update

        # Verify that the effective kernel matches expectation.
        self.assertAllClose(
            ops.convert_to_numpy(layer.kernel), effective_kernel_expected
        )

    @pytest.mark.requires_trainable_backend
    def test_lora_weight_name(self):
        class MyModel(models.Model):
            def __init__(self):
                super().__init__(name="mymodel")
                self.dense = layers.Dense(16, name="dense")

            def build(self, input_shape):
                self.dense.build(input_shape)

            def call(self, x):
                return self.dense(x)

        model = MyModel()
        model.build((None, 8))
        model.dense.enable_lora(4)
        self.assertEqual(
            model.dense.lora_kernel_a.path, "mymodel/dense/lora_kernel_a"
        )

    @pytest.mark.requires_trainable_backend
    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "lora_rank": 2,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    def test_enable_lora_with_kernel_constraint(self):
        layer = layers.Dense(units=2, kernel_constraint="max_norm")
        with self.assertRaisesRegex(
            ValueError, "incompatible with kernel constraints"
        ):
            layer.enable_lora(rank=2)

    def test_enable_lora_on_unbuilt_layer(self):
        layer = layers.Dense(units=2)
        with self.assertRaisesRegex(
            ValueError, "Cannot enable lora on a layer that isn't yet built"
        ):
            layer.enable_lora(rank=2)

    def test_enable_lora_when_already_enabled(self):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(ValueError, "lora is already enabled"):
            layer.enable_lora(rank=2)

    # Test quantization-related methods.

    @parameterized.named_parameters(
        ("int8", "int8", 1e-3),
        ("int4", "int4", 2e-3),
    )
    def test_quantize_int(self, mode, error_threshold):
        if mode == "int4" and testing.tensorflow_uses_gpu():
            self.skipTest("Segfault")
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        x = np.random.random((2, 8))
        y_float = layer(x)
        layer.quantize(mode)

        # Verify the dtype of the weights.
        # The kernel's data type is int8, despite the int4 quantization, because
        # we pack the int4 values into int8.
        self.assertEqual(backend.standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            backend.standardize_dtype(layer.kernel_scale.dtype),
            layer.variable_dtype,
        )

        # Verify the correctness of the outputs.
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, error_threshold)  # A weak correctness test

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
        new_model = models.Sequential([layers.Dense(units=16)])
        new_model.build((None, 8))
        new_model.quantize(mode)
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
    )
    def test_quantize_on_unbuilt_layer(self, mode):
        layer = layers.Dense(units=2)
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize(mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
    )
    def test_quantize_on_subclass(self, mode):
        class MyDense(layers.Dense):
            pass

        layer = MyDense(units=16)
        layer.build((None, 8))
        with self.assertRaises(NotImplementedError):
            layer.quantize(mode)

        layer.quantize(mode, type_check=False)  # No error

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
    )
    def test_quantize_when_already_quantized(self, mode):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.quantize(mode)
        for m in ["int8", "int4", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

        layer = layers.Dense(units=2, dtype=f"{mode}_from_float32")
        layer.build((None, 2))
        for m in ["int8", "int4", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", 3),
        ("int4", "int4_from_float32", 3),  # bias + packed kernel + scale
        ("float8", "float8_from_float32", 8),
    )
    @pytest.mark.skipif(testing.tensorflow_uses_gpu(), reason="Segfault")
    def test_quantize_by_setting_dtype_policy(
        self, policy, expected_num_variables
    ):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.dtype_policy = policy
        self.assertLen(layer.variables, expected_num_variables)

    @parameterized.named_parameters(
        ("int7", "int7"),
        ("float7", "float7"),
    )
    def test_quantize_invalid_mode(self, mode):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        x = np.random.random((1, 2))
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

    @parameterized.named_parameters(
        ("int8", "int8_from_mixed_bfloat16", 1, 2),
        ("int4", "int4_from_mixed_bfloat16", 1, 2),
        ("float8", "float8_from_mixed_bfloat16", 8, 0),
    )
    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(testing.tensorflow_uses_gpu(), reason="Segfault")
    def test_quantize_dtype_argument(
        self, dtype, num_trainable_weights, num_non_trainable_weights
    ):
        self.run_layer_test(
            layers.Dense,
            init_kwargs={"units": 5, "dtype": dtype},
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.named_parameters(
        ("int8", "int8", 3, 2, 5),
        ("int4", "int4", 3, 2, 5),
    )
    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(testing.tensorflow_uses_gpu(), reason="Segfault")
    def test_quantize_lora_integration(
        self,
        mode,
        num_trainable_weights,
        num_non_trainable_weights,
        num_torch_params,
    ):
        # Note that saving and loading with lora_enabled and quantized are
        # lossy, so we use a weak correctness test for model outputs (atol=0.5).
        config = dict(units=16)
        layer = layers.Dense(**config)
        layer.build((None, 8))
        layer.enable_lora(4)
        layer.quantize(mode)
        self.assertLen(layer.trainable_weights, num_trainable_weights)
        self.assertLen(layer.non_trainable_weights, num_non_trainable_weights)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, num_torch_params)

        # Try calling fit()
        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
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
        new_model = models.Sequential([layers.Dense(**config)])
        new_model.build((None, 8))
        new_model.quantize(mode)
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
            ref_input = tf.random.normal((2, 8))
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
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

    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(testing.tensorflow_uses_gpu(), reason="Segfault")
    def test_quantize_float8(self):
        import ml_dtypes

        from keras.src import quantizers

        layer = layers.Dense(units=32)
        layer.build((None, 16))
        layer.quantize("float8")
        optimizer = optimizers.AdamW(learning_rate=0.1)
        optimizer.build(layer.trainable_variables)

        def loss_fn(x, dy):
            y = layer(x, training=True)
            loss = y * ops.cast(dy, y.dtype)
            return ops.sum(loss)

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            @tf.function(jit_compile=True)
            def train_one_step(x, dy):
                with tf.GradientTape() as tape:
                    loss = loss_fn(x, dy)
                grads = tape.gradient(loss, layer.trainable_variables)
                optimizer.apply(grads, layer.trainable_variables)

        elif backend.backend() == "jax":
            import jax

            def stateless_loss_fn(trainable_variables, x, dy):
                y = layer.stateless_call(
                    trainable_variables, [], x, training=True
                )[0]
                loss = y * ops.cast(dy, y.dtype)
                return ops.sum(loss)

            grad_fn = jax.jit(jax.grad(stateless_loss_fn))

            def train_one_step(x, dy):
                trainable_variables = [
                    v.value for v in layer.trainable_variables
                ]
                optimizer_variables = [v.value for v in optimizer.variables]
                grads = grad_fn(trainable_variables, x, dy)
                trainable_variables, optimizer_variables = (
                    optimizer.stateless_apply(
                        optimizer_variables, grads, trainable_variables
                    )
                )
                for variable, value in zip(
                    layer.trainable_variables, trainable_variables
                ):
                    variable.assign(value)
                for variable, value in zip(
                    optimizer.variables, optimizer_variables
                ):
                    variable.assign(value)

        elif backend.backend() == "torch":

            def train_one_step(x, dy):
                layer.zero_grad()
                loss = loss_fn(x, dy)
                loss.backward()
                grads = [v.value.grad for v in layer.trainable_variables]
                optimizer.apply(grads, layer.trainable_variables)

        scale_x, amax_history_x = ops.ones(()), ops.zeros((1024,))
        scale_k, amax_history_k = ops.ones(()), ops.zeros((1024,))
        scale_g, amax_history_g = ops.ones(()), ops.zeros((1024,))
        e4m3_max = ops.cast(
            float(ml_dtypes.finfo("float8_e4m3fn").max), "float32"
        )
        e5m2_max = ops.cast(
            float(ml_dtypes.finfo("float8_e5m2").max), "float32"
        )

        for _ in range(3):
            x = random.normal((16, 16), dtype="float32")
            g = random.normal((16, 32), dtype="float32")
            k = ops.convert_to_tensor(layer._kernel)

            # Manually compute the expected amax history and scaling factors.
            amax_from_history_x = ops.max(amax_history_x)
            amax_from_history_k = ops.max(amax_history_k)
            amax_from_history_g = ops.max(amax_history_g)
            scale_x = quantizers.compute_float8_scale(
                amax_from_history_x, scale_x, e4m3_max
            )
            scale_k = quantizers.compute_float8_scale(
                amax_from_history_k, scale_k, e4m3_max
            )
            scale_g = quantizers.compute_float8_scale(
                amax_from_history_g, scale_g, e5m2_max
            )
            amax_history_x = quantizers.compute_float8_amax_history(
                x, amax_history_x
            )
            amax_history_k = quantizers.compute_float8_amax_history(
                k, amax_history_k
            )
            amax_history_g = quantizers.compute_float8_amax_history(
                g, amax_history_g
            )

            train_one_step(x, g)

            self.assertAllClose(layer.inputs_amax_history, amax_history_x)
            self.assertAllClose(layer.kernel_amax_history, amax_history_k)
            self.assertAllClose(layer.outputs_grad_amax_history, amax_history_g)
            self.assertAllClose(layer.inputs_scale, scale_x)
            self.assertAllClose(layer.kernel_scale, scale_k)
            self.assertAllClose(layer.outputs_grad_scale, scale_g)

    @pytest.mark.requires_trainable_backend
    def test_quantize_float8_fitting(self):
        config = dict(units=16)
        layer = layers.Dense(**config)
        layer.build((None, 8))
        layer.quantize("float8")
        self.assertLen(layer.trainable_weights, 8)
        self.assertLen(layer.non_trainable_weights, 0)

        # Try calling fit()
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_float8_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_float8_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.Dense(**config)])
        new_model.build((None, 8))
        new_model.quantize("float8")
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal((2, 8))
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
            self.assertAllClose(reloaded_layer(ref_input), ref_output)
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    def test_quantize_float8_inference(self):
        config = dict(units=16)
        layer = layers.Dense(**config)
        layer.build((None, 8))
        layer.quantize("float8")

        # Try calling with `training=False` and the result must match
        # `training=True` because there is no update.
        x = np.random.random((64, 8))
        y_inference = layer(x, training=False)
        y_training = layer(x, training=True)
        self.assertAllClose(y_inference, y_training)

    def test_gptq_serialization(self):
        """Test that a GPTQ-quantized layer can be serialized and deserialized
        correctly."""
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        config = layer.get_config()
        new_layer = layers.Dense.from_config(config)
        new_layer.build((None, 8))
        self.assertEqual(new_layer.quantization_mode, "gptq")

    def test_int4_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 kernel."""
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.quantize("int4")
        packed_kernel = layer._kernel
        self.assertAllClose(
            layer.kernel, quantizers.unpack_int4(packed_kernel, 2)
        )

    def test_legacy_load_own_variables(self):
        # In previous versions, `load_own_variables` accepted a store with
        # numeric keys.
        float32_store = {
            "0": np.random.random((8, 16)).astype("float32"),
            "1": np.random.random((16,)).astype("float32"),
        }
        int8_store = {
            "0": np.random.randint(-128, 127, size=(8, 16), dtype="int8"),
            "1": np.random.random((16,)).astype("float32"),
            "2": np.random.random((16,)).astype("float32"),  # kernel_scale.
        }
        int4_store = {
            "0": np.random.randint(-128, 127, size=(4, 16), dtype="int8"),
            "1": np.random.random((16,)).astype("float32"),
            "2": np.random.random((16,)).astype("float32"),  # kernel_scale.
        }
        float8_store = {
            "0": np.random.random((8, 16)).astype("float32"),
            "1": np.random.random((16,)).astype("float32"),
            # inputs_scale.
            "2": np.random.random(()).astype("float32"),
            # inputs_amax_history.
            "3": np.random.random((1024,)).astype("float32"),
            # kernel_scale.
            "4": np.random.random(()).astype("float32"),
            # kernel_amax_history.
            "5": np.random.random((1024,)).astype("float32"),
            # outputs_grad_scale.
            "6": np.random.random(()).astype("float32"),
            # outputs_grad_amax_history.
            "7": np.random.random((1024,)).astype("float32"),
        }
        gptq_store = {
            # bias
            "0": np.random.random((16,)).astype("float32"),
            # quantized_kernel
            "1": np.random.randint(0, 16, size=(8, 8), dtype="uint8"),
            # kernel_scale.
            "2": np.random.random((16, 1)).astype("float32"),
            # kernel_zero
            "3": np.random.random((16, 1)).astype("uint8"),
            # g_idx
            "4": np.random.random((8,)).astype("float32"),
        }

        # Test float32 layer.
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.load_own_variables(float32_store)
        self.assertAllClose(layer._kernel, float32_store["0"])
        self.assertAllClose(layer.bias, float32_store["1"])

        # Test int8-quantized layer.
        layer = layers.Dense(units=16, dtype="int8_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(int8_store)
        self.assertAllClose(layer._kernel, int8_store["0"])
        self.assertAllClose(layer.bias, int8_store["1"])
        self.assertAllClose(layer.kernel_scale, int8_store["2"])

        # Test int4-quantized layer.
        layer = layers.Dense(units=16, dtype="int4_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(int4_store)
        self.assertAllClose(layer._kernel, int4_store["0"])
        self.assertAllClose(layer.bias, int4_store["1"])
        self.assertAllClose(layer.kernel_scale, int4_store["2"])

        # Test float8-quantized layer.
        layer = layers.Dense(units=16, dtype="float8_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(float8_store)
        self.assertAllClose(layer._kernel, float8_store["0"])
        self.assertAllClose(layer.bias, float8_store["1"])
        self.assertAllClose(layer.inputs_scale, float8_store["2"])
        self.assertAllClose(layer.inputs_amax_history, float8_store["3"])
        self.assertAllClose(layer.kernel_scale, float8_store["4"])
        self.assertAllClose(layer.kernel_amax_history, float8_store["5"])
        self.assertAllClose(layer.outputs_grad_scale, float8_store["6"])
        self.assertAllClose(layer.outputs_grad_amax_history, float8_store["7"])

        # Test gptq-quantized layer.
        layer = layers.Dense(units=16, dtype="gptq/4/8_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(gptq_store)
        self.assertTrue(layer.is_gptq_calibrated)
        self.assertAllClose(layer.bias, gptq_store["0"])
        self.assertAllClose(layer.quantized_kernel, gptq_store["1"])
        self.assertAllClose(layer.kernel_scale, gptq_store["2"])
        self.assertAllClose(layer.kernel_zero, gptq_store["3"])
        self.assertAllClose(layer.g_idx, gptq_store["4"])

    def test_int4_gptq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 GPTQ
        kernel."""
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        layer.is_gptq_calibrated = True  # Bypass calibration check
        packed_kernel = layer.quantized_kernel
        self.assertAllClose(
            layer.kernel, quantizers.unpack_int4(packed_kernel, 2)
        )

    def test_gptq_kernel_packing(self):
        """Validates that 4-bit GPTQ packing reduces the kernel size."""
        layer = layers.Dense(units=16, use_bias=False)
        layer.build((None, 8))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(quantized_kernel_params, original_kernel_params // 2)

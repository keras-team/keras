import os

from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import quantizers
from keras.src import random
from keras.src import saving
from keras.src import testing

# We use a fixed version of `ReversibleEmbedding` from KerasNLP to ensure the
# changes from Keras doesn't break the downstream.


class ReversibleEmbedding(layers.Embedding):
    """An embedding layer from KerasNLP"""

    def __init__(
        self,
        input_dim,
        output_dim,
        tie_weights=True,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        reverse_dtype=None,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            **kwargs,
        )
        self.tie_weights = tie_weights
        self.reverse_dtype = reverse_dtype

    def build(self, inputs_shape=None):
        super().build(inputs_shape)
        if (
            not self.tie_weights
            and getattr(self, "quantization_mode", None) != "int8"
        ):
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(self.output_dim, self.input_dim),
                initializer=self.embeddings_initializer,
                dtype=self.dtype,
            )

    def call(self, inputs, reverse=False):
        if reverse:
            if self.tie_weights:
                kernel = ops.transpose(ops.convert_to_tensor(self.embeddings))
            else:
                kernel = self.reverse_embeddings
            if self.reverse_dtype is not None:
                inputs = ops.cast(inputs, self.reverse_dtype)
                kernel = ops.cast(kernel, self.reverse_dtype)
            return ops.matmul(inputs, kernel)

        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tie_weights": self.tie_weights,
                "reverse_dtype": self.reverse_dtype,
            }
        )
        return config

    def save_own_variables(self, store):
        if not self.built:
            return
        super().save_own_variables(store)
        target_variables = []
        if not self.tie_weights:
            # Store the reverse embedding weights as the last weights.
            target_variables.append(self.reverse_embeddings)
            if getattr(self, "quantization_mode", None) == "int8":
                target_variables.append(self.reverse_embeddings_scale)
            for i, variable in enumerate(target_variables, start=len(store)):
                store[str(i)] = variable

    def load_own_variables(self, store):
        if not self.built:
            self.build()
        super().load_own_variables(store)
        if not self.tie_weights:
            # Last weights in the stores are the reverse embedding weights.
            target_variables = [self.reverse_embeddings]
            if getattr(self, "quantization_mode", None) == "int8":
                target_variables.append(self.reverse_embeddings_scale)
            for i, variable in enumerate(
                target_variables, start=len(store) - len(target_variables)
            ):
                variable.assign(store[str(i)])

    def compute_output_spec(self, inputs, reverse=False):
        output_shape = list(inputs.shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return backend.KerasTensor(output_shape, dtype=self.compute_dtype)

    # Quantization-related (int8) methods

    def quantized_call(self, inputs, reverse=False):
        # TODO (hongyu): This function could be removed once we add `*args` and
        # `**kwargs` for `Embedding.quantized_call`
        if self.quantization_mode == "int8":
            return self._int8_call(inputs, reverse=reverse)
        else:
            self._quantization_mode_error(self.quantization_mode)

    def _int8_build(
        self,
        embeddings_initializer="zeros",
        embeddings_scale_initializer="ones",
        reverse_embeddings_initializer="zeros",
        reverse_embeddings_scale_initializer="ones",
    ):
        super()._int8_build(
            embeddings_initializer, embeddings_scale_initializer
        )
        self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
        if not self.tie_weights:
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(self.output_dim, self.input_dim),
                initializer=reverse_embeddings_initializer,
                dtype="int8",
                trainable=False,
            )
            self.reverse_embeddings_scale = self.add_weight(
                name="reverse_embeddings_scale",
                shape=(self.input_dim,),
                initializer=reverse_embeddings_scale_initializer,
                trainable=False,
            )

    def _int8_call(self, inputs, reverse=False):
        if reverse:
            if self.tie_weights:
                kernel = ops.transpose(self._embeddings)
                scale = ops.transpose(self.embeddings_scale)
            else:
                kernel = self.reverse_embeddings
                scale = self.reverse_embeddings_scale
            inputs, inputs_scale = self.inputs_quantizer(inputs)
            outputs = ops.matmul(inputs, kernel)
            # De-scale outputs
            outputs = ops.cast(outputs, self.compute_dtype)
            outputs = ops.divide(outputs, ops.multiply(inputs_scale, scale))
            return outputs

        return super()._int8_call(inputs)

    def quantize(self, mode, type_check=True):
        import gc

        if type_check and type(self) is not ReversibleEmbedding:
            raise NotImplementedError(
                f"Layer {self.__class__.__name__} does not have a `quantize()` "
                "method implemented."
            )
        self._check_quantize_args(mode, self.compute_dtype)

        self._tracker.unlock()
        if mode == "int8":
            embeddings, embeddings_scale = quantizers.abs_max_quantize(
                self._embeddings, axis=-1
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            self._untrack_variable(self._embeddings)
            del self._embeddings
            if not self.tie_weights:
                reverse_embeddings, reverse_embeddings_scale = (
                    quantizers.abs_max_quantize(self.reverse_embeddings, axis=0)
                )
                reverse_embeddings_scale = ops.squeeze(
                    reverse_embeddings_scale, axis=0
                )
                self._untrack_variable(self.reverse_embeddings)
                del self.reverse_embeddings
            else:
                reverse_embeddings = None
                reverse_embeddings_scale = None
            self._int8_build(
                lambda shape, dtype: embeddings,
                lambda shape, dtype: embeddings_scale,
                lambda shape, dtype: reverse_embeddings,
                lambda shape, dtype: reverse_embeddings_scale,
            )
        else:
            raise self._quantization_mode_error(mode)
        self._tracker.lock()

        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy
        gc.collect()


class CompatibilityTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("tie_weights", True), ("untie_weights", False)
    )
    def test_reversible_embedding_quantize_int8(self, tie_weights):
        layer_config = dict(
            input_dim=100, output_dim=32, tie_weights=tie_weights
        )
        layer = ReversibleEmbedding(**layer_config)
        layer.build()
        x = random.randint(shape=(64, 100), minval=0, maxval=9)
        x_reverse = random.uniform(shape=(64, 32))
        y_float = layer(x)
        y_reverse_float = layer(x_reverse, reverse=True)
        layer.quantize("int8")

        # Verify weights dtype
        if not tie_weights:
            self.assertEqual(
                backend.standardize_dtype(layer.reverse_embeddings.dtype),
                "int8",
            )
            self.assertEqual(
                backend.standardize_dtype(layer.reverse_embeddings_scale.dtype),
                layer.variable_dtype,
            )

        # Try eager call and verify output correctness
        y_quantized = layer(x)
        y_reverse_quantized = layer(x_reverse, reverse=True)
        mse = ops.mean(ops.square(y_float - y_quantized))
        mse_reverse = ops.mean(
            ops.square(y_reverse_float - y_reverse_quantized)
        )
        self.assertLess(mse, 1e-3)  # A weak correctness test
        self.assertLess(mse_reverse, 1e-3)  # A weak correctness test

        # Try saving and reloading the model
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_reversible_embedding_quantize_dtype_argument(self, tie_weights):
        self.run_layer_test(
            ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "dtype": "int8_from_float32",
            },
            input_data=random.randint(minval=0, maxval=100, shape=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2 if tie_weights else 4,
            expected_num_non_trainable_variables=2 if tie_weights else 4,
        )

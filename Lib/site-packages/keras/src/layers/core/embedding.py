import warnings

from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Embedding")
class Embedding(Layer):
    """Turns nonnegative integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used on nonnegative integer inputs of a fixed range.

    Example:

    >>> model = keras.Sequential()
    >>> model.add(keras.layers.Embedding(1000, 64))
    >>> # The model will take as input an integer matrix of size (batch,
    >>> # input_length), and the largest integer (i.e. word index) in the input
    >>> # should be no larger than 999 (vocabulary size).
    >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
    >>> # dimension.
    >>> input_array = np.random.randint(1000, size=(32, 10))
    >>> model.compile('rmsprop', 'mse')
    >>> output_array = model.predict(input_array)
    >>> print(output_array.shape)
    (32, 10, 64)

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
            This is useful when using recurrent layers which
            may take variable length input. If this is `True`,
            then all subsequent layers in the model need
            to support masking or an exception will be raised.
            If `mask_zero` is set to `True`, as a consequence,
            index 0 cannot be used in the vocabulary (`input_dim` should
            equal size of vocabulary + 1).
        weights: Optional floating-point matrix of size
            `(input_dim, output_dim)`. The initial embeddings values
            to use.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's embeddings
            matrix to non-trainable and replaces it with a delta over the
            original matrix, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large embedding layers.
            You can also enable LoRA on an existing
            `Embedding` layer by calling `layer.enable_lora(rank)`.

    Input shape:
        2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, input_length, output_dim)`.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        weights=None,
        lora_rank=None,
        **kwargs,
    ):
        input_length = kwargs.pop("input_length", None)
        if input_length is not None:
            warnings.warn(
                "Argument `input_length` is deprecated. Just remove it."
            )
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.autocast = False
        self.lora_rank = lora_rank
        self.lora_enabled = False

        if weights is not None:
            self.build()
            if not (isinstance(weights, list) and len(weights) == 1):
                weights = [weights]
            self.set_weights(weights)

    def build(self, input_shape=None):
        if self.built:
            return
        if self.quantization_mode is not None:
            self.quantized_build(input_shape, mode=self.quantization_mode)
        if self.quantization_mode != "int8":
            self._embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name="embeddings",
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                trainable=True,
            )
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def embeddings(self):
        if self.lora_enabled:
            return self._embeddings + ops.matmul(
                self.lora_embeddings_a, self.lora_embeddings_b
            )
        return self._embeddings

    def call(self, inputs):
        if inputs.dtype != "int32" and inputs.dtype != "int64":
            inputs = ops.cast(inputs, "int32")
        outputs = ops.take(self.embeddings, inputs, axis=0)
        return ops.cast(outputs, dtype=self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.output_dim)

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.embeddings_constraint:
            raise ValueError(
                "Lora is incompatible with embedding constraints. "
                "In order to enable lora on this layer, remove the "
                "`embeddings_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. This can only be done once per layer."
            )
        self._tracker.unlock()
        self.lora_embeddings_a = self.add_weight(
            name="lora_embeddings_a",
            shape=(self.embeddings.shape[0], rank),
            initializer=initializers.get(a_initializer),
            regularizer=self.embeddings_regularizer,
        )
        self.lora_embeddings_b = self.add_weight(
            name="lora_embeddings_b",
            shape=(rank, self.embeddings.shape[1]),
            initializer=initializers.get(b_initializer),
            regularizer=self.embeddings_regularizer,
        )
        self.embeddings.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        embeddings_value, embeddings_scale = (
            self._get_embeddings_with_merged_lora()
        )
        target_variables = [embeddings_value]
        if self.quantization_mode is not None:
            if self.quantization_mode == "int8":
                target_variables.append(embeddings_scale)
            else:
                raise self._quantization_mode_error(self.quantization_mode)
        for i, variable in enumerate(target_variables):
            store[str(i)] = variable

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        target_variables = [self._embeddings]
        if self.quantization_mode is not None:
            if self.quantization_mode == "int8":
                target_variables.append(self.embeddings_scale)
            else:
                raise self._quantization_mode_error(self.quantization_mode)
        for i, variable in enumerate(target_variables):
            variable.assign(store[str(i)])
        if self.lora_enabled:
            self.lora_embeddings_a.assign(
                ops.zeros(self.lora_embeddings_a.shape)
            )
            self.lora_embeddings_b.assign(
                ops.zeros(self.lora_embeddings_b.shape)
            )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": initializers.serialize(
                self.embeddings_initializer
            ),
            "embeddings_regularizer": regularizers.serialize(
                self.embeddings_regularizer
            ),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "embeddings_constraint": constraints.serialize(
                self.embeddings_constraint
            ),
            "mask_zero": self.mask_zero,
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return {**base_config, **config}

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )

    """Quantization-related (int8) methods"""

    def _quantization_mode_error(self, mode):
        return NotImplementedError(
            "Invalid quantization mode. Expected 'int8'. "
            f"Received: quantization_mode={mode}"
        )

    def quantized_build(self, input_shape, mode):
        if mode == "int8":
            self._int8_build()
        else:
            raise self._quantization_mode_error(mode)

    def _int8_build(
        self,
        embeddings_initializer="zeros",
        embeddings_scale_initializer="ones",
    ):
        self._embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=embeddings_initializer,
            dtype="int8",
            trainable=False,
        )
        # We choose to reduce the axis of `output_dim` because, typically,
        # `input_dim` is larger than `output_dim`. This reduces quantization
        # error.
        self.embeddings_scale = self.add_weight(
            name="embeddings_scale",
            shape=(self.input_dim,),
            initializer=embeddings_scale_initializer,
            trainable=False,
        )
        self._is_quantized = True

    def quantized_call(self, *args, **kwargs):
        if self.quantization_mode != "int8":
            raise self._quantization_mode_error(self.quantization_mode)
        return super().quantized_call(*args, **kwargs)

    def _int8_call(self, inputs, training=None):
        # We cannot update quantized self._embeddings, so the custom gradient is
        # not needed
        if backend.standardize_dtype(inputs.dtype) not in ("int32", "int64"):
            inputs = ops.cast(inputs, "int32")
        embeddings_scale = ops.take(self.embeddings_scale, inputs, axis=0)
        outputs = ops.take(self._embeddings, inputs, axis=0)
        # De-scale outputs
        outputs = ops.divide(
            ops.cast(outputs, dtype=self.compute_dtype),
            ops.expand_dims(embeddings_scale, axis=-1),
        )
        if self.lora_enabled:
            lora_outputs = ops.take(self.lora_embeddings_a, inputs, axis=0)
            lora_outputs = ops.matmul(lora_outputs, self.lora_embeddings_b)
            outputs = ops.add(outputs, lora_outputs)
        return outputs

    def quantize(self, mode, type_check=True):
        # Prevent quantization of the subclasses
        if type_check and (type(self) is not Embedding):
            raise self._not_implemented_error(self.quantize)

        if mode == "int8":
            # Quantize `self._embeddings` to int8 and compute corresponding
            # scale
            embeddings_value, embeddings_scale = quantizers.abs_max_quantize(
                self._embeddings, axis=-1, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            del self._embeddings
            # Utilize a lambda expression as an initializer to prevent adding a
            # large constant to the computation graph.
            self._int8_build(embeddings_value, embeddings_scale)
        else:
            raise self._quantization_mode_error(mode)

        # Set new dtype policy
        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy

    def _get_embeddings_with_merged_lora(self):
        if self.dtype_policy.quantization_mode is not None:
            embeddings_value = self._embeddings
            embeddings_scale = self.embeddings_scale
            if self.lora_enabled:
                # Dequantize & quantize to merge lora weights into embeddings
                # Note that this is a lossy compression
                embeddings_value = ops.divide(
                    embeddings_value, ops.expand_dims(embeddings_scale, axis=-1)
                )
                embeddings_value = ops.add(
                    embeddings_value,
                    ops.matmul(self.lora_embeddings_a, self.lora_embeddings_b),
                )
                embeddings_value, embeddings_scale = (
                    quantizers.abs_max_quantize(
                        embeddings_value, axis=-1, to_numpy=True
                    )
                )
                embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            return embeddings_value, embeddings_scale
        return self.embeddings, None

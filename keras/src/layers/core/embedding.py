import warnings

from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.layers.layer import Layer
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.saving import serialization_lib


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
        lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.

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
        lora_alpha=None,
        quantization_config=None,
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
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_enabled = False
        self.quantization_config = quantization_config

        if weights is not None:
            self.build()
            if not (isinstance(weights, list) and len(weights) == 1):
                weights = [weights]
            self.set_weights(weights)

    def build(self, input_shape=None):
        if self.built:
            return
        embeddings_shape = (self.input_dim, self.output_dim)
        if self.quantization_mode:
            self.quantized_build(
                embeddings_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if self.quantization_mode not in ("int8", "int4"):
            self._embeddings = self.add_weight(
                shape=embeddings_shape,
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
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `embeddings`."
            )
        embeddings = self._embeddings
        if self.quantization_mode == "int4":
            embeddings = quantizers.unpack_int4(
                embeddings, self._orig_output_dim, axis=-1
            )
        if self.lora_enabled:
            return embeddings + (self.lora_alpha / self.lora_rank) * ops.matmul(
                self.lora_embeddings_a, self.lora_embeddings_b
            )
        return embeddings

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

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        ragged = getattr(inputs, "ragged", False)
        return KerasTensor(
            output_shape, dtype=self.compute_dtype, ragged=ragged
        )

    def enable_lora(
        self,
        rank,
        lora_alpha=None,
        a_initializer="he_uniform",
        b_initializer="zeros",
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
            shape=(self.input_dim, rank),
            initializer=initializers.get(a_initializer),
            regularizer=self.embeddings_regularizer,
        )
        self.lora_embeddings_b = self.add_weight(
            name="lora_embeddings_b",
            shape=(rank, self.output_dim),
            initializer=initializers.get(b_initializer),
            regularizer=self.embeddings_regularizer,
        )
        self.embeddings.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)

        # Embeddings plus optional merged LoRA-aware scale
        # (returns (embeddings, None) for `None` mode).
        embeddings_value, merged_kernel_scale = (
            self._get_embeddings_with_merged_lora()
        )
        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "embeddings":
                store[str(idx)] = embeddings_value
            elif name == "embeddings_scale" and mode in ("int4", "int8"):
                # For int4/int8, the merged LoRA scale (if any) comes from
                # `_get_embeddings_with_merged_lora()`
                store[str(idx)] = merged_kernel_scale
            else:
                store[str(idx)] = getattr(self, name)
            idx += 1

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)

        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "embeddings":
                self._embeddings.assign(store[str(idx)])
            else:
                getattr(self, name).assign(store[str(idx)])
            idx += 1
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
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["quantization_config"] = (
            serialization_lib.deserialize_keras_object(
                config.get("quantization_config", None)
            )
        )
        return super().from_config(config)

    def _quantization_mode_error(self, mode):
        return NotImplementedError(
            "Invalid quantization mode. Expected one of ('int8', 'int4'). "
            f"Received: quantization_mode={mode}"
        )

    @property
    def variable_serialization_spec(self):
        """Returns a dict mapping quantization modes to variable names in order.

        This spec is used by `save_own_variables` and `load_own_variables` to
        determine the correct ordering of variables during serialization for
        each quantization mode. `None` means no quantization.
        """
        return {
            None: [
                "embeddings",
            ],
            "int8": [
                "embeddings",
                "embeddings_scale",
            ],
            "int4": [
                "embeddings",
                "embeddings_scale",
            ],
        }

    def quantized_build(self, embeddings_shape, mode, config=None):
        if mode == "int8":
            self._int8_build(embeddings_shape, config)
        elif mode == "int4":
            self._int4_build(embeddings_shape, config)
        else:
            raise self._quantization_mode_error(mode)
        self._is_quantized = True

    def _int8_build(self, embeddings_shape, config=None):
        self._embeddings = self.add_weight(
            name="embeddings",
            shape=embeddings_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        # We choose to reduce the axis of `output_dim` because, typically,
        # `input_dim` is larger than `output_dim`. This reduces quantization
        # error.
        self.embeddings_scale = self.add_weight(
            name="embeddings_scale",
            shape=(self.input_dim,),
            initializer="ones",
            trainable=False,
        )

    def _int4_build(self, embeddings_shape, config=None):
        input_dim, output_dim = embeddings_shape
        packed_rows = (output_dim + 1) // 2  # ceil for odd dims

        # Embeddings are stored *packed*: each int8 byte contains two int4
        # values.
        self._embeddings = self.add_weight(
            name="embeddings",
            shape=(input_dim, packed_rows),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        self.embeddings_scale = self.add_weight(
            name="embeddings_scale",
            shape=(self.input_dim,),
            initializer="ones",
            trainable=False,
        )
        # Record original output_dim for unpacking at runtime.
        self._orig_output_dim = output_dim

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
            outputs = ops.add(
                outputs, (self.lora_alpha / self.lora_rank) * lora_outputs
            )
        return outputs

    def _int4_call(self, inputs, training=None):
        # We cannot update quantized self._embeddings, so the custom gradient is
        # not needed
        if backend.standardize_dtype(inputs.dtype) not in ("int32", "int64"):
            inputs = ops.cast(inputs, "int32")
        embeddings_scale = ops.take(self.embeddings_scale, inputs, axis=0)
        unpacked_embeddings = quantizers.unpack_int4(
            self._embeddings, self._orig_output_dim, axis=-1
        )
        outputs = ops.take(unpacked_embeddings, inputs, axis=0)
        # De-scale outputs
        outputs = ops.divide(
            ops.cast(outputs, dtype=self.compute_dtype),
            ops.expand_dims(embeddings_scale, axis=-1),
        )
        if self.lora_enabled:
            lora_outputs = ops.take(self.lora_embeddings_a, inputs, axis=0)
            lora_outputs = ops.matmul(lora_outputs, self.lora_embeddings_b)
            outputs = ops.add(
                outputs, (self.lora_alpha / self.lora_rank) * lora_outputs
            )
        return outputs

    def quantize(self, mode=None, type_check=True, config=None):
        # Prevent quantization of the subclasses.
        if type_check and (type(self) is not Embedding):
            raise self._not_implemented_error(self.quantize)

        self.quantization_config = config

        embeddings_shape = (self.input_dim, self.output_dim)
        if mode == "int8":
            # Quantize `self._embeddings` to int8 and compute corresponding
            # scale.
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                self.quantization_config,
                quantizers.AbsMaxQuantizer(axis=-1),
            )
            embeddings_value, embeddings_scale = weight_quantizer(
                self._embeddings, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            del self._embeddings
            self.quantized_build(
                embeddings_shape, mode, self.quantization_config
            )
            self._embeddings.assign(embeddings_value)
            self.embeddings_scale.assign(embeddings_scale)
        elif mode == "int4":
            # Quantize to int4 values (stored in int8 dtype, range [-8, 7]).
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                self.quantization_config,
                quantizers.AbsMaxQuantizer(
                    axis=-1,
                    value_range=(-8, 7),
                    output_dtype="int8",
                ),
            )
            embeddings_value, embeddings_scale = weight_quantizer(
                self._embeddings, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            # 2. Pack two int4 values into a single int8 byte.
            packed_embeddings_value, _, _ = quantizers.pack_int4(
                embeddings_value, axis=-1
            )
            del self._embeddings
            self.quantized_build(
                embeddings_shape, mode, self.quantization_config
            )
            self._embeddings.assign(packed_embeddings_value)
            self.embeddings_scale.assign(embeddings_scale)
        else:
            raise self._quantization_mode_error(mode)

        # Set new dtype policy.
        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy

    def _get_embeddings_with_merged_lora(self):
        """Returns the embeddings with LoRA matrices merged, for serialization.

        This method is called by `save_own_variables` to produce a single
        embeddings tensor that includes the adaptations from LoRA. This is
        useful for deploying the model or for continuing training after
        permanently applying the LoRA update.

        If the layer is quantized (`int8` or `int4`), the process is:
        1. Dequantize the base embeddings to float.
        2. Compute the LoRA delta (`lora_embeddings_a @ lora_embeddings_b`) and
            add it to the dequantized embeddings.
        3. Re-quantize the merged result back to the original quantized
            type (`int8` or packed `int4`), calculating a new scale factor.

        If the layer is not quantized, this method returns the result of the
        `embeddings` property (which computes the merge in floating-point) and a
        scale of `None`.

        If LoRA is not enabled, it returns the original embeddings and scale
        without modification.

        Returns:
            A tuple `(embeddings_value, embeddings_scale)`:
                `embeddings_value`: The merged embeddings. A quantized tensor if
                    quantization is active, otherwise a high precision tensor.
                `embeddings_scale`: The quantization scale for the merged
                    embeddings. This is `None` if the layer is not quantized.
        """
        if self.dtype_policy.quantization_mode in (None, "gptq"):
            return self.embeddings, None

        embeddings_value = self._embeddings
        embeddings_scale = self.embeddings_scale
        if not self.lora_enabled:
            return embeddings_value, embeddings_scale

        # Dequantize embeddings to float.
        if self.quantization_mode == "int4":
            unpacked_embeddings = quantizers.unpack_int4(
                embeddings_value, self._orig_output_dim, axis=-1
            )
            float_embeddings = ops.divide(
                ops.cast(unpacked_embeddings, self.compute_dtype),
                ops.expand_dims(embeddings_scale, axis=-1),
            )
            quant_range = (-8, 7)
        elif self.quantization_mode == "int8":
            float_embeddings = ops.divide(
                ops.cast(embeddings_value, self.compute_dtype),
                ops.expand_dims(embeddings_scale, axis=-1),
            )
            quant_range = (-127, 127)
        else:
            raise ValueError(
                f"Unsupported quantization mode: {self.quantization_mode}"
            )

        # Merge LoRA weights in float domain.
        lora_delta = (self.lora_alpha / self.lora_rank) * ops.matmul(
            self.lora_embeddings_a, self.lora_embeddings_b
        )
        merged_float_embeddings = ops.add(float_embeddings, lora_delta)

        # Requantize.
        requantized_embeddings, embeddings_scale = quantizers.abs_max_quantize(
            merged_float_embeddings,
            axis=-1,
            value_range=quant_range,
            dtype="int8",
            to_numpy=True,
        )
        embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)

        # Pack if int4.
        if self.quantization_mode == "int4":
            embeddings_value, _, _ = quantizers.pack_int4(
                requantized_embeddings, axis=-1
            )
        else:
            embeddings_value = requantized_embeddings
        return embeddings_value, embeddings_scale

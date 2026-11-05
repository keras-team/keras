from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.common import add_lookup_lora_delta
from keras.src.quantizers.modes.common import add_reverse_lookup_lora_delta
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.common import apply_logit_soft_cap
from keras.src.quantizers.modes.common import cast_lookup_inputs
from keras.src.quantizers.modes.common import encode_reverse_lookup
from keras.src.quantizers.modes.common import reverse_lookup_dtype
from keras.src.quantizers.modes.common import reverse_lookup_params
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.quantizers import AbsMaxQuantizer

# Symmetric int8 codes with a per-channel divisor scale.
_INT8_SCHEME = WeightScheme(code_range=(-127, 127), scale_form="divisor")


class Int8Strategy(GeometryDispatchStrategy):
    """W8A8 dynamic quantization (int8 weights times int8 activations).

    One projection implementation serves every kernel contracted against
    its inputs: the geometry says how to contract, which axes the
    quantizers reduce over, and how a scale lines up with the kernel and
    with the outputs.
    """

    name = "int8"
    config_cls = Int8QuantizationConfig
    geometry_families = ("projection", "lookup")

    # --- Projection (Dense, EinsumDense) ----------------------------------

    def _build_projection(self, layer, geometry, kernel_shape, config):
        geometry.prepare()
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config, AbsMaxQuantizer()
            )
        )
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=geometry.kernel_scale_shape(kernel_shape),
            initializer="ones",
            trainable=False,
        )

    def _call_projection(self, layer, inputs, training=None):
        geometry = layer._quantization_geometry()
        view = self._get_projection_quantized_weight(layer, geometry)

        @ops.custom_gradient
        def contract_with_inputs_gradient(inputs, *tensors):
            """Contracts against the int8 kernel with a custom gradient.

            `tensors` are the view's stored tensors. Autodiff cannot
            differentiate through the int8 kernel, so the gradient with
            respect to the inputs is taken through the dequantized kernel.
            """
            quantized_weight = view.with_tensors(tensors)

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = quantized_weight.dequantize(layer.compute_dtype)
                inputs_grad = geometry.contract_grad(upstream, float_kernel)
                return (inputs_grad,) + (None,) * len(tensors)

            # The int8 scale is stored in the outputs' layout, so it de-scales
            # the integer contraction directly.
            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=geometry.inputs_quantization_axis
                )
                output_scale = ops.multiply(
                    geometry.align_inputs_scale(inputs_scale),
                    quantized_weight.scale,
                )
            else:
                # Weight-only: contract against the int8 kernel and de-scale
                # the outputs.
                output_scale = quantized_weight.scale
            x = geometry.contract(inputs, quantized_weight.codes)
            x = ops.cast(x, layer.compute_dtype)
            x = ops.divide(x, output_scale)
            return x, grad_fn

        # Read inside the autocast scope: on TensorFlow eager the gradient
        # runs after it, and the scale variable would then read float32.
        x = contract_with_inputs_gradient(inputs, *view.read_tensors())
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        geometry.prepare()
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config, AbsMaxQuantizer(axis=geometry.kernel_reduced_axes)
        )
        kernel_value, kernel_scale = weight_quantizer(weight, to_numpy=True)
        return (
            kernel_value,
            geometry.kernel_scale_for_storage(kernel_scale),
            None,
        )

    def _get_projection_quantized_weight(self, layer, geometry):
        # A matmul kernel's scale is shared along its input axis. An einsum
        # kernel's is stored in the outputs' layout, and the geometry lays
        # it back out against the kernel.
        axis = geometry.kernel_scale_axis
        return QuantizedWeight(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=NoPack(),
            scheme=_INT8_SCHEME,
            shape=geometry.weight_shape,
            axis=axis,
            align_scale=(
                None if axis is not None else geometry.kernel_scale_for_dequant
            ),
        )

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, _ = self._encode_projection(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, "int8", config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)

    # --- Embeddings lookup (Embedding, ReversibleEmbedding) ---------------

    def _build_lookup(self, layer, geometry, embeddings_shape, config):
        layer._embeddings = layer.add_weight(
            name="embeddings",
            shape=embeddings_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        # We choose to reduce the axis of `output_dim` because, typically,
        # `input_dim` is larger than `output_dim`. This reduces quantization
        # error.
        layer.embeddings_scale = layer.add_weight(
            name="embeddings_scale",
            shape=(layer.input_dim,),
            initializer="ones",
            trainable=False,
        )
        if geometry.reversible:
            layer.inputs_quantizer = (
                QuantizationConfig.activation_quantizer_or_default(
                    config, AbsMaxQuantizer(axis=-1)
                )
            )
            if not layer.tie_weights:
                layer.reverse_embeddings = layer.add_weight(
                    name="reverse_embeddings",
                    shape=(layer.output_dim, layer.input_dim),
                    initializer="zeros",
                    dtype="int8",
                    trainable=False,
                )
                layer.reverse_embeddings_scale = layer.add_weight(
                    name="reverse_embeddings_scale",
                    shape=(layer.input_dim,),
                    initializer="ones",
                    trainable=False,
                )

    def _call_lookup(self, layer, inputs, reverse=False):
        if reverse:
            return self._reverse_lookup(layer, inputs)
        # We cannot update quantized layer._embeddings, so the custom
        # gradient is not needed
        inputs = cast_lookup_inputs(inputs)
        embeddings_scale = ops.take(layer.embeddings_scale, inputs, axis=0)
        outputs = ops.take(layer._embeddings, inputs, axis=0)
        # De-scale outputs
        outputs = ops.divide(
            ops.cast(outputs, dtype=layer.compute_dtype),
            ops.expand_dims(embeddings_scale, axis=-1),
        )
        return add_lookup_lora_delta(layer, inputs, outputs)

    def _reverse_lookup(self, layer, inputs):
        dtype = reverse_lookup_dtype(layer)
        inputs = ops.cast(inputs, dtype)
        kernel, scale, _ = reverse_lookup_params(layer)
        if layer.inputs_quantizer:
            inputs_q, inputs_scale = layer.inputs_quantizer(inputs)
        else:
            inputs_q, inputs_scale = inputs, ops.ones((1,), dtype=dtype)
        logits = ops.matmul(inputs_q, kernel)
        # De-scale outputs
        logits = ops.cast(logits, dtype)
        logits = ops.divide(logits, ops.multiply(inputs_scale, scale))
        # The scale is a float32 variable; the projection reports its own
        # dtype, as the float layer does.
        logits = ops.cast(logits, dtype)
        logits = add_reverse_lookup_lora_delta(layer, inputs, logits)
        return apply_logit_soft_cap(layer, logits)

    def _encode_lookup(self, layer, geometry, weight, config):
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config,
            AbsMaxQuantizer(axis=-1),
        )
        embeddings_value, embeddings_scale = weight_quantizer(
            weight, to_numpy=True
        )
        return embeddings_value, ops.squeeze(embeddings_scale, axis=-1), None

    def _get_lookup_quantized_weight(self, layer, geometry):
        return QuantizedWeight(
            codes=layer._embeddings,
            scale=layer.embeddings_scale,
            layout=NoPack(),
            scheme=_INT8_SCHEME,
            shape=(layer.input_dim, layer.output_dim),
            axis=-1,
        )

    def _get_reverse_lookup_quantized_weight(self, layer, geometry):
        # A tied layer's reverse table is its forward table transposed.
        codes, scale, _ = reverse_lookup_params(layer)
        return QuantizedWeight(
            codes=codes,
            scale=scale,
            layout=NoPack(),
            scheme=_INT8_SCHEME,
            shape=(layer.output_dim, layer.input_dim),
            axis=0,
        )

    def _quantize_lookup(self, layer, geometry, config):
        embeddings_shape = geometry.weight_shape
        embeddings_value, embeddings_scale, _ = self._encode_lookup(
            layer, geometry, layer._embeddings, config
        )
        del layer._embeddings
        untied = geometry.reversible and not layer.tie_weights
        if untied:
            reverse_value, reverse_scale, _ = encode_reverse_lookup(
                self, layer, geometry, config
            )
            del layer.reverse_embeddings
        layer.quantized_build(embeddings_shape, "int8", config)
        layer._embeddings.assign(embeddings_value)
        layer.embeddings_scale.assign(embeddings_scale)
        if untied:
            layer.reverse_embeddings.assign(reverse_value)
            layer.reverse_embeddings_scale.assign(reverse_scale)

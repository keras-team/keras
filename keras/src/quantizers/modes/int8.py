from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.common import add_lookup_lora_delta
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.common import apply_logit_soft_cap
from keras.src.quantizers.modes.common import cast_lookup_inputs
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer


class Int8Strategy(GeometryDispatchStrategy):
    """W8A8 dynamic quantization (int8 weights times int8 activations).

    One projection implementation serves every kernel contracted against
    its inputs: the geometry says how to contract, which axes the
    quantizers reduce over, and how a scale lines up with the kernel and
    with the outputs.
    """

    name = "int8"
    config_cls = Int8QuantizationConfig

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

        @ops.custom_gradient
        def contract_with_inputs_gradient(inputs, kernel, kernel_scale):
            """Contracts against the int8 kernel with a custom gradient.

            Autodiff cannot differentiate through the int8 kernel, so the
            gradient with respect to the inputs is taken through the
            dequantized kernel.
            """

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=layer.compute_dtype),
                    geometry.kernel_scale_for_dequant(kernel_scale),
                )
                return (
                    geometry.contract_grad(upstream, float_kernel),
                    None,
                    None,
                )

            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=geometry.inputs_quantization_axis
                )
                output_scale = ops.multiply(
                    geometry.align_inputs_scale(inputs_scale), kernel_scale
                )
            else:
                # Weight-only: contract against the int8 kernel and de-scale
                # the outputs.
                output_scale = kernel_scale
            x = geometry.contract(inputs, kernel)
            x = ops.cast(x, layer.compute_dtype)
            x = ops.divide(x, output_scale)
            return x, grad_fn

        x = contract_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(layer._kernel),
            ops.convert_to_tensor(layer.kernel_scale),
        )
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        geometry.prepare()
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config, AbsMaxQuantizer(axis=geometry.kernel_reduced_axes)
        )
        kernel_value, kernel_scale = weight_quantizer(
            layer._kernel, to_numpy=True
        )
        kernel_scale = geometry.kernel_scale_for_storage(kernel_scale)
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

    def _call_lookup(self, layer, inputs, training=None):
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

    def _call_reversible_lookup(self, layer, inputs, reverse=False):
        if not reverse:
            return self._call_lookup(layer, inputs)
        else:
            if layer.tie_weights:
                kernel = ops.transpose(layer._embeddings)
                scale = ops.transpose(layer.embeddings_scale)
            else:
                kernel = layer.reverse_embeddings
                scale = layer.reverse_embeddings_scale
            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(inputs)
            else:
                inputs_scale = ops.ones((1,), dtype=layer.compute_dtype)
            logits = ops.matmul(inputs, kernel)
            # De-scale outputs
            logits = ops.cast(logits, layer.compute_dtype)
            logits = ops.divide(logits, ops.multiply(inputs_scale, scale))
            return apply_logit_soft_cap(layer, logits)

    def _quantize_lookup(self, layer, geometry, config):
        embeddings_shape = (layer.input_dim, layer.output_dim)
        # Quantize `layer._embeddings` to int8 and compute corresponding
        # scale.
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config,
            AbsMaxQuantizer(axis=-1),
        )
        embeddings_value, embeddings_scale = weight_quantizer(
            layer._embeddings, to_numpy=True
        )
        embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
        del layer._embeddings
        untied = geometry.reversible and not layer.tie_weights
        if untied:
            reverse_weight_quantizer = (
                QuantizationConfig.weight_quantizer_or_default(
                    config,
                    AbsMaxQuantizer(axis=0),
                )
            )
            reverse_embeddings_value, reverse_embeddings_scale = (
                reverse_weight_quantizer(
                    layer.reverse_embeddings, to_numpy=True
                )
            )
            reverse_embeddings_scale = ops.squeeze(
                reverse_embeddings_scale, axis=0
            )
            del layer.reverse_embeddings
        layer.quantized_build(embeddings_shape, "int8", config)
        layer._embeddings.assign(embeddings_value)
        layer.embeddings_scale.assign(embeddings_scale)
        if untied:
            layer.reverse_embeddings.assign(reverse_embeddings_value)
            layer.reverse_embeddings_scale.assign(reverse_embeddings_scale)

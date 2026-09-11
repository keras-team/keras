from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.common import apply_bias_activation
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

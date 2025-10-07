"""
Utility functions for loading variables with sharded support.

This module provides common utilities for loading variables that may be sharded
across multiple devices, which is useful for distributed training scenarios.
"""


def get_quantized_variable_load_order(layer):
    """
    Determine the order of variables to load for quantized layers.

    This function handles the complex logic for ordering variables during legacy
    loading, which varies based on quantization mode. The ordering is important
    because the keys in the store are saved in this specific order.

    Args:
        layer: The layer instance with quantization attributes.

    Returns:
        List of variables in the order they should be loaded.

    Raises:
        ValueError: If the quantization mode is not supported.
    """
    # Determine if bias should be included and how it's accessed
    has_bias = (
        getattr(layer, "use_bias", None)
        if hasattr(layer, "use_bias")
        else (layer.bias is not None)
    )
    bias_var = layer.bias if has_bias else None

    # Start with the main kernel variable
    if layer.quantization_mode == "gptq":
        # GPTQ: bias first (if present), then quantized_kernel
        target_variables = [bias_var] if bias_var is not None else []
        target_variables.append(layer.quantized_kernel)
    else:
        # Standard case: kernel first
        target_variables = [layer._kernel]

    # Add bias if present and not already added (not GPTQ)
    if bias_var is not None and layer.quantization_mode != "gptq":
        target_variables.append(bias_var)

    # Add quantization-specific variables
    if layer.quantization_mode is not None:
        if layer.quantization_mode in ("int8", "int4"):
            target_variables.append(layer.kernel_scale)
        elif layer.quantization_mode == "float8":
            target_variables.extend(
                [
                    layer.inputs_scale,
                    layer.inputs_amax_history,
                    layer.kernel_scale,
                    layer.kernel_amax_history,
                    layer.outputs_grad_scale,
                    layer.outputs_grad_amax_history,
                ]
            )
        elif layer.quantization_mode == "gptq":
            target_variables.extend(
                [
                    layer.kernel_scale,
                    layer.kernel_zero,
                    layer.g_idx,
                ]
            )
        else:
            # This should be handled by the layer's _quantization_mode_error
            raise ValueError(
                f"Unsupported quantization mode: {layer.quantization_mode}"
            )

    return target_variables

"""
Patch for torch.onnx.export to handle dynamic shape conflicts automatically.

This module provides a patched version of torch.onnx.export that automatically
falls back to TorchScript export when TorchDynamo export fails with dynamic
shape conflicts, specifically addressing GitHub issue #22102.
"""

import contextlib
import warnings


def patched_onnx_export(
    model,
    args,
    f=None,
    export_params=True,
    verbose=None,
    training=None,
    input_names=None,
    output_names=None,
    operator_export_type=None,
    opset_version=None,
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    custom_opsets=None,
    export_modules_as_functions=False,
    autograd_inlining=True,
    dynamic_shapes=None,
    **kwargs,
):
    """
    Patched version of torch.onnx.export that handles dynamic shape conflicts.

    Attempts TorchDynamo export first, but automatically falls back to
    TorchScript export when dynamic shape conflicts occur.
    """
    import torch

    # Store original export function
    if not hasattr(torch.onnx, "_original_export"):
        torch.onnx._original_export = torch.onnx.export

    # If dynamic_shapes is specified, try TorchDynamo first, then fallback
    if dynamic_shapes is not None:
        try:
            # Attempt TorchDynamo export with suppressed warnings
            with _suppress_export_warnings():
                return torch.onnx._original_export(
                    model=model,
                    args=args,
                    f=f,
                    export_params=export_params,
                    verbose=verbose,
                    training=training,
                    input_names=input_names,
                    output_names=output_names,
                    operator_export_type=operator_export_type,
                    opset_version=opset_version,
                    do_constant_folding=do_constant_folding,
                    dynamic_axes=dynamic_axes,
                    keep_initializers_as_inputs=keep_initializers_as_inputs,
                    custom_opsets=custom_opsets,
                    export_modules_as_functions=export_modules_as_functions,
                    autograd_inlining=autograd_inlining,
                    dynamic_shapes=dynamic_shapes,
                    **kwargs,
                )
        except Exception as e:
            # Check if it's a dynamic shape conflict error
            error_msg = str(e)
            if (
                "conflicts between user-specified ranges and inferred ranges"
                in error_msg
                or "tracing inferred a static shape" in error_msg
            ):
                # Fallback to TorchScript export with suppressed warnings
                with _suppress_export_warnings():
                    return torch.onnx._original_export(
                        model=model,
                        args=args,
                        f=f,
                        export_params=export_params,
                        verbose=verbose,
                        training=(
                            torch.onnx.TrainingMode.EVAL
                            if training is None
                            else training
                        ),
                        input_names=input_names,
                        output_names=output_names,
                        operator_export_type=operator_export_type,
                        opset_version=opset_version,
                        do_constant_folding=do_constant_folding,
                        dynamic_axes=None,  # Skip dynamic_axes for TorchScript
                        keep_initializers_as_inputs=keep_initializers_as_inputs,
                        custom_opsets=custom_opsets,
                        export_modules_as_functions=export_modules_as_functions,
                        autograd_inlining=autograd_inlining,
                        dynamo=False,  # Force TorchScript
                        **kwargs,
                    )
            else:
                # Re-raise if it's not a dynamic shape conflict
                raise

    # For non-dynamic_shapes cases, use original export
    return torch.onnx._original_export(
        model=model,
        args=args,
        f=f,
        export_params=export_params,
        verbose=verbose,
        training=training,
        input_names=input_names,
        output_names=output_names,
        operator_export_type=operator_export_type,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        custom_opsets=custom_opsets,
        export_modules_as_functions=export_modules_as_functions,
        autograd_inlining=autograd_inlining,
        **kwargs,
    )


@contextlib.contextmanager
def _suppress_export_warnings():
    """Context manager to suppress common ONNX export warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Exporting a model while it is in training mode"
        )
        warnings.filterwarnings(
            "ignore",
            message="Provided key .* for dynamic axes is not a valid "
            "input/output name",
        )
        warnings.filterwarnings(
            "ignore", message="Converting a tensor to a Python.*"
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="torch.onnx"
        )
        yield


def apply_onnx_export_patch():
    """Apply the ONNX export patch to handle dynamic shape conflicts."""
    import torch

    if not hasattr(torch.onnx, "_original_export"):
        torch.onnx._original_export = torch.onnx.export
        torch.onnx.export = patched_onnx_export


def remove_onnx_export_patch():
    """Remove the ONNX export patch and restore original behavior."""
    import torch

    if hasattr(torch.onnx, "_original_export"):
        torch.onnx.export = torch.onnx._original_export
        delattr(torch.onnx, "_original_export")

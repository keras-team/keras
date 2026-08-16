"""PyTorch ExportedProgram export utilities."""

import inspect
import warnings

from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils


def export_torch(
    model,
    filepath,
    input_signature=None,
    verbose=None,
    **kwargs,
):
    """Export the model as a PyTorch ExportedProgram (`.pt2`) artifact.

    Uses `torch.export.export` to capture the model's computation graph
    in an Ahead-of-Time (AOT) fashion. The resulting artifact contains
    only ATen-level operations and can be loaded via `torch.export.load`,
    run with `ExportedProgram.module()`, or compiled further with
    `torch.compile`.

    Args:
        model: The Keras model to export.
        filepath: `str` or `pathlib.Path` object. Path to save the
            exported artifact. Must end with `.pt2`.
        input_signature: Optional input signature. If `None`, inferred
            from the model's built input spec.
        verbose: `bool`. Whether to print a message after export.
            Defaults to `True`.
        **kwargs: Additional keyword arguments forwarded to
            `torch.export.export`, including `strict`, `dynamic_shapes`,
            `prefer_deferred_runtime_asserts_over_guards`, and
            `preserve_module_call_signature`. For a single-input model,
            `dynamic_shapes` can be a dict such as
            `{0: torch.export.Dim("batch", min=1, max=128)}`.
            Note: when `dynamic_shapes` is provided, the sample input used
            for tracing uses a batch size of 2 instead of 1. This avoids a
            PyTorch limitation where dimensions of size 1 are specialized
            to constants during export.

    Example:

    ```python
    model.export("path/to/model.pt2", format="torch")

    import torch
    loaded_program = torch.export.load("path/to/model.pt2")
    output = loaded_program.module()(torch.randn(1, 10))
    ```
    """
    import torch

    filepath = str(filepath)
    if not filepath.endswith(".pt2"):
        raise ValueError(
            "The PyTorch export requires the filepath to end with "
            f"'.pt2'. Got: {filepath}"
        )

    export_kwargs = _get_export_kwargs(kwargs)
    dynamic_shapes = export_kwargs.get("dynamic_shapes")

    if input_signature is None:
        input_signature = get_input_signature(model)

    # PyTorch limitation: torch.export specializes dimensions of size 1 to
    # constants during tracing. If a dynamic dim (e.g., batch) has a concrete
    # sample value of 1, export fails with "specialized it to be a constant".
    # Using a sample value of 2 (the smallest integer > 1) avoids this.
    # See: https://github.com/pytorch/pytorch/issues/176349
    replace_none_number = 2 if dynamic_shapes is not None else 1
    sample_inputs = tree.map_structure(
        lambda x: convert_spec_to_tensor(
            x, replace_none_number=replace_none_number
        ),
        input_signature,
    )
    sample_inputs = tuple(sample_inputs)
    if dynamic_shapes is not None:
        export_kwargs["dynamic_shapes"] = _normalize_dynamic_shapes(
            dynamic_shapes, sample_inputs
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*not properly registered as a submodule.*",
        )
        try:
            exported_program = torch.export.export(
                model,
                sample_inputs,
                **export_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to export model to PyTorch format. "
                "Common causes: unsupported operations, data-dependent "
                "control flow, or shape constraints not satisfied. "
                f"Original error: {e}"
            ) from e

    torch.export.save(exported_program, filepath)

    actual_verbose = verbose if verbose is not None else True
    if actual_verbose:
        io_utils.print_msg(f"Saved PyTorch ExportedProgram at '{filepath}'.")
    return filepath


def _get_export_kwargs(user_kwargs):
    import torch

    supported_arg_names = {
        name
        for name in inspect.signature(torch.export.export).parameters
        if name not in {"mod", "args", "kwargs"}
    }
    unknown_arg_names = sorted(set(user_kwargs).difference(supported_arg_names))
    if unknown_arg_names:
        supported_args = ", ".join(sorted(supported_arg_names))
        unknown_args = ", ".join(unknown_arg_names)
        raise ValueError(
            'Unsupported arguments for `format="torch"`: '
            f"{unknown_args}. Supported arguments are: {supported_args}."
        )

    export_kwargs = {
        name: value
        for name, value in user_kwargs.items()
        if name in supported_arg_names
    }
    # Default to non-strict mode for broader Keras model compatibility.
    # Strict mode requires all operations to be fully ATen-traceable, which
    # is too restrictive for Keras models that use Python-level structure.
    export_kwargs.setdefault("strict", False)
    return export_kwargs


def _normalize_dynamic_shapes(dynamic_shapes, sample_inputs):
    if isinstance(dynamic_shapes, dict):
        if "args" in dynamic_shapes or "kwargs" in dynamic_shapes:
            return dynamic_shapes
        # Keras models use `forward(*args, **kwargs)`. torch.export treats
        # the entire `args` tuple as a single positional argument when the
        # forward signature is `*args`. Therefore dynamic_shapes must be a
        # list/tuple of length 1, where the single element is the spec for
        # the args tuple. For a single-input model, args = (tensor,), so
        # the spec is (dynamic_shapes,).
        if len(sample_inputs) == 1:
            return [(dynamic_shapes,)]
        return dynamic_shapes

    if isinstance(dynamic_shapes, (list, tuple)):
        # Already wrapped, e.g. [({0: Dim},)] or [([spec1, spec2],)].
        if len(dynamic_shapes) == 1 and isinstance(
            dynamic_shapes[0], (list, tuple)
        ):
            return dynamic_shapes
        # Bare list/tuple matching sample_inputs length (always 1 for
        # Keras). Wrap it in the list required by torch.export's *args
        # handling.
        if len(dynamic_shapes) == len(sample_inputs):
            return [tuple(dynamic_shapes)]

    return dynamic_shapes

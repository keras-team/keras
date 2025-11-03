import warnings

from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_tf_tensor_spec
from keras.src.export.saved_model import DEFAULT_ENDPOINT_NAME
from keras.src.export.saved_model import ExportArchive
from keras.src.export.tf2onnx_lib import patch_tf2onnx
from keras.src.utils import io_utils


def export_onnx(
    model,
    filepath,
    verbose=None,
    input_signature=None,
    opset_version=None,
    **kwargs,
):
    """Export the model as a ONNX artifact for inference.

    This method lets you export a model to a lightweight ONNX artifact
    that contains the model's forward pass only (its `call()` method)
    and can be served via e.g. ONNX Runtime.

    The original code of the model (including any custom layers you may
    have used) is *no longer* necessary to reload the artifact -- it is
    entirely standalone.

    Args:
        filepath: `str` or `pathlib.Path` object. The path to save the artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which uses the default value set by different backends and
            formats.
        input_signature: Optional. Specifies the shape and dtype of the model
            inputs. Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
            `backend.KerasTensor`, or backend tensor. If not provided, it will
            be automatically computed. Defaults to `None`.
        opset_version: Optional. An integer value that specifies the ONNX opset
            version. If not provided, the default version for the backend will
            be used. Defaults to `None`.
        **kwargs: Additional keyword arguments.

    **Note:** This feature is currently supported only with TensorFlow, JAX and
    Torch backends.

    **Note:** The dtype policy must be "float32" for the model. You can further
    optimize the ONNX artifact using the ONNX toolkit. Learn more here:
    [https://onnxruntime.ai/docs/performance/](https://onnxruntime.ai/docs/performance/).

    **Note:** The dynamic shape feature is not yet supported with Torch
    backend. As a result, you must fully define the shapes of the inputs using
    `input_signature`. If `input_signature` is not provided, all instances of
    `None` (such as the batch size) will be replaced with `1`.

    Example:

    ```python
    # Export the model as a ONNX artifact
    model.export("path/to/location", format="onnx")

    # Load the artifact in a different process/environment
    ort_session = onnxruntime.InferenceSession("path/to/location")
    ort_inputs = {
        k.name: v for k, v in zip(ort_session.get_inputs(), input_data)
    }
    predictions = ort_session.run(None, ort_inputs)
    ```
    """
    actual_verbose = verbose
    if actual_verbose is None:
        actual_verbose = True  # Defaults to `True` for all backends.

    if input_signature is None:
        input_signature = get_input_signature(model)
        if not input_signature or not model._called:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
            )
    input_names = [
        getattr(spec, "name", None) or f"input_{i}"
        for i, spec in enumerate(input_signature)
    ]

    if backend.backend() in ("tensorflow", "jax"):
        from keras.src.utils.module_utils import tf2onnx

        input_signature = tree.map_structure(
            make_tf_tensor_spec, input_signature
        )
        decorated_fn = get_concrete_fn(model, input_signature, **kwargs)

        # Use `tf2onnx` to convert the `decorated_fn` to the ONNX format.
        patch_tf2onnx()  # TODO: Remove this once `tf2onnx` supports numpy 2.
        tf2onnx.convert.from_function(
            decorated_fn,
            input_signature,
            opset=opset_version,
            output_path=filepath,
        )

    elif backend.backend() == "torch":
        import torch

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        sample_inputs = tuple(sample_inputs)
        # TODO: Make dict model exportable.
        if any(isinstance(x, dict) for x in sample_inputs):
            raise ValueError(
                "Currently, `export_onnx` in the torch backend doesn't support "
                "dictionaries as inputs."
            )

        if hasattr(model, "eval"):
            model.eval()
        with warnings.catch_warnings():
            # Suppress some unuseful warnings.
            warnings.filterwarnings(
                "ignore",
                message=r".*\n.*\n*.*\n*.*export will treat it as a constant.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*not properly registered as a submodule,.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*which is what 'get_attr' Nodes typically target.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*underlying reference in the owning GraphModule.*",
            )
            warnings.filterwarnings(
                "ignore", message=r".*suppressed about get_attr references.*"
            )
            try:
                # Try the TorchDynamo-based ONNX exporter first.
                onnx_program = torch.onnx.export(
                    model,
                    sample_inputs,
                    verbose=actual_verbose,
                    opset_version=opset_version,
                    input_names=input_names,
                    dynamo=True,
                )
                if hasattr(onnx_program, "optimize"):
                    onnx_program.optimize()  # Only supported by torch>=2.6.0.
                onnx_program.save(filepath)
            except:
                if verbose is None:
                    # Set to `False` due to file system leakage issue:
                    # https://github.com/keras-team/keras/issues/20826
                    actual_verbose = False

                # Fall back to the TorchScript-based ONNX exporter.
                torch.onnx.export(
                    model,
                    sample_inputs,
                    filepath,
                    verbose=actual_verbose,
                    opset_version=opset_version,
                    input_names=input_names,
                )
    else:
        raise NotImplementedError(
            "`export_onnx` is only compatible with TensorFlow, JAX and "
            "Torch backends."
        )

    if actual_verbose:
        io_utils.print_msg(f"Saved artifact at '{filepath}'.")


def _check_jax_kwargs(kwargs):
    kwargs = kwargs.copy()
    if "is_static" not in kwargs:
        kwargs["is_static"] = True
    if "jax2tf_kwargs" not in kwargs:
        # TODO: These options will be deprecated in JAX. We need to
        # find another way to export ONNX.
        kwargs["jax2tf_kwargs"] = {
            "enable_xla": False,
            "native_serialization": False,
        }
    if kwargs["is_static"] is not True:
        raise ValueError(
            "`is_static` must be `True` in `kwargs` when using the jax backend."
        )
    if kwargs["jax2tf_kwargs"]["enable_xla"] is not False:
        raise ValueError(
            "`enable_xla` must be `False` in `kwargs['jax2tf_kwargs']` "
            "when using the jax backend."
        )
    if kwargs["jax2tf_kwargs"]["native_serialization"] is not False:
        raise ValueError(
            "`native_serialization` must be `False` in "
            "`kwargs['jax2tf_kwargs']` when using the jax backend."
        )
    return kwargs


def get_concrete_fn(model, input_signature, **kwargs):
    """Get the `tf.function` associated with the model."""
    if backend.backend() == "jax":
        kwargs = _check_jax_kwargs(kwargs)
    export_archive = ExportArchive()
    export_archive.track_and_add_endpoint(
        DEFAULT_ENDPOINT_NAME, model, input_signature, **kwargs
    )
    if backend.backend() == "tensorflow":
        export_archive._filter_and_track_resources()
    return export_archive._get_concrete_fn(DEFAULT_ENDPOINT_NAME)

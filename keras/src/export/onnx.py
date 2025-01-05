from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_tf_tensor_spec
from keras.src.export.saved_model import DEFAULT_ENDPOINT_NAME
from keras.src.export.saved_model import ExportArchive
from keras.src.export.tf2onnx_lib import patch_tf2onnx


def export_onnx(model, filepath, verbose=True, input_signature=None, **kwargs):
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
            True`.
        input_signature: Optional. Specifies the shape and dtype of the model
            inputs. Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
            `backend.KerasTensor`, or backend tensor. If not provided, it will
            be automatically computed. Defaults to `None`.
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
    if input_signature is None:
        input_signature = get_input_signature(model)
        if not input_signature or not model._called:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
            )

    if backend.backend() in ("tensorflow", "jax"):
        from keras.src.utils.module_utils import tf2onnx

        input_signature = tree.map_structure(
            make_tf_tensor_spec, input_signature
        )
        decorated_fn = get_concrete_fn(model, input_signature, **kwargs)

        # Use `tf2onnx` to convert the `decorated_fn` to the ONNX format.
        patch_tf2onnx()  # TODO: Remove this once `tf2onnx` supports numpy 2.
        tf2onnx.convert.from_function(
            decorated_fn, input_signature, output_path=filepath
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

        # Convert to ONNX using TorchScript-based ONNX Exporter.
        # TODO: Use TorchDynamo-based ONNX Exporter once
        # `torch.onnx.dynamo_export()` supports Keras models.
        torch.onnx.export(model, sample_inputs, filepath, verbose=verbose)
    else:
        raise NotImplementedError(
            "`export_onnx` is only compatible with TensorFlow, JAX and "
            "Torch backends."
        )


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
            "`is_static` must be `True` in `kwargs` when using the jax "
            "backend."
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

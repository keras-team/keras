"""Library for exporting XLA SavedModel artifacts for Keras models."""

from keras.src.export.export_utils import get_input_signature
from keras.src.export.xla_saved_model_archive import DEFAULT_SERVING_SIGNATURE
from keras.src.export.xla_saved_model_archive import XLASavedModelArchive


def export_xla_saved_model(
    model, filepath, verbose=None, input_signature=None, **kwargs
):
    """Export the model as an XLA SavedModel artifact for inference.

    This method lets you export a model to the format served by the XLA
    Serving Engine (https://github.com/openxla/xla-serving). The artifact
    contains the StableHLO of the model's forward pass (captured with
    `jax.export`) and an Orbax checkpoint of its weights, and can be served
    on CPU, GPU or TPU without any Python. The forward pass is registered
    under the serving signature `"serving_default"`, which is the signature
    the XLA Serving Engine uses when a request does not specify one.

    Args:
        filepath: `str` or `pathlib.Path` object. The path to save the artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which is `True`.
        input_signature: Optional. Specifies the shape and dtype of the model
            inputs. Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
            `KerasTensor`, or backend tensor. If not provided, it will
            be automatically computed. Defaults to `None`.
        **kwargs: Additional keyword arguments forwarded to
            `ExportArchive.add_endpoint()`, e.g. `platforms`,
            `jax_export_kwargs`, `polymorphic_shapes`, `batch_options` or
            `warmup_examples`.

    **Note:** This feature requires the JAX backend and the
    `xla_saved_model` package.

    Example:

    ```python
    # Export the model as an XLA SavedModel artifact for CPU and TPU
    model.export(
        "path/to/location", format="xla_saved_model", platforms=["cpu", "tpu"]
    )
    ```
    """
    if verbose is None:
        verbose = True
    export_archive = XLASavedModelArchive()
    if input_signature is None:
        input_signature = get_input_signature(model)

    export_archive.track_and_add_endpoint(
        DEFAULT_SERVING_SIGNATURE, model, input_signature, **kwargs
    )
    export_archive.write_out(filepath, verbose=verbose)

import functools
import tempfile

from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils
from keras.src.utils.module_utils import tensorflow as tf


def export_litert(
    model,
    filepath,
    input_signature=None,
    **kwargs,
):
    """Export the model as a LiteRT artifact for inference.

    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact.
        input_signature: Optional input signature specification. If
            `None`, it will be inferred.
        **kwargs: Additional keyword arguments passed to the exporter.
    """

    exporter = LiteRTExporter(
        model=model,
        input_signature=input_signature,
        **kwargs,
    )
    exporter.export(filepath)
    io_utils.print_msg(f"Saved artifact at '{filepath}'.")


class LiteRTExporter:
    """Exporter for the LiteRT (TFLite) format.

    This class handles the conversion of Keras models for LiteRT runtime and
    generates a `.tflite` model file. For efficient inference on mobile and
    embedded devices, it creates a single callable signature based on the
    model's `call()` method.
    """

    def __init__(
        self,
        model,
        input_signature=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            model: The Keras model to export
            input_signature: Input signature specification (e.g., TensorFlow
                TensorSpec or list of TensorSpec)
            **kwargs: Additional export parameters
        """
        self.model = model
        self.input_signature = input_signature
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model
        """
        if self.input_signature is None:
            self.input_signature = get_input_signature(self.model)

        # Normalize input_signature for tf.function: must be a list/tuple
        # of specs, one per positional argument. Wrap bare structures as
        # a single argument.
        if not isinstance(self.input_signature, (list, tuple)):
            self.input_signature = [self.input_signature]

        if not filepath.endswith(".tflite"):
            raise ValueError(
                f"The LiteRT export requires the filepath to end with "
                f"'.tflite'. Got: {filepath}"
            )

        tflite_model = self._convert_to_tflite(self.input_signature)

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to TFLite format.

        Uses Keras ExportArchive as an intermediate SavedModel step.
        This aligns with TensorFlow's official Keras 3 TFLite conversion path:
        ExportArchive -> SavedModel -> from_saved_model.

        Args:
            input_signature: Input signature for the model to convert.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        from keras.src import export as keras_export

        with tempfile.TemporaryDirectory() as saved_model_dir:
            archive = keras_export.ExportArchive()
            archive.track(self.model)
            archive.add_endpoint(
                "serve",
                functools.partial(self.model.__call__, training=False),
                input_signature=input_signature,
            )
            archive.write_out(saved_model_dir, verbose=False)

            converter = tf.lite.TFLiteConverter.from_saved_model(
                saved_model_dir
            )
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.experimental_enable_resource_variables = True
            self._apply_converter_kwargs(converter)
            return converter.convert()

    def _apply_converter_kwargs(self, converter):
        """Apply additional converter settings from kwargs.

        Args:
            converter: tf.lite.TFLiteConverter instance to configure

        Raises:
            ValueError: If any kwarg is not a valid converter attribute
        """
        for attr, value in self.kwargs.items():
            if attr == "target_spec" and isinstance(value, dict):
                # Handle nested target_spec settings
                for spec_key, spec_value in value.items():
                    if hasattr(converter.target_spec, spec_key):
                        setattr(converter.target_spec, spec_key, spec_value)
                    else:
                        raise ValueError(
                            f"Unknown target_spec attribute '{spec_key}'"
                        )
            elif hasattr(converter, attr):
                setattr(converter, attr, value)
            else:
                raise ValueError(f"Unknown converter attribute '{attr}'")

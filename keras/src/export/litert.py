from keras.src import layers
from keras.src import models
from keras.src import tree
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
        # 1. Resolve / infer input signature
        if self.input_signature is None:
            # Use the standard get_input_signature which handles all model types
            # and preserves nested structures (dicts, lists, etc.)
            self.input_signature = get_input_signature(self.model)

        # 2. Determine input structure and create adapter if needed
        # There are 3 cases:
        # Case 1: Single input (not nested)
        # Case 2: Flat list of inputs (list where flattened == original)
        # Case 3: Nested structure (dicts, nested lists, etc.)

        # Special handling for Functional models: get_input_signature wraps
        # the structure in a list, so unwrap it for analysis
        input_struct = self.input_signature
        if (
            isinstance(self.input_signature, list)
            and len(self.input_signature) == 1
        ):
            input_struct = self.input_signature[0]

        if not tree.is_nested(input_struct):
            # Case 1: Single input - use as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature
        elif isinstance(input_struct, list) and len(input_struct) == len(
            tree.flatten(input_struct)
        ):
            # Case 2: Flat list of inputs - use as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature
        else:
            # Case 3: Nested structure (dict, nested lists, etc.)
            # Create adapter model that converts flat list to nested structure
            adapted_model = self._create_nested_inputs_adapter(input_struct)

            # Flatten signature for TFLite conversion
            signature_for_conversion = tree.flatten(input_struct)

            # Use adapted model and flat list signature for conversion
            model_to_convert = adapted_model

        # Store original model reference for later use
        original_model = self.model

        # Temporarily replace self.model with the model to convert
        self.model = model_to_convert

        try:
            # Convert the model to TFLite.
            tflite_model = self._convert_to_tflite(signature_for_conversion)
        finally:
            # Restore original model
            self.model = original_model

        # Save the TFLite model to the specified file path.
        if not filepath.endswith(".tflite"):
            raise ValueError(
                f"The LiteRT export requires the filepath to end with "
                f"'.tflite'. Got: {filepath}"
            )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _create_nested_inputs_adapter(self, input_signature_struct):
        """Create an adapter model that converts flat list inputs to nested
        structure.

        This adapter allows models expecting nested inputs (dicts, lists, etc.)
        to be exported to TFLite format (which only supports positional/list
        inputs).

        Args:
            input_signature_struct: Nested structure of InputSpecs (dict, list,
                etc.)

        Returns:
            A Functional model that accepts flat list inputs and converts to
            nested
        """
        # Get flat paths to preserve names and print input mapping
        paths_and_specs = tree.flatten_with_path(input_signature_struct)
        paths = [".".join(str(e) for e in p) for p, v in paths_and_specs]
        io_utils.print_msg(f"Creating adapter for inputs: {paths}")

        # Create Input layers for TFLite (flat list-based)
        input_layers = []
        for path, spec in paths_and_specs:
            # Extract the input name from spec or path
            name = (
                spec.name
                if hasattr(spec, "name") and spec.name
                else (str(path[-1]) if path else "input")
            )

            input_layer = layers.Input(
                shape=spec.shape[1:],  # Remove batch dimension
                dtype=spec.dtype,
                name=name,
            )
            input_layers.append(input_layer)

        # Reconstruct the nested structure from flat list
        inputs_structure = tree.pack_sequence_as(
            input_signature_struct, input_layers
        )

        # Call the original model with nested inputs
        outputs = self.model(inputs_structure)

        # Build as Functional model (flat list inputs -> nested -> model ->
        # output)
        adapted_model = models.Model(inputs=input_layers, outputs=outputs)

        # Preserve the original model's variables
        adapted_model._variables = self.model.variables
        adapted_model._trainable_variables = self.model.trainable_variables
        adapted_model._non_trainable_variables = (
            self.model.non_trainable_variables
        )

        return adapted_model

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to TFLite format.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        # Try direct conversion first for all models
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            # Keras 3 only supports resource variables
            converter.experimental_enable_resource_variables = True

            # Apply any additional converter settings from kwargs
            self._apply_converter_kwargs(converter)

            tflite_model = converter.convert()

            return tflite_model

        except Exception as e:
            # If direct conversion fails, raise the error with helpful message
            raise RuntimeError(
                f"Direct TFLite conversion failed. This may be due to model "
                f"complexity or unsupported operations. Error: {e}"
            ) from e

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

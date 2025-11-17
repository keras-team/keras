from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_input_spec
from keras.src.export.export_utils import make_tf_tensor_spec
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

    def _has_dict_inputs(self):
        """Check if the model expects dictionary inputs.

        Returns:
            bool: True if model has dict inputs, False otherwise.
        """
        # Check if model.inputs is a dict (most reliable for built models)
        if hasattr(self.model, "inputs") and isinstance(
            self.model.inputs, dict
        ):
            return True

        # Check if _inputs_struct is a dict (for Functional models)
        if hasattr(self.model, "_inputs_struct") and isinstance(
            self.model._inputs_struct, dict
        ):
            return True

        # Check if provided input_signature is a dict
        if self.input_signature is not None:
            if isinstance(self.input_signature, dict):
                return True
            # Check for wrapped dict (Functional model pattern)
            if (
                isinstance(self.input_signature, (list, tuple))
                and len(self.input_signature) == 1
                and isinstance(self.input_signature[0], dict)
            ):
                return True

        return False

    def _infer_dict_input_signature(self):
        """Infer input signature from a model with dict inputs.

        This reads the actual shapes and dtypes from model._inputs_struct.

        Returns:
            dict or None: Dictionary mapping input names to InputSpec, or None
        """
        # Check _inputs_struct first (preserves dict structure)
        if hasattr(self.model, "_inputs_struct") and isinstance(
            self.model._inputs_struct, dict
        ):
            return {
                name: make_input_spec(inp)
                for name, inp in self.model._inputs_struct.items()
            }

        # Fall back to model.inputs if it's a dict
        if hasattr(self.model, "inputs") and isinstance(
            self.model.inputs, dict
        ):
            return {
                name: make_input_spec(inp)
                for name, inp in self.model.inputs.items()
            }

        return None

    def export(self, filepath):
        """Exports the Keras model to a TFLite file.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model
        """
        # 1. Resolve / infer input signature
        if self.input_signature is None:
            # Try dict-specific inference first (for models with dict inputs)
            dict_signature = self._infer_dict_input_signature()
            if dict_signature is not None:
                self.input_signature = dict_signature
            else:
                # Fall back to standard inference
                self.input_signature = get_input_signature(self.model)

        # 3. Handle dictionary inputs by creating an adapter
        # Check if we have dict inputs that need adaptation
        has_dict_inputs = isinstance(self.input_signature, dict)

        if has_dict_inputs:
            # Create adapter model that converts list to dict
            adapted_model = self._create_dict_adapter(self.input_signature)

            # Convert dict signature to list for TFLite conversion
            # The adapter will handle the dict->list conversion
            input_signature_list = list(self.input_signature.values())

            # Use adapted model and list signature for conversion
            model_to_convert = adapted_model
            signature_for_conversion = input_signature_list
        else:
            # No dict inputs - use model as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature

        # Store original model reference for later use
        original_model = self.model

        # Temporarily replace self.model with the model to convert
        self.model = model_to_convert

        try:
            # 4. Convert the model to TFLite.
            tflite_model = self._convert_to_tflite(signature_for_conversion)
        finally:
            # Restore original model
            self.model = original_model

        # 4. Save the initial TFLite model to the specified file path.
        if not filepath.endswith(".tflite"):
            raise ValueError(
                "The LiteRT export requires the filepath to end with "
                "'.tflite'. Got: {filepath}"
            )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _create_dict_adapter(self, input_signature_dict):
        """Create an adapter model that converts list inputs to dict inputs.

        This adapter allows models expecting dictionary inputs to be exported
        to TFLite format (which only supports positional/list inputs).

        Args:
            input_signature_dict: Dictionary mapping input names to InputSpec

        Returns:
            A Functional model that accepts list inputs and converts to dict
        """
        io_utils.print_msg(
            f"Creating adapter for dictionary inputs: "
            f"{list(input_signature_dict.keys())}"
        )

        input_keys = list(input_signature_dict.keys())

        # Create Input layers for TFLite (list-based)
        input_layers = []
        for name in input_keys:
            spec = input_signature_dict[name]
            input_layer = tf.keras.layers.Input(
                shape=spec.shape[1:],  # Remove batch dimension
                dtype=spec.dtype,
                name=name,
            )
            input_layers.append(input_layer)

        # Create dict from list inputs
        inputs_dict = {
            name: layer for name, layer in zip(input_keys, input_layers)
        }

        # Call the original model with dict inputs
        outputs = self.model(inputs_dict)

        # Build as Functional model (list inputs -> dict -> model -> output)
        adapted_model = tf.keras.Model(inputs=input_layers, outputs=outputs)

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

        except Exception:
            return self._convert_with_wrapper(input_signature)

    def _convert_with_wrapper(self, input_signature):
        """Converts the model to TFLite using SavedModel as intermediate.

        This fallback method is used when direct Keras conversion fails.
        It uses TensorFlow's SavedModel format as an intermediate step.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        # Normalize input_signature to list format for concrete function
        if isinstance(input_signature, dict):
            # For multi-input models with dict signature, convert to
            # ordered list
            if hasattr(self.model, "inputs") and len(self.model.inputs) > 1:
                input_signature_list = []
                for input_layer in self.model.inputs:
                    input_name = input_layer.name
                    if input_name not in input_signature:
                        raise ValueError(
                            f"Missing input '{input_name}' in input_signature. "
                            f"Model expects inputs: "
                            f"{[inp.name for inp in self.model.inputs]}, "
                            f"but input_signature only has: "
                            f"{list(input_signature.keys())}"
                        )
                    input_signature_list.append(input_signature[input_name])
                input_signature = input_signature_list
            else:
                # Single-input model with dict signature
                input_signature = [input_signature]
        elif not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]

        # Convert to TensorSpec
        tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]

        # Get concrete function from the model
        @tf.function
        def model_fn(*args):
            return self.model(*args)

        concrete_func = model_fn.get_concrete_function(*tensor_specs)

        # Convert using concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func], self.model
        )
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

    def _apply_converter_kwargs(self, converter):
        """Apply additional converter settings from kwargs.

        This method applies TFLite converter settings passed via kwargs.
        Only known LiteRT/TFLite converter settings are applied. Other kwargs
        (like format-specific settings for other export formats) are ignored.

        Known LiteRT converter settings include:
        - optimizations: List of optimization options
        - representative_dataset: Dataset generator for quantization
        - experimental_new_quantizer: Enable experimental quantizer
        - allow_custom_ops: Allow custom operations
        - enable_select_tf_ops: Enable select TF ops
        - target_spec: Target specification settings
        - inference_input_type: Input type for inference
        - inference_output_type: Output type for inference
        - experimental_enable_resource_variables: Enable resource variables

        Args:
            converter: tf.lite.TFLiteConverter instance to configure
        """
        if not self.kwargs:
            return

        # Known TFLite converter attributes that can be set
        known_converter_attrs = {
            "optimizations",
            "representative_dataset",
            "experimental_new_quantizer",
            "allow_custom_ops",
            "enable_select_tf_ops",
            "target_spec",
            "inference_input_type",
            "inference_output_type",
            "experimental_enable_resource_variables",
        }

        for key, value in self.kwargs.items():
            if key == "target_spec" and isinstance(value, dict):
                # Handle nested target_spec settings
                for spec_key, spec_value in value.items():
                    if hasattr(converter.target_spec, spec_key):
                        setattr(converter.target_spec, spec_key, spec_value)
            elif key in known_converter_attrs and hasattr(converter, key):
                setattr(converter, key, value)
            elif hasattr(converter, key):
                # Allow any attribute that exists on the converter
                setattr(converter, key, value)
            else:
                io_utils.print_msg(
                    f"Warning: Unknown converter setting '{key}' - ignoring"
                )

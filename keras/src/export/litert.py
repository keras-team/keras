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
        assert filepath.endswith(".tflite"), (
            "The LiteRT export requires the filepath to end with '.tflite'. "
            f"Got: {filepath}"
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
            converter.experimental_enable_resource_variables = False

            # Apply any additional converter settings from kwargs
            self._apply_converter_kwargs(converter)

            tflite_model = converter.convert()

            return tflite_model

        except Exception:
            return self._convert_with_wrapper(input_signature)

    def _convert_with_wrapper(self, input_signature):
        """Converts the model to TFLite using the tf.Module wrapper.

        Returns:
            A bytes object containing the serialized TFLite model.
        """

        # Define the wrapper class dynamically to avoid module-level
        # tf.Module inheritance
        class KerasModelWrapper(tf.Module):
            """
            A tf.Module wrapper for a Keras model.

            This wrapper is designed to be a clean, serializable interface
            for TFLite conversion. It holds the Keras model and exposes a
            single `__call__` method that is decorated with `tf.function`.
            Crucially, it also ensures all variables from the Keras model
            are tracked by the SavedModel format, which is key to including
            them in the final TFLite model.
            """

            def __init__(self, model):
                super().__init__()
                # Store the model reference in a way that TensorFlow won't
                # try to track it. This prevents the _DictWrapper error during
                # SavedModel serialization
                object.__setattr__(self, "_model", model)

                # Track all variables from the Keras model using proper
                # tf.Module methods. This ensures proper variable handling for
                # stateful layers like BatchNorm
                with self.name_scope:
                    for i, var in enumerate(model.variables):
                        # Use a different attribute name to avoid conflicts with
                        # tf.Module's variables property
                        setattr(self, f"model_var_{i}", var)

            @tf.function
            def __call__(self, *args, **kwargs):
                """The single entry point for the exported model."""
                # Handle both single and multi-input cases
                if args and not kwargs:
                    # Called with positional arguments
                    if len(args) == 1:
                        return self._model(args[0])
                    else:
                        # Multi-input case: Functional models expect a list,
                        # not unpacked positional args
                        if (
                            hasattr(self._model, "inputs")
                            and len(self._model.inputs) > 1
                        ):
                            return self._model(list(args))
                        else:
                            return self._model(*args)
                elif kwargs and not args:
                    # Called with keyword arguments
                    if len(kwargs) == 1 and "inputs" in kwargs:
                        # Single input case
                        return self._model(kwargs["inputs"])
                    else:
                        # Multi-input case - convert to list/dict format
                        # expected by model
                        if (
                            hasattr(self._model, "inputs")
                            and len(self._model.inputs) > 1
                        ):
                            # Multi-input functional model
                            input_list = []
                            missing_inputs = []
                            for input_layer in self._model.inputs:
                                input_name = input_layer.name
                                if input_name in kwargs:
                                    input_list.append(kwargs[input_name])
                                else:
                                    missing_inputs.append(input_name)

                            if missing_inputs:
                                available = list(kwargs.keys())
                                raise ValueError(
                                    f"Missing required inputs for multi-input "
                                    f"model: {missing_inputs}. "
                                    f"Available kwargs: {available}. "
                                    f"Please provide all inputs by name."
                                )

                            return self._model(input_list)
                        else:
                            # Single input model called with named arguments
                            return self._model(list(kwargs.values())[0])
                else:
                    # Fallback to original call
                    return self._model(*args, **kwargs)

        # 1. Wrap the Keras model in our clean tf.Module.
        wrapper = KerasModelWrapper(self.model)

        # 2. Get a concrete function from the wrapper.
        # Handle dict input signatures for multi-input models
        if isinstance(input_signature, dict):
            # For Functional models with multiple inputs, convert dict to
            # ordered list matching model.inputs order
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

        tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]

        # Pass tensor specs as positional arguments to get the concrete
        # function.
        concrete_func = wrapper.__call__.get_concrete_function(*tensor_specs)

        # 3. Convert from the concrete function.

        # Try multiple conversion strategies for better inference compatibility
        conversion_strategies = [
            {
                "experimental_enable_resource_variables": False,
                "name": "without resource variables",
            },
            {
                "experimental_enable_resource_variables": True,
                "name": "with resource variables",
            },
        ]

        for strategy in conversion_strategies:
            try:
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_func], trackable_obj=wrapper
                )
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter.experimental_enable_resource_variables = strategy[
                    "experimental_enable_resource_variables"
                ]

                # Apply any additional converter settings from kwargs
                self._apply_converter_kwargs(converter)

                tflite_model = converter.convert()

                return tflite_model

            except Exception:
                continue

        # If all strategies fail, raise the last error
        raise RuntimeError(
            "All conversion strategies failed for wrapper-based conversion"
        )

    def _apply_converter_kwargs(self, converter):
        """Apply additional converter settings from kwargs.

        This method applies any TFLite converter settings passed via kwargs
        to the converter object. Common settings include:
        - optimizations: List of optimization options
          (e.g., [tf.lite.Optimize.DEFAULT])
        - representative_dataset: Dataset generator for quantization
        - target_spec: Additional target specification settings
        - inference_input_type: Input type for inference (e.g., tf.int8)
        - inference_output_type: Output type for inference (e.g., tf.int8)

        Args:
            converter: tf.lite.TFLiteConverter instance to configure
        """
        if not self.kwargs:
            return

        for key, value in self.kwargs.items():
            if key == "target_spec" and isinstance(value, dict):
                # Handle nested target_spec settings
                for spec_key, spec_value in value.items():
                    if hasattr(converter.target_spec, spec_key):
                        setattr(converter.target_spec, spec_key, spec_value)
            elif hasattr(converter, key):
                setattr(converter, key, value)
            else:
                io_utils.print_msg(
                    f"Warning: Unknown converter setting '{key}' - ignoring"
                )

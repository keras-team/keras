import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
import tempfile
import os
import numpy as np


class LiteRTExporter:
    """
    Exporter for the LiteRT (TFLite) format that creates a single,
    callable signature for `model.call`.
    """

    def __init__(self, model, input_signature=None, verbose=None, max_sequence_length=512, **kwargs):
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose or 0
        self.max_sequence_length = max_sequence_length
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file."""
        if self.verbose:
            print("Starting LiteRT export...")
            print(f"Model: {type(self.model)} - built: {self.model.built}")

        # 1. Ensure the model is built by calling it if necessary
        self._ensure_model_built()

        # 2. Resolve / infer input signature with bounded sequence length.
        if self.input_signature is None:
            if self.verbose:
                print(f"Inferring input signature with max_sequence_length={self.max_sequence_length}.")
            self.input_signature = get_input_signature(self.model, self.max_sequence_length)
        
        # 3. Convert the model to TFLite.
        tflite_model = self._convert_to_tflite(self.input_signature)
        
        if self.verbose:
            final_size_mb = len(tflite_model) / (1024*1024)
            print(f"LiteRT model converted successfully. Size: {final_size_mb:.2f} MB")
        
        # 4. Save the model to the specified file path.
        if not filepath.endswith('.tflite'):
            filepath += '.tflite'
        
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported model to {filepath}")

    def _ensure_model_built(self):
        """
        Ensures the model is fully traced by performing a forward pass.

        This is critical because `model.built` can be True even if the model
        has not been traced with concrete input shapes, which is required for
        TFLite conversion. This method guarantees a forward pass happens.
        """
        if self.verbose:
            print("Ensuring model is traced by performing a forward pass...")

        try:
            # Debug information
            if self.verbose:
                print(f"Model type: {type(self.model)}")
                print(f"Model built: {self.model.built}")
                if hasattr(self.model, '_functional'):
                    print(f"Sequential _functional: {self.model._functional}")

            # Generate dummy inputs based on the model's specification
            dummy_inputs = []
            # Prioritize `model.inputs` as it's the most reliable source
            if hasattr(self.model, 'inputs') and self.model.inputs:
                if self.verbose:
                    print(f"Generating inputs from `model.inputs` ({len(self.model.inputs)} input(s)).")
                for input_layer in self.model.inputs:
                    shape = [1 if dim is None else dim for dim in input_layer.shape]
                    dummy_input = tf.zeros(shape, dtype=input_layer.dtype or tf.float32)
                    dummy_inputs.append(dummy_input)
                    if self.verbose:
                        print(f"  Input shape: {shape}, dtype: {input_layer.dtype or tf.float32}")
            else:
                # Fallback for pure Sequential models without an Input layer
                if self.verbose:
                    print("Model has no `inputs` attribute. Assuming pure Sequential and inferring shape.")
                input_shape = self._infer_sequential_input_shape()
                if input_shape:
                    if self.verbose:
                        print(f"Inferred input shape for Sequential model: {input_shape}")
                    dummy_inputs.append(tf.zeros(input_shape, dtype=tf.float32))
                else:
                    raise ValueError(
                        "Cannot build Sequential model: unable to infer input shape. "
                        "Please add an `Input` layer or specify `input_shape` in the first layer."
                    )

            # Debug the dummy inputs
            if self.verbose:
                print(f"About to call model with {len(dummy_inputs)} inputs")
                for i, inp in enumerate(dummy_inputs):
                    print(f"  Input {i}: shape={inp.shape}, dtype={inp.dtype}")

            # Perform a direct call in inference mode to trace the model.
            # This is more robust than a simple call() and avoids the
            # overhead of model.predict().
            if len(dummy_inputs) == 1:
                result = self.model(dummy_inputs[0], training=False)
            else:
                result = self.model(dummy_inputs, training=False)

            if self.verbose:
                print("Model successfully traced via direct call with training=False.")
                print(f"Output shape: {result.shape if hasattr(result, 'shape') else type(result)}")

        except Exception as e:
            if self.verbose:
                print(f"Error during model call: {e}")
                import traceback
                traceback.print_exc()
            raise ValueError(f"Failed to trace model with error: {e}")

        # Final, critical check
        if not self.model.built:
            raise ValueError(
                "Model could not be built even after a direct call. "
                "Please check the model's definition and input specification."
            )

    def _infer_sequential_input_shape(self):
        """Infer input shape for Sequential models."""
        try:
            # First, look for Input layer
            for layer in self.model.layers:
                if hasattr(layer, '__class__') and layer.__class__.__name__ == 'InputLayer':
                    if hasattr(layer, 'batch_input_shape') and layer.batch_input_shape:
                        input_shape = layer.batch_input_shape
                        return (1,) + input_shape[1:] if input_shape[0] is None else input_shape

            # If no Input layer, try to get from first layer
            if hasattr(self.model, 'layers') and self.model.layers:
                first_layer = self.model.layers[0]

                # Check various ways to get input shape
                for attr in ['input_shape', 'batch_input_shape', '_batch_input_shape']:
                    if hasattr(first_layer, attr):
                        input_shape = getattr(first_layer, attr)
                        if input_shape:
                            return (1,) + input_shape[1:] if input_shape[0] is None else input_shape

                # Try to infer from layer configuration without hardcoded fallbacks
                if hasattr(first_layer, '__class__'):
                    class_name = first_layer.__class__.__name__

                    if class_name == 'Dense':
                        # For Dense layers, try to infer from input_dim
                        if hasattr(first_layer, 'input_dim') and first_layer.input_dim:
                            return (1, first_layer.input_dim)

                    elif class_name == 'Dropout':
                        # For Dropout, look at the next layer to infer shape
                        if len(self.model.layers) > 1:
                            next_layer = self.model.layers[1]
                            if hasattr(next_layer, '__class__'):
                                next_class = next_layer.__class__.__name__
                                if next_class == 'Dense':
                                    if hasattr(next_layer, 'input_dim') and next_layer.input_dim:
                                        return (1, next_layer.input_dim)

                    elif class_name in ['BatchNormalization', 'LayerNormalization']:
                        # For normalization layers, try to infer from previous layer
                        if len(self.model.layers) > 1:
                            prev_layer = self.model.layers[0]  # The normalization layer itself
                            if hasattr(prev_layer, 'units'):
                                return (1, prev_layer.units)

                    # For other layer types, we cannot reliably infer without hardcoded values
                    # Return None to indicate inference failed
                    if self.verbose:
                        print(f"Cannot infer input shape for layer type: {class_name}")

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not infer Sequential input shape: {e}")

        return None

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to a TFLite model."""
        is_sequential = isinstance(self.model, tf.keras.Sequential)

        # For Sequential models, direct conversion is unreliable.
        # We will always use the wrapper-based approach.
        if is_sequential:
            if self.verbose:
                print("Sequential model detected. Using wrapper-based conversion for reliability.")
            return self._convert_with_wrapper(input_signature)

        # For Functional models, try direct conversion first.
        try:
            if self.verbose:
                print("Functional model detected. Trying direct conversion...")
            
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.experimental_enable_resource_variables = False
            tflite_model = converter.convert()
            
            if self.verbose:
                print("Direct conversion successful.")
            return tflite_model
            
        except Exception as direct_error:
            if self.verbose:
                print(f"Direct conversion failed for Functional model: {direct_error}")
                print("Falling back to wrapper-based conversion...")
            
            return self._convert_with_wrapper(input_signature)

    def _convert_with_wrapper(self, input_signature):
        """Converts the model to TFLite using the tf.Module wrapper."""
        # 1. Wrap the Keras model in our clean tf.Module.
        wrapper = _KerasModelWrapper(self.model)

        # 2. Get a concrete function from the wrapper.
        if not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]
        
        tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]
        
        # Pass tensor specs as positional arguments to get the concrete function.
        concrete_func = wrapper.__call__.get_concrete_function(*tensor_specs)

        # 3. Convert from the concrete function.
        if self.verbose:
            print("Converting concrete function to TFLite format...")
        
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func], 
            trackable_obj=wrapper
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_enable_resource_variables = False
        tflite_model = converter.convert()

        return tflite_model


class _KerasModelWrapper(tf.Module):
    """
    A tf.Module wrapper for a Keras model.

    This wrapper is designed to be a clean, serializable interface for TFLite
    conversion. It holds the Keras model and exposes a single `__call__`
    method that is decorated with `tf.function`. Crucially, it also ensures
    all variables from the Keras model are tracked by the SavedModel format,
    which is key to including them in the final TFLite model.
    """

    def __init__(self, model):
        super().__init__()
        # Store the model reference in a way that TensorFlow won't try to track it
        # This prevents the _DictWrapper error during SavedModel serialization
        object.__setattr__(self, '_model', model)

        # Explicitly track all variables from the Keras model by assigning
        # them as individual attributes of this wrapper. This ensures they are
        # properly included in the SavedModel and TFLite conversion.
        for i, var in enumerate(model.variables):
            setattr(self, f'_var_{i}', var)

    @tf.function
    def __call__(self, *args, **kwargs):
        """The single entry point for the exported model."""
        # Handle both single and multi-input cases
        if args and not kwargs:
            # Called with positional arguments
            if len(args) == 1:
                return self._model(args[0])
            else:
                return self._model(list(args))
        elif kwargs and not args:
            # Called with keyword arguments
            if len(kwargs) == 1 and 'inputs' in kwargs:
                # Single input case
                return self._model(kwargs['inputs'])
            else:
                # Multi-input case - convert to list/dict format expected by model
                if hasattr(self._model, 'inputs') and len(self._model.inputs) > 1:
                    # Multi-input functional model
                    input_list = []
                    for input_layer in self._model.inputs:
                        input_name = input_layer.name
                        if input_name in kwargs:
                            input_list.append(kwargs[input_name])
                        else:
                            # Try to match by position
                            keys = list(kwargs.keys())
                            idx = len(input_list)
                            if idx < len(keys):
                                input_list.append(kwargs[keys[idx]])
                    return self._model(input_list)
                else:
                    # Single input model called with named arguments
                    return self._model(list(kwargs.values())[0])
        else:
            # Fallback to original call
            return self._model(*args, **kwargs)
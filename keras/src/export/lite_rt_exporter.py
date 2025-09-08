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
        """Ensure the model is built by calling it with dummy data if necessary."""
        if not self.model.built:
            if self.verbose:
                print("Model not built, building with dummy data...")
            
            # For Sequential models, we need to build them by calling them
            if hasattr(self.model, '_is_graph_network') and not self.model._is_graph_network:
                # This is a Sequential model
                self._build_sequential_model()
            else:
                # This is a Functional model
                self._build_functional_model()
        else:
            # Model is already built, but let's make sure it has outputs
            if not hasattr(self.model, 'outputs') or not self.model.outputs:
                if self.verbose:
                    print("Model built but no outputs found, rebuilding...")
                # For Sequential models, we need to build them by calling them
                if hasattr(self.model, '_is_graph_network') and not self.model._is_graph_network:
                    # This is a Sequential model
                    self._build_sequential_model()
                else:
                    # This is a Functional model
                    self._build_functional_model()
        
        # Always make a prediction call with random inputs to ensure model is fully built
        self._make_prediction_call()

    def _make_prediction_call(self):
        """Make a prediction call with random inputs to ensure model is fully built."""
        try:
            if self.verbose:
                print("Making prediction call with random inputs...")
            
            # Generate random inputs based on model's input specs
            if hasattr(self.model, 'inputs') and self.model.inputs:
                # Multi-input or single input functional model
                dummy_inputs = []
                for input_layer in self.model.inputs:
                    input_shape = input_layer.shape
                    # Replace None (batch dimension) with 1
                    shape = [1 if dim is None else dim for dim in input_shape]
                    dummy_input = np.random.random(shape).astype(np.float32)
                    dummy_inputs.append(dummy_input)
                
                if len(dummy_inputs) == 1:
                    _ = self.model.predict(dummy_inputs[0], verbose=0)
                else:
                    _ = self.model.predict(dummy_inputs, verbose=0)
                    
                if self.verbose:
                    print(f"Prediction call successful with {len(dummy_inputs)} input(s)")
            else:
                # Sequential model - try to infer input shape
                input_shape = self._infer_sequential_input_shape()
                if input_shape:
                    dummy_input = np.random.random(input_shape).astype(np.float32)
                    _ = self.model.predict(dummy_input, verbose=0)
                    if self.verbose:
                        print(f"Prediction call successful with shape: {input_shape}")
                        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Prediction call failed: {e}")

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
                
                # Fallback based on layer type
                if hasattr(first_layer, '__class__'):
                    class_name = first_layer.__class__.__name__
                    if class_name == 'Dense':
                        if hasattr(first_layer, 'input_dim') and first_layer.input_dim:
                            return (1, first_layer.input_dim)
                        else:
                            return (1, 10)  # Default for Dense
                    elif class_name == 'Conv2D':
                        return (1, 28, 28, 1)  # Default for Conv2D
                    elif 'LSTM' in class_name or 'GRU' in class_name:
                        return (1, 20, 50)  # Default for RNN
                        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not infer Sequential input shape: {e}")
        
        return None

    def _build_sequential_model(self):
        """Build a Sequential model by intelligently inferring input shape from layers."""
        try:
            # First, look for Input layer in the model (most reliable)
            for layer in self.model.layers:
                if hasattr(layer, '__class__') and layer.__class__.__name__ == 'InputLayer':
                    if hasattr(layer, 'batch_input_shape') and layer.batch_input_shape:
                        input_shape = layer.batch_input_shape
                        if input_shape[0] is None:
                            dummy_shape = (1,) + input_shape[1:]
                        else:
                            dummy_shape = input_shape
                        
                        dummy_input = tf.ones(dummy_shape, dtype=tf.float32)
                        _ = self.model(dummy_input)
                        if self.verbose:
                            print(f"Sequential model built from InputLayer with shape: {dummy_shape}")
                        return
            
            # If no Input layer found, try to get input shape from the first layer
            if hasattr(self.model, 'layers') and self.model.layers:
                first_layer = self.model.layers[0]
                
                # Try to get input shape from the first layer
                input_shape = None
                
                # Check various ways to get input shape
                if hasattr(first_layer, 'input_shape') and first_layer.input_shape:
                    input_shape = first_layer.input_shape
                elif hasattr(first_layer, 'batch_input_shape') and first_layer.batch_input_shape:
                    input_shape = first_layer.batch_input_shape
                elif hasattr(first_layer, '_batch_input_shape') and first_layer._batch_input_shape:
                    input_shape = first_layer._batch_input_shape
                
                # If we have an input shape, use it
                if input_shape:
                    # Create dummy input with batch dimension
                    if input_shape[0] is None:  # Batch dimension is None
                        dummy_shape = (1,) + input_shape[1:]
                    else:
                        dummy_shape = input_shape
                    
                    dummy_input = tf.ones(dummy_shape, dtype=tf.float32)
                    _ = self.model(dummy_input)
                    if self.verbose:
                        print(f"Sequential model built with shape: {dummy_shape}")
                    return
                
                # If no explicit input shape, try to infer from layer configuration
                if hasattr(first_layer, 'units') and hasattr(first_layer, '__class__'):
                    # Dense layer - need to know input dimension
                    if first_layer.__class__.__name__ == 'Dense':
                        # For Dense layers, we need to know the input dimension
                        # Try to infer from layer configuration or use a reasonable default
                        if hasattr(first_layer, 'input_dim') and first_layer.input_dim:
                            dummy_shape = (1, first_layer.input_dim)
                        else:
                            # Use a reasonable default for Dense layers
                            dummy_shape = (1, 10)  # Common for simple models
                        
                        dummy_input = tf.ones(dummy_shape, dtype=tf.float32)
                        _ = self.model(dummy_input)
                        if self.verbose:
                            print(f"Sequential model (Dense) built with shape: {dummy_shape}")
                        return
                
                elif hasattr(first_layer, 'filters') and hasattr(first_layer, 'kernel_size'):
                    # Conv2D layer - need image dimensions
                    if first_layer.__class__.__name__ == 'Conv2D':
                        # For Conv2D, we need (height, width, channels)
                        # Use a reasonable default for image models
                        dummy_shape = (1, 28, 28, 1)  # MNIST-like
                        
                        dummy_input = tf.ones(dummy_shape, dtype=tf.float32)
                        _ = self.model(dummy_input)
                        if self.verbose:
                            print(f"Sequential model (Conv2D) built with shape: {dummy_shape}")
                        return
                
                elif hasattr(first_layer, 'units') and hasattr(first_layer, 'return_sequences'):
                    # RNN layer - need sequence dimensions
                    if 'LSTM' in first_layer.__class__.__name__ or 'GRU' in first_layer.__class__.__name__:
                        # For RNN layers, we need (sequence_length, features)
                        # Use reasonable defaults for sequence models
                        dummy_shape = (1, 20, 50)  # Common for sequence models
                        
                        dummy_input = tf.ones(dummy_shape, dtype=tf.float32)
                        _ = self.model(dummy_input)
                        if self.verbose:
                            print(f"Sequential model (RNN) built with shape: {dummy_shape}")
                        return
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not build Sequential model: {e}")

    def _build_functional_model(self):
        """Build a Functional model using its input specifications."""
        try:
            # Create dummy inputs based on input specs
            dummy_inputs = []
            for input_spec in self.model.input_spec or []:
                if hasattr(input_spec, 'shape') and input_spec.shape:
                    # Create dummy data with the expected shape
                    shape = [1] + list(input_spec.shape[1:])  # Add batch dimension
                    dummy_data = tf.ones(shape, dtype=input_spec.dtype or tf.float32)
                    dummy_inputs.append(dummy_data)
            
            # If we couldn't get specs, try to infer from layers
            if not dummy_inputs and hasattr(self.model, 'layers') and self.model.layers:
                first_layer = self.model.layers[0]
                if hasattr(first_layer, 'input_spec') and first_layer.input_spec:
                    for spec in first_layer.input_spec:
                        if hasattr(spec, 'shape') and spec.shape:
                            shape = [1] + list(spec.shape[1:])
                            dummy_data = tf.ones(shape, dtype=spec.dtype or tf.float32)
                            dummy_inputs.append(dummy_data)
            
            # Build the model
            if dummy_inputs:
                try:
                    if len(dummy_inputs) == 1:
                        _ = self.model(dummy_inputs[0])
                    else:
                        _ = self.model(dummy_inputs)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not build functional model: {e}")
                        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not build functional model: {e}")

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to a TFLite model."""
        # Try direct conversion first (simpler approach)
        try:
            if self.verbose:
                print("Converting Keras model directly to TFLite format...")
            
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
            ]
            # Ensure variables are embedded
            converter.experimental_enable_resource_variables = False
            tflite_model = converter.convert()
            
            if self.verbose:
                print("Direct conversion successful")
            return tflite_model
            
        except Exception as direct_error:
            if self.verbose:
                print(f"Direct conversion failed: {direct_error}")
                print("Trying wrapper-based conversion...")
            
            # Fallback to wrapper approach
            # 1. Wrap the Keras model in our clean tf.Module.
            wrapper = _KerasModelWrapper(self.model)

            # 2. Get a concrete function from the wrapper.
            # Handle both single and multiple input signatures
            if not isinstance(input_signature, (list, tuple)):
                input_signature = [input_signature]
            
            # Convert InputSpec objects to TensorSpec objects for get_concrete_function
            tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]
            
            # Create input arguments based on the model's expected signature
            input_args = self._create_input_args(tensor_specs)

            concrete_func = wrapper.__call__.get_concrete_function(**input_args)

            # 3. Convert directly from the concrete function to TFLite.
            if self.verbose:
                print("Converting concrete function to TFLite format...")
            
            # Use the wrapper as trackable_obj to avoid deprecation warning
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete_func], 
                trackable_obj=wrapper
            )
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
            ]
            # Ensure variables are embedded
            converter.experimental_enable_resource_variables = False
            tflite_model = converter.convert()

            return tflite_model

    def _create_input_args(self, tensor_specs):
        """Create proper input arguments for the model's call signature."""
        # Determine if this is a single-input or multi-input model
        num_inputs = len(self.model.inputs) if hasattr(self.model, 'inputs') else 1
        
        if num_inputs == 1:
            # Single input model - use 'inputs' as the argument name
            if len(tensor_specs) == 1:
                return {"inputs": tensor_specs[0]}
            else:
                # Multiple specs for single input (shouldn't happen, but handle gracefully)
                return {"inputs": tensor_specs[0]}
        else:
            # Multi-input model - use the actual input names or create generic names
            input_args = {}
            
            if hasattr(self.model, 'inputs') and self.model.inputs:
                # Use the actual input names from the model
                for i, (input_layer, spec) in enumerate(zip(self.model.inputs, tensor_specs)):
                    input_name = input_layer.name
                    input_args[input_name] = spec
            else:
                # Fallback to generic names
                for i, spec in enumerate(tensor_specs):
                    input_args[f"input_{i}"] = spec
            
            return input_args


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
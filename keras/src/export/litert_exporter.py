import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
from keras.src.utils import io_utils
import tempfile
import os
import numpy as np

# Try to import LiteRT AOT compilation if available
try:
    from litert.python.aot import aot_compile
    from litert.python.aot.core import types as litert_types
    from litert.python.aot.vendors import import_vendor
    LITERT_AVAILABLE = True
except ImportError:
    LITERT_AVAILABLE = False


def export_litert(
    model,
    filepath,
    verbose=None,
    input_signature=None,
    aot_compile_targets=None,
    **kwargs,
):
    """Export the model as a Litert artifact for inference.

    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact.
        verbose: Optional; whether to log progress messages. Defaults to
            ``False`` when ``None`` is provided.
        input_signature: Optional input signature specification. If
            ``None``, it will be inferred.
        aot_compile_targets: Optional list of Litert targets for AOT
            compilation.
        **kwargs: Additional keyword arguments passed to the exporter.

    Returns:
        The filepath to the exported artifact, or the compilation result when
        AOT compilation is requested.
    """

    actual_verbose = bool(verbose) if verbose is not None else False
    exporter = LitertExporter(
        model=model,
        input_signature=input_signature,
        verbose=actual_verbose,
        aot_compile_targets=aot_compile_targets,
        **kwargs,
    )
    result = exporter.export(filepath)
    if actual_verbose:
        if hasattr(result, "models"):
            io_utils.print_msg(
                f"Saved artifact at '{filepath}'. AOT compiled "
                f"{len(result.models)} variant(s)."
            )
        else:
            io_utils.print_msg(f"Saved artifact at '{result}'.")
    return result


class LitertExporter:
    """
    Exporter for the Litert (TFLite) format that creates a single,
    callable signature for `model.call`.
    """

    def __init__(self, model, input_signature=None, verbose=False,
                 aot_compile_targets=None, **kwargs):
        """Initialize the Litert exporter.
        
        Args:
            model: The Keras model to export
            input_signature: Input signature specification
            verbose: Whether to print progress messages during export.
            aot_compile_targets: List of Litert targets for AOT compilation
            **kwargs: Additional export parameters
        """
        self.model = model
        self.input_signature = input_signature
        self.verbose = bool(verbose)
        self.aot_compile_targets = aot_compile_targets
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file and optionally performs AOT compilation.
        
        Args:
            filepath: Output path for the exported model
            
        Returns:
            Path to exported model or compiled models if AOT compilation is performed
        """
        if self.verbose:
            print("Starting Litert export...")

        # 1. Ensure the model is built by calling it if necessary
        self._ensure_model_built()

        # 2. Resolve / infer input signature
        if self.input_signature is None:
            if self.verbose:
                print("Inferring input signature from model.")
            self.input_signature = get_input_signature(self.model)
        
        # 3. Convert the model to TFLite.
        tflite_model = self._convert_to_tflite(self.input_signature)
        
        if self.verbose:
            final_size_mb = len(tflite_model) / (1024*1024)
            print(f"TFLite model converted successfully. Size: {final_size_mb:.2f} MB")
        
        # 4. Save the initial TFLite model to the specified file path.
        if not filepath.endswith('.tflite'):
            filepath += '.tflite'
        
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"TFLite model saved to {filepath}")

        # 5. Perform AOT compilation if targets are specified and LiteRT is available
        compiled_models = None
        if self.aot_compile_targets and LITERT_AVAILABLE:
            if self.verbose:
                print("Performing AOT compilation for Litert targets...")
            compiled_models = self._aot_compile(filepath)
        elif self.aot_compile_targets and not LITERT_AVAILABLE:
            if self.verbose:
                print("Warning: AOT compilation requested but LiteRT is not available. Skipping.")
        
        if self.verbose:
            print(f"Litert export completed. Base model: {filepath}")
            if compiled_models:
                print(f"AOT compiled models: {len(compiled_models.models)} variants")

        return compiled_models if compiled_models else filepath

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

            # Perform a direct call in inference mode to trace the model.
            if len(dummy_inputs) == 1:
                result = self.model(dummy_inputs[0], training=False)
            else:
                result = self.model(dummy_inputs, training=False)

            if self.verbose:
                print("Model successfully traced via direct call with training=False.")

        except Exception as e:
            if self.verbose:
                print(f"Error during model call: {e}")
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
                        # For normalization layers, look at the next layer to infer shape
                        if len(self.model.layers) > 1:
                            next_layer = self.model.layers[1]
                            if hasattr(next_layer, '__class__'):
                                next_class = next_layer.__class__.__name__
                                if next_class == 'Dense':
                                    if hasattr(next_layer, 'input_dim') and next_layer.input_dim:
                                        return (1, next_layer.input_dim)

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

        # Try direct conversion first for all models
        try:
            if self.verbose:
                model_type = "Sequential" if is_sequential else "Functional"
                print(f"{model_type} model detected. Trying direct conversion...")
            
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
                model_type = "Sequential" if is_sequential else "Functional"
                print(f"Direct conversion failed for {model_type} model: {direct_error}")
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
        
        # Try multiple conversion strategies for better inference compatibility
        conversion_strategies = [
            {"experimental_enable_resource_variables": False, "name": "without resource variables"},
            {"experimental_enable_resource_variables": True, "name": "with resource variables"},
        ]
        
        for strategy in conversion_strategies:
            try:
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_func], 
                    trackable_obj=wrapper
                )
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter.experimental_enable_resource_variables = strategy["experimental_enable_resource_variables"]
                
                if self.verbose:
                    print(f"Trying conversion {strategy['name']}...")
                
                tflite_model = converter.convert()
                
                if self.verbose:
                    print(f"Conversion successful {strategy['name']}!")
                
                return tflite_model
                
            except Exception as e:
                if self.verbose:
                    print(f"Conversion failed {strategy['name']}: {e}")
                continue
        
        # If all strategies fail, raise the last error
        raise RuntimeError("All conversion strategies failed for wrapper-based conversion")

    def _aot_compile(self, tflite_filepath):
        """Performs AOT compilation using LiteRT."""
        if not LITERT_AVAILABLE:
            raise RuntimeError("LiteRT is not available for AOT compilation")
        
        try:
            # Create a LiteRT model from the TFLite file
            litert_model = litert_types.Model.create_from_path(tflite_filepath)
            
            # Determine output directory
            base_dir = os.path.dirname(tflite_filepath)
            model_name = os.path.splitext(os.path.basename(tflite_filepath))[0]
            output_dir = os.path.join(base_dir, f"{model_name}_compiled")
            
            if self.verbose:
                print(f"AOT compiling for targets: {self.aot_compile_targets}")
                print(f"Output directory: {output_dir}")
            
            # Perform AOT compilation
            result = aot_compile.aot_compile(
                input_model=litert_model,
                output_dir=output_dir,
                target=self.aot_compile_targets,
                keep_going=True  # Continue even if some targets fail
            )
            
            if self.verbose:
                print(f"AOT compilation completed: {len(result.models)} successful, {len(result.failed_backends)} failed")
                if result.failed_backends:
                    for backend, error in result.failed_backends:
                        print(f"  Failed: {backend.id()} - {error}")
                
                # Print compilation report if available
                try:
                    report = result.compilation_report()
                    if report:
                        print("Compilation Report:")
                        print(report)
                except:
                    pass
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"AOT compilation failed: {e}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"AOT compilation failed: {e}")

    def _get_available_litert_targets(self):
        """Get available LiteRT targets for AOT compilation."""
        if not LITERT_AVAILABLE:
            return []
        
        try:
            # Get all registered targets
            targets = import_vendor.AllRegisteredTarget()
            return targets if isinstance(targets, list) else [targets]
        except Exception as e:
            if self.verbose:
                print(f"Failed to get available targets: {e}")
            return []

    @classmethod
    def export_with_aot(cls, model, filepath, targets=None, verbose=True, **kwargs):
        """
        Convenience method to export a Keras model with AOT compilation.
        
        Args:
            model: Keras model to export
            filepath: Output file path 
            targets: List of LiteRT targets for AOT compilation (e.g., ['qualcomm', 'mediatek'])
            verbose: Whether to print verbose output
            **kwargs: Additional arguments for the exporter
            
        Returns:
            CompilationResult if AOT compilation is performed, otherwise the filepath
        """
        exporter = cls(
            model=model, 
            verbose=verbose, 
            aot_compile_targets=targets,
            **kwargs
        )
        return exporter.export(filepath)

    @classmethod  
    def get_available_targets(cls):
        """Get list of available LiteRT AOT compilation targets."""
        if not LITERT_AVAILABLE:
            return []
        
        dummy_exporter = cls(model=None)
        return dummy_exporter._get_available_litert_targets()


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

        # Track all variables from the Keras model using proper tf.Module methods
        # This ensures proper variable handling for stateful layers like BatchNorm
        with self.name_scope:
            for i, var in enumerate(model.variables):
                # Use a different attribute name to avoid conflicts with tf.Module's variables property
                setattr(self, f'model_var_{i}', var)

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
                    missing_inputs = []
                    for input_layer in self._model.inputs:
                        input_name = input_layer.name
                        if input_name in kwargs:
                            input_list.append(kwargs[input_name])
                        else:
                            missing_inputs.append(input_name)
                    
                    if missing_inputs:
                        raise ValueError(
                            f"Missing required inputs for multi-input model: {missing_inputs}. "
                            f"Available kwargs: {list(kwargs.keys())}. "
                            f"Please provide all inputs by name."
                        )
                    
                    return self._model(input_list)
                else:
                    # Single input model called with named arguments
                    return self._model(list(kwargs.values())[0])
        else:
            # Fallback to original call
            return self._model(*args, **kwargs)
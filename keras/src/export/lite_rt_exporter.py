import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
from keras.src import tree
import tempfile
import os

class LiteRTExporter:
    """
    Exporter for the LiteRT (TFLite) format.
    """
    
    def __init__(self, model, input_signature=None, verbose=None, max_sequence_length=512, **kwargs):
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose or 0
        self.max_sequence_length = max_sequence_length
        self.kwargs = kwargs

    def export(self, filepath):
        if self.verbose:
            print("Starting LiteRT export...")

        # Step 1: Get input signature, applying bounded shapes for sequence models.
        if self.input_signature is None:
            input_signature = get_input_signature(self.model, self.max_sequence_length)
        else:
            input_signature = self.input_signature
        
        # Step 2: Convert to TensorFlow TensorSpecs and create a concrete function.
        concrete_fn = self._get_concrete_function(input_signature)
        
        # Step 3: Convert the concrete function to a TFLite model.
        tflite_model = self._convert_to_tflite(concrete_fn)
        
        if self.verbose:
            print(f"LiteRT model converted successfully. Size: {len(tflite_model)} bytes")
        
        # Step 4: Save the model to the specified file path.
        if not filepath.endswith('.tflite'):
            filepath = filepath + '.tflite'
        
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported model to {filepath}")

    def _get_concrete_function(self, input_signature):
        """Create a tf.function and get its concrete function."""
        
        # Create a wrapper function that handles different input structures.
        if isinstance(input_signature, dict):
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature.values()]
            input_keys = list(input_signature.keys())
            
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                input_dict = {key: tensor for key, tensor in zip(input_keys, inputs)}
                return self.model(input_dict)
        
        elif isinstance(input_signature, list):
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature]
            
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                return self.model(list(inputs))
        
        else: # Assumes a single tensor or spec
            tf_signature = [make_tf_tensor_spec(input_signature)]
            
            @tf.function(input_signature=tf_signature)
            def model_fn(input_tensor):
                return self.model(input_tensor)
        
        return model_fn.get_concrete_function()

    def _convert_to_tflite(self, concrete_fn):
        """
        Converts a concrete function to TFLite, using the most appropriate
        conversion path based on the model type.
        """
        # For complex models (like Keras-Hub transformers), the SavedModel path
        # is more robust and correctly handles variable tracking.
        if self._is_keras_hub_model():
            if self.verbose:
                print("Keras-Hub model detected. Using SavedModel conversion path for robustness.")
            return self._convert_via_saved_model(concrete_fn)

        # For standard Keras models, the direct `from_concrete_functions` path is efficient.
        else:
            if self.verbose:
                print("Standard model detected. Using direct conversion path.")
            return self._convert_direct(concrete_fn)

    def _convert_direct(self, concrete_fn):
        """Directly convert a concrete function to TFLite."""
        try:
            # Use trackable_obj if available to follow best practices.
            if hasattr(self.model, '_get_save_spec'):
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_fn], trackable_obj=self.model
                )
            else:
                # This path is deprecated but serves as a fallback.
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
            
            self._apply_common_converter_settings(converter)
            return converter.convert()
        except Exception as e:
            raise IOError(f"Direct TFLite conversion failed. Error: {e}")

    def _convert_via_saved_model(self, concrete_fn):
        """
        A more robust conversion path that first creates a temporary SavedModel.
        This is better for complex models with intricate variable tracking.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_model_path = os.path.join(temp_dir, "temp_saved_model")
            
            # Save the model with the concrete function as a signature
            tf.saved_model.save(
                self.model, saved_model_path, signatures=concrete_fn
            )
            
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            self._apply_common_converter_settings(converter, is_complex=True)
            return converter.convert()

    def _is_keras_hub_model(self):
        """
        Checks if the model is from Keras-Hub based on module path only.
        Keras-Hub models benefit from the SavedModel conversion path.
        """
        model_module = getattr(self.model.__class__, '__module__', '').lower()
        return 'keras_hub' in model_module

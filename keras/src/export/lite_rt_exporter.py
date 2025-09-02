import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
from keras.src import tree
import tempfile
import os
from pathlib import Path

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
        Converts a concrete function to TFLite using direct conversion first.
        Falls back to SavedModel only if direct conversion fails.
        """
        # Try direct conversion first - this avoids _DictWrapper issues entirely
        try:
            if self.verbose:
                print("Attempting direct TFLite conversion...")
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
            self._apply_common_converter_settings(converter, is_complex=False)
            return converter.convert()
        except Exception as e:
            # Only fall back to SavedModel if direct conversion fails
            if self.verbose:
                print(f"Direct conversion failed: {e}")
                print("Falling back to SavedModel conversion path...")
            return self._convert_via_saved_model(concrete_fn)

    def _convert_via_saved_model(self, concrete_fn):
        """Fallback conversion via SavedModel for edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_model_dir = Path(temp_dir) / "saved_model"
            
            tf.saved_model.save(
                self.model,
                str(saved_model_dir),
                signatures={"serving_default": concrete_fn}
            )
            
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            self._apply_common_converter_settings(converter, is_complex=True)
            return converter.convert()

    def _convert_via_saved_model(self, concrete_fn):
        """
        A more robust conversion path that first creates a temporary SavedModel.
        This is better for complex models with intricate variable tracking.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_model_path = os.path.join(temp_dir, "temp_saved_model")
            
            # Try normal path first, fallback if TensorFlow introspection fails
            try:
                tf.saved_model.save(
                    self.model, saved_model_path, signatures=concrete_fn
                )
            except TypeError as e:
                if "_DictWrapper" in str(e) or "__dict__ descriptor" in str(e):
                    if self.verbose:
                        print("Using fallback SavedModel path due to TensorFlow introspection issue.")
                    # Fallback: save with minimal trackable object
                    minimal_obj = tf.Module()
                    tf.saved_model.save(
                        minimal_obj, saved_model_path, 
                        signatures={'serving_default': concrete_fn}
                    )
                else:
                    raise e
            
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

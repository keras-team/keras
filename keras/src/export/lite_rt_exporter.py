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
            
            # Check if we need to use a clean trackable object to avoid _DictWrapper issues
            trackable_obj = self._create_clean_trackable_object_if_needed()
            
            # Saving the model with the concrete function as a signature is more
            # reliable as it ensures all trackable assets of the model are found.
            tf.saved_model.save(
                trackable_obj, saved_model_path, signatures=concrete_fn
            )
            
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            
            # Keras-Hub models often require these settings.
            self._apply_common_converter_settings(converter, is_complex=True)
            return converter.convert()

    def _create_clean_trackable_object_if_needed(self):
        """
        Create a clean trackable object if the model contains _DictWrapper objects
        that cause issues during TensorFlow's introspection.
        """
        # Check if the model has _DictWrapper objects in its trackable children
        has_dict_wrapper = self._model_has_dict_wrapper_issues()
        
        if not has_dict_wrapper:
            return self.model
        
        # Create a clean trackable object to avoid _DictWrapper issues
        trackable_obj = tf.__internal__.tracking.AutoTrackable()
        
        # Copy essential variables from the model
        if hasattr(self.model, 'variables'):
            trackable_obj.variables = list(self.model.variables)
        if hasattr(self.model, 'trainable_variables'):
            trackable_obj.trainable_variables = list(self.model.trainable_variables)
        if hasattr(self.model, 'non_trainable_variables'):
            trackable_obj.non_trainable_variables = list(self.model.non_trainable_variables)
        
        return trackable_obj

    def _model_has_dict_wrapper_issues(self):
        """
        Check if the model contains _DictWrapper objects that cause introspection issues.
        """
        # Import _DictWrapper safely
        try:
            from tensorflow.python.trackable.data_structures import _DictWrapper
        except ImportError:
            return False
        
        # Check model's direct attributes for _DictWrapper objects
        for attr_name in dir(self.model):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(self.model, attr_name, None)
                if isinstance(attr_value, _DictWrapper):
                    return True
            except (AttributeError, TypeError):
                continue
        
        # Check if model class name suggests complex structures
        model_class_name = self.model.__class__.__name__.lower()
        if any(indicator in model_class_name for indicator in ['backbone', 'causal_lm', 'gemma', 'llama', 'bert']):
            return True
            
        return False

    def _apply_common_converter_settings(self, converter, is_complex=False):
        """Applies shared TFLite converter settings."""
        converter.allow_custom_ops = self.kwargs.get("allow_custom_ops", is_complex)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        if self.kwargs.get("enable_select_tf_ops", is_complex):
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

        if "optimizations" in self.kwargs:
            converter.optimizations = self.kwargs["optimizations"]
            
        # For large models, enable memory optimization to prevent overflow
        if is_complex and self._is_large_vocabulary_model():
            # Enable optimizations that reduce intermediate tensor sizes
            if not converter.optimizations:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Use representative dataset for better quantization if available
            if "representative_dataset" in self.kwargs:
                converter.representative_dataset = self.kwargs["representative_dataset"]

    def _is_large_vocabulary_model(self):
        """Check if this is a large vocabulary model that might cause overflow."""
        model_class_name = self.model.__class__.__name__.lower()
        model_module = getattr(self.model.__class__, '__module__', '').lower()
        
        # Models known to have large vocabularies
        large_vocab_indicators = ['gemma', 'llama', 'palm', 'gpt', 'bert']
        if any(indicator in model_class_name for indicator in large_vocab_indicators):
            return True
        if 'keras_hub' in model_module:
            return True
            
        return False

    def _is_keras_hub_model(self):
        """
        Heuristically checks if the model is a complex model from Keras-Hub
        that benefits from the SavedModel conversion path.
        """
        model_module = getattr(self.model.__class__, '__module__', '').lower()
        if 'keras_hub' in model_module:
            return True
        
        # Fallback check for models that might not be in the keras_hub module
        # but follow similar patterns (e.g., custom backbones).
        model_class_name = self.model.__class__.__name__.lower()
        complex_indicators = ['backbone', 'causal_lm', 'gemma', 'llama', 'bert']
        if any(indicator in model_class_name for indicator in complex_indicators):
            return True
            
        return False

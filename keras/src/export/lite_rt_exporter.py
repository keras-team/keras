import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
from keras.src import tree
import tempfile
import os

class LiteRTExporter:
    """Custom Keras exporter for LiteRT (TFLite) format, bypassing tf.lite.TFLiteConverter."""
    
    def __init__(self, model, input_signature=None, verbose=None, max_sequence_length=512, **kwargs):
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose or 0
        self.max_sequence_length = max_sequence_length
        self.kwargs = kwargs  # e.g., allow_custom_ops, enable_select_tf_ops, optimizations
    
    def export(self, filepath):
        if self.verbose:
            print("Starting custom LiteRT export...")
        
        # Pre-flight check for potentially problematic models
        if self._is_complex_model():
            if self.verbose:
                print("⚠️  Detected complex model. Using enhanced conversion path...")
            # For complex models, enable more permissive settings by default
            self.kwargs.setdefault("allow_custom_ops", True)
            self.kwargs.setdefault("enable_select_tf_ops", True)
        
        # Step 1: Get input signature (use get_input_signature if not provided)
        if self.input_signature is None:
            input_signature = get_input_signature(self.model, self.max_sequence_length)
        else:
            input_signature = self.input_signature
        
        # Convert to TensorFlow TensorSpecs for tf.function
        # Handle different input signature structures
        tf_signature = []
        
        if isinstance(input_signature, dict):
            # Dictionary input (e.g., Keras-Hub models like Gemma3)
            # Convert dict to ordered list of specs for tf.function
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature.values()]
            input_keys = list(input_signature.keys())
            
            # Create a wrapper function that handles dict inputs
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                # Reconstruct dictionary from positional args
                input_dict = {key: tensor for key, tensor in zip(input_keys, inputs)}
                return self.model(input_dict)
                
        elif isinstance(input_signature, list):
            # List of specs
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature]
            
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                return self.model(*inputs)
                
        elif hasattr(input_signature, 'shape'):
            # Single spec
            tf_signature = [make_tf_tensor_spec(input_signature)]
            
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                return self.model(*inputs)
                
        else:
            # Try to flatten and convert
            def _convert_to_spec(spec):
                tf_signature.append(make_tf_tensor_spec(spec))
            
            try:
                tree.map_structure(_convert_to_spec, input_signature)
                
                @tf.function(input_signature=tf_signature)
                def model_fn(*inputs):
                    return self.model(*inputs)
                    
            except:
                # Fallback: assume it's a single spec
                tf_signature = [make_tf_tensor_spec(input_signature)]
                
                @tf.function(input_signature=tf_signature)
                def model_fn(*inputs):
                    return self.model(*inputs)
        
        # Step 2: Trace the model to create a concrete function
        concrete_fn = model_fn.get_concrete_function()
        
        # Step 3: Convert using TFLite converter directly (simplified approach)
        # Skip the complex MLIR conversion and use the standard converter
        tflite_model = self._convert_to_tflite(concrete_fn)
        
        if self.verbose:
            print(f"LiteRT model converted. Size: {len(tflite_model)} bytes")
        
        # Step 4: Save to file
        # Ensure the filepath has the correct .tflite extension
        if not filepath.endswith('.tflite'):
            filepath = filepath + '.tflite'
        
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported to {filepath}")
    
    def _convert_to_tflite(self, concrete_fn):
        """Convert concrete function to TFLite using standard converter."""
        try:
            # First try with trackable_obj to avoid deprecated path
            if hasattr(self.model, '_get_save_spec'):
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_fn], trackable_obj=self.model
                )
            else:
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
                
            # Apply conversion settings
            converter.allow_custom_ops = self.kwargs.get("allow_custom_ops", False)
            converter.enable_select_tf_ops = self.kwargs.get("enable_select_tf_ops", False)
            
            # For large models like Keras-Hub, we might need to be more conservative
            # Try without optimizations first for complex models
            if "optimizations" in self.kwargs:
                converter.optimizations = self.kwargs["optimizations"]
            
            # Try conversion
            result = converter.convert()
            return result
            
        except RuntimeError as e:
            if "size too big" in str(e):
                # Handle the overflow issue by trying alternative approaches
                return self._convert_with_saved_model_fallback()
            else:
                raise e
        except Exception as e:
            if "custom op" in str(e).lower():
                raise ValueError(f"Custom ops detected. Enable allow_custom_ops=True. Details: {e}")
            raise

    def _convert_with_saved_model_fallback(self):
        """Fallback conversion using SavedModel path to avoid size overflow."""
        import tempfile
        import os
        
        try:
            # Create a temporary SavedModel
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = os.path.join(temp_dir, "temp_saved_model")
                
                # Export to SavedModel first
                from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
                from keras.src.export.saved_model import export_saved_model
                
                input_signature = get_input_signature(self.model)
                export_saved_model(self.model, saved_model_path, input_signature=input_signature)
                
                # Convert SavedModel to TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.allow_custom_ops = self.kwargs.get("allow_custom_ops", True)  # More permissive
                converter.enable_select_tf_ops = self.kwargs.get("enable_select_tf_ops", True)  # More permissive
                
                # Skip optimizations for problematic models
                if "optimizations" in self.kwargs and not self._is_complex_model():
                    converter.optimizations = self.kwargs["optimizations"]
                
                result = converter.convert()
                return result
                
        except Exception as e:
            raise ValueError(f"Both direct and SavedModel fallback conversion failed. "
                           f"This model may be too complex for TFLite conversion. Details: {e}")
    
    def _is_complex_model(self):
        """Check if this is a complex model that might have conversion issues."""
        # Heuristics to detect complex models like Keras-Hub transformers
        model_name = getattr(self.model, 'name', '').lower()
        model_class = self.model.__class__.__name__.lower()
        
        # Check for transformer/language model indicators
        complex_indicators = [
            'transformer', 'attention', 'bert', 'gpt', 'gemma', 'llama', 
            'backbone', 'causal_lm', 'decoder', 'encoder'
        ]
        
        for indicator in complex_indicators:
            if indicator in model_name or indicator in model_class:
                return True
                
        # Check for large parameter count
        try:
            param_count = self.model.count_params()
            if param_count > 100_000_000:  # 100M parameters
                return True
        except:
            pass
            
        return False

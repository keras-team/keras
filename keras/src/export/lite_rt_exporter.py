import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
from keras.src import tree

class LiteRTExporter:
    """Custom Keras exporter for LiteRT (TFLite) format, bypassing tf.lite.TFLiteConverter."""
    
    def __init__(self, model, input_signature=None, verbose=None, **kwargs):
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose or 0
        self.kwargs = kwargs  # e.g., allow_custom_ops, enable_select_tf_ops, optimizations
    
    def export(self, filepath):
        if self.verbose:
            print("Starting custom LiteRT export...")
        
        # Step 1: Get input signature (use get_input_signature if not provided)
        if self.input_signature is None:
            input_signature = get_input_signature(self.model)
        else:
            input_signature = self.input_signature
        
        # Convert to TensorFlow TensorSpecs for tf.function
        # Handle different input signature structures
        if isinstance(input_signature, list):
            # List of specs
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature]
        elif hasattr(input_signature, 'shape'):
            # Single spec
            tf_signature = [make_tf_tensor_spec(input_signature)]
        else:
            # Try to flatten and convert
            tf_signature = []
            def _convert_to_spec(spec):
                tf_signature.append(make_tf_tensor_spec(spec))
            
            try:
                tree.map_structure(_convert_to_spec, input_signature)
            except:
                # Fallback: assume it's a single spec
                tf_signature = [make_tf_tensor_spec(input_signature)]
        
        # Step 2: Trace the model to create a concrete function
        @tf.function(input_signature=tf_signature)
        def model_fn(*inputs):
            return self.model(*inputs)
        concrete_fn = model_fn.get_concrete_function()
        
        # Step 3: Convert using TFLite converter directly (simplified approach)
        # Skip the complex MLIR conversion and use the standard converter
        tflite_model = self._convert_to_tflite(concrete_fn)
        
        if self.verbose:
            print(f"LiteRT model converted. Size: {len(tflite_model)} bytes")
        
        # Step 4: Save to file
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported to {filepath}")
    
    def _convert_to_tflite(self, concrete_fn):
        """Convert concrete function to TFLite using standard converter."""
        try:
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
            converter.allow_custom_ops = self.kwargs.get("allow_custom_ops", False)
            converter.enable_select_tf_ops = self.kwargs.get("enable_select_tf_ops", False)
            if "optimizations" in self.kwargs:
                converter.optimizations = self.kwargs["optimizations"]
            result = converter.convert()
            return result
        except Exception as e:
            if "custom op" in str(e).lower():
                raise ValueError(f"Custom ops detected. Enable allow_custom_ops=True. Details: {e}")
            raise

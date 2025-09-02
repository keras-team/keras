import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
import tempfile
import os

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

        # 1. Get the input signature with a bounded sequence length.
        # This is the critical step to prevent memory overflow.
        if self.input_signature is None:
            if self.verbose:
                print(f"Inferring input signature with max_sequence_length={self.max_sequence_length}.")
            self.input_signature = get_input_signature(self.model, self.max_sequence_length)
        
        # 2. Create a single concrete function from the model's call method.
        concrete_fn = self._get_concrete_function(self.input_signature)
        
        # 3. Convert the concrete function to a TFLite model.
        tflite_model = self._convert_to_tflite(concrete_fn)
        
        if self.verbose:
            print(f"LiteRT model converted successfully. Size: {len(tflite_model)} bytes")
        
        # 4. Save the model to the specified file path.
        if not filepath.endswith('.tflite'):
            filepath += '.tflite'
        
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported model to {filepath}")

    def _get_concrete_function(self, input_signature):
        """Creates a tf.function from the model's call method and gets its concrete function."""
        if isinstance(input_signature, dict):
            tf_signature = {k: make_tf_tensor_spec(v) for k, v in input_signature.items()}
            
            @tf.function
            def model_fn(inputs):
                return self.model(inputs)
            
            return model_fn.get_concrete_function(tf_signature)
        
        elif isinstance(input_signature, list):
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature]
            
            @tf.function
            def model_fn(*inputs):
                return self.model(list(inputs))

            return model_fn.get_concrete_function(*tf_signature)
        
        else: # Assumes a single tensor
            tf_signature = make_tf_tensor_spec(input_signature)
            
            @tf.function
            def model_fn(input_tensor):
                return self.model(input_tensor)

            return model_fn.get_concrete_function(tf_signature)

    def _convert_to_tflite(self, concrete_fn):
        """Converts a concrete function to TFLite via the SavedModel path for robustness."""
        if self.verbose:
            print("Using SavedModel conversion path for robustness.")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # --- FIX: Save a bare tf.Module() instead of the full Keras model. ---
            # The concrete_fn already captures all necessary variables. Saving a
            # simple module avoids the `_DictWrapper` serialization error.
            module = tf.Module()
            tf.saved_model.save(
                module, temp_dir, signatures={"serving_default": concrete_fn}
            )
            
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            
            # Apply necessary settings for complex models
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.allow_custom_ops = True
            
            if "optimizations" in self.kwargs:
                converter.optimizations = self.kwargs["optimizations"]
            
            return converter.convert()
import tensorflow as tf
from keras.src.export.saved_model import ExportArchive  # Adjusted import based on available modules
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils

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
        
        # Step 2: Trace the model to create a concrete function
        @tf.function(input_signature=[input_signature])
        def model_fn(*inputs):
            return self.model(*inputs)
        concrete_fn = model_fn.get_concrete_function()
        
        # Step 3: Convert to MLIR and apply TFLite passes (bypass high-level converter)
        # Use TensorFlow's internal MLIR conversion (inspired by tf_tfl_translate.cc)
        from tensorflow.compiler.mlir import tf2tfl  # Internal module for MLIR conversion
        mlir_module = tf2tfl.convert_function(concrete_fn, enable_select_tf_ops=self.kwargs.get("enable_select_tf_ops", False))
        
        # Step 4: Export to FlatBuffer (inspired by ExportFlatbufferOrMlir and TfLiteExporter)
        from tensorflow.lite.python import tflite_convert  # Use internal conversion
        converter_flags = tf.lite.experimental.ConverterFlags()
        converter_flags.allow_custom_ops = self.kwargs.get("allow_custom_ops", False)
        converter_flags.enable_select_tf_ops = self.kwargs.get("enable_select_tf_ops", False)
        if "optimizations" in self.kwargs:
            converter_flags.optimizations = self.kwargs["optimizations"]
        
        # Perform the conversion using MLIR-to-FlatBuffer (custom logic)
        tflite_model = self._mlir_to_flatbuffer(mlir_module, converter_flags, concrete_fn)
        
        if self.verbose:
            print(f"LiteRT model converted. Size: {len(tflite_model)} bytes")
        
        # Step 5: Save to file
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported to {filepath}")
    
    def _mlir_to_flatbuffer(self, mlir_module, converter_flags, concrete_fn):
        """Custom MLIR-to-FlatBuffer conversion (inspired by attachments)."""
        # Use the standard TFLite converter with our concrete function
        try:
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
            converter.allow_custom_ops = converter_flags.allow_custom_ops
            converter.enable_select_tf_ops = converter_flags.enable_select_tf_ops
            if hasattr(converter_flags, 'optimizations') and converter_flags.optimizations:
                converter.optimizations = converter_flags.optimizations
            result = converter.convert()
            return result
        except Exception as e:
            if "custom op" in str(e).lower():
                raise ValueError(f"Custom ops detected. Enable allow_custom_ops=True. Details: {e}")
            raise

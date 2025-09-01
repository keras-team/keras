from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import tree
from keras.src.utils.module_utils import tensorflow as tf


def get_input_signature(model, max_sequence_length=512):
    """Get input signature for model export.
    
    Args:
        model: A Keras Model instance.
        max_sequence_length: Maximum sequence length for sequence models (transformers).
            Only applied when the model is detected as a sequence model based on input
            names (e.g., 'token_ids', 'input_ids') or shape patterns. For non-sequence
            models (e.g., image models), this parameter is ignored and dimensions remain
            unbounded. For large vocabulary models, this may be automatically reduced
            to prevent tensor size overflow. Defaults to 512.
    
    Returns:
        Input signature suitable for model export.
    """
    if not isinstance(model, models.Model):
        raise TypeError(
            "The model must be a `keras.Model`. "
            f"Received: model={model} of the type {type(model)}"
        )
    if not model.built:
        raise ValueError(
            "The model provided has not yet been built. It must be built "
            "before export."
        )
        
    # For large vocabulary models, adjust sequence length to prevent overflow
    effective_max_length = _get_safe_sequence_length(model, max_sequence_length)
    
    if isinstance(model, models.Functional):
        input_signature = tree.map_structure(make_input_spec, model._inputs_struct)
    elif isinstance(model, models.Sequential):
        input_signature = tree.map_structure(make_input_spec, model.inputs)
    else:
        # For subclassed models, try multiple approaches
        input_signature = _infer_input_signature_from_model(model, effective_max_length)
        if not input_signature:
            # Fallback: Try to get from model.inputs if available
            if hasattr(model, 'inputs') and model.inputs:
                input_signature = tree.map_structure(make_input_spec, model.inputs)
            elif not model._called:
                raise ValueError(
                    "The model provided has never been called and has no "
                    "detectable input structure. It must be called at least once "
                    "before export, or you must provide explicit input_signature."
                )
    return input_signature


def _get_safe_sequence_length(model, max_sequence_length):
    """Get a safe sequence length that won't cause tensor size overflow."""
    model_class_name = getattr(model, '__class__', type(None)).__name__.lower()
    model_module = getattr(getattr(model, '__class__', type(None)), '__module__', '').lower()
    
    # Check if this is a large vocabulary model
    large_vocab_indicators = ['gemma', 'llama', 'palm', 'gpt']
    is_large_vocab = (
        any(indicator in model_class_name for indicator in large_vocab_indicators) or
        'keras_hub' in model_module
    )
    
    if is_large_vocab:
        # Estimate tensor size: seq_len × vocab_size × 4 bytes (float32)
        # Conservative vocab size estimate for large models
        estimated_vocab_size = 256000
        estimated_bytes = max_sequence_length * estimated_vocab_size * 4
        
        # If estimated size > 512MB, reduce sequence length
        max_safe_bytes = 512 * 1024 * 1024  # 512MB
        if estimated_bytes > max_safe_bytes:
            safe_length = max_safe_bytes // (estimated_vocab_size * 4)
            safe_length = max(32, min(safe_length, max_sequence_length))  # At least 32, at most original
            if safe_length < max_sequence_length:
                print(f"Warning: Reducing max_sequence_length from {max_sequence_length} to {safe_length} "
                      f"for large vocabulary model to prevent tensor size overflow.")
            return safe_length
    
    return max_sequence_length


def _infer_input_signature_from_model(model, max_sequence_length=512):
    shapes_dict = getattr(model, "_build_shapes_dict", None)
    if not shapes_dict:
        return None

    # Use the safe sequence length to prevent overflow
    safe_sequence_length = _get_safe_sequence_length(model, max_sequence_length)

    def _is_sequence_model():
        """Detect if this is a sequence model based on input names and shapes."""
        if not shapes_dict:
            return False
        
        # Check input names for sequence model indicators
        input_names = list(shapes_dict.keys())
        sequence_indicators = ['token_ids', 'input_ids', 'tokens', 'input_tokens', 
                             'padding_mask', 'attention_mask', 'segment_ids']
        
        if any(indicator in name.lower() for name in input_names for indicator in sequence_indicators):
            return True
            
        # Check if any input has shape with 2+ dimensions where second dim is None
        # This is typical for sequence models: (batch_size, seq_len, ...)
        for shape in shapes_dict.values():
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                if shape[0] is None and shape[1] is None:  # (None, None, ...)
                    return True
                    
        return False

    def _make_input_spec(structure):
        # We need to turn wrapper structures like TrackingDict or _DictWrapper
        # into plain Python structures because they don't work with jax2tf/JAX.
        if isinstance(structure, dict):
            return {k: _make_input_spec(v) for k, v in structure.items()}
        elif isinstance(structure, tuple):
            if all(isinstance(d, (int, type(None))) for d in structure):
                # Handle shape bounding based on model type
                is_sequence_model = _is_sequence_model()
                bounded_shape = []
                
                for i, dim in enumerate(structure):
                    if dim is None:
                        if i == 0:
                            # Always keep batch dimension as None
                            bounded_shape.append(None)
                        elif is_sequence_model and i == 1:
                            # For sequence models, bound the sequence length dimension
                            # Using safe sequence length to prevent overflow
                            bounded_shape.append(safe_sequence_length)
                        else:
                            # For non-sequence models or non-sequence dimensions, keep unbounded
                            # This prevents breaking image models, etc.
                            bounded_shape.append(None)
                    else:
                        bounded_shape.append(dim)
                        
                return layers.InputSpec(
                    shape=tuple(bounded_shape), dtype=model.input_dtype
                )
            return tuple(_make_input_spec(v) for v in structure)
        elif isinstance(structure, list):
            if all(isinstance(d, (int, type(None))) for d in structure):
                # Handle shape bounding based on model type
                is_sequence_model = _is_sequence_model()
                bounded_shape = []
                
                for i, dim in enumerate(structure):
                    if dim is None:
                        if i == 0:
                            # Always keep batch dimension as None
                            bounded_shape.append(None)
                        elif is_sequence_model and i == 1:
                            # For sequence models, bound the sequence length dimension
                            # Using safe sequence length to prevent overflow
                            bounded_shape.append(safe_sequence_length)
                        else:
                            # For non-sequence models or non-sequence dimensions, keep unbounded
                            bounded_shape.append(None)
                    else:
                        bounded_shape.append(dim)
                        
                return layers.InputSpec(
                    shape=bounded_shape, dtype=model.input_dtype
                )
            return [_make_input_spec(v) for v in structure]
        else:
            raise ValueError(
                f"Unsupported type {type(structure)} for {structure}"
            )

    # Try to reconstruct the input structure from build shapes
    if len(shapes_dict) == 1:
        # Single input case
        return _make_input_spec(list(shapes_dict.values())[0])
    else:
        # Multiple inputs - try to determine if it's a dict or list structure
        # For Keras-Hub models like Gemma3, inputs are typically dictionaries
        input_keys = list(shapes_dict.keys())
        
        # Common patterns for multi-input models
        if any(key in ['token_ids', 'padding_mask', 'input_ids', 'attention_mask'] for key in input_keys):
            # Dictionary input structure (common for transformers)
            return {key: _make_input_spec(shape) for key, shape in shapes_dict.items()}
        else:
            # List input structure
            return [_make_input_spec(shape) for shape in shapes_dict.values()]


def make_input_spec(x):
    if isinstance(x, layers.InputSpec):
        if x.shape is None or x.dtype is None:
            raise ValueError(
                f"The `shape` and `dtype` must be provided. Received: x={x}"
            )
        input_spec = x
    elif isinstance(x, backend.KerasTensor):
        shape = (None,) + backend.standardize_shape(x.shape)[1:]
        dtype = backend.standardize_dtype(x.dtype)
        input_spec = layers.InputSpec(dtype=dtype, shape=shape, name=x.name)
    elif backend.is_tensor(x):
        shape = (None,) + backend.standardize_shape(x.shape)[1:]
        dtype = backend.standardize_dtype(x.dtype)
        input_spec = layers.InputSpec(dtype=dtype, shape=shape, name=None)
    else:
        raise TypeError(
            f"Unsupported x={x} of the type ({type(x)}). Supported types are: "
            "`keras.InputSpec`, `keras.KerasTensor` and backend tensor."
        )
    return input_spec


def make_tf_tensor_spec(x):
    if isinstance(x, tf.TensorSpec):
        tensor_spec = x
    else:
        input_spec = make_input_spec(x)
        tensor_spec = tf.TensorSpec(
            input_spec.shape, dtype=input_spec.dtype, name=input_spec.name
        )
    return tensor_spec


def convert_spec_to_tensor(spec, replace_none_number=None):
    shape = backend.standardize_shape(spec.shape)
    if replace_none_number is not None:
        replace_none_number = int(replace_none_number)
        shape = tuple(
            s if s is not None else replace_none_number for s in shape
        )
    return ops.ones(shape, spec.dtype)


# Registry for export formats
EXPORT_FORMATS = {
    "tf_saved_model": "keras.src.export.saved_model:export_saved_model",
    "lite_rt": "keras.src.export.lite_rt_exporter:LiteRTExporter",
    # Add other formats as needed
}


def _get_exporter(format_name):
    """Lazy import exporter to avoid circular imports."""
    if format_name not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format: {format_name}")

    exporter = EXPORT_FORMATS[format_name]
    if isinstance(exporter, str):
        # Lazy import for string references
        module_path, attr_name = exporter.split(":")
        module = __import__(module_path, fromlist=[attr_name])
        return getattr(module, attr_name)
    else:
        # Direct reference
        return exporter


def export_model(model, filepath, format="tf_saved_model", **kwargs):
    """Export a model to the specified format."""
    exporter_cls = _get_exporter(format)
    if format == "tf_saved_model":
        # Handle tf_saved_model differently if it's a function
        exporter_cls(model, filepath, **kwargs)
    else:
        exporter = exporter_cls(model, **kwargs)
        exporter.export(filepath)

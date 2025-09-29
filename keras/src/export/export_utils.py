from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import tree
from keras.src.utils.module_utils import tensorflow as tf


def get_input_signature(model):
    """Get input signature for model export.
    
    Args:
        model: A Keras Model instance.
    
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
        
    if isinstance(model, models.Functional):
        input_signature = tree.map_structure(make_input_spec, model._inputs_struct)
    elif isinstance(model, models.Sequential):
        input_signature = tree.map_structure(make_input_spec, model.inputs)
    else:
        # For subclassed models, try multiple approaches
        input_signature = _infer_input_signature_from_model(model)
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


def _infer_input_signature_from_model(model):
    shapes_dict = getattr(model, "_build_shapes_dict", None)
    if not shapes_dict:
        return None

    def _make_input_spec(structure):
        # We need to turn wrapper structures like TrackingDict or _DictWrapper
        # into plain Python structures because they don't work with jax2tf/JAX.
        if isinstance(structure, dict):
            return {k: _make_input_spec(v) for k, v in structure.items()}
        elif isinstance(structure, tuple):
            if all(isinstance(d, (int, type(None))) for d in structure):
                # Keep batch dimension unbounded, keep other dimensions as they are
                bounded_shape = []
                
                for i, dim in enumerate(structure):
                    if dim is None and i == 0:
                        # Always keep batch dimension as None
                        bounded_shape.append(None)
                    else:
                        # Keep other dimensions as they are (None or specific size)
                        bounded_shape.append(dim)
                        
                return layers.InputSpec(
                    shape=tuple(bounded_shape), dtype=model.input_dtype
                )
            return tuple(_make_input_spec(v) for v in structure)
        elif isinstance(structure, list):
            if all(isinstance(d, (int, type(None))) for d in structure):
                # Keep batch dimension unbounded, keep other dimensions as they are
                bounded_shape = []
                
                for i, dim in enumerate(structure):
                    if dim is None and i == 0:
                        # Always keep batch dimension as None
                        bounded_shape.append(None)
                    else:
                        # Keep other dimensions as they are
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
        # Return as dictionary by default to preserve input names
        return {key: _make_input_spec(shape) for key, shape in shapes_dict.items()}


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
    "litert": "keras.src.export.litert_exporter:export_litert",
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
    exporter = _get_exporter(format)

    if isinstance(exporter, type):
        exporter_instance = exporter(model, **kwargs)
        return exporter_instance.export(filepath)

    return exporter(model, filepath, **kwargs)
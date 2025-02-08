from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import tree
from keras.src.utils.module_utils import tensorflow as tf


def get_input_signature(model):
    print("DEBUG: Entering get_input_signature")
    if not isinstance(model, models.Model):
        raise TypeError(
            "The model must be a `keras.Model`. "
            f"Received: model={model} of the type {type(model)}"
        )
    print("DEBUG: Model type check passed.")
    if not model.built:
        raise ValueError(
            "The model provided has not yet been built. It must be built "
            "before export."
        )
    print("DEBUG: Model is built.")
    if isinstance(model, (models.Functional, models.Sequential)):
        print("DEBUG: Model is Functional or Sequential.")
        print("DEBUG: model.inputs =", model.inputs)
        if hasattr(model, "_input_names"):
            print("DEBUG: model._input_names =", model._input_names)
        else:
            print("DEBUG: model has no attribute '_input_names'.")

        if hasattr(model, "_input_names") and model._input_names:
            print("DEBUG: Using _input_names to create input_signature.")
            if isinstance(model._inputs_struct, dict):
                # Create dictionary input signature while
                # preserving order.
                input_signature = {
                    name: make_input_spec(tensor)
                    for name, tensor in zip(model._input_names, model.inputs)
                }
            else:
                input_signature = tree.map_structure(
                    make_input_spec, model.inputs
                )
        else:
            print("DEBUG: Fallback to tree.map_structure.")
            input_signature = tree.map_structure(make_input_spec, model.inputs)
    else:
        input_signature = _infer_input_signature_from_model(model)
        print("DEBUG: Inferred input_signature:", input_signature)
        if not input_signature or not model._called:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
            )
    print("DEBUG: Exiting get_input_signature with:", input_signature)
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
                return layers.InputSpec(
                    shape=(None,) + structure[1:], dtype=model.input_dtype
                )
            return tuple(_make_input_spec(v) for v in structure)
        elif isinstance(structure, list):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return layers.InputSpec(
                    shape=[None] + structure[1:], dtype=model.input_dtype
                )
            return [_make_input_spec(v) for v in structure]
        else:
            raise ValueError(
                f"Unsupported type {type(structure)} for {structure}"
            )

    return [_make_input_spec(value) for value in shapes_dict.values()]


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
        print("DEBUG: x is a TensorSpec")
        return x
    if isinstance(x, dict):
        print("DEBUG: x is a dictionary")
        # Convert dict to ordered list with names preserved.
        return {
            name: tf.TensorSpec(shape=spec.shape, dtype=spec.dtype, name=name)
            for name, spec in x.items()
        }
    elif isinstance(x, layers.InputSpec):
        print("DEBUG: x is an InputSpec")
        return tf.TensorSpec(shape=x.shape, dtype=x.dtype, name=x.name)
    else:
        print("DEBUG: x is other type")
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return tf.TensorSpec(
                shape=x.shape, dtype=x.dtype, name=getattr(x, "name", None)
            )
        raise TypeError(
            f"Unsupported x={x} of the type ({type(x)}). Supported types are: "
            "`keras.InputSpec`, `keras.KerasTensor` and backend tensor."
        )


def convert_spec_to_tensor(spec, replace_none_number=None):
    shape = backend.standardize_shape(spec.shape)
    if replace_none_number is not None:
        replace_none_number = int(replace_none_number)
        shape = tuple(
            s if s is not None else replace_none_number for s in shape
        )
    return ops.ones(shape, spec.dtype)

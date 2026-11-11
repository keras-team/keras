import itertools
import string

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import tree
from keras.src.backend.common import KerasTensor
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common import standardize_shape
from keras.src.utils.module_utils import tensorflow as tf


def get_input_signature(model):
    """Get input signature for model export.

    Args:
        model: A Keras Model instance.

    Returns:
        Input signature suitable for model export (always a tuple or list).
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
        # Functional models expect a single positional argument `inputs`
        # containing the full nested input structure. We keep the
        # original behavior of returning a single-element list that
        # wraps the mapped structure so that downstream exporters
        # build a tf.function with one positional argument.
        input_signature = [
            tree.map_structure(make_input_spec, model._inputs_struct)
        ]
    elif isinstance(model, models.Sequential):
        input_signature = tree.map_structure(make_input_spec, model.inputs)
    else:
        # Subclassed models: rely on recorded shapes from the first call.
        input_signature = _infer_input_signature_from_model(model)
        if not input_signature or not model._called:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
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

    # Always return a flat list preserving the order of shapes_dict values
    return [_make_input_spec(value) for value in shapes_dict.values()]


def make_input_spec(x):
    if isinstance(x, layers.InputSpec):
        if x.shape is None or x.dtype is None:
            raise ValueError(
                f"The `shape` and `dtype` must be provided. Received: x={x}"
            )
        input_spec = x
    elif isinstance(x, KerasTensor):
        shape = (None,) + standardize_shape(x.shape)[1:]
        dtype = standardize_dtype(x.dtype)
        input_spec = layers.InputSpec(dtype=dtype, shape=shape, name=x.name)
    elif backend.ops.is_tensor(x):
        shape = (None,) + standardize_shape(x.shape)[1:]
        dtype = standardize_dtype(x.dtype)
        input_spec = layers.InputSpec(dtype=dtype, shape=shape, name=None)
    else:
        raise TypeError(
            f"Unsupported x={x} of the type ({type(x)}). Supported types are: "
            "`keras.InputSpec`, `keras.KerasTensor` and backend tensor."
        )
    return input_spec


def make_tf_tensor_spec(x, dynamic_batch=False):
    """Create a TensorSpec from various input types.

    Args:
        x: Input to convert (tf.TensorSpec, KerasTensor, or backend tensor).
        dynamic_batch: If True, set the batch dimension to None.

    Returns:
        A tf.TensorSpec instance.
    """
    if isinstance(x, tf.TensorSpec):
        tensor_spec = x
        # Adjust batch dimension if needed
        if dynamic_batch and len(tensor_spec.shape) > 0:
            shape = tuple(
                None if i == 0 else s for i, s in enumerate(tensor_spec.shape)
            )
            tensor_spec = tf.TensorSpec(
                shape, dtype=tensor_spec.dtype, name=tensor_spec.name
            )
    else:
        input_spec = make_input_spec(x)
        shape = input_spec.shape
        # Adjust batch dimension if needed and shape is not None
        if dynamic_batch and shape is not None and len(shape) > 0:
            shape = tuple(None if i == 0 else s for i, s in enumerate(shape))
        tensor_spec = tf.TensorSpec(
            shape, dtype=input_spec.dtype, name=input_spec.name
        )
    return tensor_spec


def convert_spec_to_tensor(spec, replace_none_number=None):
    shape = standardize_shape(spec.shape)
    if replace_none_number is not None:
        replace_none_number = int(replace_none_number)
        shape = tuple(
            s if s is not None else replace_none_number for s in shape
        )
    return ops.ones(shape, spec.dtype)


def to_polymorphic_shape(struct, allow_none=True):
    """Converts a structure of specs to JAX polymorphic shape strings.

    Each leaf with a `shape` attribute (e.g. `keras.InputSpec`,
    `tf.TensorSpec`, `KerasTensor`) is turned into a string such as
    `"(batch, 10)"` that `jax2tf.convert` and `jax.export.symbolic_shape`
    understand. Known dimensions are kept as ints; the leading `None`
    dimension is named `batch` (shared across all leaves) and the other
    `None` dimensions get a unique symbol each (`a`, `b`, ..., `aa`, ...).

    Args:
        struct: A (possibly nested) structure of specs.
        allow_none: Whether `None` dimensions are allowed. If `False`, a
            `ValueError` is raised for any `None` dimension.

    Returns:
        A structure of strings with the same layout as `struct`.
    """
    if allow_none:
        # Generates unique names: a, b, ... z, aa, ab, ... az, ba, ... zz
        # for unknown non-batch dims. Defined here to be scope per endpoint.
        dim_names = itertools.chain(
            string.ascii_lowercase,
            itertools.starmap(
                lambda a, b: a + b,
                itertools.product(string.ascii_lowercase, repeat=2),
            ),
        )

    def convert_shape(x):
        poly_shape = []
        for index, dim in enumerate(list(x.shape)):
            if dim is not None:
                poly_shape.append(str(dim))
            elif not allow_none:
                raise ValueError(
                    f"Illegal None dimension in {x} with shape {x.shape}"
                )
            elif index == 0:
                poly_shape.append("batch")
            else:
                poly_shape.append(next(dim_names))
        return f"({', '.join(poly_shape)})"

    return tree.map_structure(convert_shape, struct)

from tensorflow import nest

from keras_core.api_export import keras_core_export
from keras_core.utils.naming import auto_name


@keras_core_export("keras_core.KerasTensor")
class KerasTensor:
    """Symbolic tensor -- encapsulates a shape and a dtype.

    You can use `KerasTensor` instances to build computation
    graphs of Keras operations, such as `keras_core.Function`
    objects or Functional `keras_core.models.Model` objects.

    Example:

    >>> x = keras_core.KerasTensor(shape=(3, 4), dtype="float32")
    >>> x.shape
    (3, 4)
    >>> x.dtype
    float32

    Calling a Keras operation (including a layer or a model)
    on a `KerasTensor` instance will return another `KerasTensor`
    instance with the appropriate shape and dtype. This is
    called a "symbolic call" (since there is no actual data
    involved). The computation of the correct output shape and
    dtype is called "static shape inference".
    """

    def __init__(self, shape, dtype="float32", record_history=True, name=None):
        from keras_core import backend

        shape = backend.standardize_shape(
            shape,
            allow_all_dynamic=backend.DYNAMIC_SHAPES_OK,
        )
        self.shape = shape
        self.dtype = backend.standardize_dtype(dtype)
        self.name = name or auto_name(self.__class__.__name__)
        self.record_history = record_history

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self, new_shape):
        from keras_core import operations

        return operations.Reshape(new_shape)(self)

    def squeeze(self, axis=None):
        from keras_core import operations

        return operations.Squeeze(axis)(self)

    def __array__(self):
        raise ValueError(
            "A KerasTensor is symbolic: it's a placeholder for a shape "
            "an a dtype. It doesn't have any actual numerical value. "
            "You cannot convert it to a NumPy array."
        )

    def __jax_array__(self):
        raise ValueError(
            "A KerasTensor cannot be used as input to a JAX function. "
            "A KerasTensor is a symbolic placeholder for a shape and dtype, "
            "used when constructing Keras Functional models "
            "or Keras Functions. You can only use it as input to a Keras layer "
            "or a Keras operation (from the namespaces `keras_core.layers` "
            "and `keras_core.operations`). "
            "You are likely doing something like:\n\n"
            "```\n"
            "x = Input(...)\n"
            "...\n"
            "jax_fn(x)  # Invalid.\n"
            "```\n\n"
            "What you should do instead is wrap `jax_fn` in a layer:\n\n"
            "```\n"
            "class MyLayer(Layer):\n"
            "    def call(self, x):\n"
            "        return jax_fn(x)\n\n"
            "x = MyLayer()(x)"
            "```\n"
        )

    def __tf_tensor__(self, dtype=None, name=None):
        raise ValueError(
            "A KerasTensor cannot be used as input to a TensorFlow function. "
            "A KerasTensor is a symbolic placeholder for a shape and dtype, "
            "used when constructing Keras Functional models "
            "or Keras Functions. You can only use it as input to a Keras layer "
            "or a Keras operation (from the namespaces `keras_core.layers` "
            "and `keras_core.operations`). "
            "You are likely doing something like:\n\n"
            "```\n"
            "x = Input(...)\n"
            "...\n"
            "tf_fn(x)  # Invalid.\n"
            "```\n\n"
            "What you should do instead is wrap `tf_fn` in a layer:\n\n"
            "```\n"
            "class MyLayer(Layer):\n"
            "    def call(self, x):\n"
            "        return tf_fn(x)\n\n"
            "x = MyLayer()(x)"
            "```\n"
        )

    def __repr__(self):
        return (
            f"<KerasTensor shape={self.shape}, dtype={self.dtype}, "
            f"name={self.name}>"
        )

    def __iter__(self):
        raise NotImplementedError(
            "Iterating over a symbolic KerasTensor is not supported."
        )

    def __bool__(self):
        raise TypeError("A symbolic KerasTensor cannot be used as a boolean.")

    def __add__(self, other):
        from keras_core import operations

        return operations.Add().symbolic_call(self, other)

    def __radd__(self, other):
        from keras_core import operations

        return operations.Add().symbolic_call(other, self)

    def __sub__(self, other):
        from keras_core import operations

        return operations.Subtract().symbolic_call(self, other)

    def __rsub__(self, other):
        from keras_core import operations

        return operations.Subtract().symbolic_call(other, self)

    def __mul__(self, other):
        from keras_core import operations

        return operations.Multiply().symbolic_call(self, other)

    def __rmul__(self, other):
        from keras_core import operations

        return operations.Multiply().symbolic_call(other, self)

    def __matmul__(self, other):
        from keras_core import operations

        return operations.Matmul().symbolic_call(self, other)

    def __rmatmul__(self, other):
        from keras_core import operations

        return operations.Matmul().symbolic_call(other, self)

    def __div__(self, other):
        from keras_core import operations

        return operations.Divide().symbolic_call(self, other)

    def __rdiv__(self, other):
        from keras_core import operations

        return operations.Divide().symbolic_call(other, self)

    def __truediv__(self, other):
        from keras_core import operations

        return operations.TrueDivide().symbolic_call(self, other)

    def __rtruediv__(self, other):
        from keras_core import operations

        return operations.TrueDivide().symbolic_call(other, self)

    def __neg__(self):
        from keras_core import operations

        return operations.Negative().symbolic_call(self)

    def __abs__(self):
        from keras_core import operations

        return operations.Absolute().symbolic_call(self)

    def __pow__(self, other):
        from keras_core import operations

        return operations.Power().symbolic_call(self, other)

    def __rpow__(self, other):
        from keras_core import operations

        return operations.Power().symbolic_call(other, self)

    def __getitem__(self, key):
        from keras_core import operations

        return operations.GetItem().symbolic_call(self, key)

    # TODO
    #   "__floordiv__",
    #   "__rfloordiv__",
    #   "__mod__",
    #   "__rmod__",
    #   "__lt__",
    #   "__le__",
    #   "__gt__",
    #   "__ge__",
    #   "__ne__",
    #   "__eq__",
    #   "__and__",
    #   "__rand__",
    #   "__or__",
    #   "__ror__",
    #   "__xor__",
    #   "__rxor__",
    #   "__invert__",
    #   "broadcast_to"
    #   "astype"
    #   a few more NumPy ones...


def any_symbolic_tensors(args=None, kwargs=None):
    args = args or ()
    kwargs = kwargs or {}
    for x in nest.flatten((args, kwargs)):
        if isinstance(x, KerasTensor):
            return True
    return False


@keras_core_export("keras_core.utils.is_keras_tensor")
def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a *symbolic tensor*, such as a tensor
    that was created via `Input()`. A "symbolic tensor"
    can be understood as a placeholder -- it does not
    contain any actual numerical data, only a shape and dtype.
    It can be used for building Functional models, but it
    cannot be used in actual computations.
    """
    return isinstance(x, KerasTensor)

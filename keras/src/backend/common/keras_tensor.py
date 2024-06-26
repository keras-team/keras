from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name


@keras_export("keras.KerasTensor")
class KerasTensor:
    """Symbolic tensor -- encapsulates a shape and a dtype.

    You can use `KerasTensor` instances to build computation
    graphs of Keras operations, such as `keras.Function`
    objects or Functional `keras.models.Model` objects.

    Example:

    >>> x = keras.KerasTensor(shape=(3, 4), dtype="float32")
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

    def __init__(
        self,
        shape,
        dtype="float32",
        sparse=False,
        record_history=True,
        name=None,
    ):
        from keras.src import backend

        self._shape = backend.standardize_shape(shape)
        self._dtype = backend.standardize_dtype(dtype)
        self._sparse = bool(sparse)
        self.name = name or auto_name(self.__class__.__name__)
        self.record_history = record_history

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        raise AttributeError(
            f"The shape of {self.__class__.__name__} is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        raise AttributeError(
            f"The dtype of {self.__class__.__name__} is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        raise AttributeError(
            f"The sparse of {self.__class__.__name__} is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self, newshape):
        from keras.src import ops

        return ops.Reshape(newshape)(self)

    def squeeze(self, axis=None):
        from keras.src import ops

        return ops.Squeeze(axis)(self)

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
            "or a Keras operation (from the namespaces `keras.layers` "
            "and `keras.operations`). "
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
            "x = MyLayer()(x)\n"
            "```\n"
        )

    def __tf_tensor__(self, dtype=None, name=None):
        raise ValueError(
            "A KerasTensor cannot be used as input to a TensorFlow function. "
            "A KerasTensor is a symbolic placeholder for a shape and dtype, "
            "used when constructing Keras Functional models "
            "or Keras Functions. You can only use it as input to a Keras layer "
            "or a Keras operation (from the namespaces `keras.layers` "
            "and `keras.operations`). "
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
            "x = MyLayer()(x)\n"
            "```\n"
        )

    def __repr__(self):
        return (
            f"<KerasTensor shape={self.shape}, dtype={self.dtype}, "
            f"sparse={self.sparse}, name={self.name}>"
        )

    def __iter__(self):
        raise NotImplementedError(
            "Iterating over a symbolic KerasTensor is not supported."
        )

    def __bool__(self):
        raise TypeError("A symbolic KerasTensor cannot be used as a boolean.")

    def __add__(self, other):
        from keras.src import ops

        return ops.Add().symbolic_call(self, other)

    def __radd__(self, other):
        from keras.src import ops

        return ops.Add().symbolic_call(other, self)

    def __sub__(self, other):
        from keras.src import ops

        return ops.Subtract().symbolic_call(self, other)

    def __rsub__(self, other):
        from keras.src import ops

        return ops.Subtract().symbolic_call(other, self)

    def __mul__(self, other):
        from keras.src import ops

        return ops.Multiply().symbolic_call(self, other)

    def __rmul__(self, other):
        from keras.src import ops

        return ops.Multiply().symbolic_call(other, self)

    def __matmul__(self, other):
        from keras.src import ops

        return ops.Matmul().symbolic_call(self, other)

    def __rmatmul__(self, other):
        from keras.src import ops

        return ops.Matmul().symbolic_call(other, self)

    def __div__(self, other):
        from keras.src import ops

        return ops.Divide().symbolic_call(self, other)

    def __rdiv__(self, other):
        from keras.src import ops

        return ops.Divide().symbolic_call(other, self)

    def __truediv__(self, other):
        from keras.src import ops

        return ops.TrueDivide().symbolic_call(self, other)

    def __rtruediv__(self, other):
        from keras.src import ops

        return ops.TrueDivide().symbolic_call(other, self)

    def __neg__(self):
        from keras.src import ops

        return ops.Negative().symbolic_call(self)

    def __abs__(self):
        from keras.src import ops

        return ops.Absolute().symbolic_call(self)

    def __pow__(self, other):
        from keras.src import ops

        return ops.Power().symbolic_call(self, other)

    def __rpow__(self, other):
        from keras.src import ops

        return ops.Power().symbolic_call(other, self)

    def __floordiv__(self, other):
        from keras.src import ops

        return ops.FloorDivide().symbolic_call(self, other)

    def __rfloordiv__(self, other):
        from keras.src import ops

        return ops.FloorDivide().symbolic_call(other, self)

    def __mod__(self, other):
        from keras.src import ops

        return ops.Mod().symbolic_call(self, other)

    def __rmod__(self, other):
        from keras.src import ops

        return ops.Mod().symbolic_call(other, self)

    def __lt__(self, other):
        from keras.src import ops

        return ops.Less().symbolic_call(self, other)

    def __le__(self, other):
        from keras.src import ops

        return ops.LessEqual().symbolic_call(self, other)

    def __gt__(self, other):
        from keras.src import ops

        return ops.Greater().symbolic_call(self, other)

    def __ge__(self, other):
        from keras.src import ops

        return ops.GreaterEqual().symbolic_call(self, other)

    def __ne__(self, other):
        from keras.src import ops

        return ops.NotEqual().symbolic_call(self, other)

    def __and__(self, other):
        from keras.src import ops

        return ops.LogicalAnd().symbolic_call(self, other)

    def __rand__(self, other):
        from keras.src import ops

        return ops.LogicalAnd().symbolic_call(other, self)

    def __or__(self, other):
        from keras.src import ops

        return ops.LogicalOr().symbolic_call(self, other)

    def __ror__(self, other):
        from keras.src import ops

        return ops.LogicalOr().symbolic_call(other, self)

    def __invert__(self):
        from keras.src import ops

        return ops.LogicalNot().symbolic_call(self)

    def __xor__(self, other):
        from keras.src import ops

        return ops.LogicalXor().symbolic_call(self, other)

    def __rxor__(self, other):
        from keras.src import ops

        return ops.LogicalXor().symbolic_call(other, self)

    def __getitem__(self, key):
        from keras.src import ops

        return ops.GetItem().symbolic_call(self, key)


def any_symbolic_tensors(args=None, kwargs=None):
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if isinstance(x, KerasTensor):
            return True
    return False


@keras_export(["keras.utils.is_keras_tensor", "keras.backend.is_keras_tensor"])
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

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export


@keras_export("keras.constraints.Constraint")
class Constraint:
    """Base class for weight constraints.

    A `Constraint` instance works like a stateless function.
    Users who subclass this
    class should override the `__call__()` method, which takes a single
    weight parameter and return a projected version of that parameter
    (e.g. normalized or clipped). Constraints can be used with various Keras
    layers via the `kernel_constraint` or `bias_constraint` arguments.

    Here's a simple example of a non-negative weight constraint:

    >>> class NonNegative(keras.constraints.Constraint):
    ...
    ...  def __call__(self, w):
    ...    return w * ops.cast(ops.greater_equal(w, 0.), dtype=w.dtype)

    >>> weight = ops.convert_to_tensor((-1.0, 1.0))
    >>> NonNegative()(weight)
    [0.,  1.]

    Usage in a layer:

    >>> keras.layers.Dense(4, kernel_constraint=NonNegative())
    """

    def __call__(self, w):
        """Applies the constraint to the input weight variable.

        By default, the inputs weight variable is not modified.
        Users should override this method to implement their own projection
        function.

        Args:
            w: Input weight variable.

        Returns:
            Projected variable (by default, returns unmodified inputs).
        """
        return w

    def get_config(self):
        """Returns a Python dict of the object config.

        A constraint config is a Python dictionary (JSON-serializable) that can
        be used to reinstantiate the same object.

        Returns:
            Python dict containing the configuration of the constraint object.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates a weight constraint from a configuration dictionary.

        Example:

        ```python
        constraint = UnitNorm()
        config = constraint.get_config()
        constraint = UnitNorm.from_config(config)
        ```

        Args:
            config: A Python dictionary, the output of `get_config()`.

        Returns:
            A `keras.constraints.Constraint` instance.
        """
        return cls(**config)


@keras_export(["keras.constraints.MaxNorm", "keras.constraints.max_norm"])
class MaxNorm(Constraint):
    """MaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.

    Also available via the shortcut function `keras.constraints.max_norm`.

    Args:
        max_value: the maximum norm value for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.

    """

    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        norms = ops.sqrt(ops.sum(ops.square(w), axis=self.axis, keepdims=True))
        desired = ops.clip(norms, 0, self.max_value)
        return ops.cast(w, norms.dtype) * (
            desired / (backend.epsilon() + norms)
        )

    def get_config(self):
        return {"max_value": self.max_value, "axis": self.axis}


@keras_export(["keras.constraints.NonNeg", "keras.constraints.non_neg"])
class NonNeg(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        return ops.multiply(w, ops.greater_equal(w, 0.0))


@keras_export(["keras.constraints.UnitNorm", "keras.constraints.unit_norm"])
class UnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.

    Args:
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        norms = ops.sqrt(ops.sum(ops.square(w), axis=self.axis, keepdims=True))
        return ops.cast(w, norms.dtype) / (backend.epsilon() + norms)

    def get_config(self):
        return {"axis": self.axis}


@keras_export(
    ["keras.constraints.MinMaxNorm", "keras.constraints.min_max_norm"]
)
class MinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.

    Args:
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield
            `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
            Effectively, this means that rate=1.0 stands for strict
            enforcement of the constraint, while rate<1.0 means that
            weights will be rescaled at each step to slowly move
            towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        norms = ops.sqrt(ops.sum(ops.square(w), axis=self.axis, keepdims=True))
        desired = (
            self.rate * ops.clip(norms, self.min_value, self.max_value)
            + (1 - self.rate) * norms
        )
        return ops.cast(w, norms.dtype) * (
            desired / (backend.epsilon() + norms)
        )

    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "rate": self.rate,
            "axis": self.axis,
        }

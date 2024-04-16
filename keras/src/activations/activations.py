from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export


@keras_export("keras.activations.relu")
def relu(x, negative_slope=0.0, max_value=None, threshold=0.0):
    """Applies the rectified linear unit activation function.

    With default values, this returns the standard ReLU activation:
    `max(x, 0)`, the element-wise maximum of 0 and the input tensor.

    Modifying default parameters allows you to use non-zero thresholds,
    change the max value of the activation,
    and to use a non-zero multiple of the input for values below the threshold.

    Examples:

    >>> x = [-10, -5, 0.0, 5, 10]
    >>> keras.activations.relu(x)
    [ 0.,  0.,  0.,  5., 10.]
    >>> keras.activations.relu(x, negative_slope=0.5)
    [-5. , -2.5,  0. ,  5. , 10. ]
    >>> keras.activations.relu(x, max_value=5.)
    [0., 0., 0., 5., 5.]
    >>> keras.activations.relu(x, threshold=5.)
    [-0., -0.,  0.,  0., 10.]

    Args:
        x: Input tensor.
        negative_slope: A `float` that controls the slope
            for values lower than the threshold.
        max_value: A `float` that sets the saturation threshold (the largest
            value the function will return).
        threshold: A `float` giving the threshold value of the activation
            function below which values will be damped or set to zero.

    Returns:
        A tensor with the same shape and dtype as input `x`.
    """
    if backend.any_symbolic_tensors((x,)):
        return ReLU(
            negative_slope=negative_slope,
            max_value=max_value,
            threshold=threshold,
        )(x)
    return ReLU.static_call(
        x,
        negative_slope=negative_slope,
        max_value=max_value,
        threshold=threshold,
    )


class ReLU(ops.Operation):
    def __init__(
        self, negative_slope=0.0, max_value=None, threshold=0.0, name=None
    ):
        super().__init__(name=name)
        self.negative_slope = negative_slope
        self.max_value = max_value
        self.threshold = threshold

    def call(self, x):
        return self.static_call(
            x,
            negative_slope=self.negative_slope,
            max_value=self.max_value,
            threshold=self.threshold,
        )

    def compute_output_spec(self, x):
        return backend.KerasTensor(x.shape, x.dtype)

    @staticmethod
    def static_call(x, negative_slope=0.0, max_value=None, threshold=0.0):
        x = backend.convert_to_tensor(x)
        if negative_slope != 0.0:
            if max_value is None and threshold == 0:
                return backend.nn.leaky_relu(x, negative_slope=negative_slope)

            if threshold != 0:
                negative_part = backend.nn.relu(-x + threshold)
            else:
                negative_part = backend.nn.relu(-x)

        clip_max = max_value is not None
        if threshold != 0:
            # computes x for x > threshold else 0
            threshold = ops.cast(threshold, dtype=x.dtype)
            x = x * backend.cast(
                backend.numpy.greater(x, threshold), dtype=x.dtype
            )
        elif max_value == 6:
            # if no threshold, then can use nn.relu6 native op for performance
            x = backend.nn.relu6(x)
            clip_max = False
        else:
            x = backend.nn.relu(x)

        if clip_max:
            min_value = ops.cast(0.0, dtype=x.dtype)
            max_value = ops.cast(max_value, dtype=x.dtype)
            x = backend.numpy.clip(x, min_value, max_value)

        if negative_slope != 0.0:
            x -= negative_slope * negative_part
        return x


@keras_export("keras.activations.leaky_relu")
def leaky_relu(x, negative_slope=0.2):
    """Leaky relu activation function.

    Args:
        x: Input tensor.
        negative_slope: A `float` that controls the slope
            for values lower than the threshold.
    """
    return ops.leaky_relu(x, negative_slope=negative_slope)


@keras_export("keras.activations.relu6")
def relu6(x):
    """Relu6 activation function.

    It's the ReLU function, but truncated to a maximum value of 6.

    Args:
        x: Input tensor.
    """
    return ops.relu6(x)


@keras_export("keras.activations.softmax")
def softmax(x, axis=-1):
    """Softmax converts a vector of values to a probability distribution.

    The elements of the output vector are in range `[0, 1]` and sum to 1.

    Each input vector is handled independently.
    The `axis` argument sets which axis of the input the function
    is applied along.

    Softmax is often used as the activation for the last
    layer of a classification network because the result could be interpreted as
    a probability distribution.

    The softmax of each vector x is computed as
    `exp(x) / sum(exp(x))`.

    The input values in are the log-odds of the resulting probability.

    Args:
        x: Input tensor.
        axis: Integer, axis along which the softmax is applied.
    """
    output = ops.softmax(x, axis=axis)
    # Cache the logits to use for crossentropy loss.
    try:
        output._keras_logits = x
    except AttributeError:
        # We're dealing with a C-type.
        pass
    return output


@keras_export("keras.activations.elu")
def elu(x, alpha=1.0):
    """Exponential Linear Unit.

    The exponential linear unit (ELU) with `alpha > 0` is define as:

    - `x` if `x > 0`
    - alpha * `exp(x) - 1` if `x < 0`

    ELUs have negative values which pushes the mean of the activations
    closer to zero.

    Mean activations that are closer to zero enable faster learning as they
    bring the gradient closer to the natural gradient.
    ELUs saturate to a negative value when the argument gets smaller.
    Saturation means a small derivative which decreases the variation
    and the information that is propagated to the next layer.

    Args:
        x: Input tensor.

    Reference:

    - [Clevert et al., 2016](https://arxiv.org/abs/1511.07289)
    """
    return ops.elu(x, alpha=alpha)


@keras_export("keras.activations.selu")
def selu(x):
    """Scaled Exponential Linear Unit (SELU).

    The Scaled Exponential Linear Unit (SELU) activation function is defined as:

    - `scale * x` if `x > 0`
    - `scale * alpha * (exp(x) - 1)` if `x < 0`

    where `alpha` and `scale` are pre-defined constants
    (`alpha=1.67326324` and `scale=1.05070098`).

    Basically, the SELU activation function multiplies `scale` (> 1) with the
    output of the `keras.activations.elu` function to ensure a slope larger
    than one for positive inputs.

    The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `keras.initializers.LecunNormal` initializer)
    and the number of input units is "large enough"
    (see reference paper for more information).

    Args:
        x: Input tensor.

    Notes:

    - To be used together with the
        `keras.initializers.LecunNormal` initializer.
    - To be used together with the dropout variant
        `keras.layers.AlphaDropout` (rather than regular dropout).

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """
    return ops.selu(x)


@keras_export("keras.activations.softplus")
def softplus(x):
    """Softplus activation function.

    It is defined as: `softplus(x) = log(exp(x) + 1)`.

    Args:
        x: Input tensor.
    """
    return ops.softplus(x)


@keras_export("keras.activations.softsign")
def softsign(x):
    """Softsign activation function.

    Softsign is defined as: `softsign(x) = x / (abs(x) + 1)`.

    Args:
        x: Input tensor.
    """
    return ops.softsign(x)


@keras_export(["keras.activations.silu", "keras.activations.swish"])
def silu(x):
    """Swish (or Silu) activation function.

    It is defined as: `swish(x) = x * sigmoid(x)`.

    The Swish (or Silu) activation function is a smooth,
    non-monotonic function that is unbounded above and
    bounded below.

    Args:
        x: Input tensor.

    Reference:

    - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
    """
    return ops.silu(x)


@keras_export("keras.activations.gelu")
def gelu(x, approximate=False):
    """Gaussian error linear unit (GELU) activation function.

    The Gaussian error linear unit (GELU) is defined as:

    `gelu(x) = x * P(X <= x)` where `P(X) ~ N(0, 1)`,
    i.e. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.

    GELU weights inputs by their value, rather than gating
    inputs by their sign as in ReLU.

    Args:
        x: Input tensor.
        approximate: A `bool`, whether to enable approximation.

    Reference:

    - [Hendrycks et al., 2016](https://arxiv.org/abs/1606.08415)
    """
    return ops.gelu(x, approximate=approximate)


@keras_export("keras.activations.tanh")
def tanh(x):
    """Hyperbolic tangent activation function.

    It is defined as:
    `tanh(x) = sinh(x) / cosh(x)`, i.e.
    `tanh(x) = ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))`.

    Args:
        x: Input tensor.
    """
    return ops.tanh(x)


@keras_export("keras.activations.sigmoid")
def sigmoid(x):
    """Sigmoid activation function.

    It is defined as: `sigmoid(x) = 1 / (1 + exp(-x))`.

    For small values (<-5),
    `sigmoid` returns a value close to zero, and for large values (>5)
    the result of the function gets close to 1.

    Sigmoid is equivalent to a 2-element softmax, where the second element is
    assumed to be zero. The sigmoid function always returns a value between
    0 and 1.

    Args:
        x: Input tensor.
    """
    output = ops.sigmoid(x)
    # Cache the logits to use for crossentropy loss.
    try:
        output._keras_logits = x
    except AttributeError:
        # We're dealing with a C-type.
        pass
    return output


@keras_export("keras.activations.exponential")
def exponential(x):
    """Exponential activation function.

    Args:
        x: Input tensor.
    """
    return ops.exp(x)


@keras_export("keras.activations.hard_sigmoid")
def hard_sigmoid(x):
    """Hard sigmoid activation function.

    The hard sigmoid activation is defined as:

    - `0` if `if x <= -3`
    - `1` if `x >= 3`
    - `(x/6) + 0.5` if `-3 < x < 3`

    It's a faster, piecewise linear approximation
    of the sigmoid activation.

    Args:
        x: Input tensor.

    Reference:

    - [Wikipedia "Hard sigmoid"](https://en.wikipedia.org/wiki/Hard_sigmoid)
    """
    return ops.hard_sigmoid(x)


@keras_export(["keras.activations.hard_silu", "keras.activations.hard_swish"])
def hard_silu(x):
    """Hard SiLU activation function, also known as Hard Swish.

    It is defined as:

    - `0` if `if x < -3`
    - `x` if `x > 3`
    - `x * (x + 3) / 6` if `-3 <= x <= 3`

    It's a faster, piecewise linear approximation of the silu activation.

    Args:
        x: Input tensor.

    Reference:

    - [A Howard, 2019](https://arxiv.org/abs/1905.02244)
    """
    x = backend.convert_to_tensor(x)
    return ops.hard_silu(x)


@keras_export("keras.activations.linear")
def linear(x):
    """Linear activation function (pass-through).

    A "linear" activation is an identity function:
    it returns the input, unmodified.

    Args:
        x: Input tensor.
    """
    return x


class Mish(ops.Operation):
    def call(self, x):
        return self.static_call(x)

    def compute_output_spec(self, x):
        return backend.KerasTensor(x.shape, x.dtype)

    @staticmethod
    def static_call(x):
        return x * backend.nn.tanh(backend.nn.softplus(x))


@keras_export("keras.activations.mish")
def mish(x):
    """Mish activation function.

    It is defined as:

    `mish(x) = x * tanh(softplus(x))`

    where `softplus` is defined as:

    `softplus(x) = log(exp(x) + 1)`

    Args:
        x: Input tensor.

    Reference:

    - [Misra, 2019](https://arxiv.org/abs/1908.08681)
    """
    x = backend.convert_to_tensor(x)
    return Mish.static_call(x)


@keras_export("keras.activations.log_softmax")
def log_softmax(x, axis=-1):
    """Log-Softmax activation function.

    Each input vector is handled independently.
    The `axis` argument sets which axis of the input the function
    is applied along.

    Args:
        x: Input tensor.
        axis: Integer, axis along which the softmax is applied.
    """
    return ops.log_softmax(x, axis=axis)

"""Built-in activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import warnings
from . import backend as K
from .utils.generic_utils import deserialize_keras_object
from .engine import Layer


def softmax(x, axis=-1):
    """Softmax activation function.

    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.

    # Returns
        Tensor, output of softmax transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)


def elu(x, alpha=1.0):
    """Exponential linear unit.

    # Arguments
        x: Input tensor.
        alpha: A scalar, slope of negative section.

    # Returns
        The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.

    # References
        - [Fast and Accurate Deep Network Learning by Exponential
        Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    """
    return K.elu(x, alpha)


def selu(x):
    """Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are pre-defined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).

    # Arguments
        x: A tensor or variable to compute the activation function for.

    # Returns
       The scaled exponential unit activation: `scale * elu(x, alpha)`.

    # Note
        - To be used together with the initialization "lecun_normal".
        - To be used together with the dropout variant "AlphaDropout".

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


def softplus(x):
    """Softplus activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The softplus activation: `log(exp(x) + 1)`.
    """
    return K.softplus(x)


def softsign(x):
    """Softsign activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The softplus activation: `x / (abs(x) + 1)`.
    """
    return K.softsign(x)


def relu(x, alpha=0., max_value=None):
    """Rectified Linear Unit.

    # Arguments
        x: Input tensor.
        alpha: Slope of the negative part. Defaults to zero.
        max_value: Maximum value for the output.

    # Returns
        The (leaky) rectified linear unit activation: `x` if `x > 0`,
        `alpha * x` if `x < 0`. If `max_value` is defined, the result
        is truncated to this value.
    """
    return K.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
    """Hyperbolic tangent activation function.
    """
    return K.tanh(x)


def sigmoid(x):
    """Sigmoid activation function.
    """
    return K.sigmoid(x)


def hard_sigmoid(x):
    """Hard sigmoid activation function.

    Faster to compute than sigmoid activation.

    # Arguments
        x: Input tensor.

    # Returns
        Hard sigmoid activation:

        - `0` if `x < -2.5`
        - `1` if `x > 2.5`
        - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
    """
    return K.hard_sigmoid(x)


def linear(x):
    """Linear (i.e. identity) activation function.
    """
    return x


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='activation function')


def get(identifier):
    """Get the `identifier` activation function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The activation function, `linear` if `identifier` is None.

    # Raises
        ValueError if unknown identifier
    """
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'.format(
                    identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'activation function identifier:', identifier)

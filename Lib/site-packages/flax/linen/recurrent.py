# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recurrent neural network modules.

THe RNNCell modules can be scanned using lifted transforms. For more information
see: https://flax.readthedocs.io/en/latest/developer_notes/lift.html.
"""

from functools import partial  # pylint: disable=g-importing-member
from typing import (
  Any,
  TypeVar,
)
from collections.abc import Callable, Mapping, Sequence

import jax
import numpy as np
from absl import logging
from jax import numpy as jnp
from jax import random
from typing_extensions import Protocol

from flax.core.frozen_dict import FrozenDict
from flax.core.scope import CollectionFilter, PRNGSequenceFilter
from flax.linen import initializers, transforms
from flax.linen.activation import sigmoid, tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import Conv, Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
from flax.typing import (
  Array,
  PRNGKey,
  Dtype,
  InOutScanAxis,
  Initializer,
  PrecisionLike,
)

A = TypeVar('A')
Carry = Any
CarryHistory = Any
Output = Any


class RNNCellBase(Module):
  """RNN cell base class."""

  @nowrap
  def initialize_carry(
    self, rng: PRNGKey, input_shape: tuple[int, ...]
  ) -> Carry:
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    raise NotImplementedError

  @property
  def num_feature_axes(self) -> int:
    """Returns the number of feature axes of the RNN cell."""
    raise NotImplementedError


class LSTMCell(RNNCellBase):
  r"""LSTM cell.

  The mathematical definition of the cell is as follows

  .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

  where x is the input, h is the output of the previous time step, and c is
  the memory.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (2, 3))
    >>> layer = nn.LSTMCell(features=4)
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

  Attributes:
    features: number of output features.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """

  features: int
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Initializer = default_kernel_init
  recurrent_kernel_init: Initializer = initializers.orthogonal()
  bias_init: Initializer = initializers.zeros_init()
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  carry_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(self, carry, inputs):
    r"""A long short-term memory (LSTM) cell.

    Args:
      carry: the hidden state of the LSTM cell,
        initialized using ``LSTMCell.initialize_carry``.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(
      Dense,
      features=hidden_features,
      use_bias=True,
      kernel_init=self.recurrent_kernel_init,
      bias_init=self.bias_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
    )
    dense_i = partial(
      Dense,
      features=hidden_features,
      use_bias=False,
      kernel_init=self.kernel_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
    )
    i = self.gate_fn(dense_i(name='ii')(inputs) + dense_h(name='hi')(h))
    f = self.gate_fn(dense_i(name='if')(inputs) + dense_h(name='hf')(h))
    g = self.activation_fn(dense_i(name='ig')(inputs) + dense_h(name='hg')(h))
    o = self.gate_fn(dense_i(name='io')(inputs) + dense_h(name='ho')(h))
    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @nowrap
  def initialize_carry(
    self, rng: PRNGKey, input_shape: tuple[int, ...]
  ) -> tuple[Array, Array]:
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.
    Returns:
      An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (self.features,)
    c = self.carry_init(key1, mem_shape, self.param_dtype)
    h = self.carry_init(key2, mem_shape, self.param_dtype)
    return (c, h)

  @property
  def num_feature_axes(self) -> int:
    return 1


class DenseParams(Module):
  """Dummy module for creating parameters matching ``flax.linen.Dense``."""

  features: int
  use_bias: bool = True
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(self, inputs: Array) -> tuple[Array, Array | None]:
    k = self.param(
      'kernel',
      self.kernel_init,
      (inputs.shape[-1], self.features),
      self.param_dtype,
    )
    if self.use_bias:
      b = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
    else:
      b = None
    return k, b


class OptimizedLSTMCell(RNNCellBase):
  r"""More efficient LSTM Cell that concatenates state components before matmul.

  The parameters are compatible with ``LSTMCell``. Note that this cell is often
  faster than ``LSTMCell`` as long as the hidden size is roughly <= 2048 units.

  The mathematical definition of the cell is the same as ``LSTMCell`` and as
  follows

  .. math::

      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

  where x is the input, h is the output of the previous time step, and c is
  the memory.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (2, 3))
    >>> layer = nn.OptimizedLSTMCell(features=4)
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init()).
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """

  features: int
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Initializer = default_kernel_init
  recurrent_kernel_init: Initializer = initializers.orthogonal()
  bias_init: Initializer = initializers.zeros_init()
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  carry_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(
    self, carry: tuple[Array, Array], inputs: Array
  ) -> tuple[tuple[Array, Array], Array]:
    r"""An optimized long short-term memory (LSTM) cell.

    Args:
      carry: the hidden state of the LSTM cell, initialized using
        ``LSTMCell.initialize_carry``.
      inputs: an ndarray with the input for the current time step. All
        dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]

    def _concat_dense(
      inputs: Array,
      params: Mapping[str, tuple[Array, Array | None]],
      use_bias: bool = True,
    ) -> dict[str, Array]:
      # Concatenates the individual kernels and biases, given in params, into a
      # single kernel and single bias for efficiency before applying them using
      # dot_general.
      kernels = [kernel for kernel, _ in params.values()]
      kernel = jnp.concatenate(kernels, axis=-1)
      if use_bias:
        biases = []
        for _, bias in params.values():
          if bias is None:
            raise ValueError('bias is None but use_bias is True.')
          biases.append(bias)
        bias = jnp.concatenate(biases, axis=-1)
      else:
        bias = None
      inputs, kernel, bias = promote_dtype(
        inputs, kernel, bias, dtype=self.dtype
      )
      y = jnp.dot(inputs, kernel)
      if use_bias:
        # This assert is here since mypy can't infer that bias cannot be None
        assert bias is not None
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

      # Split the result back into individual (i, f, g, o) outputs.
      split_indices = np.cumsum([kernel.shape[-1] for kernel in kernels[:-1]])
      ys = jnp.split(y, split_indices, axis=-1)
      return dict(zip(params.keys(), ys))

    # Create params with the same names/shapes as `LSTMCell` for compatibility.
    dense_params_h = {}
    dense_params_i = {}
    for component in ['i', 'f', 'g', 'o']:
      dense_params_i[component] = DenseParams(
        features=hidden_features,
        use_bias=False,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name=f'i{component}',  # type: ignore[call-arg]
      )(inputs)
      dense_params_h[component] = DenseParams(
        features=hidden_features,
        use_bias=True,
        param_dtype=self.param_dtype,
        kernel_init=self.recurrent_kernel_init,
        bias_init=self.bias_init,
        name=f'h{component}',  # type: ignore[call-arg]
      )(h)
    dense_h = _concat_dense(h, dense_params_h, use_bias=True)
    dense_i = _concat_dense(inputs, dense_params_i, use_bias=False)

    i = self.gate_fn(dense_h['i'] + dense_i['i'])
    f = self.gate_fn(dense_h['f'] + dense_i['f'])
    g = self.activation_fn(dense_h['g'] + dense_i['g'])
    o = self.gate_fn(dense_h['o'] + dense_i['o'])

    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @nowrap
  def initialize_carry(
    self, rng: PRNGKey, input_shape: tuple[int, ...]
  ) -> tuple[Array, Array]:
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (self.features,)
    c = self.carry_init(key1, mem_shape, self.param_dtype)
    h = self.carry_init(key2, mem_shape, self.param_dtype)
    return c, h

  @property
  def num_feature_axes(self) -> int:
    return 1


class SimpleCell(RNNCellBase):
  r"""Simple cell.

  The mathematical definition of the cell is as follows

  .. math::

      \begin{array}{ll}
      h' = \tanh(W_i x + b_i + W_h h)
      \end{array}

  where x is the input and h is the output of the previous time step.

  If `residual` is `True`,

  .. math::

      \begin{array}{ll}
      h' = \tanh(W_i x + b_i + W_h h + h)
      \end{array}

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (2, 3))
    >>> layer = nn.SimpleCell(features=4)
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

  Attributes:
    features: number of output features.
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    residual: pre-activation residual connection (https://arxiv.org/abs/1801.06105).
  """

  features: int
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Initializer = default_kernel_init
  recurrent_kernel_init: Initializer = initializers.orthogonal()
  bias_init: Initializer = initializers.zeros_init()
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  carry_init: Initializer = initializers.zeros_init()
  residual: bool = False

  @compact
  def __call__(self, carry, inputs):
    """Simple cell.

    Args:
      carry: the hidden state of the Simple cell,
        initialized using ``SimpleCell.initialize_carry``.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    hidden_features = carry.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(
      Dense,
      features=hidden_features,
      use_bias=False,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.recurrent_kernel_init,
    )
    dense_i = partial(
      Dense,
      features=hidden_features,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
    )
    new_carry = dense_i(name='i')(inputs) + dense_h(name='h')(carry)
    if self.residual:
      new_carry += carry
    new_carry = self.activation_fn(new_carry)
    return new_carry, new_carry

  @nowrap
  def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    mem_shape = batch_dims + (self.features,)
    return self.carry_init(rng, mem_shape, self.param_dtype)

  @property
  def num_feature_axes(self) -> int:
    return 1


class GRUCell(RNNCellBase):
  r"""GRU cell.

  The mathematical definition of the cell is as follows

  .. math::

      \begin{array}{ll}
      r = \sigma(W_{ir} x + b_{ir} + W_{hr} h) \\
      z = \sigma(W_{iz} x + b_{iz} + W_{hz} h) \\
      n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
      h' = (1 - z) * n + z * h \\
      \end{array}

  where x is the input and h is the output of the previous time step.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (2, 3))
    >>> layer = nn.GRUCell(features=4)
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

  Attributes:
    features: number of output features.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """

  features: int
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Initializer = default_kernel_init
  recurrent_kernel_init: Initializer = initializers.orthogonal()
  bias_init: Initializer = initializers.zeros_init()
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  carry_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(self, carry, inputs):
    """Gated recurrent unit (GRU) cell.

    Args:
      carry: the hidden state of the GRU cell,
        initialized using ``GRUCell.initialize_carry``.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(
      Dense,
      features=hidden_features,
      use_bias=False,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.recurrent_kernel_init,
      bias_init=self.bias_init,
    )
    dense_i = partial(
      Dense,
      features=hidden_features,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
    )
    r = self.gate_fn(dense_i(name='ir')(inputs) + dense_h(name='hr')(h))
    z = self.gate_fn(dense_i(name='iz')(inputs) + dense_h(name='hz')(h))
    # add bias because the linear transformations aren't directly summed.
    n = self.activation_fn(
      dense_i(name='in')(inputs) + r * dense_h(name='hn', use_bias=True)(h)
    )
    new_h = (1.0 - z) * n + z * h
    return new_h, new_h

  @nowrap
  def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    mem_shape = batch_dims + (self.features,)
    return self.carry_init(rng, mem_shape, self.param_dtype)

  @property
  def num_feature_axes(self) -> int:
    return 1


class MGUCell(RNNCellBase):
  r"""MGU cell (https://arxiv.org/pdf/1603.09420.pdf).

  The mathematical definition of the cell is as follows

  .. math::

      \begin{array}{ll}
      f = \sigma(W_{if} x + b_{if} + W_{hf} h) \\
      n = \tanh(W_{in} x + b_{in} + f * (W_{hn} h + b_{hn})) \\
      h' = (1 - f) * n + f * h \\
      \end{array}

  where x is the input and h is the output of the previous time step.

  If ``reset_gate`` is false, the above becomes

  .. math::

      \begin{array}{ll}
      f = \sigma(W_{if} x + b_{if} + W_{hf} h) \\
      n = \tanh(W_{in} x + b_{in} + W_{hn} h) \\
      h' = (1 - f) * n + f * h \\
      \end{array}

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (2, 3))
    >>> layer = nn.MGUCell(features=4)
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

  Attributes:
    features: number of output features.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    forget_bias_init: initializer for the bias parameters of the forget gate.
      The default is set to initializers.ones_init() because this prevents
      vanishing gradients. See https://proceedings.mlr.press/v37/jozefowicz15.pdf,
      section 2.2 for more details.
    activation_bias_init: initializer for the bias parameters of the activation
      output (default: initializers.zeros_init()).
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    reset_gate: flag for applying reset gating.
  """

  features: int
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Initializer = default_kernel_init
  recurrent_kernel_init: Initializer = initializers.orthogonal()
  forget_bias_init: Initializer = initializers.ones_init()
  activation_bias_init: Initializer = initializers.zeros_init()
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  carry_init: Initializer = initializers.zeros_init()
  reset_gate: bool = True

  @compact
  def __call__(self, carry, inputs):
    """Minimal gated unit (MGU) cell.

    Args:
      carry: the hidden state of the MGU cell,
        initialized using ``MGUCell.initialize_carry``.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(
      Dense,
      features=hidden_features,
      use_bias=False,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.recurrent_kernel_init,
      bias_init=self.activation_bias_init,
    )
    dense_i = partial(
      Dense,
      features=hidden_features,
      use_bias=True,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
    )
    f = self.gate_fn(
      dense_i(name='if', bias_init=self.forget_bias_init)(inputs)
      + dense_h(name='hf')(h)
    )
    # add bias when the linear transformations aren't directly summed.
    x = dense_h(name="hn", use_bias=self.reset_gate)(h)
    if self.reset_gate:
      x *= f
    n = self.activation_fn(
      dense_i(name="in", bias_init=self.activation_bias_init)(inputs) + x
    )
    new_h = (1.0 - f) * n + f * h
    return new_h, new_h

  @nowrap
  def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    mem_shape = batch_dims + (self.features,)
    return self.carry_init(rng, mem_shape, self.param_dtype)

  @property
  def num_feature_axes(self) -> int:
    return 1


class ConvLSTMCell(RNNCellBase):
  r"""A convolutional LSTM cell.

  The implementation is based on xingjian2015convolutional.
  Given x_t and the previous state (h_{t-1}, c_{t-1})
  the core computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} * x_t + W_{hf} * h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} * x_t + W_{ho} * h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where * denotes the convolution operator;
  i_t, f_t, o_t are input, forget and output gate activations,
  and g_t is a vector of cell updates.

  .. note::
    Forget gate initialization:
      Following jozefowicz2015empirical we add 1.0 to b_f
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (3, 5, 5))
    >>> layer = nn.ConvLSTMCell(features=4, kernel_size=(2, 2))
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of ``n`` integers, representing the inter-window
      strides.
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
      and after each spatial dimension.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """

  features: int
  kernel_size: Sequence[int]
  strides: Sequence[int] | None = None
  padding: str | Sequence[tuple[int, int]] = 'SAME'
  use_bias: bool = True
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  carry_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(self, carry, inputs):
    """Constructs a convolutional LSTM.

    Args:
      carry: the hidden state of the Conv2DLSTM cell,
        initialized using ``Conv2DLSTM.initialize_carry``.
      inputs: input data with dimensions (batch, spatial_dims..., features).
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    input_to_hidden = partial(
      Conv,
      features=4 * self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      name='ih',
    )

    hidden_to_hidden = partial(
      Conv,
      features=4 * self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      name='hh',
    )

    gates = input_to_hidden()(inputs) + hidden_to_hidden()(h)
    i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

    f = sigmoid(f + 1)
    new_c = f * c + sigmoid(i) * jnp.tanh(g)
    new_h = sigmoid(o) * jnp.tanh(new_c)
    return (new_c, new_h), new_h

  @nowrap
  def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    # (*batch_dims, *signal_dims, features)
    signal_dims = input_shape[-self.num_feature_axes : -1]
    batch_dims = input_shape[: -self.num_feature_axes]
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + signal_dims + (self.features,)
    c = self.carry_init(key1, mem_shape, self.param_dtype)
    h = self.carry_init(key2, mem_shape, self.param_dtype)
    return c, h

  @property
  def num_feature_axes(self) -> int:
    return len(self.kernel_size) + 1


class RNN(Module):
  """The ``RNN`` module takes any :class:`RNNCellBase` instance and applies it over a sequence

  using :func:`flax.linen.scan`.

  Example::

    >>> import jax.numpy as jnp
    >>> import jax
    >>> import flax.linen as nn

    >>> x = jnp.ones((10, 50, 32)) # (batch, time, features)
    >>> lstm = nn.RNN(nn.LSTMCell(64))
    >>> variables = lstm.init(jax.random.key(0), x)
    >>> y = lstm.apply(variables, x)
    >>> y.shape # (batch, time, cell_size)
    (10, 50, 64)

  As shown above, RNN uses the ``cell_size`` argument to set the ``size``
  argument for the cell's
  ``initialize_carry`` method, in practice this is typically the number of
  hidden units you want
  for the cell. However, this may vary depending on the cell you are using, for
  example the
  :class:`ConvLSTMCell` requires a ``size`` argument of the form
  ``(kernel_height, kernel_width, features)``::

    >>> x = jnp.ones((10, 50, 32, 32, 3)) # (batch, time, height, width, features)
    >>> conv_lstm = nn.RNN(nn.ConvLSTMCell(64, kernel_size=(3, 3)))
    >>> y, variables = conv_lstm.init_with_output(jax.random.key(0), x)
    >>> y.shape # (batch, time, height, width, features)
    (10, 50, 32, 32, 64)

  By default RNN expect the time dimension after the batch dimension (``(*batch,
  time, *features)``),
  if you set ``time_major=True`` RNN will instead expect the time dimesion to be
  at the beginning
  (``(time, *batch, *features)``)::

    >>> x = jnp.ones((50, 10, 32)) # (time, batch, features)
    >>> lstm = nn.RNN(nn.LSTMCell(64), time_major=True)
    >>> variables = lstm.init(jax.random.key(0), x)
    >>> y = lstm.apply(variables, x)
    >>> y.shape # (time, batch, cell_size)
    (50, 10, 64)

  The output is an array of shape ``(*batch, time, *cell_size)`` by default
  (typically), however
  if you set ``return_carry=True`` it will instead return a tuple of the final
  carry and the output::

    >>> x = jnp.ones((10, 50, 32)) # (batch, time, features)
    >>> lstm = nn.RNN(nn.LSTMCell(64), return_carry=True)
    >>> variables = lstm.init(jax.random.key(0), x)
    >>> carry, y = lstm.apply(variables, x)
    >>> jax.tree_util.tree_map(jnp.shape, carry) # ((batch, cell_size), (batch, cell_size))
    ((10, 64), (10, 64))
    >>> y.shape # (batch, time, cell_size)
    (10, 50, 64)

  To support variable length sequences, you can pass a ``seq_lengths`` which is
  an integer
  array of shape ``(*batch)`` where each element is the length of the sequence
  in the batch.
  For example::

    >>> seq_lengths = jnp.array([3, 2, 5])

  The output elements corresponding to padding elements are NOT zeroed out. If
  ``return_carry``
  is set to ``True`` the carry will be the state of the last valid element of
  each sequence.

  RNN also accepts some of the arguments of :func:`flax.linen.scan`, by default
  they are set to
  work with cells like :class:`LSTMCell` and :class:`GRUCell` but they can be
  overriden as needed.
  Overriding default values to scan looks like this::

    >>> lstm = nn.RNN(
    ...   nn.LSTMCell(64),
    ...   unroll=1, variable_axes={}, variable_broadcast='params',
    ...   variable_carry=False, split_rngs={'params': False})

  Attributes:
    cell: an instance of :class:`RNNCellBase`.
    time_major: if ``time_major=False`` (default) it will expect inputs with
      shape ``(*batch, time, *features)``, else it will expect inputs with shape
      ``(time, *batch, *features)``.
    return_carry: if ``return_carry=False`` (default) only the output sequence
      is returned, else it will return a tuple of the final carry and the output
      sequence.
    reverse: if ``reverse=False`` (default) the sequence is processed from left
      to right and returned in the original order, else it will be processed
      from right to left, and returned in reverse order. If ``seq_lengths`` is
      passed, padding will always remain at the end of the sequence.
    keep_order: if ``keep_order=True``, when ``reverse=True`` the output will be
      reversed back to the original order after processing, this is useful to
      align sequences in bidirectional RNNs. If ``keep_order=False`` (default),
      the output will remain in the order specified by ``reverse``.
    unroll: how many scan iterations to unroll within a single iteration of a
      loop, defaults to 1. This argument will be passed to ``nn.scan``.
    variable_axes: a dictionary mapping each collection to either an integer
      ``i`` (meaning we scan over dimension ``i``) or ``None`` (replicate rather
      than scan). This argument is forwarded to ``nn.scan``.
    variable_broadcast: Specifies the broadcasted variable collections. A
      broadcasted variable should not depend on any computation that cannot be
      lifted out of the loop. This is typically used to define shared parameters
      inside the fn. This argument is forwarded to ``nn.scan``.
    variable_carry: Specifies the variable collections that are carried through
      the loop. Mutations to these variables are carried to the next iteration
      and will be preserved when the scan finishes. This argument is forwarded
      to ``nn.scan``.
    split_rngs: a mapping from PRNGSequenceFilter to bool specifying whether a
      collection's PRNG key should be split such that its values are different
      at each step, or replicated such that its values remain the same at each
      step. This argument is forwarded to ``nn.scan``.
  """

  cell: RNNCellBase
  time_major: bool = False
  return_carry: bool = False
  reverse: bool = False
  keep_order: bool = False
  unroll: int = 1
  variable_axes: Mapping[
    CollectionFilter, InOutScanAxis
  ] = FrozenDict()
  variable_broadcast: CollectionFilter = 'params'
  variable_carry: CollectionFilter = False
  split_rngs: Mapping[PRNGSequenceFilter, bool] = FrozenDict(
    {'params': False}
  )

  def __call__(
    self,
    inputs: jax.Array,
    *,
    initial_carry: Carry | None = None,
    init_key: PRNGKey | None = None,
    seq_lengths: Array | None = None,
    return_carry: bool | None = None,
    time_major: bool | None = None,
    reverse: bool | None = None,
    keep_order: bool | None = None,
  ) -> Output | tuple[Carry, Output]:
    """
    Applies the RNN to the inputs.

    ``__call__`` allows you to optionally override some attributes like ``return_carry``
    and ``time_major`` defined in the constructor.

    Arguments:
      inputs: the input sequence.
      initial_carry: the initial carry, if not provided it will be initialized
        using the cell's :meth:`RNNCellBase.initialize_carry` method.
      init_key: a PRNG key used to initialize the carry, if not provided
        ``jax.random.key(0)`` will be used. Most cells will ignore this
        argument.
      seq_lengths: an optional integer array of shape ``(*batch)`` indicating
        the length of each sequence, elements whose index in the time dimension
        is greater than the corresponding length will be considered padding and
        will be ignored.
      return_carry: if ``return_carry=False`` (default) only the output sequence is returned,
        else it will return a tuple of the final carry and the output sequence.
      time_major: if ``time_major=False`` (default) it will expect inputs with shape
        ``(*batch, time, *features)``, else it will expect inputs with shape ``(time, *batch, *features)``.
      reverse: overrides the ``reverse`` attribute, if ``reverse=False`` (default) the sequence is
        processed from left to right and returned in the original order, else it will be processed
        from right to left, and returned in reverse order. If ``seq_lengths`` is passed,
        padding will always remain at the end of the sequence.
      keep_order: overrides the ``keep_order`` attribute, if ``keep_order=True``, when ``reverse=True``
        the output will be reversed back to the original order after processing, this is
        useful to align sequences in bidirectional RNNs. If ``keep_order=False`` (default),
        the output will remain in the order specified by ``reverse``.
    Returns:
      if ``return_carry=False`` (default) only the output sequence is returned,
      else it will return a tuple of the final carry and the output sequence.
    """

    if return_carry is None:
      return_carry = self.return_carry
    if time_major is None:
      time_major = self.time_major
    if reverse is None:
      reverse = self.reverse
    if keep_order is None:
      keep_order = self.keep_order

    # Infer the number of batch dimensions from the input shape.
    # Cells like ConvLSTM have additional spatial dimensions.
    time_axis = (
      0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)
    )

    # make time_axis positive
    if time_axis < 0:
      time_axis += inputs.ndim

    if time_major:
      # we add +1 because we moved the time axis to the front
      batch_dims = inputs.shape[1 : -self.cell.num_feature_axes]
    else:
      batch_dims = inputs.shape[:time_axis]

    # maybe reverse the sequence
    if reverse:
      inputs = jax.tree_util.tree_map(
          lambda x: flip_sequences(
              x,
              seq_lengths,
              num_batch_dims=len(batch_dims),
              time_major=time_major,  # type: ignore
          ),
          inputs,
      )

    carry: Carry
    if initial_carry is None:
      if init_key is None:
        init_key = random.key(0)

      input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
      carry = self.cell.initialize_carry(init_key, input_shape)
    else:
      carry = initial_carry

    slice_carry = seq_lengths is not None and return_carry

    def scan_fn(
      cell: RNNCellBase, carry: Carry, x: Array
    ) -> tuple[Carry, Array] | tuple[Carry, tuple[Carry, Array]]:
      carry, y = cell(carry, x)
      # When we have a segmentation mask we return the carry as an output
      # so that we can select the last carry for each sequence later.
      # This uses more memory but is faster than using jnp.where at each
      # iteration. As a small optimization do this when we really need it.
      if slice_carry:
        return carry, (carry, y)
      else:
        return carry, y

    scan = transforms.scan(
      scan_fn,
      in_axes=time_axis,
      out_axes=(0, time_axis) if slice_carry else time_axis,
      unroll=self.unroll,
      variable_axes=self.variable_axes,
      variable_broadcast=self.variable_broadcast,
      variable_carry=self.variable_carry,
      split_rngs=self.split_rngs,
    )

    scan_output = scan(self.cell, carry, inputs)

    # Next we select the final carry. If a segmentation mask was provided and
    # return_carry is True we slice the carry history and select the last valid
    # carry for each sequence. Otherwise we just use the last carry.
    if slice_carry:
      assert seq_lengths is not None
      _, (carries, outputs) = scan_output
      # seq_lengths[None] expands the shape of the mask to match the
      # number of dimensions of the carry.
      carry = _select_last_carry(carries, seq_lengths)
    else:
      carry, outputs = scan_output

    if reverse and keep_order:
      outputs = jax.tree_util.tree_map(
          lambda x: flip_sequences(
              x,
              seq_lengths,
              num_batch_dims=len(batch_dims),
              time_major=time_major,  # type: ignore
          ),
          outputs,
      )

    if return_carry:
      return carry, outputs
    else:
      return outputs


def _select_last_carry(sequence: A, seq_lengths: jnp.ndarray) -> A:
  last_idx = seq_lengths - 1

  def _slice_array(x: jnp.ndarray):
    return x[last_idx, jnp.arange(x.shape[1])]

  return jax.tree_util.tree_map(_slice_array, sequence)


def _expand_dims_like(x, target):
  """Expands the shape of `x` to match `target`'s shape by adding singleton dimensions."""
  return x.reshape(list(x.shape) + [1] * (target.ndim - x.ndim))


def flip_sequences(
  inputs: Array,
  seq_lengths: Array | None,
  num_batch_dims: int,
  time_major: bool,
) -> Array:
  """Flips a sequence of inputs along the time axis.

  This function can be used to prepare inputs for the reverse direction of a
  bidirectional LSTM. It solves the issue that, when naively flipping multiple
  padded sequences stored in a matrix, the first elements would be padding
  values for those sequences that were padded. This function keeps the padding
  at the end, while flipping the rest of the elements.

  Example:
  ```python
  inputs = [[1, 0, 0],
            [2, 3, 0]
            [4, 5, 6]]
  lengths = [1, 2, 3]
  flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
  ```

  Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].

  Returns:
    An ndarray with the flipped inputs.
  """
  # Compute the indices to put the inputs in flipped order as per above example.
  time_axis = 0 if time_major else num_batch_dims
  max_steps = inputs.shape[time_axis]

  if seq_lengths is None:
    # reverse inputs and return
    inputs = jnp.flip(inputs, axis=time_axis)
    return inputs

  seq_lengths = jnp.expand_dims(seq_lengths, axis=time_axis)

  # create indexes
  idxs = jnp.arange(max_steps - 1, -1, -1)  # [max_steps]
  if time_major:
    idxs = jnp.reshape(idxs, [max_steps] + [1] * num_batch_dims)
  else:
    idxs = jnp.reshape(
      idxs, [1] * num_batch_dims + [max_steps]
    )  # [1, ..., max_steps]
  idxs = (idxs + seq_lengths) % max_steps  # [*batch, max_steps]
  idxs = _expand_dims_like(
    idxs, target=inputs
  )  # [*batch, max_steps, *features]
  # Select the inputs in flipped order.
  outputs = jnp.take_along_axis(inputs, idxs, axis=time_axis)

  return outputs


def _concatenate(a: Array, b: Array) -> Array:
  """Concatenates two arrays along the last dimension."""
  return jnp.concatenate([a, b], axis=-1)


class RNNBase(Protocol):
  def __call__(
    self,
    inputs: jax.Array,
    *,
    initial_carry: Carry | None = None,
    init_key: PRNGKey | None = None,
    seq_lengths: Array | None = None,
    return_carry: bool | None = None,
    time_major: bool | None = None,
    reverse: bool | None = None,
    keep_order: bool | None = None,
  ) -> Output | tuple[Carry, Output]:
    ...


class Bidirectional(Module):
  """Processes the input in both directions and merges the results.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Bidirectional(nn.RNN(nn.GRUCell(4)), nn.RNN(nn.GRUCell(4)))
    >>> x = jnp.ones((2, 3))
    >>> variables = layer.init(jax.random.key(0), x)
    >>> out = layer.apply(variables, x)
  """

  forward_rnn: RNNBase
  backward_rnn: RNNBase
  merge_fn: Callable[[Array, Array], Array] = _concatenate
  time_major: bool = False
  return_carry: bool = False

  def __call__(
    self,
    inputs: jax.Array,
    *,
    initial_carry: Carry | None = None,
    init_key: PRNGKey | None = None,
    seq_lengths: Array | None = None,
    return_carry: bool | None = None,
    time_major: bool | None = None,
    reverse: bool | None = None,
    keep_order: bool | None = None,
  ) -> Output | tuple[Carry, Output]:
    if time_major is None:
      time_major = self.time_major
    if return_carry is None:
      return_carry = self.return_carry
    if init_key is not None:
      key_forward, key_backward = random.split(init_key)
    else:
      key_forward = key_backward = None
    if initial_carry is not None:
      initial_carry_forward, initial_carry_backward = initial_carry
    else:
      initial_carry_forward = initial_carry_backward = None
    # Throw a warning in case the user accidentally re-uses the forward RNN
    # for the backward pass and does not intend for them to share parameters.
    if self.forward_rnn is self.backward_rnn:
      logging.warning(
          'forward_rnn and backward_rnn is the same object, so '
          'they will share parameters.'
      )

    # Encode in the forward direction.
    carry_forward, outputs_forward = self.forward_rnn(
      inputs,
      initial_carry=initial_carry_forward,
      init_key=key_forward,
      seq_lengths=seq_lengths,
      return_carry=True,
      time_major=time_major,
      reverse=False,
    )

    carry_backward, outputs_backward = self.backward_rnn(
      inputs,
      initial_carry=initial_carry_backward,
      init_key=key_backward,
      seq_lengths=seq_lengths,
      return_carry=True,
      time_major=time_major,
      reverse=True,
      keep_order=True,
    )

    carry = (carry_forward, carry_backward)
    outputs = jax.tree_util.tree_map(
        self.merge_fn, outputs_forward, outputs_backward
    )

    if return_carry:
      return carry, outputs
    else:
      return outputs

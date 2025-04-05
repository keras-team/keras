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

"""RNN modules for Flax."""

from typing import Any, TypeVar
from collections.abc import Mapping
from collections.abc import Callable
from functools import partial
from typing_extensions import Protocol
from absl import logging

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import filterlib, rnglib
from flax.nnx.module import Module
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import Linear
from flax.nnx.nn.activations import sigmoid
from flax.nnx.nn.activations import tanh
from flax.nnx.transforms import iteration
from flax.typing import (
    Dtype,
    Initializer,
    Shape
)

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any

class RNNCellBase(Module):
    """RNN cell base class."""

    def initialize_carry(
        self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None
    ) -> Carry:
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        raise NotImplementedError

    def __call__(
        self,
        carry: Carry,
        inputs: Array
    ) -> tuple[Carry, Array]:
        """Run the RNN cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        raise NotImplementedError

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        raise NotImplementedError

def modified_orthogonal(key: Array, shape: Shape, dtype: Dtype = jnp.float32) -> Array:
    """Modified orthogonal initializer for compatibility with half precision."""
    initializer = initializers.orthogonal()
    return initializer(key, shape).astype(dtype)

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
  """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: Callable[..., Any] = sigmoid,
        activation_fn: Callable[..., Any] = tanh,
        kernel_init: Initializer = default_kernel_init,
        recurrent_kernel_init: Initializer = modified_orthogonal,
        bias_init: Initializer = initializers.zeros_init(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.rngs = rngs

        # input and recurrent layers are summed so only one needs a bias.
        dense_i = partial(
            Linear,
            in_features=in_features,
            out_features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        dense_h = partial(
            Linear,
            in_features=hidden_features,
            out_features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.ii = dense_i()
        self.if_ = dense_i()
        self.ig = dense_i()
        self.io = dense_i()
        self.hi = dense_h()
        self.hf = dense_h()
        self.hg = dense_h()
        self.ho = dense_h()

    def __call__(self, carry: tuple[Array, Array], inputs: Array) -> tuple[tuple[Array, Array], Array]: # type: ignore[override]
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
        i = self.gate_fn(self.ii(inputs) + self.hi(h))
        f = self.gate_fn(self.if_(inputs) + self.hf(h))
        g = self.activation_fn(self.ig(inputs) + self.hg(h))
        o = self.gate_fn(self.io(inputs) + self.ho(h))
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> tuple[Array, Array]: # type: ignore[override]
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.
        Returns:
          An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        if rngs is None:
            rngs = self.rngs
        mem_shape = batch_dims + (self.hidden_features,)
        c = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class OptimizedLSTMCell(RNNCellBase):
  r"""More efficient LSTM Cell that concatenates state components before matmul.

    The parameters are compatible with ``LSTMCell``. Note that this cell is often
    faster than ``LSTMCell`` as long as the hidden size is roughly <= 2048 units.

    The mathematical definition of the cell is the same as ``LSTMCell`` and as
    follows:

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

    Args:
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

  def __init__(
    self,
    in_features: int,
    hidden_features: int,
    *,
    gate_fn: Callable[..., Any] = sigmoid,
    activation_fn: Callable[..., Any] = tanh,
    kernel_init: Initializer = default_kernel_init,
    recurrent_kernel_init: Initializer = initializers.orthogonal(),
    bias_init: Initializer = initializers.zeros_init(),
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    carry_init: Initializer = initializers.zeros_init(),
    rngs: rnglib.Rngs,
  ):
    self.in_features = in_features
    self.hidden_features = hidden_features
    self.gate_fn = gate_fn
    self.activation_fn = activation_fn
    self.kernel_init = kernel_init
    self.recurrent_kernel_init = recurrent_kernel_init
    self.bias_init = bias_init
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.carry_init = carry_init
    self.rngs = rngs

    # input and recurrent layers are summed so only one needs a bias.
    self.dense_i = Linear(
      in_features=in_features,
      out_features=4 * hidden_features,
      use_bias=False,
      kernel_init=self.kernel_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      rngs=rngs,
    )

    self.dense_h = Linear(
      in_features=hidden_features,
      out_features=4 * hidden_features,
      use_bias=True,
      kernel_init=self.recurrent_kernel_init,
      bias_init=self.bias_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      rngs=rngs,
    )

  def __call__(
    self, carry: tuple[Array, Array], inputs: Array
  ) -> tuple[tuple[Array, Array], Array]:  # type: ignore[override]
    r"""An optimized long short-term memory (LSTM) cell.

    Args:
      carry: the hidden state of the LSTM cell, initialized using
        ``LSTMCell.initialize_carry``.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry

    # Compute combined transformations for inputs and hidden state
    y = self.dense_i(inputs) + self.dense_h(h)

    # Split the combined transformations into individual gates
    i, f, g, o = jnp.split(y, indices_or_sections=4, axis=-1)

    # Apply gate activations
    i = self.gate_fn(i)
    f = self.gate_fn(f)
    g = self.activation_fn(g)
    o = self.gate_fn(o)

    # Update cell state and hidden state
    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  def initialize_carry(
    self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None
  ) -> tuple[Array, Array]:  # type: ignore[override]
    """Initialize the RNN cell carry.

    Args:
      rngs: random number generator passed to the init_fn.
      input_shape: a tuple providing the shape of the input to the cell.

    Returns:
      An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    if rngs is None:
      rngs = self.rngs
    mem_shape = batch_dims + (self.hidden_features,)
    c = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
    h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
    return (c, h)

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
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,  # not inferred from carry for now
        *,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        residual: bool = False,
        activation_fn: Callable[..., Any] = tanh,
        kernel_init: Initializer = initializers.lecun_normal(),
        recurrent_kernel_init: Initializer = initializers.orthogonal(),
        bias_init: Initializer = initializers.zeros_init(),
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.residual = residual
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        # self.hidden_features = carry.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        self.dense_h = Linear(
            in_features=self.hidden_features,
            out_features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            rngs=rngs,
        )
        self.dense_i = Linear(
            in_features=self.in_features,
            out_features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]: # type: ignore[override]
        new_carry = self.dense_i(inputs) + self.dense_h(carry)
        if self.residual:
            new_carry += carry
        new_carry = self.activation_fn(new_carry)
        return new_carry, new_carry

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> Array: # type: ignore[override]
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        if rngs is None:
            rngs = self.rngs
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)
        return self.carry_init(rngs.carry(), mem_shape, self.param_dtype)

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

    Args:
        in_features: number of input features.
        hidden_features: number of output features.
        gate_fn: activation function used for gates (default: sigmoid).
        activation_fn: activation function used for output and memory update
          (default: tanh).
        kernel_init: initializer function for the kernels that transform
          the input (default: lecun_normal).
        recurrent_kernel_init: initializer function for the kernels that transform
          the hidden state (default: initializers.orthogonal()).
        bias_init: initializer for the bias parameters (default: initializers.zeros_init()).
        dtype: the dtype of the computation (default: None).
        param_dtype: the dtype passed to parameter initializers (default: float32).
    """

  def __init__(
    self,
    in_features: int,
    hidden_features: int,
    *,
    gate_fn: Callable[..., Any] = sigmoid,
    activation_fn: Callable[..., Any] = tanh,
    kernel_init: Initializer = default_kernel_init,
    recurrent_kernel_init: Initializer = initializers.orthogonal(),
    bias_init: Initializer = initializers.zeros_init(),
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    carry_init: Initializer = initializers.zeros_init(),
    rngs: rnglib.Rngs,
  ):
    self.in_features = in_features
    self.hidden_features = hidden_features
    self.gate_fn = gate_fn
    self.activation_fn = activation_fn
    self.kernel_init = kernel_init
    self.recurrent_kernel_init = recurrent_kernel_init
    self.bias_init = bias_init
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.carry_init = carry_init
    self.rngs = rngs

    # Combine input transformations into a single linear layer
    self.dense_i = Linear(
      in_features=in_features,
      out_features=3 * hidden_features,  # r, z, n
      use_bias=True,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      rngs=rngs,
    )

    self.dense_h = Linear(
      in_features=hidden_features,
      out_features=3 * hidden_features,  # r, z, n
      use_bias=False,
      kernel_init=self.recurrent_kernel_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      rngs=rngs,
    )

  def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:  # type: ignore[override]
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

    # Compute combined transformations for inputs and hidden state
    x_transformed = self.dense_i(inputs)
    h_transformed = self.dense_h(h)

    # Split the combined transformations into individual components
    xi_r, xi_z, xi_n = jnp.split(x_transformed, 3, axis=-1)
    hh_r, hh_z, hh_n = jnp.split(h_transformed, 3, axis=-1)

    # Compute gates
    r = self.gate_fn(xi_r + hh_r)
    z = self.gate_fn(xi_z + hh_z)

    # Compute n with an additional linear transformation on h
    n = self.activation_fn(xi_n + r * hh_n)

    # Update hidden state
    new_h = (1.0 - z) * n + z * h
    return new_h, new_h

  def initialize_carry(
    self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None
  ) -> Array:  # type: ignore[override]
    """Initialize the RNN cell carry.

    Args:
        rngs: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.

    Returns:
        An initialized carry for the given RNN cell.
    """
    batch_dims = input_shape[:-1]
    if rngs is None:
      rngs = self.rngs
    mem_shape = batch_dims + (self.hidden_features,)
    h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
    return h

  @property
  def num_feature_axes(self) -> int:
    return 1


class RNN(Module):
  """The ``RNN`` module takes any :class:`RNNCellBase` instance and applies it over a sequence

  using :func:`flax.nnx.scan`.
  """

  state_axes: dict[str, int | type[iteration.Carry] | None]

  def __init__(
    self,
    cell: RNNCellBase,
    time_major: bool = False,
    return_carry: bool = False,
    reverse: bool = False,
    keep_order: bool = False,
    unroll: int = 1,
    rngs: rnglib.Rngs | None = None,
    state_axes: Mapping[str, int | type[iteration.Carry] | None] | None = None,
    broadcast_rngs: filterlib.Filter = None,
  ):
    self.cell = cell
    self.time_major = time_major
    self.return_carry = return_carry
    self.reverse = reverse
    self.keep_order = keep_order
    self.unroll = unroll
    if rngs is None:
      rngs = rnglib.Rngs(0)
    self.rngs = rngs
    self.state_axes = state_axes or {...: iteration.Carry}  # type: ignore
    self.broadcast_rngs = broadcast_rngs

  def __call__(
    self,
    inputs: Array,
    *,
    initial_carry: Carry | None = None,
    seq_lengths: Array | None = None,
    return_carry: bool | None = None,
    time_major: bool | None = None,
    reverse: bool | None = None,
    keep_order: bool | None = None,
    rngs: rnglib.Rngs | None = None,
  ):
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
    time_axis = 0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)

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
    if rngs is None:
      rngs = self.rngs
    carry: Carry = (
            self.cell.initialize_carry(
                inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :], rngs
            )
            if initial_carry is None
            else initial_carry
        )

    slice_carry = seq_lengths is not None and return_carry
    broadcast_rngs = nnx.All(nnx.RngState, self.broadcast_rngs)
    state_axes = iteration.StateAxes({broadcast_rngs: None, **self.state_axes})  # type: ignore[misc]

    # we use split_rngs with splits=1 and squeeze=True to get unique rngs
    # every time RNN is called
    @nnx.split_rngs(splits=1, only=self.broadcast_rngs, squeeze=True)
    @nnx.scan(
      in_axes=(state_axes, iteration.Carry, time_axis),
      out_axes=(iteration.Carry, (0, time_axis))
      if slice_carry
      else (iteration.Carry, time_axis),
      unroll=self.unroll,
    )
    def scan_fn(
      cell: RNNCellBase, carry: Carry, x: Array
    ) -> tuple[Carry, Array] | tuple[Carry, tuple[Carry, Array]]:
      carry, y = cell(carry, x)
      if slice_carry:
        return carry, (carry, y)
      return carry, y

    scan_output = scan_fn(self.cell, carry, inputs)

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

    Example::

      >>> from flax.nnx.nn.recurrent import flip_sequences
      >>> from jax import numpy as jnp
      >>> inputs = jnp.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
      >>> lengths = jnp.array([1, 2, 3])
      >>> flip_sequences(inputs, lengths, 1, False)
      Array([[1, 0, 0],
             [3, 2, 0],
             [6, 5, 4]], dtype=int32)


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
    idxs = _expand_dims_like(idxs, target=inputs)  # [*batch, max_steps, *features]
    # Select the inputs in flipped order.
    outputs = jnp.take_along_axis(inputs, idxs, axis=time_axis)

    return outputs


def _concatenate(a: Array, b: Array) -> Array:
    """Concatenates two arrays along the last dimension."""
    return jnp.concatenate([a, b], axis=-1)


class RNNBase(Protocol):
    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: Carry | None = None,
        rngs: rnglib.Rngs | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
    ) -> Output | tuple[Carry, Output]: ...


class Bidirectional(Module):
    """Processes the input in both directions and merges the results.

    Example usage::

      >>> from flax import nnx
      >>> import jax
      >>> import jax.numpy as jnp

      >>> # Define forward and backward RNNs
      >>> forward_rnn = RNN(GRUCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(0)))
      >>> backward_rnn = RNN(GRUCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(0)))

      >>> # Create Bidirectional layer
      >>> layer = Bidirectional(forward_rnn=forward_rnn, backward_rnn=backward_rnn)

      >>> # Input data
      >>> x = jnp.ones((2, 3, 3))

      >>> # Apply the layer
      >>> out = layer(x)
      >>> print(out.shape)
      (2, 3, 8)

    """

    forward_rnn: RNNBase
    backward_rnn: RNNBase
    merge_fn: Callable[[Array, Array], Array] = _concatenate
    time_major: bool = False
    return_carry: bool = False

    def __init__(
        self,
        forward_rnn: RNNBase,
        backward_rnn: RNNBase,
        *,
        merge_fn: Callable[[Array, Array], Array] = _concatenate,
        time_major: bool = False,
        return_carry: bool = False,
        rngs: rnglib.Rngs | None = None,
    ):
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        self.merge_fn = merge_fn
        self.time_major = time_major
        self.return_carry = return_carry
        if rngs is None:
            rngs = rnglib.Rngs(0)
        self.rngs = rngs

    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: tuple[Carry, Carry] | None = None,
        rngs: rnglib.Rngs | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,  # unused
        keep_order: bool | None = None,  # unused
    ) -> Output | tuple[tuple[Carry, Carry], Output]:
        if time_major is None:
            time_major = self.time_major
        if return_carry is None:
            return_carry = self.return_carry
        if rngs is None:
            rngs = self.rngs
        if initial_carry is not None:
            initial_carry_forward, initial_carry_backward = initial_carry
        else:
            initial_carry_forward = None
            initial_carry_backward = None
        # Throw a warning in case the user accidentally re-uses the forward RNN
        # for the backward pass and does not intend for them to share parameters.
        if self.forward_rnn is self.backward_rnn:
            logging.warning(
                "forward_rnn and backward_rnn is the same object, so "
                "they will share parameters."
            )

        # Encode in the forward direction.
        carry_forward, outputs_forward = self.forward_rnn(
            inputs,
            initial_carry=initial_carry_forward,
            rngs=rngs,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=time_major,
            reverse=False,
        )

        # Encode in the backward direction.
        carry_backward, outputs_backward = self.backward_rnn(
            inputs,
            initial_carry=initial_carry_backward,
            rngs=rngs,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=time_major,
            reverse=True,
            keep_order=True,
        )

        carry = (carry_forward, carry_backward) if return_carry else None
        outputs = jax.tree_util.tree_map(
            self.merge_fn, outputs_forward, outputs_backward
        )

        if return_carry:
            return carry, outputs
        else:
            return outputs

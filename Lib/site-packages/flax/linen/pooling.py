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

"""Pooling modules."""

import jax.numpy as jnp
import numpy as np
from jax import lax


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
  """Helper function to define pooling functions.

  Pooling functions are implemented using the ReduceWindow XLA op.

  .. note::
    Be aware that pooling is not generally differentiable.
    That means providing a reduce_fn that is differentiable does not imply that
    pool is differentiable.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    init: the initial value for the reduction
    reduce_fn: a reduce function of the form ``(T, T) -> T``.
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of ``n`` integers, representing the inter-window
      strides (default: ``(1, ..., 1)``).
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
      and after each spatial dimension.
  Returns:
    The output of the reduction for each window slice.
  """
  num_batch_dims = inputs.ndim - (len(window_shape) + 1)
  strides = strides or (1,) * len(window_shape)
  assert len(window_shape) == len(
    strides
  ), f'len({window_shape}) must equal len({strides})'
  strides = (1,) * num_batch_dims + strides + (1,)
  dims = (1,) * num_batch_dims + window_shape + (1,)

  is_single_input = False
  if num_batch_dims == 0:
    # add singleton batch dimension because lax.reduce_window always
    # needs a batch dimension.
    inputs = inputs[None]
    strides = (1,) + strides
    dims = (1,) + dims
    is_single_input = True

  assert inputs.ndim == len(dims), f'len({inputs.shape}) != len({dims})'
  if not isinstance(padding, str):
    padding = tuple(map(tuple, padding))
    assert len(padding) == len(window_shape), (
      f'padding {padding} must specify pads for same number of dims as '
      f'window_shape {window_shape}'
    )
    assert all(
      [len(x) == 2 for x in padding]
    ), f'each entry in padding {padding} must be length 2'
    padding = ((0, 0),) + padding + ((0, 0),)
  y = lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
  if is_single_input:
    y = jnp.squeeze(y, axis=0)
  return y


def avg_pool(
  inputs, window_shape, strides=None, padding='VALID', count_include_pad=True
):
  """Pools the input by taking the average over a window.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of ``n`` integers, representing the inter-window
      strides (default: ``(1, ..., 1)``).
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
      and after each spatial dimension (default: ``'VALID'``).
    count_include_pad: a boolean whether to include padded tokens
      in the average calculation (default: ``True``).
  Returns:
    The average for each window slice.
  """
  y = pool(inputs, 0.0, lax.add, window_shape, strides, padding)
  if count_include_pad:
    y = y / np.prod(window_shape)
  else:
    div_shape = inputs.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(window_shape):
      div_shape = (1,) + div_shape[1:]
    y = y / pool(
      jnp.ones(div_shape), 0.0, lax.add, window_shape, strides, padding
    )
  return y


def max_pool(inputs, window_shape, strides=None, padding='VALID'):
  """Pools the input by taking the maximum of a window slice.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of ``n`` integers, representing the inter-window
      strides (default: ``(1, ..., 1)``).
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
      and after each spatial dimension (default: ``'VALID'``).
  Returns:
    The maximum for each window slice.
  """
  y = pool(inputs, -jnp.inf, lax.max, window_shape, strides, padding)
  return y


def min_pool(inputs, window_shape, strides=None, padding='VALID'):
  """Pools the input by taking the minimum of a window slice.

  Args:
    inputs: Input data with dimensions (batch, window dims..., features).
    window_shape: A shape tuple defining the window to reduce over.
    strides: A sequence of ``n`` integers, representing the inter-window strides
      (default: ``(1, ..., 1)``).
    padding: Either the string ``'SAME'``, the string ``'VALID'``, or a sequence of
      ``n`` ``(low, high)`` integer pairs that give the padding to apply before and
      after each spatial dimension (default: ``'VALID'``).

  Returns:
    The minimum for each window slice.
  """
  return pool(inputs, jnp.inf, lax.min, window_shape, strides, padding)

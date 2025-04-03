# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import typing
from typing import Any, Literal, Protocol

import numpy as np

import jax.numpy as jnp
from jax import lax
from jax import random
from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array, ArrayLike
from jax._src.util import set_module

export = set_module('jax.nn.initializers')

# TODO: Import or define these to match
# https://github.com/numpy/numpy/blob/main/numpy/typing/_dtype_like.py.
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float

@export
@typing.runtime_checkable
class Initializer(Protocol):
  @staticmethod
  def __call__(key: Array,
               shape: core.Shape,
               dtype: DTypeLikeInexact = jnp.float_) -> Array:
    raise NotImplementedError

@export
def zeros(key: Array,
          shape: core.Shape,
          dtype: DTypeLikeInexact = jnp.float_) -> Array:
  """An initializer that returns a constant array full of zeros.

  The ``key`` argument is ignored.

  >>> import jax, jax.numpy as jnp
  >>> jax.nn.initializers.zeros(jax.random.key(42), (2, 3), jnp.float32)
  Array([[0., 0., 0.],
         [0., 0., 0.]], dtype=float32)
  """
  return jnp.zeros(shape, dtypes.canonicalize_dtype(dtype))

@export
def ones(key: Array,
         shape: core.Shape,
         dtype: DTypeLikeInexact = jnp.float_) -> Array:
  """An initializer that returns a constant array full of ones.

  The ``key`` argument is ignored.

  >>> import jax, jax.numpy as jnp
  >>> jax.nn.initializers.ones(jax.random.key(42), (3, 2), jnp.float32)
  Array([[1., 1.],
         [1., 1.],
         [1., 1.]], dtype=float32)
  """
  return jnp.ones(shape, dtypes.canonicalize_dtype(dtype))

@export
def constant(value: ArrayLike,
             dtype: DTypeLikeInexact = jnp.float_
             ) -> Initializer:
  """Builds an initializer that returns arrays full of a constant ``value``.

  Args:
    value: the constant value with which to fill the initializer.
    dtype: optional; the initializer's default dtype.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.constant(-7)
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)
  Array([[-7., -7., -7.],
         [-7., -7., -7.]], dtype=float32)
  """
  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.full(shape, value, dtype=dtype)
  return init

@export
def uniform(scale: RealNumeric = 1e-2,
            dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds an initializer that returns real uniformly-distributed random arrays.

  Args:
    scale: optional; the upper bound of the random distribution.
    dtype: optional; the initializer's default dtype.

  Returns:
    An initializer that returns arrays whose values are uniformly distributed in
    the range ``[0, scale)``.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.uniform(10.0)
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[7.298188 , 8.691938 , 8.7230015],
         [2.0818567, 1.8662417, 5.5022564]], dtype=float32)
  """
  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype) * jnp.array(scale, dtype)
  return init

@export
def normal(stddev: RealNumeric = 1e-2,
           dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds an initializer that returns real normally-distributed random arrays.

  Args:
    stddev: optional; the standard deviation of the distribution.
    dtype: optional; the initializer's default dtype.

  Returns:
    An initializer that returns arrays whose values are normally distributed
    with mean ``0`` and standard deviation ``stddev``.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.normal(5.0)
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 3.0613258 ,  5.6129413 ,  5.6866574 ],
         [-4.063663  , -4.4520254 ,  0.63115686]], dtype=float32)
  """
  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.normal(key, shape, dtype) * jnp.array(stddev, dtype)
  return init

@export
def truncated_normal(stddev: RealNumeric = 1e-2,
                     dtype: DTypeLikeInexact = jnp.float_,
                     lower: RealNumeric = -2.0,
                     upper: RealNumeric = 2.0) -> Initializer:
  r"""Builds an initializer that returns truncated-normal random arrays.

  Args:
    stddev: optional; the standard deviation of the untruncated distribution.
      Note that this function does not apply the stddev correction as is done in
      the variancescaling initializers, and users are expected to apply this
      correction themselves via the stddev arg if they wish to employ it.
    dtype: optional; the initializer's default dtype.
    lower: Float representing the lower bound for truncation. Applied before
      the output is multiplied by the stddev.
    upper: Float representing the upper bound for truncation. Applied before
      the output is multiplied by the stddev.

  Returns:
    An initializer that returns arrays whose values follow the truncated normal
    distribution with mean ``0`` and standard deviation ``stddev``, and range
    :math:`\rm{lower * stddev} < x < \rm{upper * stddev}`.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.truncated_normal(5.0)
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 2.9047365,  5.2338114,  5.29852  ],
         [-3.836303 , -4.192359 ,  0.6022964]], dtype=float32)
  """

  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.truncated_normal(
        key, lower, upper, shape, dtype) * jnp.array(stddev, dtype)
  return init

@export
def _compute_fans(shape: Sequence[int],
                  in_axis: int | Sequence[int] = -2,
                  out_axis: int | Sequence[int] = -1,
                  batch_axis: int | Sequence[int] = ()
                  ) -> tuple[float, float]:
  """
  Compute effective input and output sizes for a linear or convolutional layer.

  Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the
  "receptive field" of a convolution (kernel spatial dimensions).
  """
  if len(shape) <= 1:
    raise ValueError(f"Can't compute input and output sizes of a {len(shape)}"
                     "-dimensional weights tensor. Must be at least 2D.")

  if isinstance(in_axis, int):
    in_size = shape[in_axis]
  else:
    in_size = math.prod([shape[i] for i in in_axis])
  if isinstance(out_axis, int):
    out_size = shape[out_axis]
  else:
    out_size = math.prod([shape[i] for i in out_axis])
  if isinstance(batch_axis, int):
    batch_size = shape[batch_axis]
  else:
    batch_size = math.prod([shape[i] for i in batch_axis])
  receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
  fan_in = in_size * receptive_field_size
  fan_out = out_size * receptive_field_size
  return fan_in, fan_out

def _complex_uniform(key: Array,
                     shape: Sequence[int],
                     dtype: DTypeLikeInexact) -> Array:
  """
  Sample uniform random values within a disk on the complex plane,
  with zero mean and unit variance.
  """
  key_r, key_theta = random.split(key)
  real_dtype = np.array(0, dtype).real.dtype
  dtype = dtypes.to_complex_dtype(real_dtype)
  r = jnp.sqrt(2 * random.uniform(key_r, shape, real_dtype)).astype(dtype)
  theta = 2 * jnp.pi * random.uniform(key_theta, shape, real_dtype).astype(dtype)
  return r * jnp.exp(1j * theta)

def _complex_truncated_normal(key: Array, upper: ArrayLike,
                              shape: Sequence[int],
                              dtype: DTypeLikeInexact) -> Array:
  """
  Sample random values from a centered normal distribution on the complex plane,
  whose modulus is truncated to `upper`, and the variance before the truncation
  is one.
  """
  key_r, key_theta = random.split(key)
  real_dtype = np.array(0, dtype).real.dtype
  dtype = dtypes.to_complex_dtype(real_dtype)
  t = ((1 - jnp.exp(jnp.array(-(upper ** 2), dtype)))
       * random.uniform(key_r, shape, real_dtype).astype(dtype))
  r = jnp.sqrt(-jnp.log(1 - t))
  theta = 2 * jnp.pi * random.uniform(key_theta, shape, real_dtype).astype(dtype)
  return r * jnp.exp(1j * theta)

@export
def variance_scaling(
  scale: RealNumeric,
  mode: Literal["fan_in"] | Literal["fan_out"] | Literal["fan_avg"],
  distribution: (Literal["truncated_normal"] | Literal["normal"] |
                      Literal["uniform"]),
  in_axis: int | Sequence[int] = -2,
  out_axis: int | Sequence[int] = -1,
  batch_axis: Sequence[int] = (),
  dtype: DTypeLikeInexact = jnp.float_
) -> Initializer:
  r"""
  Initializer that adapts its scale to the shape of the weights tensor.

  With ``distribution="truncated_normal"`` or ``distribution="normal"``, samples
  are drawn from a (truncated) normal distribution with a mean of zero
  and a standard deviation (after truncation, if applicable) of
  :math:`\sqrt{\frac{scale}{n}}`, where `n` is:

  * the number of input units in the weights tensor, if ``mode="fan_in"``,
  * the number of output units, if ``mode="fan_out"``, or
  * the average of the numbers of input and output units, if ``mode="fan_avg"``.

  This initializer can be configured with ``in_axis``, ``out_axis``, and
  ``batch_axis`` to work with general convolutional or dense layers; axes that
  are not in any of those arguments are assumed to be the "receptive field"
  (convolution kernel spatial axes).

  With ``distribution="truncated_normal"``, the absolute values of the samples
  are truncated at 2 standard deviations before scaling.

  With ``distribution="uniform"``, samples are drawn from:

  * a uniform interval, if `dtype` is real, or
  * a uniform disk, if `dtype` is complex,

  with a mean of zero and a standard deviation of :math:`\sqrt{\frac{scale}{n}}`
  where `n` is defined above.

  Args:
    scale: scaling factor (positive float).
    mode: one of ``"fan_in"``, ``"fan_out"``, and ``"fan_avg"``.
    distribution: random distribution to use. One of ``"truncated_normal"``,
      ``"normal"`` and ``"uniform"``.
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.
  """

  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    shape = core.canonicalize_shape(shape)
    dtype = dtypes.canonicalize_dtype(dtype)
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
    if mode == "fan_in": denominator = fan_in
    elif mode == "fan_out": denominator = fan_out
    elif mode == "fan_avg": denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        f"invalid mode for variance scaling initializer: {mode}")
    variance = jnp.array(scale / denominator, dtype=dtype)

    if distribution == "truncated_normal":
      if jnp.issubdtype(dtype, jnp.floating):
        # constant is stddev of standard normal truncated to (-2, 2)
        stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
        return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
      else:
        # constant is stddev of complex standard normal truncated to 2
        stddev = jnp.sqrt(variance) / jnp.array(.95311164380491208, dtype)
        return _complex_truncated_normal(key, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return random.normal(key, shape, dtype) * jnp.sqrt(variance)
    elif distribution == "uniform":
      if jnp.issubdtype(dtype, jnp.floating):
        return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
      else:
        return _complex_uniform(key, shape, dtype) * jnp.sqrt(variance)
    else:
      raise ValueError(f"invalid distribution for variance scaling initializer: {distribution}")

  return init

@export
def glorot_uniform(in_axis: int | Sequence[int] = -2,
                   out_axis: int | Sequence[int] = -1,
                   batch_axis: Sequence[int] = (),
                   dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds a Glorot uniform initializer (aka Xavier uniform initializer).

  A `Glorot uniform initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
  ``mode="fan_avg"``, and ``distribution="uniform"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.glorot_uniform()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 0.50350785,  0.8088631 ,  0.81566876],
         [-0.6393332 , -0.6865721 ,  0.11003882]], dtype=float32)

  .. _Glorot uniform initializer: http://proceedings.mlr.press/v9/glorot10a.html
  """
  return variance_scaling(1.0, "fan_avg", "uniform", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

xavier_uniform = glorot_uniform

@export
def glorot_normal(in_axis: int | Sequence[int] = -2,
                  out_axis: int | Sequence[int] = -1,
                  batch_axis: Sequence[int] = (),
                  dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds a Glorot normal initializer (aka Xavier normal initializer).

  A `Glorot normal initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
  ``mode="fan_avg"``, and ``distribution="truncated_normal"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.glorot_normal()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 0.41770416,  0.75262755,  0.7619329 ],
         [-0.5516644 , -0.6028657 ,  0.08661086]], dtype=float32)

  .. _Glorot normal initializer: http://proceedings.mlr.press/v9/glorot10a.html
  """
  return variance_scaling(1.0, "fan_avg", "truncated_normal", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

xavier_normal = glorot_normal

@export
def lecun_uniform(in_axis: int | Sequence[int] = -2,
                  out_axis: int | Sequence[int] = -1,
                  batch_axis: Sequence[int] = (),
                  dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds a Lecun uniform initializer.

  A `Lecun uniform initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
  ``mode="fan_in"``, and ``distribution="uniform"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.lecun_uniform()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 0.56293887,  0.90433645,  0.9119454 ],
         [-0.71479625, -0.7676109 ,  0.12302713]], dtype=float32)

  .. _Lecun uniform initializer: https://arxiv.org/abs/1706.02515
  """
  return variance_scaling(1.0, "fan_in", "uniform", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

@export
def lecun_normal(in_axis: int | Sequence[int] = -2,
                 out_axis: int | Sequence[int] = -1,
                 batch_axis: Sequence[int] = (),
                 dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds a Lecun normal initializer.

  A `Lecun normal initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
  ``mode="fan_in"``, and ``distribution="truncated_normal"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.lecun_normal()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 0.46700746,  0.8414632 ,  0.8518669 ],
         [-0.61677957, -0.67402434,  0.09683388]], dtype=float32)

  .. _Lecun normal initializer: https://arxiv.org/abs/1706.02515
  """
  return variance_scaling(1.0, "fan_in", "truncated_normal", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

@export
def he_uniform(in_axis: int | Sequence[int] = -2,
               out_axis: int | Sequence[int] = -1,
               batch_axis: Sequence[int] = (),
               dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds a He uniform initializer (aka Kaiming uniform initializer).

  A `He uniform initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 2.0``,
  ``mode="fan_in"``, and ``distribution="uniform"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.he_uniform()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 0.79611576,  1.2789248 ,  1.2896855 ],
         [-1.0108745 , -1.0855657 ,  0.17398663]], dtype=float32)

  .. _He uniform initializer: https://arxiv.org/abs/1502.01852
  """
  return variance_scaling(2.0, "fan_in", "uniform", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

kaiming_uniform = he_uniform

@export
def he_normal(in_axis: int | Sequence[int] = -2,
              out_axis: int | Sequence[int] = -1,
              batch_axis: Sequence[int] = (),
              dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """Builds a He normal initializer (aka Kaiming normal initializer).

  A `He normal initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 2.0``,
  ``mode="fan_in"``, and ``distribution="truncated_normal"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.he_normal()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 0.6604483 ,  1.1900088 ,  1.2047218 ],
         [-0.87225807, -0.95321447,  0.1369438 ]], dtype=float32)

  .. _He normal initializer: https://arxiv.org/abs/1502.01852
  """
  return variance_scaling(2.0, "fan_in", "truncated_normal", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

kaiming_normal = he_normal

@export
def orthogonal(scale: RealNumeric = 1.0,
               column_axis: int = -1,
               dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """
  Builds an initializer that returns uniformly distributed orthogonal matrices.

  If the shape is not square, the matrices will have orthonormal rows or columns
  depending on which side is smaller.

  Args:
    scale: the upper bound of the uniform distribution.
    column_axis: the axis that contains the columns that should be orthogonal.
    dtype: the default dtype of the weights.

  Returns:
    An orthogonal initializer.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.orthogonal()
  >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
  Array([[ 3.9026976e-01,  7.2495741e-01, -5.6756169e-01],
         [ 8.8047469e-01, -4.7409311e-01, -1.3157725e-04]],            dtype=float32)
  """
  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    if len(shape) < 2:
      raise ValueError("orthogonal initializer requires at least a 2D shape")
    n_rows, n_cols = math.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_cols, n_rows) if n_rows < n_cols else (n_rows, n_cols)
    A = random.normal(key, matrix_shape, dtype)
    Q, R = jnp.linalg.qr(A)
    diag_sign = lax.broadcast_to_rank(jnp.sign(jnp.diag(R)), rank=Q.ndim)
    Q *= diag_sign # needed for a uniform distribution
    if n_rows < n_cols: Q = Q.T
    Q = jnp.reshape(Q, tuple(np.delete(shape, column_axis)) + (shape[column_axis],))
    Q = jnp.moveaxis(Q, -1, column_axis)
    return jnp.array(scale, dtype) * Q
  return init

@export
def delta_orthogonal(
  scale: RealNumeric = 1.0,
  column_axis: int = -1,
  dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
  """
  Builds an initializer for delta orthogonal kernels.

  Args:
    scale: the upper bound of the uniform distribution.
    column_axis: the axis that contains the columns that should be orthogonal.
    dtype: the default dtype of the weights.

  Returns:
    A `delta orthogonal initializer`_. The shape passed to the initializer must
    be 3D, 4D, or 5D.

  Examples:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.delta_orthogonal()
  >>> initializer(jax.random.key(42), (3, 3, 3), jnp.float32)  # doctest: +SKIP
  Array([[[ 0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ]],
  <BLANKLINE>
         [[ 0.27858758, -0.7949833 , -0.53887904],
          [ 0.9120717 ,  0.04322892,  0.40774566],
          [-0.30085585, -0.6050892 ,  0.73712474]],
  <BLANKLINE>
         [[ 0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ]]], dtype=float32)


  .. _delta orthogonal initializer: https://arxiv.org/abs/1806.05393
  """
  def init(key: Array,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Delta orthogonal initializer requires a 3D, 4D or 5D "
                       "shape.")
    if shape[-1] < shape[-2]:
      raise ValueError("`fan_in` must be less or equal than `fan_out`. ")
    ortho_init = orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)
    ortho_matrix = ortho_init(key, shape[-2:])
    W = jnp.zeros(shape, dtype=dtype)
    if len(shape) == 3:
      k = shape[0]
      return W.at[(k-1)//2, ...].set(ortho_matrix)
    elif len(shape) == 4:
      k1, k2 = shape[:2]
      return W.at[(k1-1)//2, (k2-1)//2, ...].set(ortho_matrix)
    else:
      k1, k2, k3 = shape[:3]
      return W.at[(k1-1)//2, (k2-1)//2, (k3-1)//2, ...].set(ortho_matrix)
  return init

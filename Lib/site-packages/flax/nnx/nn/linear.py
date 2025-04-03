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
from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import opt_einsum

from flax.core.frozen_dict import FrozenDict
from flax import nnx
from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)

Array = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, tp.Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    ' int or pair of ints.'
  )


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: tp.Sequence[int] | int) -> tuple[int, ...]:
  if isinstance(x, tp.Iterable):
    return tuple(x)
  else:
    return (x,)


class LinearGeneral(Module):
  """A linear transformation with flexible axes.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp
    ...
    >>> # equivalent to `nnx.Linear(2, 4)`
    >>> layer = nnx.LinearGeneral(2, 4, rngs=nnx.Rngs(0))
    >>> layer.kernel.value.shape
    (2, 4)
    >>> # output features (4, 5)
    >>> layer = nnx.LinearGeneral(2, (4, 5), rngs=nnx.Rngs(0))
    >>> layer.kernel.value.shape
    (2, 4, 5)
    >>> layer.bias.value.shape
    (4, 5)
    >>> # apply transformation on the the second and last axes
    >>> layer = nnx.LinearGeneral((2, 3), (4, 5), axis=(1, -1), rngs=nnx.Rngs(0))
    >>> layer.kernel.value.shape
    (2, 3, 4, 5)
    >>> layer.bias.value.shape
    (4, 5)
    >>> y = layer(jnp.ones((16, 2, 3)))
    >>> y.shape
    (16, 4, 5)

  Args:
    in_features: int or tuple with number of input features.
    out_features: int or tuple with number of output features.
    axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    batch_axis: mapping of batch axis indices to axis size.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    rngs: rng key.
  """

  def __init__(
    self,
    in_features: Size | tp.Sequence[Size],
    out_features: Size | tp.Sequence[Size],
    *,
    axis: Axis | tp.Sequence[Axis] = -1,
    batch_axis: tp.Mapping[Axis, Size] = FrozenDict({}),
    use_bias: bool = True,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    precision: PrecisionLike = None,
    # Deprecated. Will be removed.
    dot_general: DotGeneralT | None = None,
    dot_general_cls: tp.Any = None,
    rngs: rnglib.Rngs,
  ):
    self.in_features = _canonicalize_tuple(in_features)
    self.out_features = _canonicalize_tuple(out_features)
    self.axis = _canonicalize_tuple(axis)
    self.batch_axis = FrozenDict[Axis, Size](batch_axis)
    self.use_bias = use_bias
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.precision = precision
    self.dot_general = dot_general
    self.dot_general_cls = dot_general_cls

    if len(self.in_features) != len(self.axis):
      raise ValueError(
        'in_features and axis must have the same length. '
        f'Got {self.in_features} and {self.axis}.'
      )

    if batch_axis:
      batch_dims = tuple(batch_axis.keys())
      max_dim = np.max(batch_dims)
      if set(batch_dims) != set(range(max_dim + 1)):
        raise ValueError(
          'batch_dims %s must be consecutive leading '
          'dimensions starting from 0.' % str(batch_dims)
        )

    n_batch_axis = len(self.batch_axis)
    n_in_features = len(self.in_features)
    n_out_features = len(self.out_features)

    def kernel_init_wrap(rng, shape, dtype):
      flat_shape = (
        np.prod(shape[:n_batch_axis])
        * np.prod(shape[n_batch_axis : n_in_features + n_batch_axis]),
        np.prod(shape[-n_out_features:]),
      )
      flat_shape = jax.tree.map(int, flat_shape)
      kernel = self.kernel_init(rng, flat_shape, dtype)
      if isinstance(kernel, variablelib.VariableMetadata):
        kernel.raw_value = jnp.reshape(kernel.raw_value, shape)
      else:
        kernel = jnp.reshape(kernel, shape)

      return kernel

    batch_shape = tuple(self.batch_axis.values())
    kernel_shape = (
      *batch_shape,
      *self.in_features,
      *self.out_features,
    )
    self.kernel = nnx.Param(
      kernel_init_wrap(rngs.params(), kernel_shape, self.param_dtype)
    )

    self.bias: nnx.Param[jax.Array] | None
    if self.use_bias:

      def bias_init_wrap(rng, shape, dtype):
        flat_shape = (int(np.prod(shape)),)
        bias = self.bias_init(rng, flat_shape, dtype)
        if isinstance(bias, variablelib.VariableMetadata):
          bias.raw_value = jnp.reshape(bias.raw_value, shape)
        else:
          bias = jnp.reshape(bias, shape)
        return bias

      bias_shape = (*batch_shape, *self.out_features)
      self.bias = nnx.Param(
        bias_init_wrap(rngs.params(), bias_shape, self.param_dtype)
      )
    else:
      self.bias = None

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    ndim = inputs.ndim
    n_batch_dims = len(self.batch_axis)
    axis = _normalize_axes(self.axis, ndim)
    batch_axis = _normalize_axes(tuple(self.batch_axis.keys()), ndim)
    n_axis = len(axis)

    # batch and non-contracting dims of input with 1s for batch dims.
    expanded_batch_shape = tuple(
      inputs.shape[ax] if ax in batch_axis else 1
      for ax in range(inputs.ndim)
      if ax not in axis
    )
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None

    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))

    inputs, kernel, bias = dtypes.promote_dtype(
      (inputs, kernel, bias), dtype=self.dtype
    )

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    out = dot_general(
      inputs,
      kernel,
      ((axis, contract_ind), (batch_axis, batch_ind)),
      precision=self.precision,
    )
    # dot_general output has shape [batch_dims/group_dims] + [feature_dims]
    if bias is not None:
      # expand bias shape to broadcast bias over batch dims.
      bias = jnp.reshape(bias, (*expanded_batch_shape, *self.out_features))
      out += bias
    return out


class Linear(Module):
  """A linear transformation applied over the last dimension of the input.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.Linear(in_features=3, out_features=4, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(4,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(3, 4)
      )
    })

  Args:
    in_features: the number of input features.
    out_features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    dot_general: dot product function.
    rngs: rng key.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    dot_general: DotGeneralT = lax.dot_general,
    rngs: rnglib.Rngs,
  ):
    kernel_key = rngs.params()
    self.kernel = nnx.Param(
      kernel_init(kernel_key, (in_features, out_features), param_dtype)
    )
    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
    else:
      self.bias = None

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.dot_general = dot_general

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None

    inputs, kernel, bias = dtypes.promote_dtype(
      (inputs, kernel, bias), dtype=self.dtype
    )
    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    assert self.use_bias == (bias is not None)
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class Einsum(Module):
  """An einsum transformation with learnable kernel and bias.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> layer = nnx.Einsum('nta,hab->nthb', (8, 2, 4), (8, 4), rngs=nnx.Rngs(0))
    >>> layer.kernel.value.shape
    (8, 2, 4)
    >>> layer.bias.value.shape
    (8, 4)
    >>> y = layer(jnp.ones((16, 11, 2)))
    >>> y.shape
    (16, 11, 8, 4)

  Args:
    einsum_str: a string to denote the einsum equation. The equation must
      have exactly two operands, the lhs being the input passed in, and
      the rhs being the learnable kernel. Exactly one of ``einsum_str``
      in the constructor argument and call argument must be not None,
      while the other must be None.
    kernel_shape: the shape of the kernel.
    bias_shape: the shape of the bias. If this is None, a bias won't be used.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    rngs: rng key.
  """

  def __init__(
    self,
    einsum_str: str,
    kernel_shape: Shape,
    bias_shape: tp.Optional[Shape] = None,
    *,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    rngs: rnglib.Rngs,
  ):
    einsum_str = einsum_str.replace(' ', '')
    self._einsum_str_check(einsum_str)

    kernel_key = rngs.params()
    self.kernel = nnx.Param(kernel_init(kernel_key, kernel_shape, param_dtype))

    self.bias: nnx.Param | None
    if bias_shape is not None:
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = None

    self.einsum_str = einsum_str
    self.kernel_shape = kernel_shape
    self.bias_shape = bias_shape
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init

  def __call__(
    self, inputs: Array, einsum_str: tp.Optional[str] = None
  ) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      einsum_str: a string to denote the einsum equation. The equation must
        have exactly two operands, the lhs being the input passed in, and
        the rhs being the learnable kernel. Exactly one of ``einsum_str``
        in the constructor argument and call argument must be not None,
        while the other must be None.

    Returns:
      The transformed input.
    """
    einsum_str = first_from(
      einsum_str,
      self.einsum_str,
      error_msg="""No `einsum_str` argument was provided to Einsum
        as either a __call__ argument, or class attribute.""",
    )
    einsum_str = einsum_str.replace(' ', '')
    self._einsum_str_check(einsum_str)

    inputs, kernel, bias = dtypes.promote_dtype(
      (
        inputs,
        self.kernel.value,
        self.bias.value if self.bias is not None else self.bias,
      ),
      dtype=self.dtype,
    )

    y = jnp.einsum(einsum_str, inputs, kernel, precision=self.precision)

    if bias is not None:
      broadcasted_bias_shape = self._infer_broadcasted_bias_shape(
        einsum_str, inputs, kernel
      )
      y += jnp.reshape(bias, broadcasted_bias_shape)
    return y

  def _infer_broadcasted_bias_shape(
    self, einsum_str: str, lhs: Array, rhs: Array
  ):
    """Infer the broadcasted bias shape given the ``einsum_str``, ``lhs``
    and ``rhs`` arrays. This is needed reshaping the bias and it to the
    output during forward inference.

    This function first replaces all ellipses with actual letter characters,
    then computes the broadcasted bias shape by checking to see which axes in
    the rhs array remain in the resulting array after einsumming. These axes
    are the embedding/feature dimensions, and all other axes in rhs are
    reduction axes.
    """
    # More details on the parsing function: https://github.com/dgasmith/opt_einsum/blob/c826bb7df16f470a69f7bf90598fc27586209d11/opt_einsum/parser.py#L246
    # returns the einsum string representation of the operands and result, with
    # ellipsis replaced by actual letter characters
    operands_str, result_str, _ = opt_einsum.parser.parse_einsum_input(
      (einsum_str, lhs, rhs)
    )

    # rhs_dict is a dict{character:index} mapping that maps every character in
    # the rhs einsum string representation to its corresponding index position in the string
    rhs_dict = {c: i for i, c in enumerate(operands_str.split(',')[1])}
    assert len(rhs_dict) == len(self.kernel_shape)

    broadcasted_bias_shape = [1] * len(result_str)
    for i, c in enumerate(result_str):
      if c in rhs_dict:
        broadcasted_bias_shape[i] = self.kernel_shape[rhs_dict[c]]

    return broadcasted_bias_shape

  def _einsum_str_check(self, einsum_str):
    if '->' not in einsum_str:
      raise ValueError(
        '`einsum_str` equation must be explicit and include "->".'
      )
    if einsum_str.count(',') != 1:
      raise ValueError(
        '`einsum_str` equation must have exactly two operands and '
        'therefore, exactly one comma character, instead of '
        f'{einsum_str.count(",")}'
      )


class Conv(Module):
  """Convolution Module wrapping ``lax.conv_general_dilated``.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> rngs = nnx.Rngs(0)
    >>> x = jnp.ones((1, 8, 3))

    >>> # valid padding
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  padding='VALID', rngs=rngs)
    >>> layer.kernel.value.shape
    (3, 3, 4)
    >>> layer.bias.value.shape
    (4,)
    >>> out = layer(x)
    >>> out.shape
    (1, 6, 4)

    >>> # circular padding with stride 2
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3, 3),
    ...                  strides=2, padding='CIRCULAR', rngs=rngs)
    >>> layer.kernel.value.shape
    (3, 3, 3, 4)
    >>> layer.bias.value.shape
    (4,)
    >>> out = layer(x)
    >>> out.shape
    (1, 4, 4)

    >>> # apply lower triangle mask
    >>> mask = jnp.tril(jnp.ones((3, 3, 4)))
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  mask=mask, padding='VALID', rngs=rngs)
    >>> out = layer(x)

  Args:
    in_features: int or tuple with number of input features.
    out_features: int or tuple with number of output features.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer, which will be interpreted
      as a tuple of the single integer. For all other cases, it must be a
      sequence of integers.
    strides: an integer or a sequence of ``n`` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs``
      (default: 1). Convolution with input dilation ``d`` is equivalent to
      transposed convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    rngs: rng key.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | tp.Sequence[int],
    strides: tp.Union[None, int, tp.Sequence[int]] = 1,
    *,
    padding: PaddingLike = 'SAME',
    input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    mask: tp.Optional[Array] = None,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
    rngs: rnglib.Rngs,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    kernel_shape = kernel_size + (
      in_features // feature_group_count,
      out_features,
    )
    kernel_key = rngs.params()
    self.kernel_shape = kernel_shape
    self.kernel = nnx.Param(kernel_init(kernel_key, kernel_shape, param_dtype))

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_shape = (out_features,)
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = None

    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.mask = mask
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.conv_general_dilated = conv_general_dilated

  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions ``(*batch_dims, spatial_dims..., features)``.
        This is the channels-last convention, i.e. NHWC for a 2d convolution and
        NDHWC for a 3D convolution. Note: this is different from the input convention
        used by ``lax.conv_general_dilated``, which puts the spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    assert isinstance(self.kernel_size, tuple)
    kernel_size = self.kernel_size

    def maybe_broadcast(
      x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
    ) -> tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: tp.List[tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    # One shared convolutional kernel for all pixels in the output.
    assert self.in_features % self.feature_group_count == 0

    if self.mask is not None and self.mask.shape != self.kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {self.kernel_shape}'
      )

    kernel = self.kernel.value

    if self.mask is not None:
      kernel *= self.mask

    bias = self.bias.value if self.bias is not None else None

    inputs, kernel, bias = dtypes.promote_dtype(
      (inputs, kernel, bias), dtype=self.dtype
    )

    y = self.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y


class ConvTranspose(Module):
  """Convolution Module wrapping ``lax.conv_transpose``.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> rngs = nnx.Rngs(0)
    >>> x = jnp.ones((1, 8, 3))

    >>> # valid padding
    >>> layer = nnx.ConvTranspose(in_features=3, out_features=4, kernel_size=(3,),
    ...                           padding='VALID', rngs=rngs)
    >>> layer.kernel.value.shape
    (3, 3, 4)
    >>> layer.bias.value.shape
    (4,)
    >>> out = layer(x)
    >>> out.shape
    (1, 10, 4)

    >>> # circular padding with stride 2
    >>> layer = nnx.ConvTranspose(in_features=3, out_features=4, kernel_size=(6, 6),
    ...                           strides=(2, 2), padding='CIRCULAR',
    ...                           transpose_kernel=True, rngs=rngs)
    >>> layer.kernel.value.shape
    (6, 6, 4, 3)
    >>> layer.bias.value.shape
    (4,)
    >>> out = layer(jnp.ones((1, 15, 15, 3)))
    >>> out.shape
    (1, 30, 30, 4)

    >>> # apply lower triangle mask
    >>> mask = jnp.tril(jnp.ones((3, 3, 4)))
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  mask=mask, padding='VALID', rngs=rngs)
    >>> out = layer(x)

  Args:
    in_features: int or tuple with number of input features.
    out_features: int or tuple with number of output features.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer, which will be interpreted
      as a tuple of the single integer. For all other cases, it must be a
      sequence of integers.
    strides: an integer or a sequence of ``n`` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    transpose_kernel: if ``True`` flips spatial axes and swaps the input/output
      channel axes of the kernel.
    rngs: rng key.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | tp.Sequence[int],
    strides: int | tp.Sequence[int] | None = None,
    *,
    padding: PaddingLike = 'SAME',
    kernel_dilation: int | tp.Sequence[int] | None = None,
    use_bias: bool = True,
    mask: Array | None = None,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike | None = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    transpose_kernel: bool = False,
    rngs: rnglib.Rngs,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    self.kernel_size = kernel_size
    self.in_features = in_features
    self.out_features = out_features
    self.strides = strides
    self.padding = padding
    self.kernel_dilation = kernel_dilation
    self.use_bias = use_bias
    self.mask = mask
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.transpose_kernel = transpose_kernel

    if self.transpose_kernel:
      kernel_shape = kernel_size + (self.out_features, in_features)
    else:
      kernel_shape = kernel_size + (in_features, self.out_features)

    self.kernel_shape = kernel_shape
    self.kernel = nnx.Param(
      self.kernel_init(rngs.params(), kernel_shape, self.param_dtype)
    )

    self.bias: nnx.Param | None
    if self.use_bias:
      self.bias = nnx.Param(
        self.bias_init(rngs.params(), (self.out_features,), self.param_dtype)
      )
    else:
      self.bias = None

  def __call__(self, inputs: Array) -> Array:
    """Applies a transposed convolution to the inputs.

    Behaviour mirrors of ``jax.lax.conv_transpose``.

    Args:
      inputs: input data with dimensions ``(*batch_dims, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution and NDHWC
        for a 3D convolution. Note: this is different from the input convention used by
        ``lax.conv_general_dilated``, which puts the spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """
    kernel_size = self.kernel_size

    def maybe_broadcast(
      x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
    ) -> tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    strides = maybe_broadcast(self.strides)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    kernel_shape = self.kernel_shape

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    kernel = self.kernel.value

    if self.mask is not None:
      kernel *= self.mask

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      padding_lax = 'VALID'

    bias = self.bias.value if self.bias is not None else None

    inputs, kernel, bias = dtypes.promote_dtype(
      (inputs, kernel, bias), dtype=self.dtype
    )

    y = lax.conv_transpose(
      inputs,
      kernel,
      strides,
      padding_lax,
      rhs_dilation=kernel_dilation,
      transpose_kernel=self.transpose_kernel,
      precision=self.precision,
    )

    if self.padding == 'CIRCULAR':
      # For circular padding, we need to identify the size of the final output
      # ("period") along each spatial dimension, pad each dimension to an
      # integer number of periods, and wrap the array periodically around each
      # dimension. Padding should be done in such a way that the start of the
      # original input data inside the padded array is located at integer
      # number of periods - otherwise the result would be circularly shifted.

      # Compute period along each spatial dimension - it's input size scaled
      # by the stride.
      scaled_x_dims = [
        x_dim * stride
        for x_dim, stride in zip(jnp.shape(inputs)[1:-1], strides)
      ]
      # Compute difference between the current size of y and the final output
      # size, and complement this difference to 2 * period - that gives how
      # much we need to pad.
      size_diffs = [
        -(y_dim - x_dim) % (2 * x_dim)
        for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
      ]
      if self.transpose_kernel:
        # If the kernel is transposed, the "+1" is put on the right to
        # mirror the regular convolution. If the same kernel parameters are used
        # as for Conv, this layer then computes the proper transpose convolution.
        total_pad = [
          (size_diff // 2, (size_diff + 1) // 2) for size_diff in size_diffs
        ]
      else:
        # Divide the padding equally between left and right. The choice to put
        # "+1" on the left (and not on the right) represents a convention for
        # aligning even-sized kernels.
        total_pad = [
          ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
        ]
      y = jnp.pad(y, [(0, 0)] + total_pad + [(0, 0)])
      # Wrap the result periodically around each spatial dimension,
      # one by one.
      for i in range(1, y.ndim - 1):
        y = y.reshape(
          y.shape[:i] + (-1, scaled_x_dims[i - 1]) + y.shape[i + 1 :]
        )
        y = y.sum(axis=i)

    if self.use_bias:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))  # type: ignore

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)

    return y


default_embed_init = initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)


class Embed(Module):
  """Embedding Module.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> layer = nnx.Embed(num_embeddings=5, features=3, rngs=nnx.Rngs(0))
    >>> nnx.state(layer)
    State({
      'embedding': VariableState( # 15 (60 B)
        type=Param,
        value=Array([[ 0.57966787, -0.523274  , -0.43195742],
               [-0.676289  , -0.50300646,  0.33996582],
               [ 0.41796115, -0.59212935,  0.95934135],
               [-1.0917838 , -0.7441663 ,  0.07713798],
               [-0.66570747,  0.13815777,  1.007365  ]], dtype=float32)
      )
    })
    >>> # get the first three and last three embeddings
    >>> indices_input = jnp.array([[0, 1, 2], [-1, -2, -3]])
    >>> layer(indices_input)
    Array([[[ 0.57966787, -0.523274  , -0.43195742],
            [-0.676289  , -0.50300646,  0.33996582],
            [ 0.41796115, -0.59212935,  0.95934135]],
    <BLANKLINE>
           [[-0.66570747,  0.13815777,  1.007365  ],
            [-1.0917838 , -0.7441663 ,  0.07713798],
            [ 0.41796115, -0.59212935,  0.95934135]]], dtype=float32)

  A parameterized function from integers [0, ``num_embeddings``) to
  ``features``-dimensional vectors. This ``Module`` will create an ``embedding``
  matrix with shape ``(num_embeddings, features)``. When calling this layer,
  the input values will be used to 0-index into the ``embedding`` matrix.
  Indexing on a value greater than or equal to ``num_embeddings`` will result
  in ``nan`` values. When ``num_embeddings`` equals to 1, it will
  broadcast the ``embedding`` matrix to input shape with ``features``
  dimension appended.

  Args:
    num_embeddings: number of embeddings / vocab size.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: same as embedding).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
    rngs: rng key.
  """

  def __init__(
    self,
    num_embeddings: int,
    features: int,
    *,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    embedding_init: Initializer = default_embed_init,
    rngs: rnglib.Rngs,
  ):
    self.embedding = nnx.Param(
      embedding_init(rngs.params(), (num_embeddings, features), param_dtype)
    )

    self.num_embeddings = num_embeddings
    self.features = features
    self.dtype = dtype or self.embedding.value.dtype
    self.param_dtype = param_dtype
    self.embedding_init = embedding_init

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.
        Values in the input array must be integers.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional ``features`` dimension appended.
    """
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    # Use take because fancy indexing numpy arrays with JAX indices does not
    # work correctly.
    (embedding,) = dtypes.promote_dtype(
      (self.embedding.value,), dtype=self.dtype, inexact=False
    )
    if self.num_embeddings == 1:
      return jnp.where(
        jnp.broadcast_to(inputs[..., None], inputs.shape + (self.features,))
        == 0,
        embedding,
        jnp.nan,
      )
    return jnp.take(embedding, inputs, axis=0)

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth ``features`` of the
        embedding.

    Returns:
      An array with final dim ``num_embeddings`` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    query, embedding = dtypes.promote_dtype(
      (query, self.embedding.value), dtype=self.dtype
    )
    return jnp.dot(query, embedding.T)

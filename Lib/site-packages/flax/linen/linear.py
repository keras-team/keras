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

"""Linear modules."""

from typing import (
  Any,
)
from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import eval_shape, lax
from jax.core import ShapedArray

import opt_einsum

from flax.core import meta
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen import module
from flax.linen.module import Module, compact
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)


default_kernel_init = initializers.lecun_normal()


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Sequence[int] | int) -> tuple[int, ...]:
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(Module):
  """A linear transformation with flexible axes.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> # equivalent to `nn.Dense(features=4)`
    >>> layer = nn.DenseGeneral(features=4)
    >>> # output features (4, 5)
    >>> layer = nn.DenseGeneral(features=(4, 5))
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, params)
    {'params': {'bias': (4, 5), 'kernel': (3, 4, 5)}}
    >>> # apply transformation on the the second and last axes
    >>> layer = nn.DenseGeneral(features=(4, 5), axis=(1, -1))
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3, 6, 7)))
    >>> jax.tree_util.tree_map(jnp.shape, params)
    {'params': {'bias': (4, 5), 'kernel': (3, 7, 4, 5)}}

  Attributes:
    features: int or tuple with number of output features.
    axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    batch_dims: tuple with batch axes.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
  """

  features: int | Sequence[int]
  axis: int | Sequence[int] = -1
  batch_dims: Sequence[int] = ()
  use_bias: bool = True
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  precision: PrecisionLike = None
  # Deprecated. Will be removed.
  dot_general: DotGeneralT | None = None
  dot_general_cls: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)
    batch_dims = _canonicalize_tuple(self.batch_dims)
    if batch_dims:
      max_dim = np.max(batch_dims)
      if set(batch_dims) != set(range(max_dim + 1)):
        raise ValueError(
          'batch_dims %s must be consecutive leading '
          'dimensions starting from 0.' % str(batch_dims)
        )

    ndim = inputs.ndim
    n_batch_dims = len(batch_dims)
    axis = _normalize_axes(axis, ndim)
    batch_dims = _normalize_axes(batch_dims, ndim)
    n_axis, n_features = len(axis), len(features)

    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      flat_shape = (
        np.prod(shape[:n_batch_dims])
        * np.prod(shape[n_batch_dims : n_axis + n_batch_dims]),
        np.prod(shape[-n_features:]),
      )
      flat_shape = jax.tree_util.tree_map(int, flat_shape)
      kernel = self.kernel_init(rng, flat_shape, dtype)
      if isinstance(kernel, meta.AxisMetadata):
        return meta.replace_boxed(kernel, jnp.reshape(kernel.unbox(), shape))
      return jnp.reshape(kernel, shape)

    batch_shape = tuple(inputs.shape[ax] for ax in batch_dims)
    # batch and non-contracting dims of input with 1s for batch dims.
    expanded_batch_shape = tuple(
      inputs.shape[ax] if ax in batch_dims else 1
      for ax in range(inputs.ndim)
      if ax not in axis
    )
    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel = self.param(
      'kernel', kernel_init_wrap, batch_shape + kernel_shape, self.param_dtype
    )

    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))

    if self.use_bias:

      def bias_init_wrap(rng, shape, dtype=jnp.float32):
        flat_shape = (
          np.prod(shape[:n_batch_dims]) * np.prod(shape[-n_features:]),
        )
        flat_shape = jax.tree_util.tree_map(int, flat_shape)
        bias = self.bias_init(rng, flat_shape, dtype)
        if isinstance(bias, meta.AxisMetadata):
          return meta.replace_boxed(bias, jnp.reshape(bias.unbox(), shape))
        return jnp.reshape(bias, shape)

      bias = self.param(
        'bias', bias_init_wrap, batch_shape + features, self.param_dtype
      )
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    out = dot_general(
      inputs,
      kernel,
      ((axis, contract_ind), (batch_dims, batch_ind)),
      precision=self.precision,
    )
    # dot_general output has shape [batch_dims/group_dims] + [feature_dims]
    if self.use_bias:
      # expand bias shape to broadcast bias over batch dims.
      bias = jnp.reshape(bias, expanded_batch_shape + features)
      out += bias
    return out


class Dense(Module):
  """A linear transformation applied over the last dimension of the input.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Dense(features=4)
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, params)
    {'params': {'bias': (4,), 'kernel': (3, 4)}}

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  features: int
  use_bias: bool = True
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  # Deprecated. Will be removed.
  dot_general: DotGeneralT | None = None
  dot_general_cls: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param(
      'kernel',
      self.kernel_init,
      (jnp.shape(inputs)[-1], self.features),
      self.param_dtype,
    )
    if self.use_bias:
      bias = self.param(
        'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    y = dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class Einsum(Module):
  """An einsum transformation with learnable kernel and bias.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Einsum((5, 6, 7), 'abc,cde->abde')
    >>> variables = layer.init(jax.random.key(0), jnp.ones((3, 4, 5)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (6, 7), 'kernel': (5, 6, 7)}}

  Attributes:
    shape: the shape of the kernel.
    einsum_str: a string to denote the einsum equation. The equation must
      have exactly two operands, the lhs being the input passed in, and
      the rhs being the learnable kernel. Exactly one of ``einsum_str``
      in the constructor argument and call argument must be not None,
      while the other must be None.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  shape: Shape
  einsum_str: str | None = None
  use_bias: bool = True
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(self, inputs: Array, einsum_str: str | None = None) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      einsum_str: a string to denote the einsum equation. The equation must
        have exactly two operands, the lhs being the input passed in, and
        the rhs being the learnable kernel. The ``einsum_str`` passed into
        the call method will take precedence over the ``einsum_str`` passed
        into the constructor.

    Returns:
      The transformed input.
    """
    einsum_str = module.merge_param('einsum_str', self.einsum_str, einsum_str)

    einsum_str = einsum_str.replace(' ', '')
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

    kernel = self.param(
      'kernel',
      self.kernel_init,
      self.shape,
      self.param_dtype,
    )

    if self.use_bias:
      bias_shape, broadcasted_bias_shape = self._get_bias_shape(
        einsum_str, inputs, kernel
      )
      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    y = jnp.einsum(einsum_str, inputs, kernel, precision=self.precision)

    if bias is not None:
      y += jnp.reshape(bias, broadcasted_bias_shape)
    return y

  def _get_bias_shape(self, einsum_str: str, lhs: Array, rhs: Array):
    """Infer the bias shape and broadcasted bias shape given the ``einsum_str``,
    ``lhs`` and ``rhs`` arrays. This is needed for instantiating the bias
    parameter and adding the bias to the output during forward inference.

    This function first replaces all ellipses with actual letter characters,
    then computes the bias shape by checking to see which axes in the rhs
    array remain in the resulting array after einsumming. These axes are the
    embedding/feature dimensions, and all other axes in rhs are reduction axes.
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
    assert len(rhs_dict) == len(self.shape)

    broadcasted_bias_shape = [1] * len(result_str)
    bias_shape = []
    for i, c in enumerate(result_str):
      if c in rhs_dict:
        broadcasted_bias_shape[i] = self.shape[rhs_dict[c]]
        bias_shape.append(self.shape[rhs_dict[c]])

    return bias_shape, broadcasted_bias_shape


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, Sequence) and len(padding) == rank:
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


class _Conv(Module):
  """Convolution Module wrapping ``lax.conv_general_dilated``.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. An integer will be
      interpreted as a tuple of the single integer.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), or a sequence of ``n`` ``(low,
      high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same padding
      in all dims and assign a single int in a sequence causes the same padding
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
  """

  features: int
  kernel_size: int | Sequence[int]
  strides: None | int | Sequence[int] = 1
  padding: PaddingLike = 'SAME'
  input_dilation: None | int | Sequence[int] = 1
  kernel_dilation: None | int | Sequence[int] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: Array | None = None
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  # Deprecated. Will be removed.
  conv_general_dilated: ConvGeneralDilatedT | None = None
  conv_general_dilated_cls: Any = None

  @property
  def shared_weights(self) -> bool:  # type: ignore
    """Defines whether weights are shared or not between different pixels.

    Returns:
      ``True`` to use shared weights in convolution (regular convolution).
      ``False`` to use different weights at different pixels, a.k.a.
      "locally connected layer", "unshared convolution", or "local convolution".

    """
    ...

  @compact
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

    kernel_size: Sequence[int]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(
      x: int | Sequence[int] | None,
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
      zero_pad: list[tuple[int, int]] = [(0, 0)]
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
    in_features = jnp.shape(inputs)[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
        in_features // self.feature_group_count,
        self.features,
      )

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
          '`lax.conv_general_dilated_local` does not support '
          f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      if self.conv_general_dilated_cls is not None:
        conv_general_dilated = self.conv_general_dilated_cls()
      elif self.conv_general_dilated is not None:
        conv_general_dilated = self.conv_general_dilated
      else:
        conv_general_dilated = lax.conv_general_dilated
      conv_output_shape = eval_shape(
        lambda lhs, rhs: conv_general_dilated(  # pylint: disable=g-long-lambda
          lhs=lhs,
          rhs=rhs,
          window_strides=strides,
          padding=padding_lax,
          dimension_numbers=dimension_numbers,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
        ),
        inputs,
        ShapedArray(kernel_size + (in_features, self.features), inputs.dtype),
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (
        np.prod(kernel_size) * in_features,
        self.features,
      )

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    kernel = self.param(
      'kernel', self.kernel_init, kernel_shape, self.param_dtype
    )

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = conv_output_shape[1:]

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      if self.conv_general_dilated_cls is not None:
        conv_general_dilated = self.conv_general_dilated_cls()
      elif self.conv_general_dilated is not None:
        conv_general_dilated = self.conv_general_dilated
      else:
        conv_general_dilated = lax.conv_general_dilated
      y = conv_general_dilated(
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
    else:
      y = lax.conv_general_dilated_local(
        lhs=inputs,
        rhs=kernel,
        window_strides=strides,
        padding=padding_lax,
        filter_shape=kernel_size,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        precision=self.precision,
      )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y


class Conv(_Conv):
  """Convolution Module wrapping ``lax.conv_general_dilated``.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> # valid padding
    >>> layer = nn.Conv(features=4, kernel_size=(3,), padding='VALID')
    >>> out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 8, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (4,), 'kernel': (3, 3, 4)}}
    >>> out.shape
    (1, 6, 4)
    >>> # circular padding with stride 2
    >>> layer = nn.Conv(features=4, kernel_size=(3, 3), strides=2, padding='CIRCULAR')
    >>> out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 8, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (4,), 'kernel': (3, 3, 3, 4)}}
    >>> out.shape
    (1, 4, 4)
    >>> # apply lower triangle mask
    >>> mask = jnp.tril(jnp.ones((3, 3, 4)))
    >>> layer = nn.Conv(features=4, kernel_size=(3,), mask=mask, padding='VALID')
    >>> variables = layer.init(jax.random.key(0), jnp.ones((1, 8, 3)))

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. An integer will be
      interpreted as a tuple of the single integer.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and
      after each spatial dimension. A single int is interpreted as applying the
      same padding in all dims and assign a single int in a sequence causes the
      same padding to be used on both sides. ``'CAUSAL'`` padding for a 1D
      convolution will left-pad the convolution axis, resulting in same-sized
      output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs`` (default:
      1). Convolution with input dilation ``d`` is equivalent to transposed
      convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation is also known as
      'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
      be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """

  @property
  def shared_weights(self) -> bool:
    return True


class ConvLocal(_Conv):
  """Local convolution Module wrapping ``lax.conv_general_dilated_local``.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> # valid padding
    >>> layer = nn.ConvLocal(features=4, kernel_size=(3,), padding='VALID')
    >>> out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 8, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (6, 4), 'kernel': (6, 9, 4)}}
    >>> out.shape
    (1, 6, 4)
    >>> # circular padding with stride 2
    >>> layer = nn.ConvLocal(features=4, kernel_size=(3, 3), strides=2, padding='CIRCULAR')
    >>> out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 8, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (1, 4, 4), 'kernel': (1, 4, 27, 4)}}
    >>> out.shape
    (1, 4, 4)
    >>> # apply lower triangle mask
    >>> mask = jnp.tril(jnp.ones((6, 9, 4)))
    >>> layer = nn.ConvLocal(features=4, kernel_size=(3,), mask=mask, padding='VALID')
    >>> variables = layer.init(jax.random.key(0), jnp.ones((1, 8, 3)))

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. An integer will be
      interpreted as a tuple of the single integer.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and
      after each spatial dimension. A single int is interpreted as applying the
      same padding in all dims and assign a single int in a sequence causes the
      same padding to be used on both sides. ``'CAUSAL'`` padding for a 1D
      convolution will left-pad the convolution axis, resulting in same-sized
      output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs`` (default:
      1). Convolution with input dilation ``d`` is equivalent to transposed
      convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation is also known as
      'atrous convolution'.
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
  """

  @property
  def shared_weights(self) -> bool:
    return False


class ConvTranspose(Module):
  """Convolution Module wrapping ``lax.conv_transpose``.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> # valid padding
    >>> layer = nn.ConvTranspose(features=4, kernel_size=(3,), padding='VALID')
    >>> out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 8, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (4,), 'kernel': (3, 3, 4)}}
    >>> out.shape
    (1, 10, 4)
    >>> # circular padding with stride 2
    >>> layer = nn.ConvTranspose(features=4, kernel_size=(6, 6), strides=(2, 2), padding='CIRCULAR', transpose_kernel=True)
    >>> out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 15, 15, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (4,), 'kernel': (6, 6, 4, 3)}}
    >>> out.shape
    (1, 30, 30, 4)
    >>> # apply lower triangle mask
    >>> mask = jnp.tril(jnp.ones((3, 3, 4)))
    >>> layer = nn.ConvTranspose(features=4, kernel_size=(3,), mask=mask, padding='VALID')
    >>> variables = layer.init(jax.random.key(0), jnp.ones((1, 8, 3)))

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution, the
      kernel size can be passed as an integer, which will be interpreted as a
      tuple of the single integer. For all other cases, it must be a sequence of
      integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides.
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same
      padding in all dims and assign a single int in a sequence causes the same
      padding to be used on both sides.
    kernel_dilation: ``None``, or an integer or a sequence of ``n`` integers,
      giving the dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
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
  """

  features: int
  kernel_size: int | Sequence[int]
  strides: Sequence[int] | None = None
  padding: PaddingLike = 'SAME'
  kernel_dilation: Sequence[int] | None = None
  use_bias: bool = True
  mask: Array | None = None
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  transpose_kernel: bool = False

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a transposed convolution to the inputs.

    Behaviour mirrors of ``jax.lax.conv_transpose``.

    Args:
      inputs: input data with dimensions ``(*batch_dims, spatial_dims..., features).``
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
    kernel_size: tuple[int, ...]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(
      x: int | Sequence[int] | None,
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

    in_features = jnp.shape(inputs)[-1]
    if self.transpose_kernel:
      kernel_shape = kernel_size + (self.features, in_features)
    else:
      kernel_shape = kernel_size + (in_features, self.features)

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    kernel = self.param(
      'kernel', self.kernel_init, kernel_shape, self.param_dtype
    )

    if self.mask is not None:
      kernel *= self.mask

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      padding_lax = 'VALID'

    if self.use_bias:
      bias = self.param(
        'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

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

  A parameterized function from integers [0, ``num_embeddings``) to
  ``features``-dimensional vectors. This ``Module`` will create an ``embedding``
  matrix with shape ``(num_embeddings, features)``. When calling this layer,
  the input values will be used to 0-index into the ``embedding`` matrix.
  Indexing on a value greater than or equal to ``num_embeddings`` will result
  in ``nan`` values. When ``num_embeddings`` equals to 1, it will
  broadcast the ``embedding`` matrix to input shape with ``features``
  dimension appended.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Embed(num_embeddings=5, features=3)
    >>> indices_input = jnp.array([[0, 1, 2], [-1, -2, -3]])
    >>> variables = layer.init(jax.random.key(0), indices_input)
    >>> variables
    {'params': {'embedding': Array([[ 0.04396089, -0.9328513 , -0.97328115],
           [ 0.41147125,  0.66334754,  0.49469155],
           [ 0.09719624,  0.49861377,  0.49519277],
           [-0.13316602,  0.6697022 ,  0.3710195 ],
           [-0.5039532 ,  0.287319  ,  1.4369922 ]], dtype=float32)}}
    >>> # get the first three and last three embeddings
    >>> layer.apply(variables, indices_input)
    Array([[[ 0.04396089, -0.9328513 , -0.97328115],
            [ 0.41147125,  0.66334754,  0.49469155],
            [ 0.09719624,  0.49861377,  0.49519277]],
    <BLANKLINE>
           [[-0.5039532 ,  0.287319  ,  1.4369922 ],
            [-0.13316602,  0.6697022 ,  0.3710195 ],
            [ 0.09719624,  0.49861377,  0.49519277]]], dtype=float32)

  Attributes:
    num_embeddings: number of embeddings / vocab size.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: same as embedding).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
  """

  num_embeddings: int
  features: int
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  embedding_init: Initializer = default_embed_init

  def setup(self):
    self.embedding = self.param(
      'embedding',
      self.embedding_init,
      (self.num_embeddings, self.features),
      self.param_dtype,
    )

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
    (embedding,) = promote_dtype(
      self.embedding, dtype=self.dtype, inexact=False
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
    query, embedding = promote_dtype(query, self.embedding, dtype=self.dtype)
    return jnp.dot(query, embedding.T)

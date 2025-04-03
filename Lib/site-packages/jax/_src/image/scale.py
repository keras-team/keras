# Copyright 2020 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import enum

import numpy as np

from jax import jit
from jax import lax
from jax import numpy as jnp
from jax._src import core
from jax._src.util import canonicalize_axis
from jax._src.numpy.util import promote_dtypes_inexact


def _fill_lanczos_kernel(radius, x):
  y = radius * jnp.sin(np.pi * x) * jnp.sin(np.pi * x / radius)
  #  out = y / (np.pi ** 2 * x ** 2) where x >1e-3, 1 otherwise
  out = jnp.where(x > 1e-3, jnp.divide(y, jnp.where(x != 0, np.pi**2 * x**2, 1)), 1)
  return jnp.where(x > radius, 0., out)


def _fill_keys_cubic_kernel(x):
  # http://ieeexplore.ieee.org/document/1163711/
  # R. G. Keys. Cubic convolution interpolation for digital image processing.
  # IEEE Transactions on Acoustics, Speech, and Signal Processing,
  # 29(6):1153–1160, 1981.
  out = ((1.5 * x - 2.5) * x) * x + 1.
  out = jnp.where(x >= 1., ((-0.5 * x + 2.5) * x - 4.) * x + 2., out)
  return jnp.where(x >= 2., 0., out)


def _fill_triangle_kernel(x):
  return jnp.maximum(0, 1 - jnp.abs(x))


def compute_weight_mat(input_size: core.DimSize,
                       output_size: core.DimSize,
                       scale,
                       translation,
                       kernel: Callable,
                       antialias: bool):
  dtype = jnp.result_type(scale, translation)
  inv_scale = 1. / scale
  # When downsampling the kernel should be scaled since we want to low pass
  # filter and interpolate, but when upsampling it should not be since we only
  # want to interpolate.
  kernel_scale = jnp.maximum(inv_scale, 1.) if antialias else 1.
  sample_f = ((jnp.arange(output_size, dtype=dtype) + 0.5) * inv_scale -
              translation * inv_scale - 0.5)
  x = (
      jnp.abs(sample_f[jnp.newaxis, :] -
              jnp.arange(input_size, dtype=dtype)[:, jnp.newaxis]) /
      kernel_scale)
  weights = kernel(x)

  total_weight_sum = jnp.sum(weights, axis=0, keepdims=True)
  weights = jnp.where(
      jnp.abs(total_weight_sum) > 1000. * float(np.finfo(np.float32).eps),
      jnp.divide(weights, jnp.where(total_weight_sum != 0,  total_weight_sum, 1)),
      0)
  # Zero out weights where the sample location is completely outside the input
  # range.
  # Note sample_f has already had the 0.5 removed, hence the weird range below.
  input_size_minus_0_5 = core.dimension_as_value(input_size) - 0.5
  return jnp.where(
      jnp.logical_and(sample_f >= -0.5,
                      sample_f <= input_size_minus_0_5)[jnp.newaxis, :], weights, 0)


def _scale_and_translate(x, output_shape: core.Shape,
                         spatial_dims: Sequence[int], scale, translation,
                         kernel, antialias: bool, precision):
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  assert len(spatial_dims) == len(scale)
  assert len(spatial_dims) == len(translation)
  if len(spatial_dims) == 0:
    return x
  contractions = []
  in_indices = list(range(len(output_shape)))
  out_indices = list(range(len(output_shape)))
  for i, d in enumerate(spatial_dims):
    d = canonicalize_axis(d, x.ndim)
    m = input_shape[d]
    n = output_shape[d]
    w = compute_weight_mat(m, n, scale[i], translation[i],
                           kernel, antialias).astype(x.dtype)
    contractions.append(w)
    contractions.append([d, len(output_shape) + i])
    out_indices[d] = len(output_shape) + i
  contractions.append(out_indices)
  return jnp.einsum(x, in_indices, *contractions, precision=precision)


class ResizeMethod(enum.Enum):
  """Image resize method.

  Possible values are:

  NEAREST:
    Nearest-neighbor interpolation.

  LINEAR:
    `Linear interpolation`_.

  LANCZOS3:
    `Lanczos resampling`_, using a kernel of radius 3.

  LANCZOS5:
    `Lanczos resampling`_, using a kernel of radius 5.

  CUBIC:
    `Cubic interpolation`_, using the Keys cubic kernel.

  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling
  """

  NEAREST = 0
  LINEAR = 1
  LANCZOS3 = 2
  LANCZOS5 = 3
  CUBIC = 4

  # Caution: The current resize implementation assumes that the resize kernels
  # are interpolating, i.e. for the identity warp the output equals the input.
  # This is not true for, e.g. a Gaussian kernel, so if such kernels are added
  # the implementation will need to be changed.

  @staticmethod
  def from_string(s: str):
    if s == 'nearest':
      return ResizeMethod.NEAREST
    if s in ['linear', 'bilinear', 'trilinear', 'triangle']:
      return ResizeMethod.LINEAR
    elif s == 'lanczos3':
      return ResizeMethod.LANCZOS3
    elif s == 'lanczos5':
      return ResizeMethod.LANCZOS5
    elif s in ['cubic', 'bicubic', 'tricubic']:
      return ResizeMethod.CUBIC
    else:
      raise ValueError(f'Unknown resize method "{s}"')

_kernels = {
    ResizeMethod.LINEAR: _fill_triangle_kernel,
    ResizeMethod.LANCZOS3: lambda x: _fill_lanczos_kernel(3., x),
    ResizeMethod.LANCZOS5: lambda x: _fill_lanczos_kernel(5., x),
    ResizeMethod.CUBIC: _fill_keys_cubic_kernel
}


# scale and translation here are scalar elements of an np.array, what is the
# correct type annotation?
def scale_and_translate(image, shape: core.Shape,
                        spatial_dims: Sequence[int],
                        scale, translation,
                        method: str | ResizeMethod,
                        antialias: bool = True,
                        precision=lax.Precision.HIGHEST):
  """Apply a scale and translation to an image.

  Generates a new image of shape 'shape' by resampling from the input image
  using the sampling method corresponding to method. For 2D images, this
  operation transforms a location in the input images, (x, y), to a location
  in the output image according to::

    (x * scale[1] + translation[1], y * scale[0] + translation[0])

  (Note the *inverse* warp is used to generate the sample locations.)
  Assumes half-centered pixels, i.e the pixel at integer location ``row, col``
  has coordinates ``y, x = row + 0.5, col + 0.5``, and similarly for other input
  image dimensions.

  If an output location(pixel) maps to an input sample location that is outside
  the input boundaries then the value for the output location will be set to
  zero.

  The ``method`` argument expects one of the following resize methods:

  ``ResizeMethod.LINEAR``, ``"linear"``, ``"bilinear"``, ``"trilinear"``,
    ``"triangle"`` `Linear interpolation`_. If ``antialias`` is ``True``, uses a
    triangular filter when downsampling.

  ``ResizeMethod.CUBIC``, ``"cubic"``, ``"bicubic"``, ``"tricubic"``
    `Cubic interpolation`_, using the Keys cubic kernel.

  ``ResizeMethod.LANCZOS3``, ``"lanczos3"``
    `Lanczos resampling`_, using a kernel of radius 3.

  ``ResizeMethod.LANCZOS5``, ``"lanczos5"``
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  Args:
    image: a JAX array.
    shape: the output shape, as a sequence of integers with length equal to the
      number of dimensions of `image`.
    spatial_dims: A length K tuple specifying the spatial dimensions that the
      passed scale and translation should be applied to.
    scale: A [K] array with the same number of dimensions as image, containing
      the scale to apply in each dimension.
    translation: A [K] array with the same number of dimensions as image,
      containing the translation to apply in each dimension.
    method: the resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC.
    antialias: Should an antialiasing filter be used when downsampling? Defaults
      to ``True``. Has no effect when upsampling.

  Returns:
    The scale and translated image.
  """
  shape = core.canonicalize_shape(shape)
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  if isinstance(method, str):
    method = ResizeMethod.from_string(method)
  if method == ResizeMethod.NEAREST:
    # Nearest neighbor is currently special-cased for straight resize, so skip
    # for now.
    raise ValueError('Nearest neighbor resampling is not currently supported '
                     'for scale_and_translate.')
  assert isinstance(method, ResizeMethod)

  kernel = _kernels[method]
  image, = promote_dtypes_inexact(image)
  scale, translation = promote_dtypes_inexact(scale, translation)
  return _scale_and_translate(image, shape, spatial_dims, scale, translation,
                              kernel, antialias, precision)


def _resize_nearest(x, output_shape: core.Shape):
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  spatial_dims = tuple(i for i in range(len(input_shape))
                       if not core.definitely_equal(input_shape[i], output_shape[i]))
  for d in spatial_dims:
    m = input_shape[d]
    n = output_shape[d]
    offsets = (jnp.arange(n, dtype=np.float32) + 0.5) * core.dimension_as_value(m) / core.dimension_as_value(n)
    # TODO(b/206898375): this computation produces the wrong result on
    # CPU and GPU when using float64. Use float32 until the bug is fixed.
    offsets = jnp.floor(offsets.astype(np.float32)).astype(np.int32)
    indices = [slice(None)] * len(input_shape)
    indices[d] = offsets
    x = x[tuple(indices)]
  return x


@partial(jit, static_argnums=(1, 2, 3, 4))
def _resize(image, shape: core.Shape, method: str | ResizeMethod,
            antialias: bool, precision):
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  if isinstance(method, str):
    method = ResizeMethod.from_string(method)
  if method == ResizeMethod.NEAREST:
    return _resize_nearest(image, shape)
  assert isinstance(method, ResizeMethod)
  kernel = _kernels[method]

  image, = promote_dtypes_inexact(image)
  # Skip dimensions that have scale=1 and translation=0, this is only possible
  # since all of the current resize methods (kernels) are interpolating, so the
  # output = input under an identity warp.
  spatial_dims = tuple(i for i in range(len(shape))
                       if not core.definitely_equal(image.shape[i], shape[i]))
  scale = [1.0 if core.definitely_equal(shape[d], 0) else core.dimension_as_value(shape[d]) / core.dimension_as_value(image.shape[d])
           for d in spatial_dims]
  return _scale_and_translate(image, shape, spatial_dims,
                              scale, [0.] * len(spatial_dims), kernel,
                              antialias, precision)


def resize(image, shape: core.Shape, method: str | ResizeMethod,
           antialias: bool = True,
           precision = lax.Precision.HIGHEST):
  """Image resize.

  The ``method`` argument expects one of the following resize methods:

  ``ResizeMethod.NEAREST``, ``"nearest"``
    `Nearest neighbor interpolation`_. The values of ``antialias`` and
    ``precision`` are ignored.

  ``ResizeMethod.LINEAR``, ``"linear"``, ``"bilinear"``, ``"trilinear"``, ``"triangle"``
    `Linear interpolation`_. If ``antialias`` is ``True``, uses a triangular
    filter when downsampling.

  ``ResizeMethod.CUBIC``, ``"cubic"``, ``"bicubic"``, ``"tricubic"``
    `Cubic interpolation`_, using the Keys cubic kernel.

  ``ResizeMethod.LANCZOS3``, ``"lanczos3"``
    `Lanczos resampling`_, using a kernel of radius 3.

  ``ResizeMethod.LANCZOS5``, ``"lanczos5"``
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Nearest neighbor interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  Args:
    image: a JAX array.
    shape: the output shape, as a sequence of integers with length equal to
      the number of dimensions of `image`. Note that :func:`resize` does not
      distinguish spatial dimensions from batch or channel dimensions, so this
      includes all dimensions of the image. To represent a batch or a channel
      dimension, simply leave that element of the shape unchanged.
    method: the resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC.
    antialias: should an antialiasing filter be used when downsampling? Defaults
      to ``True``. Has no effect when upsampling.
  Returns:
    The resized image.
  """
  return _resize(image, core.canonicalize_shape(shape), method, antialias,
                 precision)

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Private utilities for locally-connected layers."""

from keras import backend
from keras.utils import conv_utils
import numpy as np
import tensorflow.compat.v2 as tf


def get_locallyconnected_mask(
    input_shape, kernel_shape, strides, padding, data_format
):
    """Return a mask representing connectivity of a locally-connected operation.

    This method returns a masking numpy array of 0s and 1s (of type `np.float32`)
    that, when element-wise multiplied with a fully-connected weight tensor, masks
    out the weights between disconnected input-output pairs and thus implements
    local connectivity through a sparse fully-connected weight tensor.

    Assume an unshared convolution with given parameters is applied to an input
    having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
    to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
    by layer parameters such as `strides`).

    This method returns a mask which can be broadcast-multiplied (element-wise)
    with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer between
    (N+1)-D activations (N spatial + 1 channel dimensions for input and output)
    to make it perform an unshared convolution with given `kernel_shape`,
    `strides`, `padding` and `data_format`.

    Args:
      input_shape: tuple of size N: `(d_in1, ..., d_inN)` spatial shape of the
        input.
      kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
        receptive field.
      strides: tuple of size N, strides along each spatial dimension.
      padding: type of padding, string `"same"` or `"valid"`.
      data_format: a string, `"channels_first"` or `"channels_last"`.

    Returns:
      a `np.float32`-type `np.ndarray` of shape
      `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
      if `data_format == `"channels_first"`, or
      `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
      if `data_format == "channels_last"`.

    Raises:
      ValueError: if `data_format` is neither `"channels_first"` nor
                  `"channels_last"`.
    """
    mask = conv_utils.conv_kernel_mask(
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        strides=strides,
        padding=padding,
    )

    ndims = int(mask.ndim / 2)

    if data_format == "channels_first":
        mask = np.expand_dims(mask, 0)
        mask = np.expand_dims(mask, -ndims - 1)

    elif data_format == "channels_last":
        mask = np.expand_dims(mask, ndims)
        mask = np.expand_dims(mask, -1)

    else:
        raise ValueError("Unrecognized data_format: " + str(data_format))

    return mask


def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
    """Apply N-D convolution with un-shared weights using a single matmul call.

    This method outputs `inputs . (kernel * kernel_mask)`
    (with `.` standing for matrix-multiply and `*` for element-wise multiply)
    and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
    hence perform the same operation as a convolution with un-shared
    (the remaining entries in `kernel`) weights. It also does the necessary
    reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.

    Args:
        inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
          d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
        kernel: the unshared weights for N-D convolution,
            an (N+2)-D tensor of shape: `(d_in1, ..., d_inN, channels_in, d_out2,
              ..., d_outN, channels_out)` or `(channels_in, d_in1, ..., d_inN,
              channels_out, d_out2, ..., d_outN)`, with the ordering of channels
              and spatial dimensions matching that of the input. Each entry is the
              weight between a particular input and output location, similarly to
              a fully-connected weight matrix.
        kernel_mask: a float 0/1 mask tensor of shape: `(d_in1, ..., d_inN, 1,
          d_out2, ..., d_outN, 1)` or `(1, d_in1, ..., d_inN, 1, d_out2, ...,
          d_outN)`, with the ordering of singleton and spatial dimensions matching
          that of the input. Mask represents the connectivity pattern of the layer
          and is
             precomputed elsewhere based on layer parameters: stride, padding, and
               the receptive field shape.
        output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
          d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
          spatial dimensions matching that of the input.

    Returns:
        Output (N+2)-D tensor with shape `output_shape`.
    """
    inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))

    kernel = kernel_mask * kernel
    kernel = make_2d(kernel, split_dim=backend.ndim(kernel) // 2)

    output_flat = tf.matmul(inputs_flat, kernel, b_is_sparse=True)
    output = backend.reshape(
        output_flat,
        [
            backend.shape(output_flat)[0],
        ]
        + output_shape.as_list()[1:],
    )
    return output


def local_conv_sparse_matmul(
    inputs, kernel, kernel_idxs, kernel_shape, output_shape
):
    """Apply N-D convolution with un-shared weights using a single sparse matmul.

    This method outputs `inputs . tf.sparse.SparseTensor(indices=kernel_idxs,
    values=kernel, dense_shape=kernel_shape)`, with `.` standing for
    matrix-multiply. It also reshapes `inputs` to 2-D and `output` to (N+2)-D.

    Args:
        inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
          d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
        kernel: a 1-D tensor with shape `(len(kernel_idxs),)` containing all the
          weights of the layer.
        kernel_idxs:  a list of integer tuples representing indices in a sparse
          matrix performing the un-shared convolution as a matrix-multiply.
        kernel_shape: a tuple `(input_size, output_size)`, where `input_size =
          channels_in * d_in1 * ... * d_inN` and `output_size = channels_out *
          d_out1 * ... * d_outN`.
        output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
          d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
          spatial dimensions matching that of the input.

    Returns:
        Output (N+2)-D dense tensor with shape `output_shape`.
    """
    inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))
    output_flat = tf.sparse.sparse_dense_matmul(
        sp_a=tf.SparseTensor(kernel_idxs, kernel, kernel_shape),
        b=inputs_flat,
        adjoint_b=True,
    )
    output_flat_transpose = backend.transpose(output_flat)

    output_reshaped = backend.reshape(
        output_flat_transpose,
        [
            backend.shape(output_flat_transpose)[0],
        ]
        + output_shape.as_list()[1:],
    )
    return output_reshaped


def make_2d(tensor, split_dim):
    """Reshapes an N-dimensional tensor into a 2D tensor.

    Dimensions before (excluding) and after (including) `split_dim` are grouped
    together.

    Args:
      tensor: a tensor of shape `(d0, ..., d(N-1))`.
      split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).

    Returns:
      Tensor of shape
      `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
    """
    shape = tf.shape(tensor)
    in_dims = shape[:split_dim]
    out_dims = shape[split_dim:]

    in_size = tf.reduce_prod(in_dims)
    out_size = tf.reduce_prod(out_dims)

    return tf.reshape(tensor, (in_size, out_size))

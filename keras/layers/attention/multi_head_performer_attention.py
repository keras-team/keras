# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras-based multi-head Performer attention layer."""

import collections
import math
import string

import numpy as np
import tensorflow.compat.v2 as tf

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.layers import activation
from keras.layers import core
from keras.layers import regularization
from keras.utils import tf_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

_CHR_IDX = string.ascii_lowercase

# -------------------- PERFORMERS AUXILIARY FUNCTIONS -------------------------#


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


BIG_CONSTANT = 1e8


def next_seed(current_seed):
  if current_seed is None:
    return None
  else:
    return current_seed + 1


def create_projection_matrix(nb_random_projections, dim, seed=0, scaling=0):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{dim} or length taken from the \chi(dim) distribution (in the latter
  case marginal distributions of the projections are dim-dimensional Gaussian
  vectors with associated identity covariance matrix).

  Args:
    nb_random_projections: number of random projections.
    dim: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{dim}, 0 if the lengths of random projections should follow
      \chi(dim) distribution.

  Returns:
    The matrix of random projections of the shape [nb_random_projections, dim].
  """
  if nb_random_projections == 0:
    return None
  nb_full_blocks = nb_random_projections // dim
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    unstructured_block = tf.random.normal((dim, dim), seed=current_seed)
    q, _ = tf.linalg.qr(unstructured_block)
    q = tf.transpose(q)
    block_list.append(q)
    current_seed = next_seed(current_seed)
  remaining_rows = nb_random_projections - nb_full_blocks * dim
  if remaining_rows > 0:
    unstructured_block = tf.random.normal((dim, dim), seed=current_seed)
    q, _ = tf.linalg.qr(unstructured_block)
    q = tf.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = tf.concat(block_list, 0)
  current_seed = next_seed(current_seed)

  if scaling == 0:
    squares = tf.math.square(
        tf.random.normal((nb_random_projections, dim), seed=current_seed))
    squared_lengths = tf.math.reduce_sum(squares, axis=1)
    multiplier = tf.math.sqrt(squared_lengths)
  elif scaling == 1:
    multiplier = tf.math.sqrt(float(dim)) * tf.ones((nb_random_projections))
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.

  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  if projection_matrix is None:
    return tf.nn.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], projection_matrix.dtype))
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.nn.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  projection_matrix = tf.cast(projection_matrix, data.dtype)
  data_normalizer = 1.0 / tf.math.sqrt(
      (tf.math.sqrt(tf.dtypes.cast(data.shape[-1], data.dtype))))
  ratio = 1.0 / tf.math.sqrt(
      tf.dtypes.cast(projection_matrix.shape[0], data.dtype))
  data_dash = tf.einsum("blhd,md->blhm", data_normalizer * data,
                        projection_matrix)
  diag_data = tf.math.square(data)
  diag_data = tf.math.reduce_sum(diag_data, axis=-1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = tf.expand_dims(diag_data, axis=-1)
  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
  else:
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash)) +
        numerical_stabilizer)

  return data_dash


def cossim_kernel_transformation(data,
                                 is_query,
                                 projection_matrix=None,
                                 numerical_stabilizer=0.0,
                                 randomized=False):
  """Computes features for the softmax kernel with FAVOR+ cossim mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    randomized: whether randomized version of the cos similarity is used.

  Returns:
    Corresponding kernel feature map.
  """
  if is_query:
    r = tf.math.sqrt(tf.dtypes.cast(data.shape[-1], data.dtype))
    if randomized:
      projection_matrix = tf.cast(projection_matrix, data.dtype)
      ratio = 1.0 / tf.math.sqrt(
          tf.cast(tf.shape(projection_matrix)[0], data.dtype))
      return ratio * (
          tf.einsum("blhd,md->blhm", r * tf.math.l2_normalize(data, axis=[-1]),
                    projection_matrix) +
          tf.cast(numerical_stabilizer, data.dtype))
    else:
      return r * tf.math.l2_normalize(data, axis=[-1])
  else:
    if randomized:
      projection_matrix = tf.cast(projection_matrix, data.dtype)
      ratio = 1.0 / tf.math.sqrt(
          tf.cast(tf.shape(projection_matrix)[0], data.dtype))
      return ratio * tf.einsum("blhd,md->blhm",
                               tf.math.l2_normalize(data, axis=[-1]),
                               projection_matrix)
    else:
      return tf.math.l2_normalize(data, axis=[-1])


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
  return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in noncausal attention.
  """
  ks_sum = tf.reduce_sum(ks, axis=0)
  return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)


@tf.custom_gradient
def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])
    result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, ...])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    gr_sums = sums

    q_grads = []
    k_grads = []
    v_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, ...])
      grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
      k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, ...])
      v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, ...])
      gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)
    v_grads = tf.concat(v_grads[::-1], axis=0)

    return q_grads, k_grads, v_grads

  return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = tf.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, ...])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    k_grad = tf.zeros_like(ks[0])

    gr_sums = sums

    q_grads = []
    k_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, ...])
      k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
      k_grads.append(k_grad[None, ...])
      gr_sums = gr_sums - ks[index]

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)

    return q_grads, k_grads

  return result, grad


_ITER_CHUNK_SIZE = 64


def chunked_causal_numerator_func(qs, ks, vs):
  """Forward pass of not-normalized FAVOR causal attention using chunks.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    Last prefix sum state.
  """

  result = []
  sums = tf.zeros_like(ks[0])[..., None] * tf.zeros_like(vs[0])[..., None, :]

  for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):

    end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)

    chunk = tf.einsum("sijk,sijl->sijkl", ks[start_index:end_index],
                      vs[start_index:end_index])
    chunk = sums[None, ...] + tf.math.cumsum(chunk, axis=0)
    sums = chunk[-1]

    result_elem = tf.einsum("sijkl,sijk->sijl", chunk,
                            qs[start_index:end_index])
    result.append(result_elem)

  result = tf.concat(result, axis=0)

  return result, sums


def chunked_causal_numerator_grad(qs, ks, vs, sums, res_grad):
  """Backward pass of not-normalized FAVOR causal attention using chunks.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
    sums: last prefix sum state.
    res_grad: gradient of the last prefix sum state.

  Returns:
    Gradient of qs.
    Gradient of ks.
    Gradient of vs.
  """

  grads = tf.zeros_like(ks[0])[..., None] * tf.zeros_like(vs[0])[..., None, :]
  gr_sums = sums

  q_grads = []
  k_grads = []
  v_grads = []

  res_grad = res_grad[::-1]
  qs_rev = qs[::-1]
  ks_rev = ks[::-1]
  vs_rev = vs[::-1]

  for start_index in range(0, qs_rev.shape[0], _ITER_CHUNK_SIZE):

    end_index = min(qs_rev.shape[0], start_index + _ITER_CHUNK_SIZE)

    chunk = tf.einsum("sijk,sijl->sijkl", ks_rev[start_index:end_index - 1],
                      vs_rev[start_index:end_index - 1])
    chunk = tf.concat([tf.zeros_like(gr_sums[None, ...]), chunk], axis=0)
    chunk = gr_sums[None, ...] - tf.math.cumsum(chunk, axis=0)
    gr_sums = chunk[-1] - tf.einsum("ijk,ijl->ijkl", ks_rev[end_index - 1],
                                    vs_rev[end_index - 1])

    q_grads.append(
        tf.einsum("sijkl,sijl->sijk", chunk, res_grad[start_index:end_index]))

    grad_chunk = tf.einsum("sijk,sijl->sijkl", qs_rev[start_index:end_index],
                           res_grad[start_index:end_index])
    grad_chunk = grads[None, ...] + tf.math.cumsum(grad_chunk, axis=0)
    grads = grad_chunk[-1]

    k_grads.append(
        tf.einsum("sijkl,sijl->sijk", grad_chunk,
                  vs_rev[start_index:end_index]))
    v_grads.append(
        tf.einsum("sijkl,sijk->sijl", grad_chunk,
                  ks_rev[start_index:end_index]))

  q_grads = tf.concat(q_grads, axis=0)[::-1]
  k_grads = tf.concat(k_grads, axis=0)[::-1]
  v_grads = tf.concat(v_grads, axis=0)[::-1]

  return q_grads, k_grads, v_grads


@tf.custom_gradient  # ALLOW_CUSTOM_GRADIENT
def chunked_causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V using chunks.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """
  result, sums = chunked_causal_numerator_func(qs, ks, vs)

  def grad(res_grad):
    return chunked_causal_numerator_grad(qs, ks, vs, sums, res_grad)

  return result, grad


def chunked_causal_denominator_func(qs, ks):
  """Forward pass of FAVOR normalizer in causal attention using chunks.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    Last prefix sum state.
  """

  result = []
  sums = tf.zeros_like(ks[0])

  for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):

    end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)

    chunk = ks[start_index:end_index]
    chunk = sums[None, ...] + tf.math.cumsum(chunk, axis=0)
    sums = chunk[-1]

    result_elem = tf.reduce_sum(qs[start_index:end_index] * chunk, axis=3)
    result.append(result_elem)

  result = tf.concat(result, axis=0)

  return result, sums


def chunked_causal_denominator_grad(qs, ks, sums, res_grad):
  """Backward pass of FAVOR normalizer in causal attention using chunks.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    sums: last prefix sum state.
    res_grad: last prefix sum state's grad.

  Returns:
    Gradients of qs.
    Gradients of ks.
  """

  k_grad = tf.zeros_like(ks[0])
  gr_sums = sums

  q_grads = []
  k_grads = []

  res_grad = res_grad[::-1]
  qs_rev = qs[::-1]
  ks_rev = ks[::-1]

  for start_index in range(0, qs_rev.shape[0], _ITER_CHUNK_SIZE):

    end_index = min(qs_rev.shape[0], start_index + _ITER_CHUNK_SIZE)

    chunk = ks_rev[start_index:end_index - 1]
    chunk = tf.concat([tf.zeros_like(gr_sums[None, ...]), chunk], axis=0)
    chunk = gr_sums[None, ...] - tf.math.cumsum(chunk, axis=0)
    gr_sums = chunk[-1] - ks_rev[end_index - 1]

    q_grads.append(
        tf.einsum("sijk,sij->sijk", chunk, res_grad[start_index:end_index]))

    k_grad_chunk = tf.einsum("sijk,sij->sijk", qs_rev[start_index:end_index],
                             res_grad[start_index:end_index])
    k_grad_chunk = k_grad[None, ...] + tf.math.cumsum(k_grad_chunk, axis=0)
    k_grad = k_grad_chunk[-1]

    k_grads.append(k_grad_chunk)

  q_grads = tf.concat(q_grads, axis=0)[::-1]
  k_grads = tf.concat(k_grads, axis=0)[::-1]

  return q_grads, k_grads


@tf.custom_gradient  # ALLOW_CUSTOM_GRADIENT
def chunked_causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention using chunks.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in causal attention.
  """

  result, sums = chunked_causal_denominator_func(qs, ks)

  def grad(res_grad):
    return chunked_causal_denominator_grad(qs, ks, sums, res_grad)

  return result, grad


def favor_attention(query,
                    key,
                    value,
                    paddings,
                    kernel_transformation,
                    causal,
                    projection_matrix=None,
                    use_chunked_causal=False):
  """Computes FAVOR normalized attention.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    paddings: paddings tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.
    use_chunked_causal: whether to use (faster) chunked causal attention.

  Returns:
    FAVOR normalized attention.
  """
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  if paddings is not None:
    b, l, h, m = get_shape_list(key_prime)
    paddings = tf.tile(tf.reshape(paddings, [b, l, 1, 1]), [1, 1, h, m])
    key_prime *= tf.cast(1.0 - paddings, key_prime.dtype)
  query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
  key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
  value = tf.transpose(value, [1, 0, 2, 3])  # [L,B,H,D]
  # TODO(kchoro): Get rid of the transpose operations, at least in the
  # bidirectional variant.

  if causal:
    if use_chunked_causal:
      av_attention = chunked_causal_numerator(query_prime, key_prime, value)
      attention_normalizer = chunked_causal_denominator(query_prime, key_prime)
    else:
      av_attention = causal_numerator(query_prime, key_prime, value)
      attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
  # TODO(kchoro): Add more comments.
  av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
  attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
  attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
  return av_attention / attention_normalizer


# -----------------------------------------------------------------------------#


def _build_attention_equation(rank, attn_axes):
  """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
    dims>, <query attention dims>, num_heads, channels)`

    Args:
      rank: Rank of query, key, value tensors.
      attn_axes: List/tuple of axes, `[-1, rank)`, that attention will be
        applied to.

    Returns:
      Einsum equations.
  """
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ""
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (
      source_notation,
      target_notation,
      product_notation,
  )
  attn_scores_rank = len(product_notation)
  combine_equation = "%s,%s->%s" % (
      product_notation,
      source_notation,
      target_notation,
  )
  return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
  """Builds an einsum equation for projections inside multi-head attention."""
  input_str = ""
  kernel_str = ""
  output_str = ""
  bias_axes = ""
  letter_offset = 0
  for i in range(free_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    output_str += char

  letter_offset += free_dims
  for i in range(bound_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    kernel_str += char

  letter_offset += bound_dims
  for i in range(output_dims):
    char = _CHR_IDX[i + letter_offset]
    kernel_str += char
    output_str += char
    bias_axes += char
  equation = f"{input_str},{kernel_str}->{output_str}"

  return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


@keras_export("keras.layers.MultiHeadPerformerAttention")
class MultiHeadPerformerAttention(Layer):
  """MultiHeadPerformerAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.

    When using `MultiHeadPerformerAttention` inside a custom layer, the custom
    layer must implement its own `build()` method and call
    `MultiHeadPerformerAttention`'s `_build_from_signature()` there.
    This enables weights to be restored correctly when the model is loaded.

    Examples:

    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.

    >>> layer = MultiHeadPerformerAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

    >>> layer = MultiHeadPerformerAttention(
    ...     num_heads=2, key_dim=2, attention_axes=(2, 3))
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer(input_tensor, input_tensor)
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Args:
      num_heads: Number of attention heads.
      key_dim: Size of each attention head for query and key.
      value_dim: Size of each attention head for value.
      dropout: Dropout probability.
      use_bias: Boolean, whether the dense layers use bias vectors/matrices.
      output_shape: The expected shape of an output tensor, besides the batch
        and sequence dims. If not specified, projects back to the key feature
        dim.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      performer_type: Type of the Performer-attention to be used ('' if regular
        attention should be applied). Potential values are: '', 'softmax',
        'relu'.  Call arguments:
      query: Query `Tensor` of shape `(B, T, dim)`.
      value: Value `Tensor` of shape `(B, S, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      return_attention_scores: A boolean to indicate whether the output should
        be `(attention_output, attention_scores)` if `True`, or
        `attention_output` if `False`. Defaults to `False`.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
        Defaults to either using the training mode of the parent layer/model, or
        False (inference) if there is no parent layer.
      use_causal_mask: A boolean to indicate whether to apply a causal mask to
        prevent tokens from attending to future tokens (e.g., used in a decoder
        Transformer).

    Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are projected to the shape specified by `output_shape`.
      attention_scores: [Optional] multi-head attention coefficients over
        attention axes.
  """

  def __init__(
      self,
      num_heads,
      key_dim,
      value_dim=None,
      dropout=0.0,
      use_bias=True,
      output_shape=None,
      attention_axes=None,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      performer_type="",
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.supports_masking = True
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim if value_dim else key_dim
    self._dropout = dropout
    self._use_bias = use_bias
    self._output_shape = output_shape
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._kernel_regularizer = regularizers.get(kernel_regularizer)
    self._bias_regularizer = regularizers.get(bias_regularizer)
    self._activity_regularizer = regularizers.get(activity_regularizer)
    self._kernel_constraint = constraints.get(kernel_constraint)
    self._bias_constraint = constraints.get(bias_constraint)
    self._performer_type = performer_type
    if attention_axes is not None and not isinstance(attention_axes,
                                                     collections.abc.Sized):
      self._attention_axes = (attention_axes,)
    else:
      self._attention_axes = attention_axes
    self._built_from_signature = False
    self._query_shape, self._key_shape, self._value_shape = None, None, None

  def get_config(self):
    config = {
        "num_heads":
            self._num_heads,
        "key_dim":
            self._key_dim,
        "value_dim":
            self._value_dim,
        "dropout":
            self._dropout,
        "use_bias":
            self._use_bias,
        "output_shape":
            self._output_shape,
        "attention_axes":
            self._attention_axes,
        "kernel_initializer":
            initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            constraints.serialize(self._bias_constraint),
        "query_shape":
            self._query_shape,
        "key_shape":
            self._key_shape,
        "value_shape":
            self._value_shape,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # If the layer has a different build() function from the Keras default,
    # we need to trigger the customized build to create weights.
    query_shape = config.pop("query_shape")
    key_shape = config.pop("key_shape")
    value_shape = config.pop("value_shape")
    layer = cls(**config)
    if None in [query_shape, key_shape, value_shape]:
      logging.warning(
          "One of dimensions of the input shape is missing. It "
          "should have been memorized when the layer was serialized. "
          "%s is created without weights.",
          str(cls),
      )
    else:
      layer._build_from_signature(query_shape, value_shape, key_shape)
    return layer

  def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.

        Once the method is called, self._built_from_signature will be set to
        True.

        Args:
          query: Query tensor or TensorShape.
          value: Value tensor or TensorShape.
          key: Key tensor or TensorShape.
    """
    self._built_from_signature = True
    if hasattr(query, "shape"):
      self._query_shape = tf.TensorShape(query.shape)
    else:
      self._query_shape = tf.TensorShape(query)
    if hasattr(value, "shape"):
      self._value_shape = tf.TensorShape(value.shape)
    else:
      self._value_shape = tf.TensorShape(value)
    if key is None:
      self._key_shape = self._value_shape
    elif hasattr(key, "shape"):
      self._key_shape = tf.TensorShape(key.shape)
    else:
      self._key_shape = tf.TensorShape(key)

    # Any setup work performed only once should happen in an `init_scope`
    # to avoid creating symbolic Tensors that will later pollute any eager
    # operations.
    with tf_utils.maybe_init_scope(self):
      free_dims = self._query_shape.rank - 1
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          free_dims, bound_dims=1, output_dims=2)
      self._query_dense = core.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="query",
          **self._get_common_kwargs_for_sublayer(),
      )
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          self._key_shape.rank - 1, bound_dims=1, output_dims=2)
      self._key_dense = core.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="key",
          **self._get_common_kwargs_for_sublayer(),
      )
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          self._value_shape.rank - 1, bound_dims=1, output_dims=2)
      self._value_dense = core.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._value_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="value",
          **self._get_common_kwargs_for_sublayer(),
      )

      # Builds the attention computations for multi-head dot product
      # attention.  These computations could be wrapped into the keras
      # attention layer once it supports mult-head einsum computations.
      self._build_attention(output_rank)
      self._output_dense = self._make_output_dense(
          free_dims,
          self._get_common_kwargs_for_sublayer(),
          "attention_output",
      )

  def _get_common_kwargs_for_sublayer(self):
    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
    )
    # Create new clone of kernel/bias initializer, so that we don't reuse
    # the initializer instance, which could lead to same init value since
    # initializer is stateless.
    kernel_initializer = self._kernel_initializer.__class__.from_config(
        self._kernel_initializer.get_config())
    bias_initializer = self._bias_initializer.__class__.from_config(
        self._bias_initializer.get_config())
    common_kwargs["kernel_initializer"] = kernel_initializer
    common_kwargs["bias_initializer"] = bias_initializer
    return common_kwargs

  def _make_output_dense(self, free_dims, common_kwargs, name=None):
    """Builds the output projection matrix.

        Args:
          free_dims: Number of free dimensions for einsum equation building.
          common_kwargs: Common keyword arguments for einsum layer.
          name: Name for the projection layer.

        Returns:
          Projection layer.
    """
    if self._output_shape:
      if not isinstance(self._output_shape, collections.abc.Sized):
        output_shape = [self._output_shape]
      else:
        output_shape = self._output_shape
    else:
      output_shape = [self._query_shape[-1]]
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=2, output_dims=len(output_shape))
    return core.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1, output_shape),
        bias_axes=bias_axes if self._use_bias else None,
        name=name,
        **common_kwargs,
    )

  def _build_attention(self, rank):
    """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
          rank: the rank of query, key, value tensors.
    """
    if self._attention_axes is None:
      self._attention_axes = tuple(range(1, rank - 2))
    else:
      self._attention_axes = tuple(self._attention_axes)
    (
        self._dot_product_equation,
        self._combine_equation,
        attn_scores_rank,
    ) = _build_attention_equation(
        rank, attn_axes=self._attention_axes)
    norm_axes = tuple(
        range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = activation.Softmax(axis=norm_axes)
    self._dropout_layer = regularization.Dropout(rate=self._dropout)

  def _masked_softmax(self, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, T, S]
    if attention_mask is not None:
      # The expand dim happens starting from the `num_heads` dimension,
      # (<batch_dims>, num_heads, <query_attention_dims,
      # key_attention_dims>)
      mask_expansion_axis = -len(self._attention_axes) * 2 - 1
      for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
        attention_mask = tf.expand_dims(
            attention_mask, axis=mask_expansion_axis)
    return self._softmax(attention_scores, attention_mask)

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         attention_mask=None,
                         training=None):
    """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. It is generally not needed if the
            `query` and `value` (and/or `key`) are masked.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
    """
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum(self._dot_product_equation, key, query)

    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum(self._combine_equation,
                                 attention_scores_dropout, value)

    return attention_output, attention_scores

  def _compute_performers_attention(self,
                                    query,
                                    key,
                                    value,
                                    use_causal_mask=False,
                                    training=None,
                                    cache=None,
                                    num_rand_features=0):
    """Applies Performers' Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. It is generally not needed if the
            `query` and `value` (and/or `key`) are masked.
          use_causal_mask: Python boolean indicating whether causal masking
            should be applied.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
          cache: dictionary storing Performers' superstate.
          num_rand_features: number of random features used.

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
    """
    paddings = None
    if self._performer_type == "relu":
      kernel_transformation = relu_kernel_transformation
      projection_matrix = None
    elif self._performer_type == "softmax":
      kernel_transformation = softmax_kernel_transformation
      projection_matrix = create_projection_matrix(num_rand_features,
                                                   query.shape[-1], 0)
    else:
      logging.info("Attention type is not supported, returning query.")
      return query, None

    if not training and use_causal_mask:
      query_prime = kernel_transformation(query, True, projection_matrix)
      key_prime = kernel_transformation(key, False, projection_matrix)
      # update superstate (cache)
      res_cache_numerator = tf.einsum("...lhm,...lhd->...hmd", key_prime, value)
      cache["num"] = cache["num"] + res_cache_numerator
      cache["den"] = cache["den"] + tf.squeeze(key_prime, axis=-3)
      # computing embedding for the newly coming token
      x_num = tf.einsum("...lhm,...hmd->...lhd", query_prime, cache["num"])
      x_den = tf.einsum("...lhm,...hm->...lh", query_prime, cache["den"])
      x_den = tf.expand_dims(x_den, len(x_den.shape))
      attention_output = x_num / x_den
    else:
      attention_output = favor_attention(query, key, value, paddings,
                                         kernel_transformation, use_causal_mask,
                                         projection_matrix)
    return attention_output, cache

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None,
           use_causal_mask=False,
           cache=None,
           num_rand_features=0):
    attention_mask = self._compute_attention_mask(
        query,
        value,
        key=key,
        attention_mask=attention_mask,
        use_causal_mask=use_causal_mask,
    )

    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    query_is_ragged = isinstance(query, tf.RaggedTensor)
    if query_is_ragged:
      query_lengths = query.nested_row_lengths()
      query = query.to_tensor()

    key_is_ragged = isinstance(key, tf.RaggedTensor)
    value_is_ragged = isinstance(value, tf.RaggedTensor)
    if key_is_ragged and value_is_ragged:
      # Ensure they have the same shape.
      bounding_shape = tf.math.maximum(key.bounding_shape(),
                                       value.bounding_shape())
      key = key.to_tensor(shape=bounding_shape)
      value = value.to_tensor(shape=bounding_shape)
    elif key_is_ragged:
      key = key.to_tensor(shape=tf.shape(value))
    elif value_is_ragged:
      value = value.to_tensor(shape=tf.shape(key))

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, H]
    value = self._value_dense(value)

    if self._performer_type == "":
      attention_output, attention_scores = self._compute_attention(
          query, key, value, attention_mask, training=training)
    else:
      attention_output, cache = self._compute_performers_attention(
          query,
          key,
          value,
          use_causal_mask,
          training=training,
          cache=cache,
          num_rand_features=num_rand_features)

    attention_output = self._output_dense(attention_output)

    if query_is_ragged:
      attention_output = tf.RaggedTensor.from_tensor(
          attention_output, lengths=query_lengths)

    if self._performer_type == "":
      if return_attention_scores:
        return attention_output, attention_scores
      return attention_output
    else:
      return attention_output, cache

  def _compute_attention_mask(self,
                              query,
                              value,
                              key=None,
                              attention_mask=None,
                              use_causal_mask=False):
    """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions.
          use_causal_mask: A boolean to indicate whether to apply a causal mask
            to prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

        Returns:
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions, based on the Keras masks of the
            `query`, `key`, `value`, and `attention_mask` tensors, and the
            causal mask if `use_causal_mask=True`.
    """
    query_mask = getattr(query, "_keras_mask", None)
    value_mask = getattr(value, "_keras_mask", None)
    key_mask = getattr(key, "_keras_mask", None)
    auto_mask = None
    if query_mask is not None:
      query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
      # B = batch size, T = max query length
      auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
    if value_mask is not None:
      value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
      # B = batch size, S == max value length
      mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
      auto_mask = mask if auto_mask is None else auto_mask & mask
    if key_mask is not None:
      key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
      # B == batch size, S == max key length == max value length
      mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
      auto_mask = mask if auto_mask is None else auto_mask & mask
    if use_causal_mask:
      # the shape of the causal mask is [1, T, S]
      mask = self._compute_causal_mask(query, value)
      auto_mask = mask if auto_mask is None else auto_mask & mask
    if auto_mask is not None:
      # merge attention_mask & automatic mask, to shape [B, T, S]
      attention_mask = (
          auto_mask if attention_mask is None else
          tf.cast(attention_mask, bool) & auto_mask)
    return attention_mask

  def _compute_causal_mask(self, query, value=None):
    """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean `Tensor` equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```
        Args:
          query: query `Tensor` of shape `(B, T, ...)`.
          value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
            query).

        Returns:
          mask: a boolean `Tensor` of shape [1, T, S] containing a lower
                triangular matrix of shape [T, S].
    """
    q_seq_length = tf.shape(query)[1]
    v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
    return tf.linalg.band_part(  # creates a lower triangular matrix
        tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0)

  def compute_output_shape(self, query_shape, value_shape, key_shape=None):

    if key_shape is None:
      key_shape = value_shape

    query_shape = tf.TensorShape(query_shape)
    value_shape = tf.TensorShape(value_shape)
    key_shape = tf.TensorShape(key_shape)

    if query_shape[-1] != value_shape[-1]:
      raise ValueError(
          "The last dimension of `query_shape` and `value_shape` "
          f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
          "Received: query_shape={query_shape}, value_shape={value_shape}")

    if value_shape[1:-1] != key_shape[1:-1]:
      raise ValueError(
          "All dimensions of `value` and `key`, except the last one, "
          f"must be equal. Received {value_shape} and "
          f"{key_shape}")

    if self._output_shape:
      return query_shape[:-1].concatenate(self._output_shape)

    return query_shape


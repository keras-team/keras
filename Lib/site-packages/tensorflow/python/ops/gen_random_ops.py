"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar, List, Any
from typing_extensions import Annotated

TV_Multinomial_T = TypeVar("TV_Multinomial_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Multinomial_output_dtype = TypeVar("TV_Multinomial_output_dtype", _atypes.Int32, _atypes.Int64)

def multinomial(logits: Annotated[Any, TV_Multinomial_T], num_samples: Annotated[Any, _atypes.Int32], seed:int=0, seed2:int=0, output_dtype:TV_Multinomial_output_dtype=_dtypes.int64, name=None) -> Annotated[Any, TV_Multinomial_output_dtype]:
  r"""Draws samples from a multinomial distribution.

  Args:
    logits: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
      represents the unnormalized log probabilities for all classes.
    num_samples: A `Tensor` of type `int32`.
      0-D.  Number of independent samples to draw for each row slice.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the internal random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    output_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Multinomial", name, logits, num_samples, "seed", seed, "seed2",
        seed2, "output_dtype", output_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return multinomial_eager_fallback(
          logits, num_samples, seed=seed, seed2=seed2,
          output_dtype=output_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if output_dtype is None:
    output_dtype = _dtypes.int64
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Multinomial", logits=logits, num_samples=num_samples, seed=seed,
                       seed2=seed2, output_dtype=output_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "T", _op._get_attr_type("T"),
              "output_dtype", _op._get_attr_type("output_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Multinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Multinomial = tf_export("raw_ops.Multinomial")(_ops.to_raw_op(multinomial))


def multinomial_eager_fallback(logits: Annotated[Any, TV_Multinomial_T], num_samples: Annotated[Any, _atypes.Int32], seed: int, seed2: int, output_dtype: TV_Multinomial_output_dtype, name, ctx) -> Annotated[Any, TV_Multinomial_output_dtype]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if output_dtype is None:
    output_dtype = _dtypes.int64
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _attr_T, (logits,) = _execute.args_to_matching_eager([logits], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  num_samples = _ops.convert_to_tensor(num_samples, _dtypes.int32)
  _inputs_flat = [logits, num_samples]
  _attrs = ("seed", seed, "seed2", seed2, "T", _attr_T, "output_dtype",
  output_dtype)
  _result = _execute.execute(b"Multinomial", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Multinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ParameterizedTruncatedNormal_dtype = TypeVar("TV_ParameterizedTruncatedNormal_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_ParameterizedTruncatedNormal_T = TypeVar("TV_ParameterizedTruncatedNormal_T", _atypes.Int32, _atypes.Int64)

def parameterized_truncated_normal(shape: Annotated[Any, TV_ParameterizedTruncatedNormal_T], means: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], stdevs: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], minvals: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], maxvals: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_ParameterizedTruncatedNormal_dtype]:
  r"""Outputs random values from a normal distribution. The parameters may each be a

  scalar which applies to the entire output, or a vector of length shape[0] which
  stores the parameters for each batch.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor. Batches are indexed by the 0th dimension.
    means: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The mean parameter of each batch.
    stdevs: A `Tensor`. Must have the same type as `means`.
      The standard deviation parameter of each batch. Must be greater than 0.
    minvals: A `Tensor`. Must have the same type as `means`.
      The minimum cutoff. May be -infinity.
    maxvals: A `Tensor`. Must have the same type as `means`.
      The maximum cutoff. May be +infinity, and must be more than the minval
      for each batch.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `means`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParameterizedTruncatedNormal", name, shape, means, stdevs,
        minvals, maxvals, "seed", seed, "seed2", seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parameterized_truncated_normal_eager_fallback(
          shape, means, stdevs, minvals, maxvals, seed=seed, seed2=seed2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParameterizedTruncatedNormal", shape=shape, means=means,
                                        stdevs=stdevs, minvals=minvals,
                                        maxvals=maxvals, seed=seed,
                                        seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "dtype",
              _op._get_attr_type("dtype"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParameterizedTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParameterizedTruncatedNormal = tf_export("raw_ops.ParameterizedTruncatedNormal")(_ops.to_raw_op(parameterized_truncated_normal))


def parameterized_truncated_normal_eager_fallback(shape: Annotated[Any, TV_ParameterizedTruncatedNormal_T], means: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], stdevs: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], minvals: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], maxvals: Annotated[Any, TV_ParameterizedTruncatedNormal_dtype], seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_ParameterizedTruncatedNormal_dtype]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([means, stdevs, minvals, maxvals], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (means, stdevs, minvals, maxvals) = _inputs_dtype
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [shape, means, stdevs, minvals, maxvals]
  _attrs = ("seed", seed, "seed2", seed2, "dtype", _attr_dtype, "T", _attr_T)
  _result = _execute.execute(b"ParameterizedTruncatedNormal", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParameterizedTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomGamma_S = TypeVar("TV_RandomGamma_S", _atypes.Int32, _atypes.Int64)
TV_RandomGamma_T = TypeVar("TV_RandomGamma_T", _atypes.Float32, _atypes.Float64, _atypes.Half)

def random_gamma(shape: Annotated[Any, TV_RandomGamma_S], alpha: Annotated[Any, TV_RandomGamma_T], seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomGamma_T]:
  r"""Outputs random values from the Gamma distribution(s) described by alpha.

  This op uses the algorithm by Marsaglia et al. to acquire samples via
  transformation-rejection from pairs of uniform and normal random variables.
  See http://dl.acm.org/citation.cfm?id=358414

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D integer tensor. Shape of independent samples to draw from each
      distribution described by the shape parameters given in alpha.
    alpha: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      A tensor in which each scalar is a "shape" parameter describing the
      associated gamma distribution.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `alpha`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomGamma", name, shape, alpha, "seed", seed, "seed2", seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_gamma_eager_fallback(
          shape, alpha, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomGamma", shape=shape, alpha=alpha, seed=seed, seed2=seed2,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "S", _op._get_attr_type("S"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomGamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomGamma = tf_export("raw_ops.RandomGamma")(_ops.to_raw_op(random_gamma))


def random_gamma_eager_fallback(shape: Annotated[Any, TV_RandomGamma_S], alpha: Annotated[Any, TV_RandomGamma_T], seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomGamma_T]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_T, (alpha,) = _execute.args_to_matching_eager([alpha], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [shape, alpha]
  _attrs = ("seed", seed, "seed2", seed2, "S", _attr_S, "T", _attr_T)
  _result = _execute.execute(b"RandomGamma", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomGamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomGammaGrad_T = TypeVar("TV_RandomGammaGrad_T", _atypes.Float32, _atypes.Float64)

def random_gamma_grad(alpha: Annotated[Any, TV_RandomGammaGrad_T], sample: Annotated[Any, TV_RandomGammaGrad_T], name=None) -> Annotated[Any, TV_RandomGammaGrad_T]:
  r"""Computes the derivative of a Gamma random sample w.r.t. `alpha`.

  Args:
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    sample: A `Tensor`. Must have the same type as `alpha`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `alpha`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomGammaGrad", name, alpha, sample)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_gamma_grad_eager_fallback(
          alpha, sample, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomGammaGrad", alpha=alpha, sample=sample, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomGammaGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomGammaGrad = tf_export("raw_ops.RandomGammaGrad")(_ops.to_raw_op(random_gamma_grad))


def random_gamma_grad_eager_fallback(alpha: Annotated[Any, TV_RandomGammaGrad_T], sample: Annotated[Any, TV_RandomGammaGrad_T], name, ctx) -> Annotated[Any, TV_RandomGammaGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([alpha, sample], ctx, [_dtypes.float32, _dtypes.float64, ])
  (alpha, sample) = _inputs_T
  _inputs_flat = [alpha, sample]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RandomGammaGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomGammaGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomPoisson_S = TypeVar("TV_RandomPoisson_S", _atypes.Int32, _atypes.Int64)
TV_RandomPoisson_dtype = TypeVar("TV_RandomPoisson_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half)

def random_poisson(shape: Annotated[Any, TV_RandomPoisson_S], rate: Annotated[Any, TV_RandomPoisson_dtype], seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomPoisson_dtype]:
  r"""Use RandomPoissonV2 instead.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    rate: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `rate`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomPoisson", name, shape, rate, "seed", seed, "seed2",
        seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_poisson_eager_fallback(
          shape, rate, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomPoisson", shape=shape, rate=rate, seed=seed, seed2=seed2,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "S", _op._get_attr_type("S"),
              "dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomPoisson", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomPoisson = tf_export("raw_ops.RandomPoisson")(_ops.to_raw_op(random_poisson))


def random_poisson_eager_fallback(shape: Annotated[Any, TV_RandomPoisson_S], rate: Annotated[Any, TV_RandomPoisson_dtype], seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomPoisson_dtype]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_dtype, (rate,) = _execute.args_to_matching_eager([rate], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [shape, rate]
  _attrs = ("seed", seed, "seed2", seed2, "S", _attr_S, "dtype", _attr_dtype)
  _result = _execute.execute(b"RandomPoisson", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomPoisson", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomPoissonV2_S = TypeVar("TV_RandomPoissonV2_S", _atypes.Int32, _atypes.Int64)
TV_RandomPoissonV2_R = TypeVar("TV_RandomPoissonV2_R", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_RandomPoissonV2_dtype = TypeVar("TV_RandomPoissonV2_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def random_poisson_v2(shape: Annotated[Any, TV_RandomPoissonV2_S], rate: Annotated[Any, TV_RandomPoissonV2_R], seed:int=0, seed2:int=0, dtype:TV_RandomPoissonV2_dtype=_dtypes.int64, name=None) -> Annotated[Any, TV_RandomPoissonV2_dtype]:
  r"""Outputs random values from the Poisson distribution(s) described by rate.

  This op uses two algorithms, depending on rate. If rate >= 10, then
  the algorithm by Hormann is used to acquire samples via
  transformation-rejection.
  See http://www.sciencedirect.com/science/article/pii/0167668793909974.

  Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
  random variables.
  See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
  Programming, Volume 2. Addison Wesley

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D integer tensor. Shape of independent samples to draw from each
      distribution described by the shape parameters given in rate.
    rate: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
      A tensor in which each scalar is a "rate" parameter describing the
      associated poisson distribution.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomPoissonV2", name, shape, rate, "seed", seed, "seed2",
        seed2, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_poisson_v2_eager_fallback(
          shape, rate, seed=seed, seed2=seed2, dtype=dtype, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomPoissonV2", shape=shape, rate=rate, seed=seed, seed2=seed2,
                           dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "S", _op._get_attr_type("S"), "R",
              _op._get_attr_type("R"), "dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomPoissonV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomPoissonV2 = tf_export("raw_ops.RandomPoissonV2")(_ops.to_raw_op(random_poisson_v2))


def random_poisson_v2_eager_fallback(shape: Annotated[Any, TV_RandomPoissonV2_S], rate: Annotated[Any, TV_RandomPoissonV2_R], seed: int, seed2: int, dtype: TV_RandomPoissonV2_dtype, name, ctx) -> Annotated[Any, TV_RandomPoissonV2_dtype]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_R, (rate,) = _execute.args_to_matching_eager([rate], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ], _dtypes.float64)
  _inputs_flat = [shape, rate]
  _attrs = ("seed", seed, "seed2", seed2, "S", _attr_S, "R", _attr_R, "dtype",
  dtype)
  _result = _execute.execute(b"RandomPoissonV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomPoissonV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomShuffle_T = TypeVar("TV_RandomShuffle_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def random_shuffle(value: Annotated[Any, TV_RandomShuffle_T], seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomShuffle_T]:
  r"""Randomly shuffles a tensor along its first dimension.

    The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
    to one and only one `output[i]`. For example, a mapping that might occur for a
    3x2 tensor is:

  ```
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```

  Args:
    value: A `Tensor`. The tensor to be shuffled.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomShuffle", name, value, "seed", seed, "seed2", seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_shuffle_eager_fallback(
          value, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomShuffle", value=value, seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomShuffle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomShuffle = tf_export("raw_ops.RandomShuffle")(_ops.to_raw_op(random_shuffle))


def random_shuffle_eager_fallback(value: Annotated[Any, TV_RandomShuffle_T], seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomShuffle_T]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _inputs_flat = [value]
  _attrs = ("seed", seed, "seed2", seed2, "T", _attr_T)
  _result = _execute.execute(b"RandomShuffle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomShuffle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomStandardNormal_dtype = TypeVar("TV_RandomStandardNormal_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_RandomStandardNormal_T = TypeVar("TV_RandomStandardNormal_T", _atypes.Int32, _atypes.Int64)

def random_standard_normal(shape: Annotated[Any, TV_RandomStandardNormal_T], dtype: TV_RandomStandardNormal_dtype, seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomStandardNormal_dtype]:
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    dtype: A `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`.
      The type of the output.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomStandardNormal", name, shape, "seed", seed, "seed2",
        seed2, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_standard_normal_eager_fallback(
          shape, seed=seed, seed2=seed2, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomStandardNormal", shape=shape, dtype=dtype, seed=seed,
                                seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "dtype",
              _op._get_attr_type("dtype"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomStandardNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomStandardNormal = tf_export("raw_ops.RandomStandardNormal")(_ops.to_raw_op(random_standard_normal))


def random_standard_normal_eager_fallback(shape: Annotated[Any, TV_RandomStandardNormal_T], dtype: TV_RandomStandardNormal_dtype, seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomStandardNormal_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [shape]
  _attrs = ("seed", seed, "seed2", seed2, "dtype", dtype, "T", _attr_T)
  _result = _execute.execute(b"RandomStandardNormal", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomStandardNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomUniform_dtype = TypeVar("TV_RandomUniform_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_RandomUniform_T = TypeVar("TV_RandomUniform_T", _atypes.Int32, _atypes.Int64)

def random_uniform(shape: Annotated[Any, TV_RandomUniform_T], dtype: TV_RandomUniform_dtype, seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomUniform_dtype]:
  r"""Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    dtype: A `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`.
      The type of the output.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomUniform", name, shape, "seed", seed, "seed2", seed2,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_uniform_eager_fallback(
          shape, seed=seed, seed2=seed2, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomUniform", shape=shape, dtype=dtype, seed=seed, seed2=seed2,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "dtype",
              _op._get_attr_type("dtype"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomUniform", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomUniform = tf_export("raw_ops.RandomUniform")(_ops.to_raw_op(random_uniform))


def random_uniform_eager_fallback(shape: Annotated[Any, TV_RandomUniform_T], dtype: TV_RandomUniform_dtype, seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomUniform_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [shape]
  _attrs = ("seed", seed, "seed2", seed2, "dtype", dtype, "T", _attr_T)
  _result = _execute.execute(b"RandomUniform", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomUniform", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomUniformInt_Tout = TypeVar("TV_RandomUniformInt_Tout", _atypes.Int32, _atypes.Int64)
TV_RandomUniformInt_T = TypeVar("TV_RandomUniformInt_T", _atypes.Int32, _atypes.Int64)

def random_uniform_int(shape: Annotated[Any, TV_RandomUniformInt_T], minval: Annotated[Any, TV_RandomUniformInt_Tout], maxval: Annotated[Any, TV_RandomUniformInt_Tout], seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomUniformInt_Tout]:
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    minval: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D.  Inclusive lower bound on the generated integers.
    maxval: A `Tensor`. Must have the same type as `minval`.
      0-D.  Exclusive upper bound on the generated integers.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomUniformInt", name, shape, minval, maxval, "seed", seed,
        "seed2", seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_uniform_int_eager_fallback(
          shape, minval, maxval, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomUniformInt", shape=shape, minval=minval, maxval=maxval,
                            seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "Tout", _op._get_attr_type("Tout"),
              "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomUniformInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomUniformInt = tf_export("raw_ops.RandomUniformInt")(_ops.to_raw_op(random_uniform_int))


def random_uniform_int_eager_fallback(shape: Annotated[Any, TV_RandomUniformInt_T], minval: Annotated[Any, TV_RandomUniformInt_Tout], maxval: Annotated[Any, TV_RandomUniformInt_Tout], seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomUniformInt_Tout]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_Tout, _inputs_Tout = _execute.args_to_matching_eager([minval, maxval], ctx, [_dtypes.int32, _dtypes.int64, ])
  (minval, maxval) = _inputs_Tout
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [shape, minval, maxval]
  _attrs = ("seed", seed, "seed2", seed2, "Tout", _attr_Tout, "T", _attr_T)
  _result = _execute.execute(b"RandomUniformInt", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomUniformInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TruncatedNormal_dtype = TypeVar("TV_TruncatedNormal_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_TruncatedNormal_T = TypeVar("TV_TruncatedNormal_T", _atypes.Int32, _atypes.Int64)

def truncated_normal(shape: Annotated[Any, TV_TruncatedNormal_T], dtype: TV_TruncatedNormal_dtype, seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_TruncatedNormal_dtype]:
  r"""Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    dtype: A `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`.
      The type of the output.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TruncatedNormal", name, shape, "seed", seed, "seed2", seed2,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return truncated_normal_eager_fallback(
          shape, seed=seed, seed2=seed2, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TruncatedNormal", shape=shape, dtype=dtype, seed=seed, seed2=seed2,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "dtype",
              _op._get_attr_type("dtype"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TruncatedNormal = tf_export("raw_ops.TruncatedNormal")(_ops.to_raw_op(truncated_normal))


def truncated_normal_eager_fallback(shape: Annotated[Any, TV_TruncatedNormal_T], dtype: TV_TruncatedNormal_dtype, seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_TruncatedNormal_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [shape]
  _attrs = ("seed", seed, "seed2", seed2, "dtype", dtype, "T", _attr_T)
  _result = _execute.execute(b"TruncatedNormal", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


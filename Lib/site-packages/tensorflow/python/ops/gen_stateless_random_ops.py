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

TV_StatelessMultinomial_T = TypeVar("TV_StatelessMultinomial_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_StatelessMultinomial_Tseed = TypeVar("TV_StatelessMultinomial_Tseed", _atypes.Int32, _atypes.Int64)
TV_StatelessMultinomial_output_dtype = TypeVar("TV_StatelessMultinomial_output_dtype", _atypes.Int32, _atypes.Int64)

def stateless_multinomial(logits: Annotated[Any, TV_StatelessMultinomial_T], num_samples: Annotated[Any, _atypes.Int32], seed: Annotated[Any, TV_StatelessMultinomial_Tseed], output_dtype:TV_StatelessMultinomial_output_dtype=_dtypes.int64, name=None) -> Annotated[Any, TV_StatelessMultinomial_output_dtype]:
  r"""Draws samples from a multinomial distribution.

  Args:
    logits: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
      represents the unnormalized log probabilities for all classes.
    num_samples: A `Tensor` of type `int32`.
      0-D.  Number of independent samples to draw for each row slice.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
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
        _ctx, "StatelessMultinomial", name, logits, num_samples, seed,
        "output_dtype", output_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_multinomial_eager_fallback(
          logits, num_samples, seed, output_dtype=output_dtype, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_dtype is None:
    output_dtype = _dtypes.int64
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessMultinomial", logits=logits, num_samples=num_samples,
                                seed=seed, output_dtype=output_dtype,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tseed",
              _op._get_attr_type("Tseed"), "output_dtype",
              _op._get_attr_type("output_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessMultinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessMultinomial = tf_export("raw_ops.StatelessMultinomial")(_ops.to_raw_op(stateless_multinomial))


def stateless_multinomial_eager_fallback(logits: Annotated[Any, TV_StatelessMultinomial_T], num_samples: Annotated[Any, _atypes.Int32], seed: Annotated[Any, TV_StatelessMultinomial_Tseed], output_dtype: TV_StatelessMultinomial_output_dtype, name, ctx) -> Annotated[Any, TV_StatelessMultinomial_output_dtype]:
  if output_dtype is None:
    output_dtype = _dtypes.int64
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _attr_T, (logits,) = _execute.args_to_matching_eager([logits], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  num_samples = _ops.convert_to_tensor(num_samples, _dtypes.int32)
  _inputs_flat = [logits, num_samples, seed]
  _attrs = ("T", _attr_T, "Tseed", _attr_Tseed, "output_dtype", output_dtype)
  _result = _execute.execute(b"StatelessMultinomial", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessMultinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessParameterizedTruncatedNormal_S = TypeVar("TV_StatelessParameterizedTruncatedNormal_S", _atypes.Int32, _atypes.Int64)
TV_StatelessParameterizedTruncatedNormal_Tseed = TypeVar("TV_StatelessParameterizedTruncatedNormal_Tseed", _atypes.Int32, _atypes.Int64)
TV_StatelessParameterizedTruncatedNormal_dtype = TypeVar("TV_StatelessParameterizedTruncatedNormal_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half)

def stateless_parameterized_truncated_normal(shape: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_S], seed: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_Tseed], means: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], stddevs: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], minvals: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], maxvals: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], name=None) -> Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype]:
  r"""TODO: add doc.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    means: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      The mean parameter of each batch.
    stddevs: A `Tensor`. Must have the same type as `means`.
      The standard deviation parameter of each batch. Must be greater than 0.
    minvals: A `Tensor`. Must have the same type as `means`.
      The minimum cutoff. May be -infinity.
    maxvals: A `Tensor`. Must have the same type as `means`.
      The maximum cutoff. May be +infinity, and must be more than the minval
      for each batch.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `means`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessParameterizedTruncatedNormal", name, shape, seed,
        means, stddevs, minvals, maxvals)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_parameterized_truncated_normal_eager_fallback(
          shape, seed, means, stddevs, minvals, maxvals, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessParameterizedTruncatedNormal", shape=shape, seed=seed,
                                                 means=means, stddevs=stddevs,
                                                 minvals=minvals,
                                                 maxvals=maxvals, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("S", _op._get_attr_type("S"), "Tseed",
              _op._get_attr_type("Tseed"), "dtype",
              _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessParameterizedTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessParameterizedTruncatedNormal = tf_export("raw_ops.StatelessParameterizedTruncatedNormal")(_ops.to_raw_op(stateless_parameterized_truncated_normal))


def stateless_parameterized_truncated_normal_eager_fallback(shape: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_S], seed: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_Tseed], means: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], stddevs: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], minvals: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], maxvals: Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype], name, ctx) -> Annotated[Any, TV_StatelessParameterizedTruncatedNormal_dtype]:
  _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([means, stddevs, minvals, maxvals], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (means, stddevs, minvals, maxvals) = _inputs_dtype
  _inputs_flat = [shape, seed, means, stddevs, minvals, maxvals]
  _attrs = ("S", _attr_S, "Tseed", _attr_Tseed, "dtype", _attr_dtype)
  _result = _execute.execute(b"StatelessParameterizedTruncatedNormal", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessParameterizedTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomBinomial_S = TypeVar("TV_StatelessRandomBinomial_S", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomBinomial_Tseed = TypeVar("TV_StatelessRandomBinomial_Tseed", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomBinomial_T = TypeVar("TV_StatelessRandomBinomial_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_StatelessRandomBinomial_dtype = TypeVar("TV_StatelessRandomBinomial_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def stateless_random_binomial(shape: Annotated[Any, TV_StatelessRandomBinomial_S], seed: Annotated[Any, TV_StatelessRandomBinomial_Tseed], counts: Annotated[Any, TV_StatelessRandomBinomial_T], probs: Annotated[Any, TV_StatelessRandomBinomial_T], dtype:TV_StatelessRandomBinomial_dtype=_dtypes.int64, name=None) -> Annotated[Any, TV_StatelessRandomBinomial_dtype]:
  r"""Outputs deterministic pseudorandom random numbers from a binomial distribution.

  Outputs random values from a binomial distribution.

  The outputs are a deterministic function of `shape`, `seed`, `counts`, and `probs`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    counts: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
      The counts of the binomial distribution. Must be broadcastable with `probs`,
      and broadcastable with the rightmost dimensions of `shape`.
    probs: A `Tensor`. Must have the same type as `counts`.
      The probability of success for the binomial distribution. Must be broadcastable
      with `counts` and broadcastable with the rightmost dimensions of `shape`.
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.int64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomBinomial", name, shape, seed, counts, probs,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_binomial_eager_fallback(
          shape, seed, counts, probs, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomBinomial", shape=shape, seed=seed, counts=counts,
                                   probs=probs, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("S", _op._get_attr_type("S"), "Tseed",
              _op._get_attr_type("Tseed"), "T", _op._get_attr_type("T"),
              "dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomBinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomBinomial = tf_export("raw_ops.StatelessRandomBinomial")(_ops.to_raw_op(stateless_random_binomial))


def stateless_random_binomial_eager_fallback(shape: Annotated[Any, TV_StatelessRandomBinomial_S], seed: Annotated[Any, TV_StatelessRandomBinomial_Tseed], counts: Annotated[Any, TV_StatelessRandomBinomial_T], probs: Annotated[Any, TV_StatelessRandomBinomial_T], dtype: TV_StatelessRandomBinomial_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomBinomial_dtype]:
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _attr_T, _inputs_T = _execute.args_to_matching_eager([counts, probs], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ], _dtypes.float64)
  (counts, probs) = _inputs_T
  _inputs_flat = [shape, seed, counts, probs]
  _attrs = ("S", _attr_S, "Tseed", _attr_Tseed, "T", _attr_T, "dtype", dtype)
  _result = _execute.execute(b"StatelessRandomBinomial", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomBinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomGammaV2_dtype = TypeVar("TV_StatelessRandomGammaV2_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessRandomGammaV2_T = TypeVar("TV_StatelessRandomGammaV2_T", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomGammaV2_Tseed = TypeVar("TV_StatelessRandomGammaV2_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_gamma_v2(shape: Annotated[Any, TV_StatelessRandomGammaV2_T], seed: Annotated[Any, TV_StatelessRandomGammaV2_Tseed], alpha: Annotated[Any, TV_StatelessRandomGammaV2_dtype], name=None) -> Annotated[Any, TV_StatelessRandomGammaV2_dtype]:
  r"""Outputs deterministic pseudorandom random numbers from a gamma distribution.

  Outputs random values from a gamma distribution.

  The outputs are a deterministic function of `shape`, `seed`, and `alpha`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    alpha: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      The concentration of the gamma distribution. Shape must match the rightmost
      dimensions of `shape`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `alpha`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomGammaV2", name, shape, seed, alpha)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_gamma_v2_eager_fallback(
          shape, seed, alpha, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomGammaV2", shape=shape, seed=seed, alpha=alpha,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "T",
              _op._get_attr_type("T"), "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomGammaV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomGammaV2 = tf_export("raw_ops.StatelessRandomGammaV2")(_ops.to_raw_op(stateless_random_gamma_v2))


def stateless_random_gamma_v2_eager_fallback(shape: Annotated[Any, TV_StatelessRandomGammaV2_T], seed: Annotated[Any, TV_StatelessRandomGammaV2_Tseed], alpha: Annotated[Any, TV_StatelessRandomGammaV2_dtype], name, ctx) -> Annotated[Any, TV_StatelessRandomGammaV2_dtype]:
  _attr_dtype, (alpha,) = _execute.args_to_matching_eager([alpha], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [shape, seed, alpha]
  _attrs = ("dtype", _attr_dtype, "T", _attr_T, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomGammaV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomGammaV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomNormal_dtype = TypeVar("TV_StatelessRandomNormal_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessRandomNormal_T = TypeVar("TV_StatelessRandomNormal_T", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomNormal_Tseed = TypeVar("TV_StatelessRandomNormal_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_normal(shape: Annotated[Any, TV_StatelessRandomNormal_T], seed: Annotated[Any, TV_StatelessRandomNormal_Tseed], dtype:TV_StatelessRandomNormal_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatelessRandomNormal_dtype]:
  r"""Outputs deterministic pseudorandom values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomNormal", name, shape, seed, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_normal_eager_fallback(
          shape, seed, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomNormal", shape=shape, seed=seed, dtype=dtype,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "T",
              _op._get_attr_type("T"), "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomNormal = tf_export("raw_ops.StatelessRandomNormal")(_ops.to_raw_op(stateless_random_normal))


def stateless_random_normal_eager_fallback(shape: Annotated[Any, TV_StatelessRandomNormal_T], seed: Annotated[Any, TV_StatelessRandomNormal_Tseed], dtype: TV_StatelessRandomNormal_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomNormal_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [shape, seed]
  _attrs = ("dtype", dtype, "T", _attr_T, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomNormal", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomPoisson_Rtype = TypeVar("TV_StatelessRandomPoisson_Rtype", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_StatelessRandomPoisson_dtype = TypeVar("TV_StatelessRandomPoisson_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_StatelessRandomPoisson_T = TypeVar("TV_StatelessRandomPoisson_T", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomPoisson_Tseed = TypeVar("TV_StatelessRandomPoisson_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_poisson(shape: Annotated[Any, TV_StatelessRandomPoisson_T], seed: Annotated[Any, TV_StatelessRandomPoisson_Tseed], lam: Annotated[Any, TV_StatelessRandomPoisson_Rtype], dtype: TV_StatelessRandomPoisson_dtype, name=None) -> Annotated[Any, TV_StatelessRandomPoisson_dtype]:
  r"""Outputs deterministic pseudorandom random numbers from a Poisson distribution.

  Outputs random values from a Poisson distribution.

  The outputs are a deterministic function of `shape`, `seed`, and `lam`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    lam: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
      The rate of the Poisson distribution. Shape must match the rightmost dimensions
      of `shape`.
    dtype: A `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomPoisson", name, shape, seed, lam, "dtype",
        dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_poisson_eager_fallback(
          shape, seed, lam, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomPoisson", shape=shape, seed=seed, lam=lam,
                                  dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Rtype", _op._get_attr_type("Rtype"), "dtype",
              _op._get_attr_type("dtype"), "T", _op._get_attr_type("T"),
              "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomPoisson", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomPoisson = tf_export("raw_ops.StatelessRandomPoisson")(_ops.to_raw_op(stateless_random_poisson))


def stateless_random_poisson_eager_fallback(shape: Annotated[Any, TV_StatelessRandomPoisson_T], seed: Annotated[Any, TV_StatelessRandomPoisson_Tseed], lam: Annotated[Any, TV_StatelessRandomPoisson_Rtype], dtype: TV_StatelessRandomPoisson_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomPoisson_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Rtype, (lam,) = _execute.args_to_matching_eager([lam], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [shape, seed, lam]
  _attrs = ("Rtype", _attr_Rtype, "dtype", dtype, "T", _attr_T, "Tseed",
  _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomPoisson", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomPoisson", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomUniform_dtype = TypeVar("TV_StatelessRandomUniform_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessRandomUniform_T = TypeVar("TV_StatelessRandomUniform_T", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomUniform_Tseed = TypeVar("TV_StatelessRandomUniform_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_uniform(shape: Annotated[Any, TV_StatelessRandomUniform_T], seed: Annotated[Any, TV_StatelessRandomUniform_Tseed], dtype:TV_StatelessRandomUniform_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatelessRandomUniform_dtype]:
  r"""Outputs deterministic pseudorandom random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomUniform", name, shape, seed, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_uniform_eager_fallback(
          shape, seed, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomUniform", shape=shape, seed=seed, dtype=dtype,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "T",
              _op._get_attr_type("T"), "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomUniform", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomUniform = tf_export("raw_ops.StatelessRandomUniform")(_ops.to_raw_op(stateless_random_uniform))


def stateless_random_uniform_eager_fallback(shape: Annotated[Any, TV_StatelessRandomUniform_T], seed: Annotated[Any, TV_StatelessRandomUniform_Tseed], dtype: TV_StatelessRandomUniform_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomUniform_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [shape, seed]
  _attrs = ("dtype", dtype, "T", _attr_T, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomUniform", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomUniform", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomUniformFullInt_dtype = TypeVar("TV_StatelessRandomUniformFullInt_dtype", _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)
TV_StatelessRandomUniformFullInt_T = TypeVar("TV_StatelessRandomUniformFullInt_T", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomUniformFullInt_Tseed = TypeVar("TV_StatelessRandomUniformFullInt_Tseed", _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)

def stateless_random_uniform_full_int(shape: Annotated[Any, TV_StatelessRandomUniformFullInt_T], seed: Annotated[Any, TV_StatelessRandomUniformFullInt_Tseed], dtype:TV_StatelessRandomUniformFullInt_dtype=_dtypes.uint64, name=None) -> Annotated[Any, TV_StatelessRandomUniformFullInt_dtype]:
  r"""Outputs deterministic pseudorandom random integers from a uniform distribution.

  The generated values are uniform integers covering the whole range of `dtype`.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`, `uint32`, `uint64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.int32, tf.int64, tf.uint32, tf.uint64`. Defaults to `tf.uint64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomUniformFullInt", name, shape, seed, "dtype",
        dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_uniform_full_int_eager_fallback(
          shape, seed, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomUniformFullInt", shape=shape, seed=seed, dtype=dtype,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "T",
              _op._get_attr_type("T"), "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomUniformFullInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomUniformFullInt = tf_export("raw_ops.StatelessRandomUniformFullInt")(_ops.to_raw_op(stateless_random_uniform_full_int))


def stateless_random_uniform_full_int_eager_fallback(shape: Annotated[Any, TV_StatelessRandomUniformFullInt_T], seed: Annotated[Any, TV_StatelessRandomUniformFullInt_Tseed], dtype: TV_StatelessRandomUniformFullInt_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomUniformFullInt_dtype]:
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.uint32, _dtypes.uint64, ], _dtypes.int64)
  _inputs_flat = [shape, seed]
  _attrs = ("dtype", dtype, "T", _attr_T, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomUniformFullInt", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomUniformFullInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomUniformInt_dtype = TypeVar("TV_StatelessRandomUniformInt_dtype", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomUniformInt_T = TypeVar("TV_StatelessRandomUniformInt_T", _atypes.Int32, _atypes.Int64)
TV_StatelessRandomUniformInt_Tseed = TypeVar("TV_StatelessRandomUniformInt_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_uniform_int(shape: Annotated[Any, TV_StatelessRandomUniformInt_T], seed: Annotated[Any, TV_StatelessRandomUniformInt_Tseed], minval: Annotated[Any, TV_StatelessRandomUniformInt_dtype], maxval: Annotated[Any, TV_StatelessRandomUniformInt_dtype], name=None) -> Annotated[Any, TV_StatelessRandomUniformInt_dtype]:
  r"""Outputs deterministic pseudorandom random integers from a uniform distribution.

  The generated values follow a uniform distribution in the range `[minval, maxval)`.

  The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    minval: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomUniformInt", name, shape, seed, minval, maxval)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_uniform_int_eager_fallback(
          shape, seed, minval, maxval, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomUniformInt", shape=shape, seed=seed, minval=minval,
                                     maxval=maxval, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "T",
              _op._get_attr_type("T"), "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomUniformInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomUniformInt = tf_export("raw_ops.StatelessRandomUniformInt")(_ops.to_raw_op(stateless_random_uniform_int))


def stateless_random_uniform_int_eager_fallback(shape: Annotated[Any, TV_StatelessRandomUniformInt_T], seed: Annotated[Any, TV_StatelessRandomUniformInt_Tseed], minval: Annotated[Any, TV_StatelessRandomUniformInt_dtype], maxval: Annotated[Any, TV_StatelessRandomUniformInt_dtype], name, ctx) -> Annotated[Any, TV_StatelessRandomUniformInt_dtype]:
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([minval, maxval], ctx, [_dtypes.int32, _dtypes.int64, ])
  (minval, maxval) = _inputs_dtype
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [shape, seed, minval, maxval]
  _attrs = ("dtype", _attr_dtype, "T", _attr_T, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomUniformInt", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomUniformInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessTruncatedNormal_dtype = TypeVar("TV_StatelessTruncatedNormal_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessTruncatedNormal_T = TypeVar("TV_StatelessTruncatedNormal_T", _atypes.Int32, _atypes.Int64)
TV_StatelessTruncatedNormal_Tseed = TypeVar("TV_StatelessTruncatedNormal_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_truncated_normal(shape: Annotated[Any, TV_StatelessTruncatedNormal_T], seed: Annotated[Any, TV_StatelessTruncatedNormal_Tseed], dtype:TV_StatelessTruncatedNormal_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatelessTruncatedNormal_dtype]:
  r"""Outputs deterministic pseudorandom values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessTruncatedNormal", name, shape, seed, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_truncated_normal_eager_fallback(
          shape, seed, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessTruncatedNormal", shape=shape, seed=seed, dtype=dtype,
                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "T",
              _op._get_attr_type("T"), "Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessTruncatedNormal = tf_export("raw_ops.StatelessTruncatedNormal")(_ops.to_raw_op(stateless_truncated_normal))


def stateless_truncated_normal_eager_fallback(shape: Annotated[Any, TV_StatelessTruncatedNormal_T], seed: Annotated[Any, TV_StatelessTruncatedNormal_Tseed], dtype: TV_StatelessTruncatedNormal_dtype, name, ctx) -> Annotated[Any, TV_StatelessTruncatedNormal_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [shape, seed]
  _attrs = ("dtype", dtype, "T", _attr_T, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessTruncatedNormal", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


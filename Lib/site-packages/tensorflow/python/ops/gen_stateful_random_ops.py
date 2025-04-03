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

TV_NonDeterministicInts_dtype = TypeVar("TV_NonDeterministicInts_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_NonDeterministicInts_shape_dtype = TypeVar("TV_NonDeterministicInts_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def non_deterministic_ints(shape: Annotated[Any, TV_NonDeterministicInts_shape_dtype], dtype:TV_NonDeterministicInts_dtype=_dtypes.int64, name=None) -> Annotated[Any, TV_NonDeterministicInts_dtype]:
  r"""Non-deterministically generates some integers.

  This op may use some OS-provided source of non-determinism (e.g. an RNG), so each execution will give different results.

  Args:
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.int64`.
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
        _ctx, "NonDeterministicInts", name, shape, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_deterministic_ints_eager_fallback(
          shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonDeterministicInts", shape=shape, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonDeterministicInts", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NonDeterministicInts = tf_export("raw_ops.NonDeterministicInts")(_ops.to_raw_op(non_deterministic_ints))


def non_deterministic_ints_eager_fallback(shape: Annotated[Any, TV_NonDeterministicInts_shape_dtype], dtype: TV_NonDeterministicInts_dtype, name, ctx) -> Annotated[Any, TV_NonDeterministicInts_dtype]:
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  _inputs_flat = [shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"NonDeterministicInts", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonDeterministicInts", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def rng_read_and_skip(resource: Annotated[Any, _atypes.Resource], alg: Annotated[Any, _atypes.Int32], delta: Annotated[Any, _atypes.UInt64], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_read_and_skip(n)` will be the same as that after `uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation choice.

  In the case that the input algorithm is RNG_ALG_AUTO_SELECT, the counter in the state needs to be of size int64[2], the current maximal counter size among algorithms. In this case, this op will manage the counter as if it is an 128-bit integer with layout [lower_64bits, higher_64bits]. If an algorithm needs less than 128 bits for the counter, it should use the left portion of the int64[2]. In this way, the int64[2] is compatible with all current RNG algorithms (Philox, ThreeFry and xla::RandomAlgorithm::RNG_DEFAULT). Downstream RNG ops can thus use this counter with any RNG algorithm.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG. The state consists of the counter followed by the key.
    alg: A `Tensor` of type `int32`. The RNG algorithm.
    delta: A `Tensor` of type `uint64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RngReadAndSkip", name, resource, alg, delta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rng_read_and_skip_eager_fallback(
          resource, alg, delta, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RngReadAndSkip", resource=resource, alg=alg, delta=delta, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RngReadAndSkip", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RngReadAndSkip = tf_export("raw_ops.RngReadAndSkip")(_ops.to_raw_op(rng_read_and_skip))


def rng_read_and_skip_eager_fallback(resource: Annotated[Any, _atypes.Resource], alg: Annotated[Any, _atypes.Int32], delta: Annotated[Any, _atypes.UInt64], name, ctx) -> Annotated[Any, _atypes.Int64]:
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  delta = _ops.convert_to_tensor(delta, _dtypes.uint64)
  _inputs_flat = [resource, alg, delta]
  _attrs = None
  _result = _execute.execute(b"RngReadAndSkip", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RngReadAndSkip", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def rng_skip(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], delta: Annotated[Any, _atypes.Int64], name=None):
  r"""Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_skip(n)` will be the same as that after `stateful_uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation detail.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    delta: A `Tensor` of type `int64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RngSkip", name, resource, algorithm, delta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rng_skip_eager_fallback(
          resource, algorithm, delta, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RngSkip", resource=resource, algorithm=algorithm, delta=delta,
                   name=name)
  return _op
RngSkip = tf_export("raw_ops.RngSkip")(_ops.to_raw_op(rng_skip))


def rng_skip_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], delta: Annotated[Any, _atypes.Int64], name, ctx):
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  delta = _ops.convert_to_tensor(delta, _dtypes.int64)
  _inputs_flat = [resource, algorithm, delta]
  _attrs = None
  _result = _execute.execute(b"RngSkip", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


TV_StatefulRandomBinomial_S = TypeVar("TV_StatefulRandomBinomial_S", _atypes.Int32, _atypes.Int64)
TV_StatefulRandomBinomial_T = TypeVar("TV_StatefulRandomBinomial_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_StatefulRandomBinomial_dtype = TypeVar("TV_StatefulRandomBinomial_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def stateful_random_binomial(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulRandomBinomial_S], counts: Annotated[Any, TV_StatefulRandomBinomial_T], probs: Annotated[Any, TV_StatefulRandomBinomial_T], dtype:TV_StatefulRandomBinomial_dtype=_dtypes.int64, name=None) -> Annotated[Any, TV_StatefulRandomBinomial_dtype]:
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    algorithm: A `Tensor` of type `int64`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    counts: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    probs: A `Tensor`. Must have the same type as `counts`.
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
        _ctx, "StatefulRandomBinomial", name, resource, algorithm, shape,
        counts, probs, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_random_binomial_eager_fallback(
          resource, algorithm, shape, counts, probs, dtype=dtype, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulRandomBinomial", resource=resource, algorithm=algorithm,
                                  shape=shape, counts=counts, probs=probs,
                                  dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("S", _op._get_attr_type("S"), "T", _op._get_attr_type("T"),
              "dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulRandomBinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulRandomBinomial = tf_export("raw_ops.StatefulRandomBinomial")(_ops.to_raw_op(stateful_random_binomial))


def stateful_random_binomial_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulRandomBinomial_S], counts: Annotated[Any, TV_StatefulRandomBinomial_T], probs: Annotated[Any, TV_StatefulRandomBinomial_T], dtype: TV_StatefulRandomBinomial_dtype, name, ctx) -> Annotated[Any, TV_StatefulRandomBinomial_dtype]:
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_T, _inputs_T = _execute.args_to_matching_eager([counts, probs], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ], _dtypes.float64)
  (counts, probs) = _inputs_T
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape, counts, probs]
  _attrs = ("S", _attr_S, "T", _attr_T, "dtype", dtype)
  _result = _execute.execute(b"StatefulRandomBinomial", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulRandomBinomial", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatefulStandardNormal_dtype = TypeVar("TV_StatefulStandardNormal_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulStandardNormal_shape_dtype = TypeVar("TV_StatefulStandardNormal_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateful_standard_normal(resource: Annotated[Any, _atypes.Resource], shape: Annotated[Any, TV_StatefulStandardNormal_shape_dtype], dtype:TV_StatefulStandardNormal_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatefulStandardNormal_dtype]:
  r"""Outputs random values from a normal distribution. This op is deprecated in favor of op 'StatefulStandardNormalV2'

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
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
        _ctx, "StatefulStandardNormal", name, resource, shape, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_standard_normal_eager_fallback(
          resource, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulStandardNormal", resource=resource, shape=shape, dtype=dtype,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulStandardNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulStandardNormal = tf_export("raw_ops.StatefulStandardNormal")(_ops.to_raw_op(stateful_standard_normal))


def stateful_standard_normal_eager_fallback(resource: Annotated[Any, _atypes.Resource], shape: Annotated[Any, TV_StatefulStandardNormal_shape_dtype], dtype: TV_StatefulStandardNormal_dtype, name, ctx) -> Annotated[Any, TV_StatefulStandardNormal_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulStandardNormal", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulStandardNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatefulStandardNormalV2_dtype = TypeVar("TV_StatefulStandardNormalV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulStandardNormalV2_shape_dtype = TypeVar("TV_StatefulStandardNormalV2_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateful_standard_normal_v2(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulStandardNormalV2_shape_dtype], dtype:TV_StatefulStandardNormalV2_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatefulStandardNormalV2_dtype]:
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
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
        _ctx, "StatefulStandardNormalV2", name, resource, algorithm, shape,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_standard_normal_v2_eager_fallback(
          resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulStandardNormalV2", resource=resource, algorithm=algorithm,
                                    shape=shape, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulStandardNormalV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulStandardNormalV2 = tf_export("raw_ops.StatefulStandardNormalV2")(_ops.to_raw_op(stateful_standard_normal_v2))


def stateful_standard_normal_v2_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulStandardNormalV2_shape_dtype], dtype: TV_StatefulStandardNormalV2_dtype, name, ctx) -> Annotated[Any, TV_StatefulStandardNormalV2_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulStandardNormalV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulStandardNormalV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatefulTruncatedNormal_dtype = TypeVar("TV_StatefulTruncatedNormal_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulTruncatedNormal_shape_dtype = TypeVar("TV_StatefulTruncatedNormal_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateful_truncated_normal(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulTruncatedNormal_shape_dtype], dtype:TV_StatefulTruncatedNormal_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatefulTruncatedNormal_dtype]:
  r"""Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
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
        _ctx, "StatefulTruncatedNormal", name, resource, algorithm, shape,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_truncated_normal_eager_fallback(
          resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulTruncatedNormal", resource=resource, algorithm=algorithm,
                                   shape=shape, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulTruncatedNormal = tf_export("raw_ops.StatefulTruncatedNormal")(_ops.to_raw_op(stateful_truncated_normal))


def stateful_truncated_normal_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulTruncatedNormal_shape_dtype], dtype: TV_StatefulTruncatedNormal_dtype, name, ctx) -> Annotated[Any, TV_StatefulTruncatedNormal_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulTruncatedNormal", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulTruncatedNormal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatefulUniform_dtype = TypeVar("TV_StatefulUniform_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulUniform_shape_dtype = TypeVar("TV_StatefulUniform_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateful_uniform(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniform_shape_dtype], dtype:TV_StatefulUniform_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatefulUniform_dtype]:
  r"""Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
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
        _ctx, "StatefulUniform", name, resource, algorithm, shape, "dtype",
        dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_uniform_eager_fallback(
          resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulUniform", resource=resource, algorithm=algorithm,
                           shape=shape, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulUniform", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulUniform = tf_export("raw_ops.StatefulUniform")(_ops.to_raw_op(stateful_uniform))


def stateful_uniform_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniform_shape_dtype], dtype: TV_StatefulUniform_dtype, name, ctx) -> Annotated[Any, TV_StatefulUniform_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulUniform", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulUniform", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatefulUniformFullInt_dtype = TypeVar("TV_StatefulUniformFullInt_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulUniformFullInt_shape_dtype = TypeVar("TV_StatefulUniformFullInt_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateful_uniform_full_int(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformFullInt_shape_dtype], dtype:TV_StatefulUniformFullInt_dtype=_dtypes.uint64, name=None) -> Annotated[Any, TV_StatefulUniformFullInt_dtype]:
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers covering the whole range of `dtype`.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.uint64`.
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
        _ctx, "StatefulUniformFullInt", name, resource, algorithm, shape,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_uniform_full_int_eager_fallback(
          resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulUniformFullInt", resource=resource, algorithm=algorithm,
                                  shape=shape, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulUniformFullInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulUniformFullInt = tf_export("raw_ops.StatefulUniformFullInt")(_ops.to_raw_op(stateful_uniform_full_int))


def stateful_uniform_full_int_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformFullInt_shape_dtype], dtype: TV_StatefulUniformFullInt_dtype, name, ctx) -> Annotated[Any, TV_StatefulUniformFullInt_dtype]:
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulUniformFullInt", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulUniformFullInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatefulUniformInt_dtype = TypeVar("TV_StatefulUniformInt_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulUniformInt_shape_dtype = TypeVar("TV_StatefulUniformInt_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateful_uniform_int(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformInt_shape_dtype], minval: Annotated[Any, TV_StatefulUniformInt_dtype], maxval: Annotated[Any, TV_StatefulUniformInt_dtype], name=None) -> Annotated[Any, TV_StatefulUniformInt_dtype]:
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    minval: A `Tensor`. Minimum value (inclusive, scalar).
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
        _ctx, "StatefulUniformInt", name, resource, algorithm, shape, minval,
        maxval)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_uniform_int_eager_fallback(
          resource, algorithm, shape, minval, maxval, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulUniformInt", resource=resource, algorithm=algorithm,
                              shape=shape, minval=minval, maxval=maxval,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulUniformInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatefulUniformInt = tf_export("raw_ops.StatefulUniformInt")(_ops.to_raw_op(stateful_uniform_int))


def stateful_uniform_int_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformInt_shape_dtype], minval: Annotated[Any, TV_StatefulUniformInt_dtype], maxval: Annotated[Any, TV_StatefulUniformInt_dtype], name, ctx) -> Annotated[Any, TV_StatefulUniformInt_dtype]:
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([minval, maxval], ctx, [], _dtypes.int64)
  (minval, maxval) = _inputs_dtype
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape, minval, maxval]
  _attrs = ("dtype", _attr_dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulUniformInt", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulUniformInt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


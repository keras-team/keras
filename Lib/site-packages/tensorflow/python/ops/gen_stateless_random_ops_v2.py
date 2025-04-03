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

TV_StatelessRandomGammaV3_dtype = TypeVar("TV_StatelessRandomGammaV3_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessRandomGammaV3_shape_dtype = TypeVar("TV_StatelessRandomGammaV3_shape_dtype", _atypes.Int32, _atypes.Int64)

def stateless_random_gamma_v3(shape: Annotated[Any, TV_StatelessRandomGammaV3_shape_dtype], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], alpha: Annotated[Any, TV_StatelessRandomGammaV3_dtype], name=None) -> Annotated[Any, TV_StatelessRandomGammaV3_dtype]:
  r"""Outputs deterministic pseudorandom random numbers from a gamma distribution.

  Outputs random values from a gamma distribution.

  The outputs are a deterministic function of the inputs.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
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
        _ctx, "StatelessRandomGammaV3", name, shape, key, counter, alg, alpha)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_gamma_v3_eager_fallback(
          shape, key, counter, alg, alpha, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomGammaV3", shape=shape, key=key, counter=counter,
                                  alg=alg, alpha=alpha, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape_dtype",
              _op._get_attr_type("shape_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomGammaV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomGammaV3 = tf_export("raw_ops.StatelessRandomGammaV3")(_ops.to_raw_op(stateless_random_gamma_v3))


def stateless_random_gamma_v3_eager_fallback(shape: Annotated[Any, TV_StatelessRandomGammaV3_shape_dtype], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], alpha: Annotated[Any, TV_StatelessRandomGammaV3_dtype], name, ctx) -> Annotated[Any, TV_StatelessRandomGammaV3_dtype]:
  _attr_dtype, (alpha,) = _execute.args_to_matching_eager([alpha], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [shape, key, counter, alg, alpha]
  _attrs = ("dtype", _attr_dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatelessRandomGammaV3", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomGammaV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def stateless_random_get_alg(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Picks the best counter-based RNG algorithm based on device.

  This op picks the best counter-based RNG algorithm based on device.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomGetAlg", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_get_alg_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomGetAlg", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomGetAlg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomGetAlg = tf_export("raw_ops.StatelessRandomGetAlg")(_ops.to_raw_op(stateless_random_get_alg))


def stateless_random_get_alg_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"StatelessRandomGetAlg", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomGetAlg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_StatelessRandomGetKeyCounterOutput = collections.namedtuple(
    "StatelessRandomGetKeyCounter",
    ["key", "counter"])


TV_StatelessRandomGetKeyCounter_Tseed = TypeVar("TV_StatelessRandomGetKeyCounter_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_get_key_counter(seed: Annotated[Any, TV_StatelessRandomGetKeyCounter_Tseed], name=None):
  r"""Scrambles seed into key and counter, using the best algorithm based on device.

  This op scrambles a shape-[2] seed into a key and a counter, both needed by counter-based RNG algorithms. The scrambing uses the best algorithm based on device. The scrambling is opaque but approximately satisfies the property that different seed results in different key/counter pair (which will in turn result in different random numbers).

  Args:
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, counter).

    key: A `Tensor` of type `uint64`.
    counter: A `Tensor` of type `uint64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomGetKeyCounter", name, seed)
      _result = _StatelessRandomGetKeyCounterOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_get_key_counter_eager_fallback(
          seed, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomGetKeyCounter", seed=seed, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomGetKeyCounter", _inputs_flat, _attrs, _result)
  _result = _StatelessRandomGetKeyCounterOutput._make(_result)
  return _result

StatelessRandomGetKeyCounter = tf_export("raw_ops.StatelessRandomGetKeyCounter")(_ops.to_raw_op(stateless_random_get_key_counter))


def stateless_random_get_key_counter_eager_fallback(seed: Annotated[Any, TV_StatelessRandomGetKeyCounter_Tseed], name, ctx):
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [seed]
  _attrs = ("Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomGetKeyCounter", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomGetKeyCounter", _inputs_flat, _attrs, _result)
  _result = _StatelessRandomGetKeyCounterOutput._make(_result)
  return _result

_StatelessRandomGetKeyCounterAlgOutput = collections.namedtuple(
    "StatelessRandomGetKeyCounterAlg",
    ["key", "counter", "alg"])


TV_StatelessRandomGetKeyCounterAlg_Tseed = TypeVar("TV_StatelessRandomGetKeyCounterAlg_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_random_get_key_counter_alg(seed: Annotated[Any, TV_StatelessRandomGetKeyCounterAlg_Tseed], name=None):
  r"""Picks the best algorithm based on device, and scrambles seed into key and counter.

  This op picks the best counter-based RNG algorithm based on device, and scrambles a shape-[2] seed into a key and a counter, both needed by the counter-based algorithm. The scrambling is opaque but approximately satisfies the property that different seed results in different key/counter pair (which will in turn result in different random numbers).

  Args:
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, counter, alg).

    key: A `Tensor` of type `uint64`.
    counter: A `Tensor` of type `uint64`.
    alg: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessRandomGetKeyCounterAlg", name, seed)
      _result = _StatelessRandomGetKeyCounterAlgOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_get_key_counter_alg_eager_fallback(
          seed, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomGetKeyCounterAlg", seed=seed, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tseed", _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomGetKeyCounterAlg", _inputs_flat, _attrs, _result)
  _result = _StatelessRandomGetKeyCounterAlgOutput._make(_result)
  return _result

StatelessRandomGetKeyCounterAlg = tf_export("raw_ops.StatelessRandomGetKeyCounterAlg")(_ops.to_raw_op(stateless_random_get_key_counter_alg))


def stateless_random_get_key_counter_alg_eager_fallback(seed: Annotated[Any, TV_StatelessRandomGetKeyCounterAlg_Tseed], name, ctx):
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [seed]
  _attrs = ("Tseed", _attr_Tseed)
  _result = _execute.execute(b"StatelessRandomGetKeyCounterAlg", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomGetKeyCounterAlg", _inputs_flat, _attrs, _result)
  _result = _StatelessRandomGetKeyCounterAlgOutput._make(_result)
  return _result


TV_StatelessRandomNormalV2_dtype = TypeVar("TV_StatelessRandomNormalV2_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessRandomNormalV2_Tshape = TypeVar("TV_StatelessRandomNormalV2_Tshape", _atypes.Int32, _atypes.Int64)

def stateless_random_normal_v2(shape: Annotated[Any, TV_StatelessRandomNormalV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype:TV_StatelessRandomNormalV2_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatelessRandomNormalV2_dtype]:
  r"""Outputs deterministic pseudorandom values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
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
        _ctx, "StatelessRandomNormalV2", name, shape, key, counter, alg,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_normal_v2_eager_fallback(
          shape, key, counter, alg, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomNormalV2", shape=shape, key=key, counter=counter,
                                   alg=alg, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomNormalV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomNormalV2 = tf_export("raw_ops.StatelessRandomNormalV2")(_ops.to_raw_op(stateless_random_normal_v2))


def stateless_random_normal_v2_eager_fallback(shape: Annotated[Any, TV_StatelessRandomNormalV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype: TV_StatelessRandomNormalV2_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomNormalV2_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [shape, key, counter, alg]
  _attrs = ("dtype", dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"StatelessRandomNormalV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomNormalV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomUniformFullIntV2_dtype = TypeVar("TV_StatelessRandomUniformFullIntV2_dtype", _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)
TV_StatelessRandomUniformFullIntV2_Tshape = TypeVar("TV_StatelessRandomUniformFullIntV2_Tshape", _atypes.Int32, _atypes.Int64)

def stateless_random_uniform_full_int_v2(shape: Annotated[Any, TV_StatelessRandomUniformFullIntV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype:TV_StatelessRandomUniformFullIntV2_dtype=_dtypes.uint64, name=None) -> Annotated[Any, TV_StatelessRandomUniformFullIntV2_dtype]:
  r"""Outputs deterministic pseudorandom random integers from a uniform distribution.

  The generated values are uniform integers covering the whole range of `dtype`.

  The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
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
        _ctx, "StatelessRandomUniformFullIntV2", name, shape, key, counter,
        alg, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_uniform_full_int_v2_eager_fallback(
          shape, key, counter, alg, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomUniformFullIntV2", shape=shape, key=key,
                                           counter=counter, alg=alg,
                                           dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomUniformFullIntV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomUniformFullIntV2 = tf_export("raw_ops.StatelessRandomUniformFullIntV2")(_ops.to_raw_op(stateless_random_uniform_full_int_v2))


def stateless_random_uniform_full_int_v2_eager_fallback(shape: Annotated[Any, TV_StatelessRandomUniformFullIntV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype: TV_StatelessRandomUniformFullIntV2_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomUniformFullIntV2_dtype]:
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [shape, key, counter, alg]
  _attrs = ("dtype", dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"StatelessRandomUniformFullIntV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomUniformFullIntV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomUniformIntV2_dtype = TypeVar("TV_StatelessRandomUniformIntV2_dtype", _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)
TV_StatelessRandomUniformIntV2_Tshape = TypeVar("TV_StatelessRandomUniformIntV2_Tshape", _atypes.Int32, _atypes.Int64)

def stateless_random_uniform_int_v2(shape: Annotated[Any, TV_StatelessRandomUniformIntV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], minval: Annotated[Any, TV_StatelessRandomUniformIntV2_dtype], maxval: Annotated[Any, TV_StatelessRandomUniformIntV2_dtype], name=None) -> Annotated[Any, TV_StatelessRandomUniformIntV2_dtype]:
  r"""Outputs deterministic pseudorandom random integers from a uniform distribution.

  The generated values follow a uniform distribution in the range `[minval, maxval)`.

  The outputs are a deterministic function of `shape`, `key`, `counter`, `alg`, `minval` and `maxval`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
    minval: A `Tensor`. Must be one of the following types: `int32`, `int64`, `uint32`, `uint64`.
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
        _ctx, "StatelessRandomUniformIntV2", name, shape, key, counter, alg,
        minval, maxval)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_uniform_int_v2_eager_fallback(
          shape, key, counter, alg, minval, maxval, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomUniformIntV2", shape=shape, key=key, counter=counter,
                                       alg=alg, minval=minval, maxval=maxval,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomUniformIntV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomUniformIntV2 = tf_export("raw_ops.StatelessRandomUniformIntV2")(_ops.to_raw_op(stateless_random_uniform_int_v2))


def stateless_random_uniform_int_v2_eager_fallback(shape: Annotated[Any, TV_StatelessRandomUniformIntV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], minval: Annotated[Any, TV_StatelessRandomUniformIntV2_dtype], maxval: Annotated[Any, TV_StatelessRandomUniformIntV2_dtype], name, ctx) -> Annotated[Any, TV_StatelessRandomUniformIntV2_dtype]:
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([minval, maxval], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.uint32, _dtypes.uint64, ])
  (minval, maxval) = _inputs_dtype
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [shape, key, counter, alg, minval, maxval]
  _attrs = ("dtype", _attr_dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"StatelessRandomUniformIntV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomUniformIntV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessRandomUniformV2_dtype = TypeVar("TV_StatelessRandomUniformV2_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessRandomUniformV2_Tshape = TypeVar("TV_StatelessRandomUniformV2_Tshape", _atypes.Int32, _atypes.Int64)

def stateless_random_uniform_v2(shape: Annotated[Any, TV_StatelessRandomUniformV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype:TV_StatelessRandomUniformV2_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatelessRandomUniformV2_dtype]:
  r"""Outputs deterministic pseudorandom random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
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
        _ctx, "StatelessRandomUniformV2", name, shape, key, counter, alg,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_random_uniform_v2_eager_fallback(
          shape, key, counter, alg, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessRandomUniformV2", shape=shape, key=key, counter=counter,
                                    alg=alg, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessRandomUniformV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessRandomUniformV2 = tf_export("raw_ops.StatelessRandomUniformV2")(_ops.to_raw_op(stateless_random_uniform_v2))


def stateless_random_uniform_v2_eager_fallback(shape: Annotated[Any, TV_StatelessRandomUniformV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype: TV_StatelessRandomUniformV2_dtype, name, ctx) -> Annotated[Any, TV_StatelessRandomUniformV2_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [shape, key, counter, alg]
  _attrs = ("dtype", dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"StatelessRandomUniformV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessRandomUniformV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessShuffle_T = TypeVar("TV_StatelessShuffle_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateless_shuffle(value: Annotated[Any, TV_StatelessShuffle_T], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_StatelessShuffle_T]:
  r"""Randomly and deterministically shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

  ```
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```

  The outputs are a deterministic function of `value`, `key`, `counter` and `alg`.

  Args:
    value: A `Tensor`. The tensor to be shuffled.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessShuffle", name, value, key, counter, alg)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_shuffle_eager_fallback(
          value, key, counter, alg, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessShuffle", value=value, key=key, counter=counter, alg=alg,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessShuffle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessShuffle = tf_export("raw_ops.StatelessShuffle")(_ops.to_raw_op(stateless_shuffle))


def stateless_shuffle_eager_fallback(value: Annotated[Any, TV_StatelessShuffle_T], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_StatelessShuffle_T]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [value, key, counter, alg]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"StatelessShuffle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessShuffle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StatelessTruncatedNormalV2_dtype = TypeVar("TV_StatelessTruncatedNormalV2_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_StatelessTruncatedNormalV2_Tshape = TypeVar("TV_StatelessTruncatedNormalV2_Tshape", _atypes.Int32, _atypes.Int64)

def stateless_truncated_normal_v2(shape: Annotated[Any, TV_StatelessTruncatedNormalV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype:TV_StatelessTruncatedNormalV2_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_StatelessTruncatedNormalV2_dtype]:
  r"""Outputs deterministic pseudorandom values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
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
        _ctx, "StatelessTruncatedNormalV2", name, shape, key, counter, alg,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_truncated_normal_v2_eager_fallback(
          shape, key, counter, alg, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessTruncatedNormalV2", shape=shape, key=key, counter=counter,
                                      alg=alg, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessTruncatedNormalV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatelessTruncatedNormalV2 = tf_export("raw_ops.StatelessTruncatedNormalV2")(_ops.to_raw_op(stateless_truncated_normal_v2))


def stateless_truncated_normal_v2_eager_fallback(shape: Annotated[Any, TV_StatelessTruncatedNormalV2_Tshape], key: Annotated[Any, _atypes.UInt64], counter: Annotated[Any, _atypes.UInt64], alg: Annotated[Any, _atypes.Int32], dtype: TV_StatelessTruncatedNormalV2_dtype, name, ctx) -> Annotated[Any, TV_StatelessTruncatedNormalV2_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  key = _ops.convert_to_tensor(key, _dtypes.uint64)
  counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
  alg = _ops.convert_to_tensor(alg, _dtypes.int32)
  _inputs_flat = [shape, key, counter, alg]
  _attrs = ("dtype", dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"StatelessTruncatedNormalV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessTruncatedNormalV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


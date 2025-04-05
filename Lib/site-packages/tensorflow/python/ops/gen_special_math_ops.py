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

TV_BesselI0_T = TypeVar("TV_BesselI0_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_i0(x: Annotated[Any, TV_BesselI0_T], name=None) -> Annotated[Any, TV_BesselI0_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselI0", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_i0_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselI0", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselI0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselI0 = tf_export("raw_ops.BesselI0")(_ops.to_raw_op(bessel_i0))


def bessel_i0_eager_fallback(x: Annotated[Any, TV_BesselI0_T], name, ctx) -> Annotated[Any, TV_BesselI0_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselI0", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselI0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselI0e_T = TypeVar("TV_BesselI0e_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_i0e(x: Annotated[Any, TV_BesselI0e_T], name=None) -> Annotated[Any, TV_BesselI0e_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselI0e", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_i0e_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselI0e", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselI0e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselI0e = tf_export("raw_ops.BesselI0e")(_ops.to_raw_op(bessel_i0e))


def bessel_i0e_eager_fallback(x: Annotated[Any, TV_BesselI0e_T], name, ctx) -> Annotated[Any, TV_BesselI0e_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselI0e", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselI0e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselI1_T = TypeVar("TV_BesselI1_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_i1(x: Annotated[Any, TV_BesselI1_T], name=None) -> Annotated[Any, TV_BesselI1_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselI1", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_i1_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselI1", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselI1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselI1 = tf_export("raw_ops.BesselI1")(_ops.to_raw_op(bessel_i1))


def bessel_i1_eager_fallback(x: Annotated[Any, TV_BesselI1_T], name, ctx) -> Annotated[Any, TV_BesselI1_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselI1", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselI1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselI1e_T = TypeVar("TV_BesselI1e_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_i1e(x: Annotated[Any, TV_BesselI1e_T], name=None) -> Annotated[Any, TV_BesselI1e_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselI1e", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_i1e_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselI1e", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselI1e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselI1e = tf_export("raw_ops.BesselI1e")(_ops.to_raw_op(bessel_i1e))


def bessel_i1e_eager_fallback(x: Annotated[Any, TV_BesselI1e_T], name, ctx) -> Annotated[Any, TV_BesselI1e_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselI1e", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselI1e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselJ0_T = TypeVar("TV_BesselJ0_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_j0(x: Annotated[Any, TV_BesselJ0_T], name=None) -> Annotated[Any, TV_BesselJ0_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselJ0", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_j0_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselJ0", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselJ0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselJ0 = tf_export("raw_ops.BesselJ0")(_ops.to_raw_op(bessel_j0))


def bessel_j0_eager_fallback(x: Annotated[Any, TV_BesselJ0_T], name, ctx) -> Annotated[Any, TV_BesselJ0_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselJ0", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselJ0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselJ1_T = TypeVar("TV_BesselJ1_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_j1(x: Annotated[Any, TV_BesselJ1_T], name=None) -> Annotated[Any, TV_BesselJ1_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselJ1", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_j1_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselJ1", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselJ1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselJ1 = tf_export("raw_ops.BesselJ1")(_ops.to_raw_op(bessel_j1))


def bessel_j1_eager_fallback(x: Annotated[Any, TV_BesselJ1_T], name, ctx) -> Annotated[Any, TV_BesselJ1_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselJ1", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselJ1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselK0_T = TypeVar("TV_BesselK0_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_k0(x: Annotated[Any, TV_BesselK0_T], name=None) -> Annotated[Any, TV_BesselK0_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselK0", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_k0_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselK0", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselK0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselK0 = tf_export("raw_ops.BesselK0")(_ops.to_raw_op(bessel_k0))


def bessel_k0_eager_fallback(x: Annotated[Any, TV_BesselK0_T], name, ctx) -> Annotated[Any, TV_BesselK0_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselK0", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselK0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselK0e_T = TypeVar("TV_BesselK0e_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_k0e(x: Annotated[Any, TV_BesselK0e_T], name=None) -> Annotated[Any, TV_BesselK0e_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselK0e", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_k0e_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselK0e", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselK0e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselK0e = tf_export("raw_ops.BesselK0e")(_ops.to_raw_op(bessel_k0e))


def bessel_k0e_eager_fallback(x: Annotated[Any, TV_BesselK0e_T], name, ctx) -> Annotated[Any, TV_BesselK0e_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselK0e", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselK0e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselK1_T = TypeVar("TV_BesselK1_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_k1(x: Annotated[Any, TV_BesselK1_T], name=None) -> Annotated[Any, TV_BesselK1_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselK1", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_k1_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselK1", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselK1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselK1 = tf_export("raw_ops.BesselK1")(_ops.to_raw_op(bessel_k1))


def bessel_k1_eager_fallback(x: Annotated[Any, TV_BesselK1_T], name, ctx) -> Annotated[Any, TV_BesselK1_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselK1", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselK1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselK1e_T = TypeVar("TV_BesselK1e_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_k1e(x: Annotated[Any, TV_BesselK1e_T], name=None) -> Annotated[Any, TV_BesselK1e_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselK1e", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_k1e_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselK1e", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselK1e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselK1e = tf_export("raw_ops.BesselK1e")(_ops.to_raw_op(bessel_k1e))


def bessel_k1e_eager_fallback(x: Annotated[Any, TV_BesselK1e_T], name, ctx) -> Annotated[Any, TV_BesselK1e_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselK1e", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselK1e", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselY0_T = TypeVar("TV_BesselY0_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_y0(x: Annotated[Any, TV_BesselY0_T], name=None) -> Annotated[Any, TV_BesselY0_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselY0", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_y0_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselY0", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselY0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselY0 = tf_export("raw_ops.BesselY0")(_ops.to_raw_op(bessel_y0))


def bessel_y0_eager_fallback(x: Annotated[Any, TV_BesselY0_T], name, ctx) -> Annotated[Any, TV_BesselY0_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselY0", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselY0", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BesselY1_T = TypeVar("TV_BesselY1_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def bessel_y1(x: Annotated[Any, TV_BesselY1_T], name=None) -> Annotated[Any, TV_BesselY1_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BesselY1", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bessel_y1_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BesselY1", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BesselY1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BesselY1 = tf_export("raw_ops.BesselY1")(_ops.to_raw_op(bessel_y1))


def bessel_y1_eager_fallback(x: Annotated[Any, TV_BesselY1_T], name, ctx) -> Annotated[Any, TV_BesselY1_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BesselY1", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BesselY1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Dawsn_T = TypeVar("TV_Dawsn_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def dawsn(x: Annotated[Any, TV_Dawsn_T], name=None) -> Annotated[Any, TV_Dawsn_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Dawsn", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dawsn_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Dawsn", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Dawsn", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Dawsn = tf_export("raw_ops.Dawsn")(_ops.to_raw_op(dawsn))


def dawsn_eager_fallback(x: Annotated[Any, TV_Dawsn_T], name, ctx) -> Annotated[Any, TV_Dawsn_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Dawsn", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Dawsn", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Expint_T = TypeVar("TV_Expint_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def expint(x: Annotated[Any, TV_Expint_T], name=None) -> Annotated[Any, TV_Expint_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Expint", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return expint_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Expint", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Expint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Expint = tf_export("raw_ops.Expint")(_ops.to_raw_op(expint))


def expint_eager_fallback(x: Annotated[Any, TV_Expint_T], name, ctx) -> Annotated[Any, TV_Expint_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Expint", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Expint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FresnelCos_T = TypeVar("TV_FresnelCos_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def fresnel_cos(x: Annotated[Any, TV_FresnelCos_T], name=None) -> Annotated[Any, TV_FresnelCos_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FresnelCos", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fresnel_cos_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FresnelCos", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FresnelCos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FresnelCos = tf_export("raw_ops.FresnelCos")(_ops.to_raw_op(fresnel_cos))


def fresnel_cos_eager_fallback(x: Annotated[Any, TV_FresnelCos_T], name, ctx) -> Annotated[Any, TV_FresnelCos_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"FresnelCos", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FresnelCos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FresnelSin_T = TypeVar("TV_FresnelSin_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def fresnel_sin(x: Annotated[Any, TV_FresnelSin_T], name=None) -> Annotated[Any, TV_FresnelSin_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FresnelSin", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fresnel_sin_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FresnelSin", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FresnelSin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FresnelSin = tf_export("raw_ops.FresnelSin")(_ops.to_raw_op(fresnel_sin))


def fresnel_sin_eager_fallback(x: Annotated[Any, TV_FresnelSin_T], name, ctx) -> Annotated[Any, TV_FresnelSin_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"FresnelSin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FresnelSin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Spence_T = TypeVar("TV_Spence_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def spence(x: Annotated[Any, TV_Spence_T], name=None) -> Annotated[Any, TV_Spence_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Spence", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return spence_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Spence", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Spence", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Spence = tf_export("raw_ops.Spence")(_ops.to_raw_op(spence))


def spence_eager_fallback(x: Annotated[Any, TV_Spence_T], name, ctx) -> Annotated[Any, TV_Spence_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Spence", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Spence", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


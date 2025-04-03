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

TV_BatchMatrixBandPart_T = TypeVar("TV_BatchMatrixBandPart_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def batch_matrix_band_part(input: Annotated[Any, TV_BatchMatrixBandPart_T], num_lower: Annotated[Any, _atypes.Int64], num_upper: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, TV_BatchMatrixBandPart_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    num_lower: A `Tensor` of type `int64`.
    num_upper: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatrixBandPart", name, input, num_lower, num_upper)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_matrix_band_part_eager_fallback(
          input, num_lower, num_upper, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatrixBandPart", input=input, num_lower=num_lower,
                               num_upper=num_upper, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatrixBandPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatrixBandPart = tf_export("raw_ops.BatchMatrixBandPart")(_ops.to_raw_op(batch_matrix_band_part))


def batch_matrix_band_part_eager_fallback(input: Annotated[Any, TV_BatchMatrixBandPart_T], num_lower: Annotated[Any, _atypes.Int64], num_upper: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, TV_BatchMatrixBandPart_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  num_lower = _ops.convert_to_tensor(num_lower, _dtypes.int64)
  num_upper = _ops.convert_to_tensor(num_upper, _dtypes.int64)
  _inputs_flat = [input, num_lower, num_upper]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BatchMatrixBandPart", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatrixBandPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchMatrixDiag_T = TypeVar("TV_BatchMatrixDiag_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def batch_matrix_diag(diagonal: Annotated[Any, TV_BatchMatrixDiag_T], name=None) -> Annotated[Any, TV_BatchMatrixDiag_T]:
  r"""TODO: add doc.

  Args:
    diagonal: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatrixDiag", name, diagonal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_matrix_diag_eager_fallback(
          diagonal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatrixDiag", diagonal=diagonal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatrixDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatrixDiag = tf_export("raw_ops.BatchMatrixDiag")(_ops.to_raw_op(batch_matrix_diag))


def batch_matrix_diag_eager_fallback(diagonal: Annotated[Any, TV_BatchMatrixDiag_T], name, ctx) -> Annotated[Any, TV_BatchMatrixDiag_T]:
  _attr_T, (diagonal,) = _execute.args_to_matching_eager([diagonal], ctx, [])
  _inputs_flat = [diagonal]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BatchMatrixDiag", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatrixDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchMatrixDiagPart_T = TypeVar("TV_BatchMatrixDiagPart_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def batch_matrix_diag_part(input: Annotated[Any, TV_BatchMatrixDiagPart_T], name=None) -> Annotated[Any, TV_BatchMatrixDiagPart_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatrixDiagPart", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_matrix_diag_part_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatrixDiagPart", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatrixDiagPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatrixDiagPart = tf_export("raw_ops.BatchMatrixDiagPart")(_ops.to_raw_op(batch_matrix_diag_part))


def batch_matrix_diag_part_eager_fallback(input: Annotated[Any, TV_BatchMatrixDiagPart_T], name, ctx) -> Annotated[Any, TV_BatchMatrixDiagPart_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BatchMatrixDiagPart", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatrixDiagPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchMatrixSetDiag_T = TypeVar("TV_BatchMatrixSetDiag_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def batch_matrix_set_diag(input: Annotated[Any, TV_BatchMatrixSetDiag_T], diagonal: Annotated[Any, TV_BatchMatrixSetDiag_T], name=None) -> Annotated[Any, TV_BatchMatrixSetDiag_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    diagonal: A `Tensor`. Must have the same type as `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatrixSetDiag", name, input, diagonal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_matrix_set_diag_eager_fallback(
          input, diagonal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatrixSetDiag", input=input, diagonal=diagonal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatrixSetDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatrixSetDiag = tf_export("raw_ops.BatchMatrixSetDiag")(_ops.to_raw_op(batch_matrix_set_diag))


def batch_matrix_set_diag_eager_fallback(input: Annotated[Any, TV_BatchMatrixSetDiag_T], diagonal: Annotated[Any, TV_BatchMatrixSetDiag_T], name, ctx) -> Annotated[Any, TV_BatchMatrixSetDiag_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, diagonal], ctx, [])
  (input, diagonal) = _inputs_T
  _inputs_flat = [input, diagonal]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BatchMatrixSetDiag", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatrixSetDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchToSpace_T = TypeVar("TV_BatchToSpace_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_BatchToSpace_Tidx = TypeVar("TV_BatchToSpace_Tidx", _atypes.Int32, _atypes.Int64)

def batch_to_space(input: Annotated[Any, TV_BatchToSpace_T], crops: Annotated[Any, TV_BatchToSpace_Tidx], block_size: int, name=None) -> Annotated[Any, TV_BatchToSpace_T]:
  r"""BatchToSpace for 4-D tensors of type T.

  This is a legacy version of the more general BatchToSpaceND.

  Rearranges (permutes) data from batch into blocks of spatial data, followed by
  cropping. This is the reverse transformation of SpaceToBatch. More specifically,
  this op outputs a copy of the input tensor where values from the `batch`
  dimension are moved in spatial blocks to the `height` and `width` dimensions,
  followed by cropping along the `height` and `width` dimensions.

  Args:
    input: A `Tensor`. 4-D tensor with shape
      `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
        depth]`. Note that the batch size of the input tensor must be divisible by
      `block_size * block_size`.
    crops: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
      how many elements to crop from the intermediate result across the spatial
      dimensions as follows:

          crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
    block_size: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchToSpace", name, input, crops, "block_size", block_size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_to_space_eager_fallback(
          input, crops, block_size=block_size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  block_size = _execute.make_int(block_size, "block_size")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchToSpace", input=input, crops=crops, block_size=block_size,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "block_size",
              _op._get_attr_int("block_size"), "Tidx",
              _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchToSpace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchToSpace = tf_export("raw_ops.BatchToSpace")(_ops.to_raw_op(batch_to_space))


def batch_to_space_eager_fallback(input: Annotated[Any, TV_BatchToSpace_T], crops: Annotated[Any, TV_BatchToSpace_Tidx], block_size: int, name, ctx) -> Annotated[Any, TV_BatchToSpace_T]:
  block_size = _execute.make_int(block_size, "block_size")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tidx, (crops,) = _execute.args_to_matching_eager([crops], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, crops]
  _attrs = ("T", _attr_T, "block_size", block_size, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"BatchToSpace", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchToSpace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchToSpaceND_T = TypeVar("TV_BatchToSpaceND_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_BatchToSpaceND_Tblock_shape = TypeVar("TV_BatchToSpaceND_Tblock_shape", _atypes.Int32, _atypes.Int64)
TV_BatchToSpaceND_Tcrops = TypeVar("TV_BatchToSpaceND_Tcrops", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export(v1=['batch_to_space_nd', 'manip.batch_to_space_nd'])
@deprecated_endpoints('batch_to_space_nd', 'manip.batch_to_space_nd')
def batch_to_space_nd(input: Annotated[Any, TV_BatchToSpaceND_T], block_shape: Annotated[Any, TV_BatchToSpaceND_Tblock_shape], crops: Annotated[Any, TV_BatchToSpaceND_Tcrops], name=None) -> Annotated[Any, TV_BatchToSpaceND_T]:
  r"""BatchToSpace for N-D tensors of type T.

  This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
  `block_shape + [batch]`, interleaves these blocks back into the grid defined by
  the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
  the input.  The spatial dimensions of this intermediate result are then
  optionally cropped according to `crops` to produce the output.  This is the
  reverse of SpaceToBatch.  See below for a precise description.

  Args:
    input: A `Tensor`.
      N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
      where spatial_shape has M dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with shape `[M]`, all values must be >= 1.
    crops: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D with shape `[M, 2]`, all values must be >= 0.
        `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
        dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
        required that
        `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.

      This operation is equivalent to the following steps:

      1. Reshape `input` to `reshaped` of shape:
           [block_shape[0], ..., block_shape[M-1],
            batch / prod(block_shape),
            input_shape[1], ..., input_shape[N-1]]

      2. Permute dimensions of `reshaped` to produce `permuted` of shape
           [batch / prod(block_shape),

            input_shape[1], block_shape[0],
            ...,
            input_shape[M], block_shape[M-1],

            input_shape[M+1], ..., input_shape[N-1]]

      3. Reshape `permuted` to produce `reshaped_permuted` of shape
           [batch / prod(block_shape),

            input_shape[1] * block_shape[0],
            ...,
            input_shape[M] * block_shape[M-1],

            input_shape[M+1],
            ...,
            input_shape[N-1]]

      4. Crop the start and end of dimensions `[1, ..., M]` of
         `reshaped_permuted` according to `crops` to produce the output of shape:
           [batch / prod(block_shape),

            input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
            ...,
            input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],

            input_shape[M+1], ..., input_shape[N-1]]

      Some examples:

      (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [0, 0]]`:

      ```
      [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
      ```

      The output tensor has shape `[1, 2, 2, 1]` and value:

      ```
      x = [[[[1], [2]], [[3], [4]]]]
      ```

      (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [0, 0]]`:

      ```
      [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
      ```

      The output tensor has shape `[1, 2, 2, 3]` and value:

      ```
      x = [[[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]]]
      ```

      (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1], [3]], [[9], [11]]],
           [[[2], [4]], [[10], [12]]],
           [[[5], [7]], [[13], [15]]],
           [[[6], [8]], [[14], [16]]]]
      ```

      The output tensor has shape `[1, 4, 4, 1]` and value:

      ```
      x = [[[[1],   [2],  [3],  [4]],
           [[5],   [6],  [7],  [8]],
           [[9],  [10], [11],  [12]],
           [[13], [14], [15],  [16]]]]
      ```

      (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [2, 0]]`:

      ```
      x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
           [[[0], [2], [4]]], [[[0], [10], [12]]],
           [[[0], [5], [7]]], [[[0], [13], [15]]],
           [[[0], [6], [8]]], [[[0], [14], [16]]]]
      ```

      The output tensor has shape `[2, 2, 4, 1]` and value:

      ```
      x = [[[[1],   [2],  [3],  [4]],
            [[5],   [6],  [7],  [8]]],
           [[[9],  [10], [11],  [12]],
            [[13], [14], [15],  [16]]]]
      ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchToSpaceND", name, input, block_shape, crops)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_batch_to_space_nd(
          (input, block_shape, crops, name,), None)
      if _result is not NotImplemented:
        return _result
      return batch_to_space_nd_eager_fallback(
          input, block_shape, crops, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            batch_to_space_nd, (), dict(input=input, block_shape=block_shape,
                                        crops=crops, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_batch_to_space_nd(
        (input, block_shape, crops, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchToSpaceND", input=input, block_shape=block_shape, crops=crops,
                          name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          batch_to_space_nd, (), dict(input=input, block_shape=block_shape,
                                      crops=crops, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tblock_shape",
              _op._get_attr_type("Tblock_shape"), "Tcrops",
              _op._get_attr_type("Tcrops"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchToSpaceND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchToSpaceND = tf_export("raw_ops.BatchToSpaceND")(_ops.to_raw_op(batch_to_space_nd))
_dispatcher_for_batch_to_space_nd = batch_to_space_nd._tf_type_based_dispatcher.Dispatch


def batch_to_space_nd_eager_fallback(input: Annotated[Any, TV_BatchToSpaceND_T], block_shape: Annotated[Any, TV_BatchToSpaceND_Tblock_shape], crops: Annotated[Any, TV_BatchToSpaceND_Tcrops], name, ctx) -> Annotated[Any, TV_BatchToSpaceND_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tblock_shape, (block_shape,) = _execute.args_to_matching_eager([block_shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tcrops, (crops,) = _execute.args_to_matching_eager([crops], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, block_shape, crops]
  _attrs = ("T", _attr_T, "Tblock_shape", _attr_Tblock_shape, "Tcrops",
  _attr_Tcrops)
  _result = _execute.execute(b"BatchToSpaceND", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchToSpaceND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Bitcast_T = TypeVar("TV_Bitcast_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Bitcast_type = TypeVar("TV_Bitcast_type", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitcast')
def bitcast(input: Annotated[Any, TV_Bitcast_T], type: TV_Bitcast_type, name=None) -> Annotated[Any, TV_Bitcast_type]:
  r"""Bitcasts a tensor from one type to another without copying data.

  Given a tensor `input`, this operation returns a tensor that has the same buffer
  data as `input` with datatype `type`.

  If the input datatype `T` is larger than the output datatype `type` then the
  shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

  If `T` is smaller than `type`, the operator requires that the rightmost
  dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
  [..., sizeof(`type`)/sizeof(`T`)] to [...].

  tf.bitcast() and tf.cast() work differently when real dtype is casted as a complex dtype
  (e.g. tf.complex64 or tf.complex128) as tf.cast() make imaginary part 0 while tf.bitcast()
  gives module error.
  For example,

  Example 1:

  >>> a = [1., 2., 3.]
  >>> equality_bitcast = tf.bitcast(a, tf.complex128)
  Traceback (most recent call last):
  ...
  InvalidArgumentError: Cannot bitcast from 1 to 18 [Op:Bitcast]
  >>> equality_cast = tf.cast(a, tf.complex128)
  >>> print(equality_cast)
  tf.Tensor([1.+0.j 2.+0.j 3.+0.j], shape=(3,), dtype=complex128)

  Example 2:

  >>> tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
  <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([255, 255, 255, 255], dtype=uint8)>

  Example 3:

  >>> x = [1., 2., 3.]
  >>> y = [0., 2., 3.]
  >>> equality= tf.equal(x,y)
  >>> equality_cast = tf.cast(equality,tf.float32)
  >>> equality_bitcast = tf.bitcast(equality_cast,tf.uint8)
  >>> print(equality)
  tf.Tensor([False True True], shape=(3,), dtype=bool)
  >>> print(equality_cast)
  tf.Tensor([0. 1. 1.], shape=(3,), dtype=float32)
  >>> print(equality_bitcast)
  tf.Tensor(
      [[  0   0   0   0]
       [  0   0 128  63]
       [  0   0 128  63]], shape=(3, 4), dtype=uint8)

  *NOTE*: Bitcast is implemented as a low-level cast, so machines with different
  endian orderings will give different results. A copy from input buffer to output
  buffer is made on BE machines when types are of different sizes in order to get
  the same casting results as on LE machines.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `complex64`, `complex128`, `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    type: A `tf.DType` from: `tf.bfloat16, tf.half, tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Bitcast", name, input, "type", type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_bitcast(
          (input, type, name,), None)
      if _result is not NotImplemented:
        return _result
      return bitcast_eager_fallback(
          input, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            bitcast, (), dict(input=input, type=type, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_bitcast(
        (input, type, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Bitcast", input=input, type=type, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          bitcast, (), dict(input=input, type=type, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "type",
              _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Bitcast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Bitcast = tf_export("raw_ops.Bitcast")(_ops.to_raw_op(bitcast))
_dispatcher_for_bitcast = bitcast._tf_type_based_dispatcher.Dispatch


def bitcast_eager_fallback(input: Annotated[Any, TV_Bitcast_T], type: TV_Bitcast_type, name, ctx) -> Annotated[Any, TV_Bitcast_type]:
  type = _execute.make_type(type, "type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int64, _dtypes.int32, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.int8, _dtypes.int16, _dtypes.complex64, _dtypes.complex128, _dtypes.qint8, _dtypes.quint8, _dtypes.qint16, _dtypes.quint16, _dtypes.qint32, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "type", type)
  _result = _execute.execute(b"Bitcast", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Bitcast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BroadcastArgs_T = TypeVar("TV_BroadcastArgs_T", _atypes.Int32, _atypes.Int64)

def broadcast_args(s0: Annotated[Any, TV_BroadcastArgs_T], s1: Annotated[Any, TV_BroadcastArgs_T], name=None) -> Annotated[Any, TV_BroadcastArgs_T]:
  r"""Return the shape of s0 op s1 with broadcast.

  Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
  broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.

  Args:
    s0: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    s1: A `Tensor`. Must have the same type as `s0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `s0`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BroadcastArgs", name, s0, s1)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return broadcast_args_eager_fallback(
          s0, s1, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BroadcastArgs", s0=s0, s1=s1, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BroadcastArgs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BroadcastArgs = tf_export("raw_ops.BroadcastArgs")(_ops.to_raw_op(broadcast_args))


def broadcast_args_eager_fallback(s0: Annotated[Any, TV_BroadcastArgs_T], s1: Annotated[Any, TV_BroadcastArgs_T], name, ctx) -> Annotated[Any, TV_BroadcastArgs_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([s0, s1], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  (s0, s1) = _inputs_T
  _inputs_flat = [s0, s1]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BroadcastArgs", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BroadcastArgs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_BroadcastGradientArgsOutput = collections.namedtuple(
    "BroadcastGradientArgs",
    ["r0", "r1"])


TV_BroadcastGradientArgs_T = TypeVar("TV_BroadcastGradientArgs_T", _atypes.Int32, _atypes.Int64)

def broadcast_gradient_args(s0: Annotated[Any, TV_BroadcastGradientArgs_T], s1: Annotated[Any, TV_BroadcastGradientArgs_T], name=None):
  r"""Return the reduction indices for computing gradients of s0 op s1 with broadcast.

  This is typically used by gradient computations for a broadcasting operation.

  Args:
    s0: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    s1: A `Tensor`. Must have the same type as `s0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r0, r1).

    r0: A `Tensor`. Has the same type as `s0`.
    r1: A `Tensor`. Has the same type as `s0`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BroadcastGradientArgs", name, s0, s1)
      _result = _BroadcastGradientArgsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return broadcast_gradient_args_eager_fallback(
          s0, s1, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BroadcastGradientArgs", s0=s0, s1=s1, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BroadcastGradientArgs", _inputs_flat, _attrs, _result)
  _result = _BroadcastGradientArgsOutput._make(_result)
  return _result

BroadcastGradientArgs = tf_export("raw_ops.BroadcastGradientArgs")(_ops.to_raw_op(broadcast_gradient_args))


def broadcast_gradient_args_eager_fallback(s0: Annotated[Any, TV_BroadcastGradientArgs_T], s1: Annotated[Any, TV_BroadcastGradientArgs_T], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([s0, s1], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  (s0, s1) = _inputs_T
  _inputs_flat = [s0, s1]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BroadcastGradientArgs", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BroadcastGradientArgs", _inputs_flat, _attrs, _result)
  _result = _BroadcastGradientArgsOutput._make(_result)
  return _result


TV_BroadcastTo_T = TypeVar("TV_BroadcastTo_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_BroadcastTo_Tidx = TypeVar("TV_BroadcastTo_Tidx", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('broadcast_to')
def broadcast_to(input: Annotated[Any, TV_BroadcastTo_T], shape: Annotated[Any, TV_BroadcastTo_Tidx], name=None) -> Annotated[Any, TV_BroadcastTo_T]:
  r"""Broadcast an array for a compatible shape.

  Broadcasting is the process of making arrays to have compatible shapes
  for arithmetic operations. Two shapes are compatible if for each
  dimension pair they are either equal or one of them is one.

  For example:

  >>> x = tf.constant([[1, 2, 3]])   # Shape (1, 3,)
  >>> y = tf.broadcast_to(x, [2, 3])
  >>> print(y)
  tf.Tensor(
      [[1 2 3]
       [1 2 3]], shape=(2, 3), dtype=int32)

  In the above example, the input Tensor with the shape of `[1, 3]`
  is broadcasted to output Tensor with shape of `[2, 3]`.

  When broadcasting, if a tensor has fewer axes than necessary its shape is
  padded on the left with ones. So this gives the same result as the previous
  example:

  >>> x = tf.constant([1, 2, 3])   # Shape (3,)
  >>> y = tf.broadcast_to(x, [2, 3])


  When doing broadcasted operations such as multiplying a tensor
  by a scalar, broadcasting (usually) confers some time or space
  benefit, as the broadcasted tensor is never materialized.

  However, `broadcast_to` does not carry with it any such benefits.
  The newly-created tensor takes the full memory of the broadcasted
  shape. (In a graph context, `broadcast_to` might be fused to
  subsequent operation and then be optimized away, however.)

  Args:
    input: A `Tensor`. A Tensor to broadcast.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An 1-D `int` Tensor. The shape of the desired output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BroadcastTo", name, input, shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_broadcast_to(
          (input, shape, name,), None)
      if _result is not NotImplemented:
        return _result
      return broadcast_to_eager_fallback(
          input, shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            broadcast_to, (), dict(input=input, shape=shape, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_broadcast_to(
        (input, shape, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BroadcastTo", input=input, shape=shape, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          broadcast_to, (), dict(input=input, shape=shape, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BroadcastTo", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BroadcastTo = tf_export("raw_ops.BroadcastTo")(_ops.to_raw_op(broadcast_to))
_dispatcher_for_broadcast_to = broadcast_to._tf_type_based_dispatcher.Dispatch


def broadcast_to_eager_fallback(input: Annotated[Any, TV_BroadcastTo_T], shape: Annotated[Any, TV_BroadcastTo_Tidx], name, ctx) -> Annotated[Any, TV_BroadcastTo_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tidx, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, shape]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"BroadcastTo", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BroadcastTo", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CheckNumerics_T = TypeVar("TV_CheckNumerics_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('debugging.check_numerics', v1=['debugging.check_numerics', 'check_numerics'])
@deprecated_endpoints('check_numerics')
def check_numerics(tensor: Annotated[Any, TV_CheckNumerics_T], message: str, name=None) -> Annotated[Any, TV_CheckNumerics_T]:
  r"""Checks a tensor for NaN and Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
  tensor.

  Example usage:

  ``` python
  a = tf.Variable(1.0)
  tf.debugging.check_numerics(a, message='')

  b = tf.Variable(np.nan)
  try:
    tf.debugging.check_numerics(b, message='Checking b')
  except Exception as e:
    assert "Checking b : Tensor had NaN values" in e.message

  c = tf.Variable(np.inf)
  try:
    tf.debugging.check_numerics(c, message='Checking c')
  except Exception as e:
    assert "Checking c : Tensor had Inf values" in e.message
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CheckNumerics", name, tensor, "message", message)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_check_numerics(
          (tensor, message, name,), None)
      if _result is not NotImplemented:
        return _result
      return check_numerics_eager_fallback(
          tensor, message=message, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            check_numerics, (), dict(tensor=tensor, message=message,
                                     name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_check_numerics(
        (tensor, message, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  message = _execute.make_str(message, "message")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CheckNumerics", tensor=tensor, message=message, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          check_numerics, (), dict(tensor=tensor, message=message, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "message",
              _op.get_attr("message"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CheckNumerics", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CheckNumerics = tf_export("raw_ops.CheckNumerics")(_ops.to_raw_op(check_numerics))
_dispatcher_for_check_numerics = check_numerics._tf_type_based_dispatcher.Dispatch


def check_numerics_eager_fallback(tensor: Annotated[Any, TV_CheckNumerics_T], message: str, name, ctx) -> Annotated[Any, TV_CheckNumerics_T]:
  message = _execute.make_str(message, "message")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T, "message", message)
  _result = _execute.execute(b"CheckNumerics", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CheckNumerics", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CheckNumericsV2_T = TypeVar("TV_CheckNumericsV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def check_numerics_v2(tensor: Annotated[Any, TV_CheckNumericsV2_T], message: str, name=None) -> Annotated[Any, TV_CheckNumericsV2_T]:
  r"""Checks a tensor for NaN, -Inf and +Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
  tensor. Unlike CheckNumerics (V1), CheckNumericsV2 distinguishes -Inf and +Inf
  in the errors it throws.

  Args:
    tensor: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CheckNumericsV2", name, tensor, "message", message)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return check_numerics_v2_eager_fallback(
          tensor, message=message, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  message = _execute.make_str(message, "message")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CheckNumericsV2", tensor=tensor, message=message, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "message",
              _op.get_attr("message"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CheckNumericsV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CheckNumericsV2 = tf_export("raw_ops.CheckNumericsV2")(_ops.to_raw_op(check_numerics_v2))


def check_numerics_v2_eager_fallback(tensor: Annotated[Any, TV_CheckNumericsV2_T], message: str, name, ctx) -> Annotated[Any, TV_CheckNumericsV2_T]:
  message = _execute.make_str(message, "message")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T, "message", message)
  _result = _execute.execute(b"CheckNumericsV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CheckNumericsV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Concat_T = TypeVar("TV_Concat_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def concat(concat_dim: Annotated[Any, _atypes.Int32], values: Annotated[List[Any], TV_Concat_T], name=None) -> Annotated[Any, TV_Concat_T]:
  r"""Concatenates tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects with the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Concat", name, concat_dim, values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return concat_eager_fallback(
          concat_dim, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'concat' Op, not %r." % values)
  _attr_N = len(values)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Concat", concat_dim=concat_dim, values=values, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Concat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Concat = tf_export("raw_ops.Concat")(_ops.to_raw_op(concat))


def concat_eager_fallback(concat_dim: Annotated[Any, _atypes.Int32], values: Annotated[List[Any], TV_Concat_T], name, ctx) -> Annotated[Any, TV_Concat_T]:
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'concat' Op, not %r." % values)
  _attr_N = len(values)
  _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
  concat_dim = _ops.convert_to_tensor(concat_dim, _dtypes.int32)
  _inputs_flat = [concat_dim] + list(values)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"Concat", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Concat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ConcatOffset_shape_type = TypeVar("TV_ConcatOffset_shape_type", _atypes.Int32, _atypes.Int64)

def concat_offset(concat_dim: Annotated[Any, _atypes.Int32], shape: Annotated[List[Any], TV_ConcatOffset_shape_type], name=None):
  r"""Computes offsets of concat inputs within its output.

  For example:

  >>> x = [2, 2, 7]
  >>> y = [2, 3, 7]
  >>> z = [2, 9, 7]
  >>> offsets = concat_offset(1, [x, y, z])
  >>> [[a.item() for a in list(off.numpy())] for off in offsets]
  [[0, 0, 0], [0, 2, 0], [0, 5, 0]]

  This is typically used by gradient computations for a concat operation.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      The dimension along which to concatenate.
    shape: A list of at least 2 `Tensor` objects with the same type in: `int32`, `int64`.
      The `N` int32 or int64 vectors representing shape of tensors being concatenated.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `shape` of `Tensor` objects with the same type as `shape`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConcatOffset", name, concat_dim, shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return concat_offset_eager_fallback(
          concat_dim, shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(shape, (list, tuple)):
    raise TypeError(
        "Expected list for 'shape' argument to "
        "'concat_offset' Op, not %r." % shape)
  _attr_N = len(shape)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConcatOffset", concat_dim=concat_dim, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "shape_type",
              _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConcatOffset", _inputs_flat, _attrs, _result)
  return _result

ConcatOffset = tf_export("raw_ops.ConcatOffset")(_ops.to_raw_op(concat_offset))


def concat_offset_eager_fallback(concat_dim: Annotated[Any, _atypes.Int32], shape: Annotated[List[Any], TV_ConcatOffset_shape_type], name, ctx):
  if not isinstance(shape, (list, tuple)):
    raise TypeError(
        "Expected list for 'shape' argument to "
        "'concat_offset' Op, not %r." % shape)
  _attr_N = len(shape)
  _attr_shape_type, shape = _execute.args_to_matching_eager(list(shape), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  concat_dim = _ops.convert_to_tensor(concat_dim, _dtypes.int32)
  _inputs_flat = [concat_dim] + list(shape)
  _attrs = ("N", _attr_N, "shape_type", _attr_shape_type)
  _result = _execute.execute(b"ConcatOffset", _attr_N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConcatOffset", _inputs_flat, _attrs, _result)
  return _result


TV_ConcatV2_T = TypeVar("TV_ConcatV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ConcatV2_Tidx = TypeVar("TV_ConcatV2_Tidx", _atypes.Int32, _atypes.Int64)

def concat_v2(values: Annotated[List[Any], TV_ConcatV2_T], axis: Annotated[Any, TV_ConcatV2_Tidx], name=None) -> Annotated[Any, TV_ConcatV2_T]:
  r"""Concatenates tensors along one dimension.

  Args:
    values: A list of at least 2 `Tensor` objects with the same type.
      List of `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [-rank(values), rank(values)).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConcatV2", name, values, axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return concat_v2_eager_fallback(
          values, axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'concat_v2' Op, not %r." % values)
  _attr_N = len(values)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConcatV2", values=values, axis=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConcatV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConcatV2 = tf_export("raw_ops.ConcatV2")(_ops.to_raw_op(concat_v2))


def concat_v2_eager_fallback(values: Annotated[List[Any], TV_ConcatV2_T], axis: Annotated[Any, TV_ConcatV2_Tidx], name, ctx) -> Annotated[Any, TV_ConcatV2_T]:
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'concat_v2' Op, not %r." % values)
  _attr_N = len(values)
  _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = list(values) + [axis]
  _attrs = ("N", _attr_N, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"ConcatV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConcatV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ConjugateTranspose_T = TypeVar("TV_ConjugateTranspose_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ConjugateTranspose_Tperm = TypeVar("TV_ConjugateTranspose_Tperm", _atypes.Int32, _atypes.Int64)

def conjugate_transpose(x: Annotated[Any, TV_ConjugateTranspose_T], perm: Annotated[Any, TV_ConjugateTranspose_Tperm], name=None) -> Annotated[Any, TV_ConjugateTranspose_T]:
  r"""Shuffle dimensions of x according to a permutation and conjugate the result.

  The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
    `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`

  Args:
    x: A `Tensor`.
    perm: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConjugateTranspose", name, x, perm)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conjugate_transpose_eager_fallback(
          x, perm, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConjugateTranspose", x=x, perm=perm, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tperm",
              _op._get_attr_type("Tperm"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConjugateTranspose", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConjugateTranspose = tf_export("raw_ops.ConjugateTranspose")(_ops.to_raw_op(conjugate_transpose))


def conjugate_transpose_eager_fallback(x: Annotated[Any, TV_ConjugateTranspose_T], perm: Annotated[Any, TV_ConjugateTranspose_Tperm], name, ctx) -> Annotated[Any, TV_ConjugateTranspose_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _attr_Tperm, (perm,) = _execute.args_to_matching_eager([perm], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x, perm]
  _attrs = ("T", _attr_T, "Tperm", _attr_Tperm)
  _result = _execute.execute(b"ConjugateTranspose", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConjugateTranspose", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Const_dtype = TypeVar("TV_Const_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def const(value, dtype: TV_Const_dtype, name=None) -> Annotated[Any, TV_Const_dtype]:
  r"""Returns a constant tensor.

  Args:
    value: A `tf.TensorProto`. Attr `value` is the tensor to return.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Const", name, "value", value, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return const_eager_fallback(
          value=value, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  value = _execute.make_tensor(value, "value")
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Const", value=value, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("value", _op.get_attr("value"), "dtype",
              _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Const", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Const = tf_export("raw_ops.Const")(_ops.to_raw_op(const))


def const_eager_fallback(value, dtype: TV_Const_dtype, name, ctx) -> Annotated[Any, TV_Const_dtype]:
  value = _execute.make_tensor(value, "value")
  dtype = _execute.make_type(dtype, "dtype")
  _inputs_flat = []
  _attrs = ("value", value, "dtype", dtype)
  _result = _execute.execute(b"Const", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Const", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugGradientIdentity_T = TypeVar("TV_DebugGradientIdentity_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_gradient_identity(input: Annotated[Any, TV_DebugGradientIdentity_T], name=None) -> Annotated[Any, TV_DebugGradientIdentity_T]:
  r"""Identity op for gradient debugging.

  This op is hidden from public in Python. It is used by TensorFlow Debugger to
  register gradient tensors for gradient debugging.
  This op operates on non-reference-type tensors.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugGradientIdentity", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_gradient_identity_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugGradientIdentity", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugGradientIdentity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugGradientIdentity = tf_export("raw_ops.DebugGradientIdentity")(_ops.to_raw_op(debug_gradient_identity))


def debug_gradient_identity_eager_fallback(input: Annotated[Any, TV_DebugGradientIdentity_T], name, ctx) -> Annotated[Any, TV_DebugGradientIdentity_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DebugGradientIdentity", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugGradientIdentity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugGradientRefIdentity_T = TypeVar("TV_DebugGradientRefIdentity_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_gradient_ref_identity(input: Annotated[Any, TV_DebugGradientRefIdentity_T], name=None) -> Annotated[Any, TV_DebugGradientRefIdentity_T]:
  r"""Identity op for gradient debugging.

  This op is hidden from public in Python. It is used by TensorFlow Debugger to
  register gradient tensors for gradient debugging.
  This op operates on reference-type tensors.

  Args:
    input: A mutable `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("debug_gradient_ref_identity op does not support eager execution. Arg 'output' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugGradientRefIdentity", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugGradientRefIdentity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugGradientRefIdentity = tf_export("raw_ops.DebugGradientRefIdentity")(_ops.to_raw_op(debug_gradient_ref_identity))


def debug_gradient_ref_identity_eager_fallback(input: Annotated[Any, TV_DebugGradientRefIdentity_T], name, ctx) -> Annotated[Any, TV_DebugGradientRefIdentity_T]:
  raise RuntimeError("debug_gradient_ref_identity op does not support eager execution. Arg 'output' is a ref.")

TV_DeepCopy_T = TypeVar("TV_DeepCopy_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def deep_copy(x: Annotated[Any, TV_DeepCopy_T], name=None) -> Annotated[Any, TV_DeepCopy_T]:
  r"""Makes a copy of `x`.

  Args:
    x: A `Tensor`. The source tensor of type `T`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeepCopy", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return deep_copy_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeepCopy", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DeepCopy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DeepCopy = tf_export("raw_ops.DeepCopy")(_ops.to_raw_op(deep_copy))


def deep_copy_eager_fallback(x: Annotated[Any, TV_DeepCopy_T], name, ctx) -> Annotated[Any, TV_DeepCopy_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DeepCopy", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DeepCopy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DepthToSpace_T = TypeVar("TV_DepthToSpace_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def depth_to_space(input: Annotated[Any, TV_DepthToSpace_T], block_size: int, data_format:str="NHWC", name=None) -> Annotated[Any, TV_DepthToSpace_T]:
  r"""DepthToSpace for tensors of type T.

  Rearranges data from depth into blocks of spatial data.
  This is the reverse transformation of SpaceToDepth. More specifically,
  this op outputs a copy of the input tensor where values from the `depth`
  dimension are moved in spatial blocks to the `height` and `width` dimensions.
  The attr `block_size` indicates the input block size and how the data is moved.

    * Chunks of data of size `block_size * block_size` from depth are rearranged
      into non-overlapping blocks of size `block_size x block_size`
    * The width of the output tensor is `input_depth * block_size`, whereas the
      height is `input_height * block_size`.
    * The Y, X coordinates within each block of the output image are determined
      by the high order component of the input channel index.
    * The depth of the input tensor must be divisible by
      `block_size * block_size`.

  The `data_format` attr specifies the layout of the input and output tensors
  with the following options:
    "NHWC": `[ batch, height, width, channels ]`
    "NCHW": `[ batch, channels, height, width ]`
    "NCHW_VECT_C":
        `qint8 [ batch, channels / 4, height, width, 4 ]`

  It is useful to consider the operation as transforming a 6-D Tensor.
  e.g. for data_format = NHWC,
       Each element in the input tensor can be specified via 6 coordinates,
       ordered by decreasing memory layout significance as:
       n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
                          within the input image, bX, bY means coordinates
                          within the output block, oC means output channels).
       The output would be the input transposed to the following layout:
       n,iY,bY,iX,bX,oC

  This operation is useful for resizing the activations between convolutions
  (but keeping all data), e.g. instead of pooling. It is also useful for training
  purely convolutional models.

  For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
  block_size = 2:

  ```
  x = [[[[1, 2, 3, 4]]]]

  ```

  This operation will output a tensor of shape `[1, 2, 2, 1]`:

  ```
     [[[[1], [2]],
       [[3], [4]]]]
  ```

  Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
  the corresponding output will have 2x2 elements and will have a depth of
  1 channel (1 = `4 / (block_size * block_size)`).
  The output element shape is `[2, 2, 1]`.

  For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

  ```
  x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
  ```

  This operation, for block size of 2, will return the following tensor of shape
  `[1, 2, 2, 3]`

  ```
     [[[[1, 2, 3], [4, 5, 6]],
       [[7, 8, 9], [10, 11, 12]]]]

  ```

  Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

  ```
  x =  [[[[1, 2, 3, 4],
         [5, 6, 7, 8]],
        [[9, 10, 11, 12],
         [13, 14, 15, 16]]]]
  ```

  the operator will return the following tensor of shape `[1 4 4 1]`:

  ```
  x = [[[ [1],   [2],  [5],  [6]],
        [ [3],   [4],  [7],  [8]],
        [ [9],  [10], [13],  [14]],
        [ [11], [12], [15],  [16]]]]

  ```

  Args:
    input: A `Tensor`.
    block_size: An `int` that is `>= 2`.
      The size of the spatial block, same as in Space2Depth.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NCHW_VECT_C"`. Defaults to `"NHWC"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DepthToSpace", name, input, "block_size", block_size,
        "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return depth_to_space_eager_fallback(
          input, block_size=block_size, data_format=data_format, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  block_size = _execute.make_int(block_size, "block_size")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DepthToSpace", input=input, block_size=block_size,
                        data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "block_size",
              _op._get_attr_int("block_size"), "data_format",
              _op.get_attr("data_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DepthToSpace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DepthToSpace = tf_export("raw_ops.DepthToSpace")(_ops.to_raw_op(depth_to_space))


def depth_to_space_eager_fallback(input: Annotated[Any, TV_DepthToSpace_T], block_size: int, data_format: str, name, ctx) -> Annotated[Any, TV_DepthToSpace_T]:
  block_size = _execute.make_int(block_size, "block_size")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "block_size", block_size, "data_format",
  data_format)
  _result = _execute.execute(b"DepthToSpace", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DepthToSpace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Dequantize_T = TypeVar("TV_Dequantize_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_Dequantize_dtype = TypeVar("TV_Dequantize_dtype", _atypes.BFloat16, _atypes.Float32)

def dequantize(input: Annotated[Any, TV_Dequantize_T], min_range: Annotated[Any, _atypes.Float32], max_range: Annotated[Any, _atypes.Float32], mode:str="MIN_COMBINED", narrow_range:bool=False, axis:int=-1, dtype:TV_Dequantize_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_Dequantize_dtype]:
  r"""Dequantize the 'input' tensor into a float or bfloat16 Tensor.

  [min_range, max_range] are scalar floats that specify the range for
  the output. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  if T == qint8: in[i] += (range(T) + 1)/ 2.0
  out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
  ```
  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  If the input comes from a QuantizedRelu6, the output type is
  quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
  0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
  Dequantize on quint8 will take each value, cast to float, and multiply
  by 6 / 255.
  Note that if quantizedtype is qint8, the operation will additionally add
  each value by 128 prior to casting.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```c++
  num_discrete_values = 1 << (# of bits in T)
  range_adjust = num_discrete_values / (num_discrete_values - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = range / num_discrete_values
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
  ```

  If the mode is `SCALED`, dequantization is performed by multiplying each
  input value by a scaling_factor. (Thus an input of 0 always maps to 0.0).

  The scaling_factor is determined from `min_range`, `max_range`, and
  `narrow_range` in a way that is compatible with `QuantizeAndDequantize{V2|V3}`
  and `QuantizeV2`, using the following algorithm:

  ```c++

    const int min_expected_T = std::numeric_limits<T>::min() +
      (narrow_range ? 1 : 0);
    const int max_expected_T = std::numeric_limits<T>::max();
    const float max_expected_T = std::numeric_limits<float>::max();

    const float scale_factor =
      (std::numeric_limits<T>::min() == 0) ? (max_range / max_expected_T)
                                           : std::max(min_range / min_expected_T,
                                                      max_range / max_expected_T);
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    mode: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST", "SCALED"`. Defaults to `"MIN_COMBINED"`.
    narrow_range: An optional `bool`. Defaults to `False`.
    axis: An optional `int`. Defaults to `-1`.
    dtype: An optional `tf.DType` from: `tf.bfloat16, tf.float32`. Defaults to `tf.float32`.
      Type of the output tensor. Currently Dequantize supports float and bfloat16.
      If 'dtype' is 'bfloat16', it only supports 'MIN_COMBINED' mode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Dequantize", name, input, min_range, max_range, "mode", mode,
        "narrow_range", narrow_range, "axis", axis, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dequantize_eager_fallback(
          input, min_range, max_range, mode=mode, narrow_range=narrow_range,
          axis=axis, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if mode is None:
    mode = "MIN_COMBINED"
  mode = _execute.make_str(mode, "mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Dequantize", input=input, min_range=min_range, max_range=max_range,
                      mode=mode, narrow_range=narrow_range, axis=axis,
                      dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "mode", _op.get_attr("mode"),
              "narrow_range", _op._get_attr_bool("narrow_range"), "axis",
              _op._get_attr_int("axis"), "dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Dequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Dequantize = tf_export("raw_ops.Dequantize")(_ops.to_raw_op(dequantize))


def dequantize_eager_fallback(input: Annotated[Any, TV_Dequantize_T], min_range: Annotated[Any, _atypes.Float32], max_range: Annotated[Any, _atypes.Float32], mode: str, narrow_range: bool, axis: int, dtype: TV_Dequantize_dtype, name, ctx) -> Annotated[Any, TV_Dequantize_dtype]:
  if mode is None:
    mode = "MIN_COMBINED"
  mode = _execute.make_str(mode, "mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_range = _ops.convert_to_tensor(min_range, _dtypes.float32)
  max_range = _ops.convert_to_tensor(max_range, _dtypes.float32)
  _inputs_flat = [input, min_range, max_range]
  _attrs = ("T", _attr_T, "mode", mode, "narrow_range", narrow_range, "axis",
  axis, "dtype", dtype)
  _result = _execute.execute(b"Dequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Dequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Diag_T = TypeVar("TV_Diag_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('linalg.tensor_diag', v1=['linalg.tensor_diag', 'diag'])
@deprecated_endpoints('diag')
def diag(diagonal: Annotated[Any, TV_Diag_T], name=None) -> Annotated[Any, TV_Diag_T]:
  r"""Returns a diagonal tensor with a given diagonal values.

  Given a `diagonal`, this operation returns a tensor with the `diagonal` and
  everything else padded with zeros. The diagonal is computed as follows:

  Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
  rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

  `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

  For example:

  ```
  # 'diagonal' is [1, 2, 3, 4]
  tf.diag(diagonal) ==> [[1, 0, 0, 0]
                         [0, 2, 0, 0]
                         [0, 0, 3, 0]
                         [0, 0, 0, 4]]
  ```

  Args:
    diagonal: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      Rank k tensor where k is at most 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Diag", name, diagonal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_diag(
          (diagonal, name,), None)
      if _result is not NotImplemented:
        return _result
      return diag_eager_fallback(
          diagonal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            diag, (), dict(diagonal=diagonal, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_diag(
        (diagonal, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Diag", diagonal=diagonal, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          diag, (), dict(diagonal=diagonal, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Diag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Diag = tf_export("raw_ops.Diag")(_ops.to_raw_op(diag))
_dispatcher_for_diag = diag._tf_type_based_dispatcher.Dispatch


def diag_eager_fallback(diagonal: Annotated[Any, TV_Diag_T], name, ctx) -> Annotated[Any, TV_Diag_T]:
  _attr_T, (diagonal,) = _execute.args_to_matching_eager([diagonal], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [diagonal]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Diag", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Diag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DiagPart_T = TypeVar("TV_DiagPart_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def diag_part(input: Annotated[Any, TV_DiagPart_T], name=None) -> Annotated[Any, TV_DiagPart_T]:
  r"""Returns the diagonal part of the tensor.

  This operation returns a tensor with the `diagonal` part
  of the `input`. The `diagonal` part is computed as follows:

  Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
  tensor of rank `k` with dimensions `[D1,..., Dk]` where:

  `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

  For example:

  ```
  # 'input' is [[1, 0, 0, 0]
                [0, 2, 0, 0]
                [0, 0, 3, 0]
                [0, 0, 0, 4]]

  tf.diag_part(input) ==> [1, 2, 3, 4]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      Rank k tensor where k is even and not zero.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DiagPart", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return diag_part_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DiagPart", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DiagPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DiagPart = tf_export("raw_ops.DiagPart")(_ops.to_raw_op(diag_part))


def diag_part_eager_fallback(input: Annotated[Any, TV_DiagPart_T], name, ctx) -> Annotated[Any, TV_DiagPart_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DiagPart", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DiagPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_EditDistance_T = TypeVar("TV_EditDistance_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def edit_distance(hypothesis_indices: Annotated[Any, _atypes.Int64], hypothesis_values: Annotated[Any, TV_EditDistance_T], hypothesis_shape: Annotated[Any, _atypes.Int64], truth_indices: Annotated[Any, _atypes.Int64], truth_values: Annotated[Any, TV_EditDistance_T], truth_shape: Annotated[Any, _atypes.Int64], normalize:bool=True, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Computes the (possibly normalized) Levenshtein Edit Distance.

  The inputs are variable-length sequences provided by SparseTensors
    (hypothesis_indices, hypothesis_values, hypothesis_shape)
  and
    (truth_indices, truth_values, truth_shape).

  The inputs are:

  Args:
    hypothesis_indices: A `Tensor` of type `int64`.
      The indices of the hypothesis list SparseTensor.
      This is an N x R int64 matrix.
    hypothesis_values: A `Tensor`.
      The values of the hypothesis list SparseTensor.
      This is an N-length vector.
    hypothesis_shape: A `Tensor` of type `int64`.
      The shape of the hypothesis list SparseTensor.
      This is an R-length vector.
    truth_indices: A `Tensor` of type `int64`.
      The indices of the truth list SparseTensor.
      This is an M x R int64 matrix.
    truth_values: A `Tensor`. Must have the same type as `hypothesis_values`.
      The values of the truth list SparseTensor.
      This is an M-length vector.
    truth_shape: A `Tensor` of type `int64`. truth indices, vector.
    normalize: An optional `bool`. Defaults to `True`.
      boolean (if true, edit distances are normalized by length of truth).

      The output is:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EditDistance", name, hypothesis_indices, hypothesis_values,
        hypothesis_shape, truth_indices, truth_values, truth_shape,
        "normalize", normalize)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return edit_distance_eager_fallback(
          hypothesis_indices, hypothesis_values, hypothesis_shape,
          truth_indices, truth_values, truth_shape, normalize=normalize,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if normalize is None:
    normalize = True
  normalize = _execute.make_bool(normalize, "normalize")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EditDistance", hypothesis_indices=hypothesis_indices,
                        hypothesis_values=hypothesis_values,
                        hypothesis_shape=hypothesis_shape,
                        truth_indices=truth_indices,
                        truth_values=truth_values, truth_shape=truth_shape,
                        normalize=normalize, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("normalize", _op._get_attr_bool("normalize"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EditDistance", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EditDistance = tf_export("raw_ops.EditDistance")(_ops.to_raw_op(edit_distance))


def edit_distance_eager_fallback(hypothesis_indices: Annotated[Any, _atypes.Int64], hypothesis_values: Annotated[Any, TV_EditDistance_T], hypothesis_shape: Annotated[Any, _atypes.Int64], truth_indices: Annotated[Any, _atypes.Int64], truth_values: Annotated[Any, TV_EditDistance_T], truth_shape: Annotated[Any, _atypes.Int64], normalize: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if normalize is None:
    normalize = True
  normalize = _execute.make_bool(normalize, "normalize")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([hypothesis_values, truth_values], ctx, [])
  (hypothesis_values, truth_values) = _inputs_T
  hypothesis_indices = _ops.convert_to_tensor(hypothesis_indices, _dtypes.int64)
  hypothesis_shape = _ops.convert_to_tensor(hypothesis_shape, _dtypes.int64)
  truth_indices = _ops.convert_to_tensor(truth_indices, _dtypes.int64)
  truth_shape = _ops.convert_to_tensor(truth_shape, _dtypes.int64)
  _inputs_flat = [hypothesis_indices, hypothesis_values, hypothesis_shape, truth_indices, truth_values, truth_shape]
  _attrs = ("normalize", normalize, "T", _attr_T)
  _result = _execute.execute(b"EditDistance", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EditDistance", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Empty_dtype = TypeVar("TV_Empty_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def empty(shape: Annotated[Any, _atypes.Int32], dtype: TV_Empty_dtype, init:bool=False, name=None) -> Annotated[Any, TV_Empty_dtype]:
  r"""Creates a tensor with the given shape.

This operation creates a tensor of `shape` and `dtype`.

  Args:
    shape: A `Tensor` of type `int32`.
      1-D. Represents the shape of the output tensor.
    dtype: A `tf.DType`.
    init: An optional `bool`. Defaults to `False`.
      If True, initialize the returned tensor with the default value of dtype.  Otherwise, the implementation is free not to initializethe tensor's content.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Empty", name, shape, "dtype", dtype, "init", init)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return empty_eager_fallback(
          shape, dtype=dtype, init=init, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if init is None:
    init = False
  init = _execute.make_bool(init, "init")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Empty", shape=shape, dtype=dtype, init=init, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "init",
              _op._get_attr_bool("init"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Empty", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Empty = tf_export("raw_ops.Empty")(_ops.to_raw_op(empty))


def empty_eager_fallback(shape: Annotated[Any, _atypes.Int32], dtype: TV_Empty_dtype, init: bool, name, ctx) -> Annotated[Any, TV_Empty_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if init is None:
    init = False
  init = _execute.make_bool(init, "init")
  shape = _ops.convert_to_tensor(shape, _dtypes.int32)
  _inputs_flat = [shape]
  _attrs = ("dtype", dtype, "init", init)
  _result = _execute.execute(b"Empty", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Empty", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_EnsureShape_T = TypeVar("TV_EnsureShape_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def ensure_shape(input: Annotated[Any, TV_EnsureShape_T], shape, name=None) -> Annotated[Any, TV_EnsureShape_T]:
  r"""Ensures that the tensor's shape matches the expected shape.

  Raises an error if the input tensor's shape does not match the specified shape.
  Returns the input tensor otherwise.

  Args:
    input: A `Tensor`. A tensor, whose shape is to be validated.
    shape: A `tf.TensorShape` or list of `ints`.
      The expected (possibly partially specified) shape of the input tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnsureShape", name, input, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ensure_shape_eager_fallback(
          input, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnsureShape", input=input, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shape", _op.get_attr("shape"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EnsureShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EnsureShape = tf_export("raw_ops.EnsureShape")(_ops.to_raw_op(ensure_shape))


def ensure_shape_eager_fallback(input: Annotated[Any, TV_EnsureShape_T], shape, name, ctx) -> Annotated[Any, TV_EnsureShape_T]:
  shape = _execute.make_shape(shape, "shape")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("shape", shape, "T", _attr_T)
  _result = _execute.execute(b"EnsureShape", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EnsureShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ExpandDims_T = TypeVar("TV_ExpandDims_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ExpandDims_Tdim = TypeVar("TV_ExpandDims_Tdim", _atypes.Int32, _atypes.Int64)

def expand_dims(input: Annotated[Any, TV_ExpandDims_T], axis: Annotated[Any, TV_ExpandDims_Tdim], name=None) -> Annotated[Any, TV_ExpandDims_T]:
  r"""Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
  zero; if you specify a negative number for `axis` it is counted backward from
  the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```
  # 't' is a tensor of shape [2]
  shape(expand_dims(t, 0)) ==> [1, 2]
  shape(expand_dims(t, 1)) ==> [2, 1]
  shape(expand_dims(t, -1)) ==> [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
  shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
  shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D (scalar). Specifies the dimension index at which to
      expand the shape of `input`. Must be in the range
      `[-rank(input) - 1, rank(input)]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExpandDims", name, input, axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return expand_dims_eager_fallback(
          input, axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExpandDims", input=input, dim=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tdim",
              _op._get_attr_type("Tdim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExpandDims", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExpandDims = tf_export("raw_ops.ExpandDims")(_ops.to_raw_op(expand_dims))


def expand_dims_eager_fallback(input: Annotated[Any, TV_ExpandDims_T], axis: Annotated[Any, TV_ExpandDims_Tdim], name, ctx) -> Annotated[Any, TV_ExpandDims_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tdim, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("T", _attr_T, "Tdim", _attr_Tdim)
  _result = _execute.execute(b"ExpandDims", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExpandDims", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ExtractImagePatches_T = TypeVar("TV_ExtractImagePatches_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def extract_image_patches(images: Annotated[Any, TV_ExtractImagePatches_T], ksizes, strides, rates, padding: str, name=None) -> Annotated[Any, TV_ExtractImagePatches_T]:
  r"""Extract `patches` from `images` and put them in the "depth" output dimension.

  Args:
    images: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `complex64`, `complex128`, `bool`.
      4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
    ksizes: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of `images`.
    strides: A list of `ints` that has length `>= 4`.
      How far the centers of two consecutive patches are in
      the images. Must be: `[1, stride_rows, stride_cols, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      Must be: `[1, rate_rows, rate_cols, 1]`. This is the
      input stride, specifying how far two consecutive patch samples are in the
      input. Equivalent to extracting patches with
      `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by
      subsampling them spatially by a factor of `rates`. This is equivalent to
      `rate` in dilated (a.k.a. Atrous) convolutions.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExtractImagePatches", name, images, "ksizes", ksizes,
        "strides", strides, "rates", rates, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return extract_image_patches_eager_fallback(
          images, ksizes=ksizes, strides=strides, rates=rates,
          padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksizes' argument to "
        "'extract_image_patches' Op, not %r." % ksizes)
  ksizes = [_execute.make_int(_i, "ksizes") for _i in ksizes]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'extract_image_patches' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'extract_image_patches' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExtractImagePatches", images=images, ksizes=ksizes, strides=strides,
                               rates=rates, padding=padding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksizes", _op.get_attr("ksizes"), "strides",
              _op.get_attr("strides"), "rates", _op.get_attr("rates"), "T",
              _op._get_attr_type("T"), "padding", _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExtractImagePatches", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExtractImagePatches = tf_export("raw_ops.ExtractImagePatches")(_ops.to_raw_op(extract_image_patches))


def extract_image_patches_eager_fallback(images: Annotated[Any, TV_ExtractImagePatches_T], ksizes, strides, rates, padding: str, name, ctx) -> Annotated[Any, TV_ExtractImagePatches_T]:
  if not isinstance(ksizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksizes' argument to "
        "'extract_image_patches' Op, not %r." % ksizes)
  ksizes = [_execute.make_int(_i, "ksizes") for _i in ksizes]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'extract_image_patches' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'extract_image_patches' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, _dtypes.bool, ])
  _inputs_flat = [images]
  _attrs = ("ksizes", ksizes, "strides", strides, "rates", rates, "T",
  _attr_T, "padding", padding)
  _result = _execute.execute(b"ExtractImagePatches", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExtractImagePatches", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ExtractVolumePatches_T = TypeVar("TV_ExtractVolumePatches_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('extract_volume_patches')
def extract_volume_patches(input: Annotated[Any, TV_ExtractVolumePatches_T], ksizes, strides, padding: str, name=None) -> Annotated[Any, TV_ExtractVolumePatches_T]:
  r"""Extract `patches` from `input` and put them in the `"depth"` output dimension. 3D extension of `extract_image_patches`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      5-D Tensor with shape `[batch, in_planes, in_rows, in_cols, depth]`.
    ksizes: A list of `ints` that has length `>= 5`.
      The size of the sliding window for each dimension of `input`.
    strides: A list of `ints` that has length `>= 5`.
      1-D of length 5. How far the centers of two consecutive patches are in
      `input`. Must be: `[1, stride_planes, stride_rows, stride_cols, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.

      The size-related attributes are specified as follows:

      ```python
      ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]
      strides = [1, stride_planes, strides_rows, strides_cols, 1]
      ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExtractVolumePatches", name, input, "ksizes", ksizes,
        "strides", strides, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_extract_volume_patches(
          (input, ksizes, strides, padding, name,), None)
      if _result is not NotImplemented:
        return _result
      return extract_volume_patches_eager_fallback(
          input, ksizes=ksizes, strides=strides, padding=padding, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            extract_volume_patches, (), dict(input=input, ksizes=ksizes,
                                             strides=strides, padding=padding,
                                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_extract_volume_patches(
        (input, ksizes, strides, padding, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksizes' argument to "
        "'extract_volume_patches' Op, not %r." % ksizes)
  ksizes = [_execute.make_int(_i, "ksizes") for _i in ksizes]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'extract_volume_patches' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExtractVolumePatches", input=input, ksizes=ksizes, strides=strides,
                                padding=padding, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          extract_volume_patches, (), dict(input=input, ksizes=ksizes,
                                           strides=strides, padding=padding,
                                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksizes", _op.get_attr("ksizes"), "strides",
              _op.get_attr("strides"), "T", _op._get_attr_type("T"),
              "padding", _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExtractVolumePatches", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExtractVolumePatches = tf_export("raw_ops.ExtractVolumePatches")(_ops.to_raw_op(extract_volume_patches))
_dispatcher_for_extract_volume_patches = extract_volume_patches._tf_type_based_dispatcher.Dispatch


def extract_volume_patches_eager_fallback(input: Annotated[Any, TV_ExtractVolumePatches_T], ksizes, strides, padding: str, name, ctx) -> Annotated[Any, TV_ExtractVolumePatches_T]:
  if not isinstance(ksizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksizes' argument to "
        "'extract_volume_patches' Op, not %r." % ksizes)
  ksizes = [_execute.make_int(_i, "ksizes") for _i in ksizes]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'extract_volume_patches' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [input]
  _attrs = ("ksizes", ksizes, "strides", strides, "T", _attr_T, "padding",
  padding)
  _result = _execute.execute(b"ExtractVolumePatches", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExtractVolumePatches", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_args', v1=['quantization.fake_quant_with_min_max_args', 'fake_quant_with_min_max_args'])
@deprecated_endpoints('fake_quant_with_min_max_args')
def fake_quant_with_min_max_args(inputs: Annotated[Any, _atypes.Float32], min:float=-6, max:float=6, num_bits:int=8, narrow_range:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same shape and type.

  
    Quantization is called fake since the output is still in floating point.
    The API converts inputs into values within the range [min and max] and returns
    as output.

  Attributes

  *   `[min; max]` define the clamping range for the `inputs` data.
  *   `inputs` values are quantized into the quantization range (
  `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
  when it is true) and then de-quantized and output as floats in `[min; max]`
  interval.
  *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:

  *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.


  Examples

  ```python

  inp = tf.constant ([10.03, -10.23, 3])
  out = tf.quantization.fake_quant_with_min_max_args(inp, min=-5, max=5,
                                                     num_bits=16)
  print(out)

  #  Output:
  #  tf.Tensor([ 4.9999237 -5.0000763  3.0000763], shape=(3,), dtype=float32)
  ```

  Raises:
    * InvalidArgumentError:
      - If num_bits are outside of range [2, 16].
      - If min >= max.
    * ValueError: If `inputs` are of any other type than float32.

  Args:
    inputs: A `Tensor` of type `float32`.
    min: An optional `float`. Defaults to `-6`.
    max: An optional `float`. Defaults to `6`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeQuantWithMinMaxArgs", name, inputs, "min", min, "max", max,
        "num_bits", num_bits, "narrow_range", narrow_range)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fake_quant_with_min_max_args(
          (inputs, min, max, num_bits, narrow_range, name,), None)
      if _result is not NotImplemented:
        return _result
      return fake_quant_with_min_max_args_eager_fallback(
          inputs, min=min, max=max, num_bits=num_bits,
          narrow_range=narrow_range, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fake_quant_with_min_max_args, (), dict(inputs=inputs, min=min,
                                                   max=max, num_bits=num_bits,
                                                   narrow_range=narrow_range,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fake_quant_with_min_max_args(
        (inputs, min, max, num_bits, narrow_range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if min is None:
    min = -6
  min = _execute.make_float(min, "min")
  if max is None:
    max = 6
  max = _execute.make_float(max, "max")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQuantWithMinMaxArgs", inputs=inputs, min=min, max=max,
                                   num_bits=num_bits,
                                   narrow_range=narrow_range, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fake_quant_with_min_max_args, (), dict(inputs=inputs, min=min,
                                                 max=max, num_bits=num_bits,
                                                 narrow_range=narrow_range,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("min", _op.get_attr("min"), "max", _op.get_attr("max"),
              "num_bits", _op._get_attr_int("num_bits"), "narrow_range",
              _op._get_attr_bool("narrow_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQuantWithMinMaxArgs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FakeQuantWithMinMaxArgs = tf_export("raw_ops.FakeQuantWithMinMaxArgs")(_ops.to_raw_op(fake_quant_with_min_max_args))
_dispatcher_for_fake_quant_with_min_max_args = fake_quant_with_min_max_args._tf_type_based_dispatcher.Dispatch


def fake_quant_with_min_max_args_eager_fallback(inputs: Annotated[Any, _atypes.Float32], min: float, max: float, num_bits: int, narrow_range: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if min is None:
    min = -6
  min = _execute.make_float(min, "min")
  if max is None:
    max = 6
  max = _execute.make_float(max, "max")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
  _inputs_flat = [inputs]
  _attrs = ("min", min, "max", max, "num_bits", num_bits, "narrow_range",
  narrow_range)
  _result = _execute.execute(b"FakeQuantWithMinMaxArgs", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeQuantWithMinMaxArgs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_args_gradient', v1=['quantization.fake_quant_with_min_max_args_gradient', 'fake_quant_with_min_max_args_gradient'])
@deprecated_endpoints('fake_quant_with_min_max_args_gradient')
def fake_quant_with_min_max_args_gradient(gradients: Annotated[Any, _atypes.Float32], inputs: Annotated[Any, _atypes.Float32], min:float=-6, max:float=6, num_bits:int=8, narrow_range:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Compute gradients for a FakeQuantWithMinMaxArgs operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
    min: An optional `float`. Defaults to `-6`.
    max: An optional `float`. Defaults to `6`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeQuantWithMinMaxArgsGradient", name, gradients, inputs,
        "min", min, "max", max, "num_bits", num_bits, "narrow_range",
        narrow_range)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fake_quant_with_min_max_args_gradient(
          (gradients, inputs, min, max, num_bits, narrow_range, name,), None)
      if _result is not NotImplemented:
        return _result
      return fake_quant_with_min_max_args_gradient_eager_fallback(
          gradients, inputs, min=min, max=max, num_bits=num_bits,
          narrow_range=narrow_range, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fake_quant_with_min_max_args_gradient, (), dict(gradients=gradients,
                                                            inputs=inputs,
                                                            min=min, max=max,
                                                            num_bits=num_bits,
                                                            narrow_range=narrow_range,
                                                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fake_quant_with_min_max_args_gradient(
        (gradients, inputs, min, max, num_bits, narrow_range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if min is None:
    min = -6
  min = _execute.make_float(min, "min")
  if max is None:
    max = 6
  max = _execute.make_float(max, "max")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQuantWithMinMaxArgsGradient", gradients=gradients, inputs=inputs,
                                           min=min, max=max,
                                           num_bits=num_bits,
                                           narrow_range=narrow_range,
                                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fake_quant_with_min_max_args_gradient, (), dict(gradients=gradients,
                                                          inputs=inputs,
                                                          min=min, max=max,
                                                          num_bits=num_bits,
                                                          narrow_range=narrow_range,
                                                          name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("min", _op.get_attr("min"), "max", _op.get_attr("max"),
              "num_bits", _op._get_attr_int("num_bits"), "narrow_range",
              _op._get_attr_bool("narrow_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQuantWithMinMaxArgsGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FakeQuantWithMinMaxArgsGradient = tf_export("raw_ops.FakeQuantWithMinMaxArgsGradient")(_ops.to_raw_op(fake_quant_with_min_max_args_gradient))
_dispatcher_for_fake_quant_with_min_max_args_gradient = fake_quant_with_min_max_args_gradient._tf_type_based_dispatcher.Dispatch


def fake_quant_with_min_max_args_gradient_eager_fallback(gradients: Annotated[Any, _atypes.Float32], inputs: Annotated[Any, _atypes.Float32], min: float, max: float, num_bits: int, narrow_range: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if min is None:
    min = -6
  min = _execute.make_float(min, "min")
  if max is None:
    max = 6
  max = _execute.make_float(max, "max")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
  inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
  _inputs_flat = [gradients, inputs]
  _attrs = ("min", min, "max", max, "num_bits", num_bits, "narrow_range",
  narrow_range)
  _result = _execute.execute(b"FakeQuantWithMinMaxArgsGradient", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeQuantWithMinMaxArgsGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_vars', v1=['quantization.fake_quant_with_min_max_vars', 'fake_quant_with_min_max_vars'])
@deprecated_endpoints('fake_quant_with_min_max_vars')
def fake_quant_with_min_max_vars(inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits:int=8, narrow_range:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Fake-quantize the 'inputs' tensor of type float via global float scalars

  Fake-quantize the `inputs` tensor of type float via global float scalars
  `min` and `max` to `outputs` tensor of same shape as `inputs`.

  Attributes

  *   `[min; max]` define the clamping range for the `inputs` data.
  *   `inputs` values are quantized into the quantization range (
  `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
  when it is true) and then de-quantized and output as floats in `[min; max]`
  interval.
  *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:

  *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

  This operation has a gradient and thus allows for training `min` and `max`
  values.

  >>> constant_input = tf.constant([[1.2, -0.3, 0.7], [2.1, 0.5, -1.0]], dtype=tf.float32)
  >>>
  >>> min_val = -0.5
  >>> max_val = 0.8
  >>> num_bits = 8
  >>> narrow_range = False #False:for the quantization range [0; 2^num_bits - 1]
  >>>
  >>> quantized_data = tf.quantization.fake_quant_with_min_max_vars(
  ...   inputs=constant_input, min=min_val, max=max_val, num_bits=num_bits, narrow_range=narrow_range
  ... )
  >>>
  >>> print("Input:\n", constant_input.numpy())
  Input:
  [[ 1.2 -0.3  0.7]
  [ 2.1  0.5 -1. ]]
  >>> print("Output:\n", quantized_data.numpy())
  Output:
  [[ 0.8003921 -0.3007843  0.6984313]
  [ 0.8003921  0.4996078 -0.4996078]]

  Args:
    inputs: A `Tensor` of type `float32`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeQuantWithMinMaxVars", name, inputs, min, max, "num_bits",
        num_bits, "narrow_range", narrow_range)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fake_quant_with_min_max_vars(
          (inputs, min, max, num_bits, narrow_range, name,), None)
      if _result is not NotImplemented:
        return _result
      return fake_quant_with_min_max_vars_eager_fallback(
          inputs, min, max, num_bits=num_bits, narrow_range=narrow_range,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fake_quant_with_min_max_vars, (), dict(inputs=inputs, min=min,
                                                   max=max, num_bits=num_bits,
                                                   narrow_range=narrow_range,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fake_quant_with_min_max_vars(
        (inputs, min, max, num_bits, narrow_range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQuantWithMinMaxVars", inputs=inputs, min=min, max=max,
                                   num_bits=num_bits,
                                   narrow_range=narrow_range, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fake_quant_with_min_max_vars, (), dict(inputs=inputs, min=min,
                                                 max=max, num_bits=num_bits,
                                                 narrow_range=narrow_range,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bits", _op._get_attr_int("num_bits"), "narrow_range",
              _op._get_attr_bool("narrow_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQuantWithMinMaxVars", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FakeQuantWithMinMaxVars = tf_export("raw_ops.FakeQuantWithMinMaxVars")(_ops.to_raw_op(fake_quant_with_min_max_vars))
_dispatcher_for_fake_quant_with_min_max_vars = fake_quant_with_min_max_vars._tf_type_based_dispatcher.Dispatch


def fake_quant_with_min_max_vars_eager_fallback(inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits: int, narrow_range: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
  min = _ops.convert_to_tensor(min, _dtypes.float32)
  max = _ops.convert_to_tensor(max, _dtypes.float32)
  _inputs_flat = [inputs, min, max]
  _attrs = ("num_bits", num_bits, "narrow_range", narrow_range)
  _result = _execute.execute(b"FakeQuantWithMinMaxVars", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeQuantWithMinMaxVars", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FakeQuantWithMinMaxVarsGradientOutput = collections.namedtuple(
    "FakeQuantWithMinMaxVarsGradient",
    ["backprops_wrt_input", "backprop_wrt_min", "backprop_wrt_max"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_vars_gradient', v1=['quantization.fake_quant_with_min_max_vars_gradient', 'fake_quant_with_min_max_vars_gradient'])
@deprecated_endpoints('fake_quant_with_min_max_vars_gradient')
def fake_quant_with_min_max_vars_gradient(gradients: Annotated[Any, _atypes.Float32], inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits:int=8, narrow_range:bool=False, name=None):
  r"""Compute gradients for a FakeQuantWithMinMaxVars operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxVars operation.
      min, max: Quantization interval, scalar floats.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization; between 2 and 8, inclusive.
    narrow_range: An optional `bool`. Defaults to `False`.
      Whether to quantize into 2^num_bits - 1 distinct values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

    backprops_wrt_input: A `Tensor` of type `float32`.
    backprop_wrt_min: A `Tensor` of type `float32`.
    backprop_wrt_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeQuantWithMinMaxVarsGradient", name, gradients, inputs, min,
        max, "num_bits", num_bits, "narrow_range", narrow_range)
      _result = _FakeQuantWithMinMaxVarsGradientOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fake_quant_with_min_max_vars_gradient(
          (gradients, inputs, min, max, num_bits, narrow_range, name,), None)
      if _result is not NotImplemented:
        return _result
      return fake_quant_with_min_max_vars_gradient_eager_fallback(
          gradients, inputs, min, max, num_bits=num_bits,
          narrow_range=narrow_range, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fake_quant_with_min_max_vars_gradient, (), dict(gradients=gradients,
                                                            inputs=inputs,
                                                            min=min, max=max,
                                                            num_bits=num_bits,
                                                            narrow_range=narrow_range,
                                                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fake_quant_with_min_max_vars_gradient(
        (gradients, inputs, min, max, num_bits, narrow_range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQuantWithMinMaxVarsGradient", gradients=gradients, inputs=inputs,
                                           min=min, max=max,
                                           num_bits=num_bits,
                                           narrow_range=narrow_range,
                                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fake_quant_with_min_max_vars_gradient, (), dict(gradients=gradients,
                                                          inputs=inputs,
                                                          min=min, max=max,
                                                          num_bits=num_bits,
                                                          narrow_range=narrow_range,
                                                          name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bits", _op._get_attr_int("num_bits"), "narrow_range",
              _op._get_attr_bool("narrow_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQuantWithMinMaxVarsGradient", _inputs_flat, _attrs, _result)
  _result = _FakeQuantWithMinMaxVarsGradientOutput._make(_result)
  return _result

FakeQuantWithMinMaxVarsGradient = tf_export("raw_ops.FakeQuantWithMinMaxVarsGradient")(_ops.to_raw_op(fake_quant_with_min_max_vars_gradient))
_dispatcher_for_fake_quant_with_min_max_vars_gradient = fake_quant_with_min_max_vars_gradient._tf_type_based_dispatcher.Dispatch


def fake_quant_with_min_max_vars_gradient_eager_fallback(gradients: Annotated[Any, _atypes.Float32], inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits: int, narrow_range: bool, name, ctx):
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
  inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
  min = _ops.convert_to_tensor(min, _dtypes.float32)
  max = _ops.convert_to_tensor(max, _dtypes.float32)
  _inputs_flat = [gradients, inputs, min, max]
  _attrs = ("num_bits", num_bits, "narrow_range", narrow_range)
  _result = _execute.execute(b"FakeQuantWithMinMaxVarsGradient", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeQuantWithMinMaxVarsGradient", _inputs_flat, _attrs, _result)
  _result = _FakeQuantWithMinMaxVarsGradientOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_vars_per_channel', v1=['quantization.fake_quant_with_min_max_vars_per_channel', 'fake_quant_with_min_max_vars_per_channel'])
@deprecated_endpoints('fake_quant_with_min_max_vars_per_channel')
def fake_quant_with_min_max_vars_per_channel(inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits:int=8, narrow_range:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Fake-quantize the 'inputs' tensor of type float via per-channel floats

  Fake-quantize the `inputs` tensor of type float per-channel and one of the
  shapes: `[d]`, `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max`
  of shape `[d]` to `outputs` tensor of same shape as `inputs`.

  Attributes

  *   `[min; max]` define the clamping range for the `inputs` data.
  *   `inputs` values are quantized into the quantization range (
  `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
  when it is true) and then de-quantized and output as floats in `[min; max]`
  interval.
  *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:

  *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

  This operation has a gradient and thus allows for training `min` and `max`
  values.

  Args:
    inputs: A `Tensor` of type `float32`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeQuantWithMinMaxVarsPerChannel", name, inputs, min, max,
        "num_bits", num_bits, "narrow_range", narrow_range)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fake_quant_with_min_max_vars_per_channel(
          (inputs, min, max, num_bits, narrow_range, name,), None)
      if _result is not NotImplemented:
        return _result
      return fake_quant_with_min_max_vars_per_channel_eager_fallback(
          inputs, min, max, num_bits=num_bits, narrow_range=narrow_range,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fake_quant_with_min_max_vars_per_channel, (), dict(inputs=inputs,
                                                               min=min,
                                                               max=max,
                                                               num_bits=num_bits,
                                                               narrow_range=narrow_range,
                                                               name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fake_quant_with_min_max_vars_per_channel(
        (inputs, min, max, num_bits, narrow_range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQuantWithMinMaxVarsPerChannel", inputs=inputs, min=min, max=max,
                                             num_bits=num_bits,
                                             narrow_range=narrow_range,
                                             name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fake_quant_with_min_max_vars_per_channel, (), dict(inputs=inputs,
                                                             min=min, max=max,
                                                             num_bits=num_bits,
                                                             narrow_range=narrow_range,
                                                             name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bits", _op._get_attr_int("num_bits"), "narrow_range",
              _op._get_attr_bool("narrow_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQuantWithMinMaxVarsPerChannel", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FakeQuantWithMinMaxVarsPerChannel = tf_export("raw_ops.FakeQuantWithMinMaxVarsPerChannel")(_ops.to_raw_op(fake_quant_with_min_max_vars_per_channel))
_dispatcher_for_fake_quant_with_min_max_vars_per_channel = fake_quant_with_min_max_vars_per_channel._tf_type_based_dispatcher.Dispatch


def fake_quant_with_min_max_vars_per_channel_eager_fallback(inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits: int, narrow_range: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
  min = _ops.convert_to_tensor(min, _dtypes.float32)
  max = _ops.convert_to_tensor(max, _dtypes.float32)
  _inputs_flat = [inputs, min, max]
  _attrs = ("num_bits", num_bits, "narrow_range", narrow_range)
  _result = _execute.execute(b"FakeQuantWithMinMaxVarsPerChannel", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeQuantWithMinMaxVarsPerChannel", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FakeQuantWithMinMaxVarsPerChannelGradientOutput = collections.namedtuple(
    "FakeQuantWithMinMaxVarsPerChannelGradient",
    ["backprops_wrt_input", "backprop_wrt_min", "backprop_wrt_max"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_vars_per_channel_gradient', v1=['quantization.fake_quant_with_min_max_vars_per_channel_gradient', 'fake_quant_with_min_max_vars_per_channel_gradient'])
@deprecated_endpoints('fake_quant_with_min_max_vars_per_channel_gradient')
def fake_quant_with_min_max_vars_per_channel_gradient(gradients: Annotated[Any, _atypes.Float32], inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits:int=8, narrow_range:bool=False, name=None):
  r"""Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
      shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
        same as `gradients`.
      min, max: Quantization interval, floats of shape `[d]`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization; between 2 and 16, inclusive.
    narrow_range: An optional `bool`. Defaults to `False`.
      Whether to quantize into 2^num_bits - 1 distinct values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

    backprops_wrt_input: A `Tensor` of type `float32`.
    backprop_wrt_min: A `Tensor` of type `float32`.
    backprop_wrt_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeQuantWithMinMaxVarsPerChannelGradient", name, gradients,
        inputs, min, max, "num_bits", num_bits, "narrow_range", narrow_range)
      _result = _FakeQuantWithMinMaxVarsPerChannelGradientOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fake_quant_with_min_max_vars_per_channel_gradient(
          (gradients, inputs, min, max, num_bits, narrow_range, name,), None)
      if _result is not NotImplemented:
        return _result
      return fake_quant_with_min_max_vars_per_channel_gradient_eager_fallback(
          gradients, inputs, min, max, num_bits=num_bits,
          narrow_range=narrow_range, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fake_quant_with_min_max_vars_per_channel_gradient, (), dict(gradients=gradients,
                                                                        inputs=inputs,
                                                                        min=min,
                                                                        max=max,
                                                                        num_bits=num_bits,
                                                                        narrow_range=narrow_range,
                                                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fake_quant_with_min_max_vars_per_channel_gradient(
        (gradients, inputs, min, max, num_bits, narrow_range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQuantWithMinMaxVarsPerChannelGradient", gradients=gradients,
                                                     inputs=inputs, min=min,
                                                     max=max,
                                                     num_bits=num_bits,
                                                     narrow_range=narrow_range,
                                                     name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fake_quant_with_min_max_vars_per_channel_gradient, (), dict(gradients=gradients,
                                                                      inputs=inputs,
                                                                      min=min,
                                                                      max=max,
                                                                      num_bits=num_bits,
                                                                      narrow_range=narrow_range,
                                                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bits", _op._get_attr_int("num_bits"), "narrow_range",
              _op._get_attr_bool("narrow_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQuantWithMinMaxVarsPerChannelGradient", _inputs_flat, _attrs, _result)
  _result = _FakeQuantWithMinMaxVarsPerChannelGradientOutput._make(_result)
  return _result

FakeQuantWithMinMaxVarsPerChannelGradient = tf_export("raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient")(_ops.to_raw_op(fake_quant_with_min_max_vars_per_channel_gradient))
_dispatcher_for_fake_quant_with_min_max_vars_per_channel_gradient = fake_quant_with_min_max_vars_per_channel_gradient._tf_type_based_dispatcher.Dispatch


def fake_quant_with_min_max_vars_per_channel_gradient_eager_fallback(gradients: Annotated[Any, _atypes.Float32], inputs: Annotated[Any, _atypes.Float32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], num_bits: int, narrow_range: bool, name, ctx):
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
  inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
  min = _ops.convert_to_tensor(min, _dtypes.float32)
  max = _ops.convert_to_tensor(max, _dtypes.float32)
  _inputs_flat = [gradients, inputs, min, max]
  _attrs = ("num_bits", num_bits, "narrow_range", narrow_range)
  _result = _execute.execute(b"FakeQuantWithMinMaxVarsPerChannelGradient", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeQuantWithMinMaxVarsPerChannelGradient", _inputs_flat, _attrs, _result)
  _result = _FakeQuantWithMinMaxVarsPerChannelGradientOutput._make(_result)
  return _result


TV_Fill_T = TypeVar("TV_Fill_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Fill_index_type = TypeVar("TV_Fill_index_type", _atypes.Int32, _atypes.Int64)

def fill(dims: Annotated[Any, TV_Fill_index_type], value: Annotated[Any, TV_Fill_T], name=None) -> Annotated[Any, TV_Fill_T]:
  r"""Creates a tensor filled with a scalar value.

  This operation creates a tensor of shape `dims` and fills it with `value`.

  For example:

  ```
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
  ```

  `tf.fill` differs from `tf.constant` in a few ways:

  *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
      Tensor values.
  *   `tf.fill` creates an Op in the computation graph that constructs the actual
      Tensor value at runtime. This is in contrast to `tf.constant` which embeds
      the entire Tensor into the graph with a `Const` node.
  *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
      based on other runtime Tensors, unlike `tf.constant`.

  Args:
    dims: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D. Represents the shape of the output tensor.
    value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.

      @compatibility(numpy)
      Equivalent to np.full
      @end_compatibility
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Fill", name, dims, value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fill_eager_fallback(
          dims, value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Fill", dims=dims, value=value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "index_type",
              _op._get_attr_type("index_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Fill", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Fill = tf_export("raw_ops.Fill")(_ops.to_raw_op(fill))


def fill_eager_fallback(dims: Annotated[Any, TV_Fill_index_type], value: Annotated[Any, TV_Fill_T], name, ctx) -> Annotated[Any, TV_Fill_T]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _attr_index_type, (dims,) = _execute.args_to_matching_eager([dims], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [dims, value]
  _attrs = ("T", _attr_T, "index_type", _attr_index_type)
  _result = _execute.execute(b"Fill", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Fill", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Fingerprint_T = TypeVar("TV_Fingerprint_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def fingerprint(data: Annotated[Any, TV_Fingerprint_T], method: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.UInt8]:
  r"""Generates fingerprint values.

  Generates fingerprint values of `data`.

  Fingerprint op considers the first dimension of `data` as the batch dimension,
  and `output[i]` contains the fingerprint value generated from contents in
  `data[i, ...]` for all `i`.

  Fingerprint op writes fingerprint values as byte arrays. For example, the
  default method `farmhash64` generates a 64-bit fingerprint value at a time.
  This 8-byte value is written out as an `uint8` array of size 8, in little-endian
  order.

  For example, suppose that `data` has data type `DT_INT32` and shape (2, 3, 4),
  and that the fingerprint method is `farmhash64`. In this case, the output shape
  is (2, 8), where 2 is the batch dimension size of `data`, and 8 is the size of
  each fingerprint value in bytes. `output[0, :]` is generated from 12 integers in
  `data[0, :, :]` and similarly `output[1, :]` is generated from other 12 integers
  in `data[1, :, :]`.

  Note that this op fingerprints the raw underlying buffer, and it does not
  fingerprint Tensor's metadata such as data type and/or shape. For example, the
  fingerprint values are invariant under reshapes and bitcasts as long as the
  batch dimension remain the same:

  ```
  Fingerprint(data) == Fingerprint(Reshape(data, ...))
  Fingerprint(data) == Fingerprint(Bitcast(data, ...))
  ```

  For string data, one should expect `Fingerprint(data) !=
  Fingerprint(ReduceJoin(data))` in general.

  Args:
    data: A `Tensor`. Must have rank 1 or higher.
    method: A `Tensor` of type `string`.
      Fingerprint method used by this op. Currently available method is
      `farmhash::fingerprint64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Fingerprint", name, data, method)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fingerprint_eager_fallback(
          data, method, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Fingerprint", data=data, method=method, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Fingerprint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Fingerprint = tf_export("raw_ops.Fingerprint")(_ops.to_raw_op(fingerprint))


def fingerprint_eager_fallback(data: Annotated[Any, TV_Fingerprint_T], method: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.UInt8]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [])
  method = _ops.convert_to_tensor(method, _dtypes.string)
  _inputs_flat = [data, method]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Fingerprint", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Fingerprint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Gather_Tparams = TypeVar("TV_Gather_Tparams", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Gather_Tindices = TypeVar("TV_Gather_Tindices", _atypes.Int32, _atypes.Int64)

def gather(params: Annotated[Any, TV_Gather_Tparams], indices: Annotated[Any, TV_Gather_Tindices], validate_indices:bool=True, name=None) -> Annotated[Any, TV_Gather_Tparams]:
  r"""Gather slices from `params` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

  ```python
      # Scalar indices
      output[:, ..., :] = params[indices, :, ... :]

      # Vector indices
      output[i, :, ..., :] = params[indices[i], :, ... :]

      # Higher rank indices
      output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
  ```

  If `indices` is a permutation and `len(indices) == params.shape[0]` then
  this operation will permute `params` accordingly.

  `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
  `indices` are always validated to be within range. If assigned to GPU,
  out-of-bound indices result in safe but unspecified behavior, which may include
  raising an error.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
  </div>

  Args:
    params: A `Tensor`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Gather", name, params, indices, "validate_indices",
        validate_indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return gather_eager_fallback(
          params, indices, validate_indices=validate_indices, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Gather", params=params, indices=indices,
                  validate_indices=validate_indices, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("validate_indices", _op._get_attr_bool("validate_indices"),
              "Tparams", _op._get_attr_type("Tparams"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Gather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Gather = tf_export("raw_ops.Gather")(_ops.to_raw_op(gather))


def gather_eager_fallback(params: Annotated[Any, TV_Gather_Tparams], indices: Annotated[Any, TV_Gather_Tindices], validate_indices: bool, name, ctx) -> Annotated[Any, TV_Gather_Tparams]:
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _attr_Tparams, (params,) = _execute.args_to_matching_eager([params], ctx, [])
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [params, indices]
  _attrs = ("validate_indices", validate_indices, "Tparams", _attr_Tparams,
  "Tindices", _attr_Tindices)
  _result = _execute.execute(b"Gather", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Gather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_GatherNd_Tparams = TypeVar("TV_GatherNd_Tparams", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_GatherNd_Tindices = TypeVar("TV_GatherNd_Tindices", _atypes.Int16, _atypes.Int32, _atypes.Int64)

def gather_nd(params: Annotated[Any, TV_GatherNd_Tparams], indices: Annotated[Any, TV_GatherNd_Tindices], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_GatherNd_Tparams]:
  r"""Gather slices from `params` into a Tensor with shape specified by `indices`.

  `indices` is a K-dimensional integer tensor, best thought of as a
  (K-1)-dimensional tensor of indices into `params`, where each element defines a
  slice of `params`:

      output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]

  Whereas in `tf.gather` `indices` defines slices into the `axis`
  dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
  first `N` dimensions of `params`, where `N = indices.shape[-1]`.

  The last dimension of `indices` can be at most the rank of
  `params`:

      indices.shape[-1] <= params.rank

  The last dimension of `indices` corresponds to elements
  (if `indices.shape[-1] == params.rank`) or slices
  (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
  of `params`.  The output tensor has shape

      indices.shape[:-1] + params.shape[indices.shape[-1]:]

  If `indices` contains any out-of-bound indices, depending on
  `bad_indices_policy`, the op will either return an error or ignore the
  out-of-bound indices. `bad_indices_policy` can be one of the following values:
  1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
     historically on CPU and GPU we handle errors in different ways, and for
     backward compatibility we keep the default behavior.
  2. "ERROR": raises error; GPU does not support this value.
  3. "IGNORE": ignore error and set the corresponding output to 0;
     supported on both CPU and GPU.

  Some examples below.

  Simple indexing into a matrix:

  ```python
      indices = [[0, 0], [1, 1]]
      params = [['a', 'b'], ['c', 'd']]
      output = ['a', 'd']
  ```

  Slice indexing into a matrix:

  ```python
      indices = [[1], [0]]
      params = [['a', 'b'], ['c', 'd']]
      output = [['c', 'd'], ['a', 'b']]
  ```

  Indexing into a 3-tensor:

  ```python
      indices = [[1]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['a1', 'b1'], ['c1', 'd1']]]


      indices = [[0, 1], [1, 0]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0', 'd0'], ['a1', 'b1']]


      indices = [[0, 0, 1], [1, 0, 1]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = ['b0', 'b1']
  ```

  Batched indexing into a matrix:

  ```python
      indices = [[[0, 0]], [[0, 1]]]
      params = [['a', 'b'], ['c', 'd']]
      output = [['a'], ['b']]
  ```

  Batched slice indexing into a matrix:

  ```python
      indices = [[[1]], [[0]]]
      params = [['a', 'b'], ['c', 'd']]
      output = [[['c', 'd']], [['a', 'b']]]
  ```

  Batched indexing into a 3-tensor:

  ```python
      indices = [[[1]], [[0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[[['a1', 'b1'], ['c1', 'd1']]],
                [[['a0', 'b0'], ['c0', 'd0']]]]

      indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['c0', 'd0'], ['a1', 'b1']],
                [['a0', 'b0'], ['c1', 'd1']]]


      indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['b0', 'b1'], ['d0', 'c1']]
  ```

  See also `tf.gather` and `tf.batch_gather`.

  Args:
    params: A `Tensor`. The tensor from which to gather values.
    indices: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      Index tensor.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GatherNd", name, params, indices, "bad_indices_policy",
        bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return gather_nd_eager_fallback(
          params, indices, bad_indices_policy=bad_indices_policy, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GatherNd", params=params, indices=indices,
                    bad_indices_policy=bad_indices_policy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tparams", _op._get_attr_type("Tparams"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GatherNd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GatherNd = tf_export("raw_ops.GatherNd")(_ops.to_raw_op(gather_nd))


def gather_nd_eager_fallback(params: Annotated[Any, TV_GatherNd_Tparams], indices: Annotated[Any, TV_GatherNd_Tindices], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_GatherNd_Tparams]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_Tparams, (params,) = _execute.args_to_matching_eager([params], ctx, [])
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [params, indices]
  _attrs = ("Tparams", _attr_Tparams, "Tindices", _attr_Tindices,
  "bad_indices_policy", bad_indices_policy)
  _result = _execute.execute(b"GatherNd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GatherNd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_GatherV2_Tparams = TypeVar("TV_GatherV2_Tparams", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_GatherV2_Tindices = TypeVar("TV_GatherV2_Tindices", _atypes.Int16, _atypes.Int32, _atypes.Int64)
TV_GatherV2_Taxis = TypeVar("TV_GatherV2_Taxis", _atypes.Int32, _atypes.Int64)

def gather_v2(params: Annotated[Any, TV_GatherV2_Tparams], indices: Annotated[Any, TV_GatherV2_Tindices], axis: Annotated[Any, TV_GatherV2_Taxis], batch_dims:int=0, name=None) -> Annotated[Any, TV_GatherV2_Tparams]:
  r"""Gather slices from `params` axis `axis` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `params.shape[:axis] +
  indices.shape[batch_dims:] + params.shape[axis + 1:]` where:

  ```python
      # Scalar indices (output is rank(params) - 1).
      output[a_0, ..., a_n, b_0, ..., b_n] =
        params[a_0, ..., a_n, indices, b_0, ..., b_n]

      # Vector indices (output is rank(params)).
      output[a_0, ..., a_n, i, b_0, ..., b_n] =
        params[a_0, ..., a_n, indices[i], b_0, ..., b_n]

      # Higher rank indices (output is rank(params) + rank(indices) - 1).
      output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
        params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
  ```

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
  </div>

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, a 0 is stored in the
  corresponding output value.

  Note that on TPU, if any dimension of `params` is of size 0 then the output will
  be the expected shape filled with zeros. On CPU and GPU an error will be
  returned.

  See also `tf.batch_gather` and `tf.gather_nd`.

  Args:
    params: A `Tensor`.
      The tensor from which to gather values. Must be at least rank
      `axis + 1`.
    indices: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      Index tensor. Must be in range `[0, params.shape[axis])`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The axis in `params` to gather `indices` from. Defaults to the first
      dimension. Supports negative indexes.
    batch_dims: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GatherV2", name, params, indices, axis, "batch_dims",
        batch_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return gather_v2_eager_fallback(
          params, indices, axis, batch_dims=batch_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if batch_dims is None:
    batch_dims = 0
  batch_dims = _execute.make_int(batch_dims, "batch_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GatherV2", params=params, indices=indices, axis=axis,
                    batch_dims=batch_dims, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("batch_dims", _op._get_attr_int("batch_dims"), "Tparams",
              _op._get_attr_type("Tparams"), "Tindices",
              _op._get_attr_type("Tindices"), "Taxis",
              _op._get_attr_type("Taxis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GatherV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GatherV2 = tf_export("raw_ops.GatherV2")(_ops.to_raw_op(gather_v2))


def gather_v2_eager_fallback(params: Annotated[Any, TV_GatherV2_Tparams], indices: Annotated[Any, TV_GatherV2_Tindices], axis: Annotated[Any, TV_GatherV2_Taxis], batch_dims: int, name, ctx) -> Annotated[Any, TV_GatherV2_Tparams]:
  if batch_dims is None:
    batch_dims = 0
  batch_dims = _execute.make_int(batch_dims, "batch_dims")
  _attr_Tparams, (params,) = _execute.args_to_matching_eager([params], ctx, [])
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  _attr_Taxis, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [params, indices, axis]
  _attrs = ("batch_dims", batch_dims, "Tparams", _attr_Tparams, "Tindices",
  _attr_Tindices, "Taxis", _attr_Taxis)
  _result = _execute.execute(b"GatherV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GatherV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_GuaranteeConst_T = TypeVar("TV_GuaranteeConst_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def guarantee_const(input: Annotated[Any, TV_GuaranteeConst_T], name=None) -> Annotated[Any, TV_GuaranteeConst_T]:
  r"""Gives a guarantee to the TF runtime that the input tensor is a constant.

  The runtime is then free to make optimizations based on this.

  Only accepts value typed tensors as inputs and rejects resource variable handles
  as input.

  Returns the input tensor without modification.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GuaranteeConst", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return guarantee_const_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GuaranteeConst", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GuaranteeConst", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GuaranteeConst = tf_export("raw_ops.GuaranteeConst")(_ops.to_raw_op(guarantee_const))


def guarantee_const_eager_fallback(input: Annotated[Any, TV_GuaranteeConst_T], name, ctx) -> Annotated[Any, TV_GuaranteeConst_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"GuaranteeConst", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GuaranteeConst", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Identity_T = TypeVar("TV_Identity_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def identity(input: Annotated[Any, TV_Identity_T], name=None) -> Annotated[Any, TV_Identity_T]:
  r"""Return a tensor with the same shape and contents as the input tensor or value.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Identity", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return identity_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Identity", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Identity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Identity = tf_export("raw_ops.Identity")(_ops.to_raw_op(identity))


def identity_eager_fallback(input: Annotated[Any, TV_Identity_T], name, ctx) -> Annotated[Any, TV_Identity_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Identity", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Identity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('identity_n')
def identity_n(input, name=None):
  r"""Returns a list of tensors with the same shapes and contents as the input

  tensors.

  This op can be used to override the gradient for complicated functions. For
  example, suppose y = f(x) and we wish to apply a custom function g for backprop
  such that dx = g(dy). In Python,

  ```python
  with tf.get_default_graph().gradient_override_map(
      {'IdentityN': 'OverrideGradientWithG'}):
    y, _ = identity_n([f(x), x])

  @tf.RegisterGradient('OverrideGradientWithG')
  def ApplyG(op, dy, _):
    return [None, g(dy)]  # Do not backprop to f(x).
  ```

  Args:
    input: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IdentityN", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_identity_n(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return identity_n_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            identity_n, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_identity_n(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IdentityN", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          identity_n, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IdentityN", _inputs_flat, _attrs, _result)
  return _result

IdentityN = tf_export("raw_ops.IdentityN")(_ops.to_raw_op(identity_n))
_dispatcher_for_identity_n = identity_n._tf_type_based_dispatcher.Dispatch


def identity_n_eager_fallback(input, name, ctx):
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"IdentityN", len(input), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IdentityN", _inputs_flat, _attrs, _result)
  return _result


TV_ImmutableConst_dtype = TypeVar("TV_ImmutableConst_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def immutable_const(dtype: TV_ImmutableConst_dtype, shape, memory_region_name: str, name=None) -> Annotated[Any, TV_ImmutableConst_dtype]:
  r"""Returns immutable tensor from memory region.

  The current implementation memmaps the tensor from a file.

  Args:
    dtype: A `tf.DType`. Type of the returned tensor.
    shape: A `tf.TensorShape` or list of `ints`. Shape of the returned tensor.
    memory_region_name: A `string`.
      Name of readonly memory region used by the tensor, see
      NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ImmutableConst", name, "dtype", dtype, "shape", shape,
        "memory_region_name", memory_region_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return immutable_const_eager_fallback(
          dtype=dtype, shape=shape, memory_region_name=memory_region_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  memory_region_name = _execute.make_str(memory_region_name, "memory_region_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ImmutableConst", dtype=dtype, shape=shape,
                          memory_region_name=memory_region_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"), "memory_region_name",
              _op.get_attr("memory_region_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ImmutableConst", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ImmutableConst = tf_export("raw_ops.ImmutableConst")(_ops.to_raw_op(immutable_const))


def immutable_const_eager_fallback(dtype: TV_ImmutableConst_dtype, shape, memory_region_name: str, name, ctx) -> Annotated[Any, TV_ImmutableConst_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  memory_region_name = _execute.make_str(memory_region_name, "memory_region_name")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape, "memory_region_name",
  memory_region_name)
  _result = _execute.execute(b"ImmutableConst", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ImmutableConst", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InplaceAdd_T = TypeVar("TV_InplaceAdd_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def inplace_add(x: Annotated[Any, TV_InplaceAdd_T], i: Annotated[Any, _atypes.Int32], v: Annotated[Any, TV_InplaceAdd_T], name=None) -> Annotated[Any, TV_InplaceAdd_T]:
  r"""Adds v into specified rows of x.

      Computes y = x; y[i, :] += v; return y.

  Args:
    x: A `Tensor`. A `Tensor` of type T.
    i: A `Tensor` of type `int32`.
      A vector. Indices into the left-most dimension of `x`.
    v: A `Tensor`. Must have the same type as `x`.
      A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InplaceAdd", name, x, i, v)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return inplace_add_eager_fallback(
          x, i, v, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InplaceAdd", x=x, i=i, v=v, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InplaceAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InplaceAdd = tf_export("raw_ops.InplaceAdd")(_ops.to_raw_op(inplace_add))


def inplace_add_eager_fallback(x: Annotated[Any, TV_InplaceAdd_T], i: Annotated[Any, _atypes.Int32], v: Annotated[Any, TV_InplaceAdd_T], name, ctx) -> Annotated[Any, TV_InplaceAdd_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, v], ctx, [])
  (x, v) = _inputs_T
  i = _ops.convert_to_tensor(i, _dtypes.int32)
  _inputs_flat = [x, i, v]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"InplaceAdd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InplaceAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InplaceSub_T = TypeVar("TV_InplaceSub_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def inplace_sub(x: Annotated[Any, TV_InplaceSub_T], i: Annotated[Any, _atypes.Int32], v: Annotated[Any, TV_InplaceSub_T], name=None) -> Annotated[Any, TV_InplaceSub_T]:
  r"""    Subtracts `v` into specified rows of `x`.

    Computes y = x; y[i, :] -= v; return y.

  Args:
    x: A `Tensor`. A `Tensor` of type T.
    i: A `Tensor` of type `int32`.
      A vector. Indices into the left-most dimension of `x`.
    v: A `Tensor`. Must have the same type as `x`.
      A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InplaceSub", name, x, i, v)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return inplace_sub_eager_fallback(
          x, i, v, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InplaceSub", x=x, i=i, v=v, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InplaceSub", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InplaceSub = tf_export("raw_ops.InplaceSub")(_ops.to_raw_op(inplace_sub))


def inplace_sub_eager_fallback(x: Annotated[Any, TV_InplaceSub_T], i: Annotated[Any, _atypes.Int32], v: Annotated[Any, TV_InplaceSub_T], name, ctx) -> Annotated[Any, TV_InplaceSub_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, v], ctx, [])
  (x, v) = _inputs_T
  i = _ops.convert_to_tensor(i, _dtypes.int32)
  _inputs_flat = [x, i, v]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"InplaceSub", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InplaceSub", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InplaceUpdate_T = TypeVar("TV_InplaceUpdate_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def inplace_update(x: Annotated[Any, TV_InplaceUpdate_T], i: Annotated[Any, _atypes.Int32], v: Annotated[Any, TV_InplaceUpdate_T], name=None) -> Annotated[Any, TV_InplaceUpdate_T]:
  r"""Updates specified rows 'i' with values 'v'.

  Computes `x[i, :] = v; return x`.

  Originally this function is mutative however for compilation we make this
  operation create / operate on a copy of `x`.

  Args:
    x: A `Tensor`. A tensor of type `T`.
    i: A `Tensor` of type `int32`.
      A vector. Indices into the left-most dimension of `x`.
    v: A `Tensor`. Must have the same type as `x`.
      A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InplaceUpdate", name, x, i, v)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return inplace_update_eager_fallback(
          x, i, v, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InplaceUpdate", x=x, i=i, v=v, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InplaceUpdate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InplaceUpdate = tf_export("raw_ops.InplaceUpdate")(_ops.to_raw_op(inplace_update))


def inplace_update_eager_fallback(x: Annotated[Any, TV_InplaceUpdate_T], i: Annotated[Any, _atypes.Int32], v: Annotated[Any, TV_InplaceUpdate_T], name, ctx) -> Annotated[Any, TV_InplaceUpdate_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, v], ctx, [])
  (x, v) = _inputs_T
  i = _ops.convert_to_tensor(i, _dtypes.int32)
  _inputs_flat = [x, i, v]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"InplaceUpdate", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InplaceUpdate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InvertPermutation_T = TypeVar("TV_InvertPermutation_T", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.invert_permutation', v1=['math.invert_permutation', 'invert_permutation'])
@deprecated_endpoints('invert_permutation')
def invert_permutation(x: Annotated[Any, TV_InvertPermutation_T], name=None) -> Annotated[Any, TV_InvertPermutation_T]:
  r"""Computes the inverse permutation of a tensor.

  This operation computes the inverse of an index permutation. It takes a 1-D
  integer tensor `x`, which represents the indices of a zero-based array, and
  swaps each value with its index position. In other words, for an output tensor
  `y` and an input tensor `x`, this operation computes the following:

  `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

  The values must include 0. There can be no duplicate values or negative values.

  For example:

  ```
  # tensor `x` is [3, 4, 0, 2, 1]
  invert_permutation(x) ==> [2, 4, 3, 0, 1]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InvertPermutation", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_invert_permutation(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return invert_permutation_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            invert_permutation, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_invert_permutation(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InvertPermutation", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          invert_permutation, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InvertPermutation", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InvertPermutation = tf_export("raw_ops.InvertPermutation")(_ops.to_raw_op(invert_permutation))
_dispatcher_for_invert_permutation = invert_permutation._tf_type_based_dispatcher.Dispatch


def invert_permutation_eager_fallback(x: Annotated[Any, TV_InvertPermutation_T], name, ctx) -> Annotated[Any, TV_InvertPermutation_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"InvertPermutation", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InvertPermutation", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_ListDiffOutput = collections.namedtuple(
    "ListDiff",
    ["out", "idx"])


TV_ListDiff_T = TypeVar("TV_ListDiff_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ListDiff_out_idx = TypeVar("TV_ListDiff_out_idx", _atypes.Int32, _atypes.Int64)

def list_diff(x: Annotated[Any, TV_ListDiff_T], y: Annotated[Any, TV_ListDiff_T], out_idx:TV_ListDiff_out_idx=_dtypes.int32, name=None):
  r"""Computes the difference between two lists of numbers or strings.

  Given a list `x` and a list `y`, this operation returns a list `out` that
  represents all values that are in `x` but not in `y`. The returned list `out`
  is sorted in the same order that the numbers appear in `x` (duplicates are
  preserved). This operation also returns a list `idx` that represents the
  position of each `out` element in `x`. In other words:

  `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

  For example, given this input:

  ```
  x = [1, 2, 3, 4, 5, 6]
  y = [1, 3, 5]
  ```

  This operation would return:

  ```
  out ==> [2, 4, 6]
  idx ==> [1, 3, 5]
  ```

  Args:
    x: A `Tensor`. 1-D. Values to keep.
    y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, idx).

    out: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ListDiff", name, x, y, "out_idx", out_idx)
      _result = _ListDiffOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return list_diff_eager_fallback(
          x, y, out_idx=out_idx, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ListDiff", x=x, y=y, out_idx=out_idx, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_idx",
              _op._get_attr_type("out_idx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ListDiff", _inputs_flat, _attrs, _result)
  _result = _ListDiffOutput._make(_result)
  return _result

ListDiff = tf_export("raw_ops.ListDiff")(_ops.to_raw_op(list_diff))


def list_diff_eager_fallback(x: Annotated[Any, TV_ListDiff_T], y: Annotated[Any, TV_ListDiff_T], out_idx: TV_ListDiff_out_idx, name, ctx):
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T, "out_idx", out_idx)
  _result = _execute.execute(b"ListDiff", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ListDiff", _inputs_flat, _attrs, _result)
  _result = _ListDiffOutput._make(_result)
  return _result


TV_LowerBound_T = TypeVar("TV_LowerBound_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LowerBound_out_type = TypeVar("TV_LowerBound_out_type", _atypes.Int32, _atypes.Int64)

def lower_bound(sorted_inputs: Annotated[Any, TV_LowerBound_T], values: Annotated[Any, TV_LowerBound_T], out_type:TV_LowerBound_out_type=_dtypes.int32, name=None) -> Annotated[Any, TV_LowerBound_out_type]:
  r"""Applies lower_bound(sorted_search_values, values) along each row.

  Each set of rows with the same index in (sorted_inputs, values) is treated
  independently.  The resulting row is the equivalent of calling
  `np.searchsorted(sorted_inputs, values, side='left')`.

  The result is not a global index to the entire
  `Tensor`, but rather just the index in the last dimension.

  A 2-D example:
    sorted_sequence = [[0, 3, 9, 9, 10],
                       [1, 2, 3, 4, 5]]
    values = [[2, 4, 9],
              [0, 2, 6]]

    result = LowerBound(sorted_sequence, values)

    result == [[1, 2, 2],
               [0, 1, 5]]

  Args:
    sorted_inputs: A `Tensor`. 2-D Tensor where each row is ordered.
    values: A `Tensor`. Must have the same type as `sorted_inputs`.
      2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
      the values that will be searched for in `sorted_search_values`.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LowerBound", name, sorted_inputs, values, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lower_bound_eager_fallback(
          sorted_inputs, values, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LowerBound", sorted_inputs=sorted_inputs, values=values,
                      out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LowerBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LowerBound = tf_export("raw_ops.LowerBound")(_ops.to_raw_op(lower_bound))


def lower_bound_eager_fallback(sorted_inputs: Annotated[Any, TV_LowerBound_T], values: Annotated[Any, TV_LowerBound_T], out_type: TV_LowerBound_out_type, name, ctx) -> Annotated[Any, TV_LowerBound_out_type]:
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([sorted_inputs, values], ctx, [])
  (sorted_inputs, values) = _inputs_T
  _inputs_flat = [sorted_inputs, values]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"LowerBound", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LowerBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixBandPart_T = TypeVar("TV_MatrixBandPart_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MatrixBandPart_Tindex = TypeVar("TV_MatrixBandPart_Tindex", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('linalg.band_part', v1=['linalg.band_part', 'matrix_band_part'])
@deprecated_endpoints('matrix_band_part')
def matrix_band_part(input: Annotated[Any, TV_MatrixBandPart_T], num_lower: Annotated[Any, TV_MatrixBandPart_Tindex], num_upper: Annotated[Any, TV_MatrixBandPart_Tindex], name=None) -> Annotated[Any, TV_MatrixBandPart_T]:
  r"""Copy a tensor setting everything outside a central band in each innermost matrix to zero.

  The `band` part is computed as follows:
  Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
  tensor with the same shape where

  `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

  The indicator function

  `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                   (num_upper < 0 || (n-m) <= num_upper)`.

  For example:

  ```
  # if 'input' is [[ 0,  1,  2, 3]
  #                [-1,  0,  1, 2]
  #                [-2, -1,  0, 1]
  #                [-3, -2, -1, 0]],

  tf.linalg.band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                         [-1,  0,  1, 2]
                                         [ 0, -1,  0, 1]
                                         [ 0,  0, -1, 0]],

  tf.linalg.band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                        [-1,  0,  1, 0]
                                        [-2, -1,  0, 1]
                                        [ 0, -2, -1, 0]]
  ```

  Useful special cases:

  ```
   tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
   tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
   tf.linalg.band_part(input, 0, 0) ==> Diagonal.
  ```

  Args:
    input: A `Tensor`. Rank `k` tensor.
    num_lower: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D tensor. Number of subdiagonals to keep. If negative, keep entire
      lower triangle.
    num_upper: A `Tensor`. Must have the same type as `num_lower`.
      0-D tensor. Number of superdiagonals to keep. If negative, keep
      entire upper triangle.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixBandPart", name, input, num_lower, num_upper)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_matrix_band_part(
          (input, num_lower, num_upper, name,), None)
      if _result is not NotImplemented:
        return _result
      return matrix_band_part_eager_fallback(
          input, num_lower, num_upper, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            matrix_band_part, (), dict(input=input, num_lower=num_lower,
                                       num_upper=num_upper, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_matrix_band_part(
        (input, num_lower, num_upper, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixBandPart", input=input, num_lower=num_lower,
                          num_upper=num_upper, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          matrix_band_part, (), dict(input=input, num_lower=num_lower,
                                     num_upper=num_upper, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindex",
              _op._get_attr_type("Tindex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixBandPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixBandPart = tf_export("raw_ops.MatrixBandPart")(_ops.to_raw_op(matrix_band_part))
_dispatcher_for_matrix_band_part = matrix_band_part._tf_type_based_dispatcher.Dispatch


def matrix_band_part_eager_fallback(input: Annotated[Any, TV_MatrixBandPart_T], num_lower: Annotated[Any, TV_MatrixBandPart_Tindex], num_upper: Annotated[Any, TV_MatrixBandPart_Tindex], name, ctx) -> Annotated[Any, TV_MatrixBandPart_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tindex, _inputs_Tindex = _execute.args_to_matching_eager([num_lower, num_upper], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  (num_lower, num_upper) = _inputs_Tindex
  _inputs_flat = [input, num_lower, num_upper]
  _attrs = ("T", _attr_T, "Tindex", _attr_Tindex)
  _result = _execute.execute(b"MatrixBandPart", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixBandPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixDiag_T = TypeVar("TV_MatrixDiag_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_diag(diagonal: Annotated[Any, TV_MatrixDiag_T], name=None) -> Annotated[Any, TV_MatrixDiag_T]:
  r"""Returns a batched diagonal tensor with a given batched diagonal values.

  Given a `diagonal`, this operation returns a tensor with the `diagonal` and
  everything else padded with zeros. The diagonal is computed as follows:

  Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
  tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

  `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

  For example:

  ```
  # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

  and diagonal.shape = (2, 4)

  tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                       [0, 2, 0, 0]
                                       [0, 0, 3, 0]
                                       [0, 0, 0, 4]],
                                      [[5, 0, 0, 0]
                                       [0, 6, 0, 0]
                                       [0, 0, 7, 0]
                                       [0, 0, 0, 8]]]

  which has shape (2, 4, 4)
  ```

  Args:
    diagonal: A `Tensor`. Rank `k`, where `k >= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixDiag", name, diagonal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_diag_eager_fallback(
          diagonal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixDiag", diagonal=diagonal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixDiag = tf_export("raw_ops.MatrixDiag")(_ops.to_raw_op(matrix_diag))


def matrix_diag_eager_fallback(diagonal: Annotated[Any, TV_MatrixDiag_T], name, ctx) -> Annotated[Any, TV_MatrixDiag_T]:
  _attr_T, (diagonal,) = _execute.args_to_matching_eager([diagonal], ctx, [])
  _inputs_flat = [diagonal]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MatrixDiag", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixDiagPart_T = TypeVar("TV_MatrixDiagPart_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_diag_part(input: Annotated[Any, TV_MatrixDiagPart_T], name=None) -> Annotated[Any, TV_MatrixDiagPart_T]:
  r"""Returns the batched diagonal part of a batched tensor.

  This operation returns a tensor with the `diagonal` part
  of the batched `input`. The `diagonal` part is computed as follows:

  Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
  tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:

  `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

  The input must be at least a matrix.

  For example:

  ```
  # 'input' is [[[1, 0, 0, 0]
                 [0, 2, 0, 0]
                 [0, 0, 3, 0]
                 [0, 0, 0, 4]],
                [[5, 0, 0, 0]
                 [0, 6, 0, 0]
                 [0, 0, 7, 0]
                 [0, 0, 0, 8]]]

  and input.shape = (2, 4, 4)

  tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

  which has shape (2, 4)
  ```

  Args:
    input: A `Tensor`. Rank `k` tensor where `k >= 2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixDiagPart", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_diag_part_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixDiagPart", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixDiagPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixDiagPart = tf_export("raw_ops.MatrixDiagPart")(_ops.to_raw_op(matrix_diag_part))


def matrix_diag_part_eager_fallback(input: Annotated[Any, TV_MatrixDiagPart_T], name, ctx) -> Annotated[Any, TV_MatrixDiagPart_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MatrixDiagPart", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixDiagPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixDiagPartV2_T = TypeVar("TV_MatrixDiagPartV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_diag_part_v2(input: Annotated[Any, TV_MatrixDiagPartV2_T], k: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagPartV2_T], name=None) -> Annotated[Any, TV_MatrixDiagPartV2_T]:
  r"""Returns the batched diagonal part of a batched tensor.

  Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
  `input`.

  Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
  Let `max_diag_len` be the maximum length among all diagonals to be extracted,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
  Let `num_diags` be the number of diagonals to extract,
  `num_diags = k[1] - k[0] + 1`.

  If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
  `[I, J, ..., L, max_diag_len]` and values:

  ```
  diagonal[i, j, ..., l, n]
    = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
      padding_value                 ; otherwise.
  ```
  where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.

  Otherwise, the output tensor has rank `r` with dimensions
  `[I, J, ..., L, num_diags, max_diag_len]` with values:

  ```
  diagonal[i, j, ..., l, m, n]
    = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
      padding_value                 ; otherwise.
  ```
  where `d = k[1] - m`, `y = max(-d, 0)`, and `x = max(d, 0)`.

  The input must be at least a matrix.

  For example:

  ```
  input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                     [5, 6, 7, 8],
                     [9, 8, 7, 6]],
                    [[5, 4, 3, 2],
                     [1, 2, 3, 4],
                     [5, 6, 7, 8]]])

  # A main diagonal from each batch.
  tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
                                  [5, 2, 7]]

  # A superdiagonal from each batch.
  tf.matrix_diag_part(input, k = 1)
    ==> [[2, 7, 6],  # Output shape: (2, 3)
         [4, 3, 8]]

  # A tridiagonal band from each batch.
  tf.matrix_diag_part(input, k = (-1, 1))
    ==> [[[2, 7, 6],  # Output shape: (2, 3, 3)
          [1, 6, 7],
          [5, 8, 0]],
         [[4, 3, 8],
          [5, 2, 7],
          [1, 6, 0]]]

  # Padding value = 9
  tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
    ==> [[[4, 9, 9],  # Output shape: (2, 3, 3)
          [3, 8, 9],
          [2, 7, 6]],
         [[2, 9, 9],
          [3, 4, 9],
          [4, 3, 8]]]
  ```

  Args:
    input: A `Tensor`. Rank `r` tensor where `r >= 2`.
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    padding_value: A `Tensor`. Must have the same type as `input`.
      The value to fill the area outside the specified diagonal band with.
      Default is 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixDiagPartV2", name, input, k, padding_value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_diag_part_v2_eager_fallback(
          input, k, padding_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixDiagPartV2", input=input, k=k, padding_value=padding_value,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixDiagPartV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixDiagPartV2 = tf_export("raw_ops.MatrixDiagPartV2")(_ops.to_raw_op(matrix_diag_part_v2))


def matrix_diag_part_v2_eager_fallback(input: Annotated[Any, TV_MatrixDiagPartV2_T], k: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagPartV2_T], name, ctx) -> Annotated[Any, TV_MatrixDiagPartV2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, padding_value], ctx, [])
  (input, padding_value) = _inputs_T
  k = _ops.convert_to_tensor(k, _dtypes.int32)
  _inputs_flat = [input, k, padding_value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MatrixDiagPartV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixDiagPartV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixDiagPartV3_T = TypeVar("TV_MatrixDiagPartV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_diag_part_v3(input: Annotated[Any, TV_MatrixDiagPartV3_T], k: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagPartV3_T], align:str="RIGHT_LEFT", name=None) -> Annotated[Any, TV_MatrixDiagPartV3_T]:
  r"""Returns the batched diagonal part of a batched tensor.

  Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
  `input`.

  Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
  Let `max_diag_len` be the maximum length among all diagonals to be extracted,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
  Let `num_diags` be the number of diagonals to extract,
  `num_diags = k[1] - k[0] + 1`.

  If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
  `[I, J, ..., L, max_diag_len]` and values:

  ```
  diagonal[i, j, ..., l, n]
    = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
      padding_value                 ; otherwise.
  ```
  where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.

  Otherwise, the output tensor has rank `r` with dimensions
  `[I, J, ..., L, num_diags, max_diag_len]` with values:

  ```
  diagonal[i, j, ..., l, m, n]
    = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
      padding_value                 ; otherwise.
  ```
  where `d = k[1] - m`, `y = max(-d, 0) - offset`, and `x = max(d, 0) - offset`.

  `offset` is zero except when the alignment of the diagonal is to the right.
  ```
  offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                             and `d >= 0`) or
                                           (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                             and `d <= 0`)
           0                          ; otherwise
  ```
  where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

  The input must be at least a matrix.

  For example:

  ```
  input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                     [5, 6, 7, 8],
                     [9, 8, 7, 6]],
                    [[5, 4, 3, 2],
                     [1, 2, 3, 4],
                     [5, 6, 7, 8]]])

  # A main diagonal from each batch.
  tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
                                  [5, 2, 7]]

  # A superdiagonal from each batch.
  tf.matrix_diag_part(input, k = 1)
    ==> [[2, 7, 6],  # Output shape: (2, 3)
         [4, 3, 8]]

  # A band from each batch.
  tf.matrix_diag_part(input, k = (-1, 2))
    ==> [[[0, 3, 8],  # Output shape: (2, 4, 3)
          [2, 7, 6],
          [1, 6, 7],
          [5, 8, 0]],
         [[0, 3, 4],
          [4, 3, 8],
          [5, 2, 7],
          [1, 6, 0]]]

  # LEFT_RIGHT alignment.
  tf.matrix_diag_part(input, k = (-1, 2), align="LEFT_RIGHT")
    ==> [[[3, 8, 0],  # Output shape: (2, 4, 3)
          [2, 7, 6],
          [1, 6, 7],
          [0, 5, 8]],
         [[3, 4, 0],
          [4, 3, 8],
          [5, 2, 7],
          [0, 1, 6]]]

  # max_diag_len can be shorter than the main diagonal.
  tf.matrix_diag_part(input, k = (-2, -1))
    ==> [[[5, 8],
          [9, 0]],
         [[1, 6],
          [5, 0]]]

  # padding_value = 9
  tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
    ==> [[[9, 9, 4],  # Output shape: (2, 3, 3)
          [9, 3, 8],
          [2, 7, 6]],
         [[9, 9, 2],
          [9, 3, 4],
          [4, 3, 8]]]

  ```

  Args:
    input: A `Tensor`. Rank `r` tensor where `r >= 2`.
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    padding_value: A `Tensor`. Must have the same type as `input`.
      The value to fill the area outside the specified diagonal band with.
      Default is 0.
    align: An optional `string` from: `"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"`. Defaults to `"RIGHT_LEFT"`.
      Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
      a string specifying how superdiagonals and subdiagonals should be aligned,
      respectively. There are four possible alignments: "RIGHT_LEFT" (default),
      "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
      to the right (left-pads the row) and subdiagonals to the left (right-pads the
      row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
      the opposite alignment.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixDiagPartV3", name, input, k, padding_value, "align",
        align)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_diag_part_v3_eager_fallback(
          input, k, padding_value, align=align, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align is None:
    align = "RIGHT_LEFT"
  align = _execute.make_str(align, "align")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixDiagPartV3", input=input, k=k, padding_value=padding_value,
                            align=align, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align", _op.get_attr("align"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixDiagPartV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixDiagPartV3 = tf_export("raw_ops.MatrixDiagPartV3")(_ops.to_raw_op(matrix_diag_part_v3))


def matrix_diag_part_v3_eager_fallback(input: Annotated[Any, TV_MatrixDiagPartV3_T], k: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagPartV3_T], align: str, name, ctx) -> Annotated[Any, TV_MatrixDiagPartV3_T]:
  if align is None:
    align = "RIGHT_LEFT"
  align = _execute.make_str(align, "align")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, padding_value], ctx, [])
  (input, padding_value) = _inputs_T
  k = _ops.convert_to_tensor(k, _dtypes.int32)
  _inputs_flat = [input, k, padding_value]
  _attrs = ("T", _attr_T, "align", align)
  _result = _execute.execute(b"MatrixDiagPartV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixDiagPartV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixDiagV2_T = TypeVar("TV_MatrixDiagV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_diag_v2(diagonal: Annotated[Any, TV_MatrixDiagV2_T], k: Annotated[Any, _atypes.Int32], num_rows: Annotated[Any, _atypes.Int32], num_cols: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagV2_T], name=None) -> Annotated[Any, TV_MatrixDiagV2_T]:
  r"""Returns a batched diagonal tensor with given batched diagonal values.

  Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
  diagonals of a matrix, with everything else padded with `padding`. `num_rows`
  and `num_cols` specify the dimension of the innermost matrix of the output. If
  both are not specified, the op assumes the innermost matrix is square and infers
  its size from `k` and the innermost dimension of `diagonal`. If only one of them
  is specified, the op assumes the unspecified value is the smallest possible
  based on other criteria.

  Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
  rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
  diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
  `r` with shape `[I, J, ..., L, num_rows, num_cols]`.

  The second innermost dimension of `diagonal` has double meaning.
  When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
  [I, J, ..., M], and the output tensor is:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
      padding_value                             ; otherwise
  ```

  Otherwise, `M` is treated as the number of diagonals for the matrix in the
  same batch (`M = k[1]-k[0]+1`), and the output tensor is:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
      padding_value                                     ; otherwise
  ```
  where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

  For example:

  ```
  # The main diagonal.
  diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                       [5, 6, 7, 8]])
  tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                                 [0, 2, 0, 0],
                                 [0, 0, 3, 0],
                                 [0, 0, 0, 4]],
                                [[5, 0, 0, 0],
                                 [0, 6, 0, 0],
                                 [0, 0, 7, 0],
                                 [0, 0, 0, 8]]]

  # A superdiagonal (per batch).
  diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_diag(diagonal, k = 1)
    ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]],
         [[0, 4, 0, 0],
          [0, 0, 5, 0],
          [0, 0, 0, 6],
          [0, 0, 0, 0]]]

  # A band of diagonals.
  diagonals = np.array([[[1, 2, 3],  # Input shape: (2, 2, 3)
                         [4, 5, 0]],
                        [[6, 7, 9],
                         [9, 1, 0]]])
  tf.matrix_diag(diagonals, k = (-1, 0))
    ==> [[[1, 0, 0],  # Output shape: (2, 3, 3)
          [4, 2, 0],
          [0, 5, 3]],
         [[6, 0, 0],
          [9, 7, 0],
          [0, 1, 9]]]

  # Rectangular matrix.
  diagonal = np.array([1, 2])  # Input shape: (2)
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
    ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
         [1, 0, 0, 0],
         [0, 2, 0, 0]]

  # Rectangular matrix with inferred num_cols and padding_value = 9.
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
    ==> [[9, 9],  # Output shape: (3, 2)
         [1, 9],
         [9, 2]]
  ```

  Args:
    diagonal: A `Tensor`. Rank `r`, where `r >= 1`
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    num_rows: A `Tensor` of type `int32`.
      The number of rows of the output matrix. If it is not provided, the op assumes
      the output matrix is a square matrix and infers the matrix size from k and the
      innermost dimension of `diagonal`.
    num_cols: A `Tensor` of type `int32`.
      The number of columns of the output matrix. If it is not provided, the op
      assumes the output matrix is a square matrix and infers the matrix size from
      k and the innermost dimension of `diagonal`.
    padding_value: A `Tensor`. Must have the same type as `diagonal`.
      The number to fill the area outside the specified diagonal band with.
      Default is 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixDiagV2", name, diagonal, k, num_rows, num_cols,
        padding_value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_diag_v2_eager_fallback(
          diagonal, k, num_rows, num_cols, padding_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixDiagV2", diagonal=diagonal, k=k, num_rows=num_rows,
                        num_cols=num_cols, padding_value=padding_value,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixDiagV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixDiagV2 = tf_export("raw_ops.MatrixDiagV2")(_ops.to_raw_op(matrix_diag_v2))


def matrix_diag_v2_eager_fallback(diagonal: Annotated[Any, TV_MatrixDiagV2_T], k: Annotated[Any, _atypes.Int32], num_rows: Annotated[Any, _atypes.Int32], num_cols: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagV2_T], name, ctx) -> Annotated[Any, TV_MatrixDiagV2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([diagonal, padding_value], ctx, [])
  (diagonal, padding_value) = _inputs_T
  k = _ops.convert_to_tensor(k, _dtypes.int32)
  num_rows = _ops.convert_to_tensor(num_rows, _dtypes.int32)
  num_cols = _ops.convert_to_tensor(num_cols, _dtypes.int32)
  _inputs_flat = [diagonal, k, num_rows, num_cols, padding_value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MatrixDiagV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixDiagV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixDiagV3_T = TypeVar("TV_MatrixDiagV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_diag_v3(diagonal: Annotated[Any, TV_MatrixDiagV3_T], k: Annotated[Any, _atypes.Int32], num_rows: Annotated[Any, _atypes.Int32], num_cols: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagV3_T], align:str="RIGHT_LEFT", name=None) -> Annotated[Any, TV_MatrixDiagV3_T]:
  r"""Returns a batched diagonal tensor with given batched diagonal values.

  Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
  diagonals of a matrix, with everything else padded with `padding`. `num_rows`
  and `num_cols` specify the dimension of the innermost matrix of the output. If
  both are not specified, the op assumes the innermost matrix is square and infers
  its size from `k` and the innermost dimension of `diagonal`. If only one of them
  is specified, the op assumes the unspecified value is the smallest possible
  based on other criteria.

  Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
  rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
  diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
  `r` with shape `[I, J, ..., L, num_rows, num_cols]`.

  The second innermost dimension of `diagonal` has double meaning.
  When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
  [I, J, ..., M], and the output tensor is:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
      padding_value                             ; otherwise
  ```

  Otherwise, `M` is treated as the number of diagonals for the matrix in the
  same batch (`M = k[1]-k[0]+1`), and the output tensor is:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
      padding_value                                     ; otherwise
  ```
  where `d = n - m`, `diag_index = [k] - d`, and
  `index_in_diag = n - max(d, 0) + offset`.

  `offset` is zero except when the alignment of the diagonal is to the right.
  ```
  offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                             and `d >= 0`) or
                                           (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                             and `d <= 0`)
           0                          ; otherwise
  ```
  where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

  For example:

  ```
  # The main diagonal.
  diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                       [5, 6, 7, 8]])
  tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                                 [0, 2, 0, 0],
                                 [0, 0, 3, 0],
                                 [0, 0, 0, 4]],
                                [[5, 0, 0, 0],
                                 [0, 6, 0, 0],
                                 [0, 0, 7, 0],
                                 [0, 0, 0, 8]]]

  # A superdiagonal (per batch).
  diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_diag(diagonal, k = 1)
    ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]],
         [[0, 4, 0, 0],
          [0, 0, 5, 0],
          [0, 0, 0, 6],
          [0, 0, 0, 0]]]

  # A tridiagonal band (per batch).
  diagonals = np.array([[[0, 8, 9],  # Input shape: (2, 2, 3)
                         [1, 2, 3],
                         [4, 5, 0]],
                        [[0, 2, 3],
                         [6, 7, 9],
                         [9, 1, 0]]])
  tf.matrix_diag(diagonals, k = (-1, 1))
    ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
          [4, 2, 9],
          [0, 5, 3]],
         [[6, 2, 0],
          [9, 7, 3],
          [0, 1, 9]]]

  # LEFT_RIGHT alignment.
  diagonals = np.array([[[8, 9, 0],  # Input shape: (2, 2, 3)
                         [1, 2, 3],
                         [0, 4, 5]],
                        [[2, 3, 0],
                         [6, 7, 9],
                         [0, 9, 1]]])
  tf.matrix_diag(diagonals, k = (-1, 1), align="LEFT_RIGHT")
    ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
          [4, 2, 9],
          [0, 5, 3]],
         [[6, 2, 0],
          [9, 7, 3],
          [0, 1, 9]]]

  # Rectangular matrix.
  diagonal = np.array([1, 2])  # Input shape: (2)
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
    ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
         [1, 0, 0, 0],
         [0, 2, 0, 0]]

  # Rectangular matrix with inferred num_cols and padding_value = 9.
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
    ==> [[9, 9],  # Output shape: (3, 2)
         [1, 9],
         [9, 2]]

  ```

  Args:
    diagonal: A `Tensor`. Rank `r`, where `r >= 1`
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    num_rows: A `Tensor` of type `int32`.
      The number of rows of the output matrix. If it is not provided, the op assumes
      the output matrix is a square matrix and infers the matrix size from k and the
      innermost dimension of `diagonal`.
    num_cols: A `Tensor` of type `int32`.
      The number of columns of the output matrix. If it is not provided, the op
      assumes the output matrix is a square matrix and infers the matrix size from
      k and the innermost dimension of `diagonal`.
    padding_value: A `Tensor`. Must have the same type as `diagonal`.
      The number to fill the area outside the specified diagonal band with.
      Default is 0.
    align: An optional `string` from: `"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"`. Defaults to `"RIGHT_LEFT"`.
      Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
      a string specifying how superdiagonals and subdiagonals should be aligned,
      respectively. There are four possible alignments: "RIGHT_LEFT" (default),
      "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
      to the right (left-pads the row) and subdiagonals to the left (right-pads the
      row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
      the opposite alignment.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixDiagV3", name, diagonal, k, num_rows, num_cols,
        padding_value, "align", align)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_diag_v3_eager_fallback(
          diagonal, k, num_rows, num_cols, padding_value, align=align,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align is None:
    align = "RIGHT_LEFT"
  align = _execute.make_str(align, "align")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixDiagV3", diagonal=diagonal, k=k, num_rows=num_rows,
                        num_cols=num_cols, padding_value=padding_value,
                        align=align, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align", _op.get_attr("align"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixDiagV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixDiagV3 = tf_export("raw_ops.MatrixDiagV3")(_ops.to_raw_op(matrix_diag_v3))


def matrix_diag_v3_eager_fallback(diagonal: Annotated[Any, TV_MatrixDiagV3_T], k: Annotated[Any, _atypes.Int32], num_rows: Annotated[Any, _atypes.Int32], num_cols: Annotated[Any, _atypes.Int32], padding_value: Annotated[Any, TV_MatrixDiagV3_T], align: str, name, ctx) -> Annotated[Any, TV_MatrixDiagV3_T]:
  if align is None:
    align = "RIGHT_LEFT"
  align = _execute.make_str(align, "align")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([diagonal, padding_value], ctx, [])
  (diagonal, padding_value) = _inputs_T
  k = _ops.convert_to_tensor(k, _dtypes.int32)
  num_rows = _ops.convert_to_tensor(num_rows, _dtypes.int32)
  num_cols = _ops.convert_to_tensor(num_cols, _dtypes.int32)
  _inputs_flat = [diagonal, k, num_rows, num_cols, padding_value]
  _attrs = ("T", _attr_T, "align", align)
  _result = _execute.execute(b"MatrixDiagV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixDiagV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixSetDiag_T = TypeVar("TV_MatrixSetDiag_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_set_diag(input: Annotated[Any, TV_MatrixSetDiag_T], diagonal: Annotated[Any, TV_MatrixSetDiag_T], name=None) -> Annotated[Any, TV_MatrixSetDiag_T]:
  r"""Returns a batched matrix tensor with new batched diagonal values.

  Given `input` and `diagonal`, this operation returns a tensor with the
  same shape and values as `input`, except for the main diagonal of the
  innermost matrices.  These will be overwritten by the values in `diagonal`.

  The output is computed as follows:

  Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
  `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
  tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:

    * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
    * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.

  Args:
    input: A `Tensor`. Rank `k+1`, where `k >= 1`.
    diagonal: A `Tensor`. Must have the same type as `input`.
      Rank `k`, where `k >= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixSetDiag", name, input, diagonal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_set_diag_eager_fallback(
          input, diagonal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixSetDiag", input=input, diagonal=diagonal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixSetDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixSetDiag = tf_export("raw_ops.MatrixSetDiag")(_ops.to_raw_op(matrix_set_diag))


def matrix_set_diag_eager_fallback(input: Annotated[Any, TV_MatrixSetDiag_T], diagonal: Annotated[Any, TV_MatrixSetDiag_T], name, ctx) -> Annotated[Any, TV_MatrixSetDiag_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, diagonal], ctx, [])
  (input, diagonal) = _inputs_T
  _inputs_flat = [input, diagonal]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MatrixSetDiag", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixSetDiag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixSetDiagV2_T = TypeVar("TV_MatrixSetDiagV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_set_diag_v2(input: Annotated[Any, TV_MatrixSetDiagV2_T], diagonal: Annotated[Any, TV_MatrixSetDiagV2_T], k: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_MatrixSetDiagV2_T]:
  r"""Returns a batched matrix tensor with new batched diagonal values.

  Given `input` and `diagonal`, this operation returns a tensor with the
  same shape and values as `input`, except for the specified diagonals of the
  innermost matrices. These will be overwritten by the values in `diagonal`.

  `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
  `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
  Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
  `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
  `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`

  The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
  If `k` is scalar or `k[0] == k[1]`:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
      input[i, j, ..., l, m, n]              ; otherwise
  ```

  Otherwise,

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
      input[i, j, ..., l, m, n]                         ; otherwise
  ```
  where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

  For example:

  ```
  # The main diagonal.
  input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]],
                    [[7, 7, 7, 7],
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]]])
  diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_set_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
                                     [7, 2, 7, 7],
                                     [7, 7, 3, 7]],
                                    [[4, 7, 7, 7],
                                     [7, 5, 7, 7],
                                     [7, 7, 6, 7]]]

  # A superdiagonal (per batch).
  tf.matrix_set_diag(diagonal, k = 1)
    ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
          [7, 7, 2, 7],
          [7, 7, 7, 3]],
         [[7, 4, 7, 7],
          [7, 7, 5, 7],
          [7, 7, 7, 6]]]

  # A band of diagonals.
  diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
                         [4, 5, 0]],
                        [[6, 1, 2],
                         [3, 4, 0]]])
  tf.matrix_set_diag(diagonals, k = (-1, 0))
    ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
          [4, 2, 7, 7],
          [0, 5, 3, 7]],
         [[6, 7, 7, 7],
          [3, 1, 7, 7],
          [7, 4, 2, 7]]]

  ```

  Args:
    input: A `Tensor`. Rank `r+1`, where `r >= 1`.
    diagonal: A `Tensor`. Must have the same type as `input`.
      Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
      `k >= 1`.
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixSetDiagV2", name, input, diagonal, k)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_set_diag_v2_eager_fallback(
          input, diagonal, k, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixSetDiagV2", input=input, diagonal=diagonal, k=k, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixSetDiagV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixSetDiagV2 = tf_export("raw_ops.MatrixSetDiagV2")(_ops.to_raw_op(matrix_set_diag_v2))


def matrix_set_diag_v2_eager_fallback(input: Annotated[Any, TV_MatrixSetDiagV2_T], diagonal: Annotated[Any, TV_MatrixSetDiagV2_T], k: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_MatrixSetDiagV2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, diagonal], ctx, [])
  (input, diagonal) = _inputs_T
  k = _ops.convert_to_tensor(k, _dtypes.int32)
  _inputs_flat = [input, diagonal, k]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MatrixSetDiagV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixSetDiagV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatrixSetDiagV3_T = TypeVar("TV_MatrixSetDiagV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def matrix_set_diag_v3(input: Annotated[Any, TV_MatrixSetDiagV3_T], diagonal: Annotated[Any, TV_MatrixSetDiagV3_T], k: Annotated[Any, _atypes.Int32], align:str="RIGHT_LEFT", name=None) -> Annotated[Any, TV_MatrixSetDiagV3_T]:
  r"""Returns a batched matrix tensor with new batched diagonal values.

  Given `input` and `diagonal`, this operation returns a tensor with the
  same shape and values as `input`, except for the specified diagonals of the
  innermost matrices. These will be overwritten by the values in `diagonal`.

  `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
  `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
  Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
  `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
  `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`

  The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
  If `k` is scalar or `k[0] == k[1]`:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
      input[i, j, ..., l, m, n]              ; otherwise
  ```

  Otherwise,

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
      input[i, j, ..., l, m, n]                         ; otherwise
  ```
  where `d = n - m`, `diag_index = k[1] - d`, and
  `index_in_diag = n - max(d, 0) + offset`.

  `offset` is zero except when the alignment of the diagonal is to the right.
  ```
  offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                             and `d >= 0`) or
                                           (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                             and `d <= 0`)
           0                          ; otherwise
  ```
  where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

  For example:

  ```
  # The main diagonal.
  input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]],
                    [[7, 7, 7, 7],
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]]])
  diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_set_diag(input, diagonal)
    ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
          [7, 2, 7, 7],
          [7, 7, 3, 7]],
         [[4, 7, 7, 7],
          [7, 5, 7, 7],
          [7, 7, 6, 7]]]

  # A superdiagonal (per batch).
  tf.matrix_set_diag(input, diagonal, k = 1)
    ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
          [7, 7, 2, 7],
          [7, 7, 7, 3]],
         [[7, 4, 7, 7],
          [7, 7, 5, 7],
          [7, 7, 7, 6]]]

  # A band of diagonals.
  diagonals = np.array([[[0, 9, 1],  # Diagonal shape: (2, 4, 3)
                         [6, 5, 8],
                         [1, 2, 3],
                         [4, 5, 0]],
                        [[0, 1, 2],
                         [5, 6, 4],
                         [6, 1, 2],
                         [3, 4, 0]]])
  tf.matrix_set_diag(input, diagonals, k = (-1, 2))
    ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
          [4, 2, 5, 1],
          [7, 5, 3, 8]],
         [[6, 5, 1, 7],
          [3, 1, 6, 2],
          [7, 4, 2, 4]]]

  # LEFT_RIGHT alignment.
  diagonals = np.array([[[9, 1, 0],  # Diagonal shape: (2, 4, 3)
                         [6, 5, 8],
                         [1, 2, 3],
                         [0, 4, 5]],
                        [[1, 2, 0],
                         [5, 6, 4],
                         [6, 1, 2],
                         [0, 3, 4]]])
  tf.matrix_set_diag(input, diagonals, k = (-1, 2), align="LEFT_RIGHT")
    ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
          [4, 2, 5, 1],
          [7, 5, 3, 8]],
         [[6, 5, 1, 7],
          [3, 1, 6, 2],
          [7, 4, 2, 4]]]

  ```

  Args:
    input: A `Tensor`. Rank `r+1`, where `r >= 1`.
    diagonal: A `Tensor`. Must have the same type as `input`.
      Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
      `k >= 1`.
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    align: An optional `string` from: `"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"`. Defaults to `"RIGHT_LEFT"`.
      Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
      a string specifying how superdiagonals and subdiagonals should be aligned,
      respectively. There are four possible alignments: "RIGHT_LEFT" (default),
      "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
      to the right (left-pads the row) and subdiagonals to the left (right-pads the
      row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
      the opposite alignment.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatrixSetDiagV3", name, input, diagonal, k, "align", align)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matrix_set_diag_v3_eager_fallback(
          input, diagonal, k, align=align, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align is None:
    align = "RIGHT_LEFT"
  align = _execute.make_str(align, "align")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatrixSetDiagV3", input=input, diagonal=diagonal, k=k, align=align,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align", _op.get_attr("align"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatrixSetDiagV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatrixSetDiagV3 = tf_export("raw_ops.MatrixSetDiagV3")(_ops.to_raw_op(matrix_set_diag_v3))


def matrix_set_diag_v3_eager_fallback(input: Annotated[Any, TV_MatrixSetDiagV3_T], diagonal: Annotated[Any, TV_MatrixSetDiagV3_T], k: Annotated[Any, _atypes.Int32], align: str, name, ctx) -> Annotated[Any, TV_MatrixSetDiagV3_T]:
  if align is None:
    align = "RIGHT_LEFT"
  align = _execute.make_str(align, "align")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, diagonal], ctx, [])
  (input, diagonal) = _inputs_T
  k = _ops.convert_to_tensor(k, _dtypes.int32)
  _inputs_flat = [input, diagonal, k]
  _attrs = ("T", _attr_T, "align", align)
  _result = _execute.execute(b"MatrixSetDiagV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatrixSetDiagV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MirrorPad_T = TypeVar("TV_MirrorPad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MirrorPad_Tpaddings = TypeVar("TV_MirrorPad_Tpaddings", _atypes.Int32, _atypes.Int64)

def mirror_pad(input: Annotated[Any, TV_MirrorPad_T], paddings: Annotated[Any, TV_MirrorPad_Tpaddings], mode: str, name=None) -> Annotated[Any, TV_MirrorPad_T]:
  r"""Pads a tensor with mirrored values.

  This operation pads a `input` with mirrored values according to the `paddings`
  you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
  the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many values to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of `input`
  in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
  than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
  (if false, respectively).

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```
  # 't' is [[1, 2, 3], [4, 5, 6]].
  # 'paddings' is [[1, 1]], [2, 2]].
  # 'mode' is SYMMETRIC.
  # rank of 't' is 2.
  pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                        [2, 1, 1, 2, 3, 3, 2]
                        [5, 4, 4, 5, 6, 6, 5]
                        [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    input: A `Tensor`. The input tensor to be padded.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
      Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
      do not include the borders, while in symmetric mode the padded regions
      do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
      is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
      it is `[1, 2, 3, 3, 2]` in symmetric mode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MirrorPad", name, input, paddings, "mode", mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mirror_pad_eager_fallback(
          input, paddings, mode=mode, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  mode = _execute.make_str(mode, "mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MirrorPad", input=input, paddings=paddings, mode=mode, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tpaddings",
              _op._get_attr_type("Tpaddings"), "mode", _op.get_attr("mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MirrorPad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MirrorPad = tf_export("raw_ops.MirrorPad")(_ops.to_raw_op(mirror_pad))


def mirror_pad_eager_fallback(input: Annotated[Any, TV_MirrorPad_T], paddings: Annotated[Any, TV_MirrorPad_Tpaddings], mode: str, name, ctx) -> Annotated[Any, TV_MirrorPad_T]:
  mode = _execute.make_str(mode, "mode")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, paddings]
  _attrs = ("T", _attr_T, "Tpaddings", _attr_Tpaddings, "mode", mode)
  _result = _execute.execute(b"MirrorPad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MirrorPad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MirrorPadGrad_T = TypeVar("TV_MirrorPadGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MirrorPadGrad_Tpaddings = TypeVar("TV_MirrorPadGrad_Tpaddings", _atypes.Int32, _atypes.Int64)

def mirror_pad_grad(input: Annotated[Any, TV_MirrorPadGrad_T], paddings: Annotated[Any, TV_MirrorPadGrad_Tpaddings], mode: str, name=None) -> Annotated[Any, TV_MirrorPadGrad_T]:
  r"""Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

  This operation folds the padded areas of `input` by `MirrorPad` according to the
  `paddings` you specify. `paddings` must be the same as `paddings` argument
  given to the corresponding `MirrorPad` op.

  The folded size of each dimension D of the output is:

  `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

  For example:

  ```
  # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
  # 'paddings' is [[0, 1]], [0, 1]].
  # 'mode' is SYMMETRIC.
  # rank of 't' is 2.
  pad(t, paddings) ==> [[ 1,  5]
                        [11, 28]]
  ```

  Args:
    input: A `Tensor`. The input tensor to be folded.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
      The mode used in the `MirrorPad` op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MirrorPadGrad", name, input, paddings, "mode", mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mirror_pad_grad_eager_fallback(
          input, paddings, mode=mode, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  mode = _execute.make_str(mode, "mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MirrorPadGrad", input=input, paddings=paddings, mode=mode, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tpaddings",
              _op._get_attr_type("Tpaddings"), "mode", _op.get_attr("mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MirrorPadGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MirrorPadGrad = tf_export("raw_ops.MirrorPadGrad")(_ops.to_raw_op(mirror_pad_grad))


def mirror_pad_grad_eager_fallback(input: Annotated[Any, TV_MirrorPadGrad_T], paddings: Annotated[Any, TV_MirrorPadGrad_Tpaddings], mode: str, name, ctx) -> Annotated[Any, TV_MirrorPadGrad_T]:
  mode = _execute.make_str(mode, "mode")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, paddings]
  _attrs = ("T", _attr_T, "Tpaddings", _attr_Tpaddings, "mode", mode)
  _result = _execute.execute(b"MirrorPadGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MirrorPadGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_OneHot_T = TypeVar("TV_OneHot_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_OneHot_TI = TypeVar("TV_OneHot_TI", _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)

def one_hot(indices: Annotated[Any, TV_OneHot_TI], depth: Annotated[Any, _atypes.Int32], on_value: Annotated[Any, TV_OneHot_T], off_value: Annotated[Any, TV_OneHot_T], axis:int=-1, name=None) -> Annotated[Any, TV_OneHot_T]:
  r"""Returns a one-hot tensor.

  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.

  If the input `indices` is rank `N`, the output will have rank `N+1`,
  The new axis is created at dimension `axis` (default: the new axis is
  appended at the end).

  If `indices` is a scalar the output shape will be a vector of length `depth`.

  If `indices` is a vector of length `features`, the output shape will be:
  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```

  If `indices` is a matrix (batch) with shape `[batch, features]`,
  the output shape will be:
  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```


  Examples
  =========

  Suppose that
  ```
    indices = [0, 2, -1, 1]
    depth = 3
    on_value = 5.0
    off_value = 0.0
    axis = -1
  ```

  Then output is `[4 x 3]`:
  ```
  output =
    [5.0 0.0 0.0]  // one_hot(0)
    [0.0 0.0 5.0]  // one_hot(2)
    [0.0 0.0 0.0]  // one_hot(-1)
    [0.0 5.0 0.0]  // one_hot(1)
  ```

  Suppose that
  ```
    indices = [0, 2, -1, 1]
    depth = 3
    on_value = 0.0
    off_value = 3.0
    axis = 0
  ```

  Then output is `[3 x 4]`:
  ```
  output =
    [0.0 3.0 3.0 3.0]
    [3.0 3.0 3.0 0.0]
    [3.0 3.0 3.0 3.0]
    [3.0 0.0 3.0 3.0]
  //  ^                one_hot(0)
  //      ^            one_hot(2)
  //          ^        one_hot(-1)
  //              ^    one_hot(1)
  ```

  Suppose that
  ```
    indices = [[0, 2], [1, -1]]
    depth = 3
    on_value = 1.0
    off_value = 0.0
    axis = -1
  ```

  Then output is `[2 x 2 x 3]`:
  ```
  output =
    [
      [1.0, 0.0, 0.0]  // one_hot(0)
      [0.0, 0.0, 1.0]  // one_hot(2)
    ][
      [0.0, 1.0, 0.0]  // one_hot(1)
      [0.0, 0.0, 0.0]  // one_hot(-1)
    ]
  ```

  Args:
    indices: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `int64`.
      A tensor of indices.
    depth: A `Tensor` of type `int32`.
      A scalar defining the depth of the one hot dimension.
    on_value: A `Tensor`.
      A scalar defining the value to fill in output when `indices[j] = i`.
    off_value: A `Tensor`. Must have the same type as `on_value`.
      A scalar defining the value to fill in output when `indices[j] != i`.
    axis: An optional `int`. Defaults to `-1`.
      The axis to fill (default: -1, a new inner-most axis).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `on_value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OneHot", name, indices, depth, on_value, off_value, "axis",
        axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return one_hot_eager_fallback(
          indices, depth, on_value, off_value, axis=axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OneHot", indices=indices, depth=depth, on_value=on_value,
                  off_value=off_value, axis=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("axis", _op._get_attr_int("axis"), "T", _op._get_attr_type("T"),
              "TI", _op._get_attr_type("TI"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OneHot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OneHot = tf_export("raw_ops.OneHot")(_ops.to_raw_op(one_hot))


def one_hot_eager_fallback(indices: Annotated[Any, TV_OneHot_TI], depth: Annotated[Any, _atypes.Int32], on_value: Annotated[Any, TV_OneHot_T], off_value: Annotated[Any, TV_OneHot_T], axis: int, name, ctx) -> Annotated[Any, TV_OneHot_T]:
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([on_value, off_value], ctx, [])
  (on_value, off_value) = _inputs_T
  _attr_TI, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  depth = _ops.convert_to_tensor(depth, _dtypes.int32)
  _inputs_flat = [indices, depth, on_value, off_value]
  _attrs = ("axis", axis, "T", _attr_T, "TI", _attr_TI)
  _result = _execute.execute(b"OneHot", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OneHot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_OnesLike_T = TypeVar("TV_OnesLike_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def ones_like(x: Annotated[Any, TV_OnesLike_T], name=None) -> Annotated[Any, TV_OnesLike_T]:
  r"""Returns a tensor of ones with the same shape and type as x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`, `complex64`, `complex128`, `bool`.
      a tensor of type T.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OnesLike", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ones_like_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OnesLike", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OnesLike", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OnesLike = tf_export("raw_ops.OnesLike")(_ops.to_raw_op(ones_like))


def ones_like_eager_fallback(x: Annotated[Any, TV_OnesLike_T], name, ctx) -> Annotated[Any, TV_OnesLike_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.uint32, _dtypes.int64, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, _dtypes.bool, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"OnesLike", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OnesLike", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Pack_T = TypeVar("TV_Pack_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def pack(values: Annotated[List[Any], TV_Pack_T], axis:int=0, name=None) -> Annotated[Any, TV_Pack_T]:
  r"""Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs the `N` tensors in `values` into a tensor with rank one higher than each
  tensor in `values`, by packing them along the `axis` dimension.
  Given a list of tensors of shape `(A, B, C)`;

  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.

  For example:

  ```
  # 'x' is [1, 4]
  # 'y' is [2, 5]
  # 'z' is [3, 6]
  pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
  pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
  ```

  This is the opposite of `unpack`.

  Args:
    values: A list of at least 1 `Tensor` objects with the same type.
      Must be of same shape and type.
    axis: An optional `int`. Defaults to `0`.
      Dimension along which to pack.  Negative values wrap around, so the
      valid range is `[-(R+1), R+1)`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Pack", name, values, "axis", axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return pack_eager_fallback(
          values, axis=axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'pack' Op, not %r." % values)
  _attr_N = len(values)
  if axis is None:
    axis = 0
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Pack", values=values, axis=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "axis", _op._get_attr_int("axis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Pack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Pack = tf_export("raw_ops.Pack")(_ops.to_raw_op(pack))


def pack_eager_fallback(values: Annotated[List[Any], TV_Pack_T], axis: int, name, ctx) -> Annotated[Any, TV_Pack_T]:
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'pack' Op, not %r." % values)
  _attr_N = len(values)
  if axis is None:
    axis = 0
  axis = _execute.make_int(axis, "axis")
  _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
  _inputs_flat = list(values)
  _attrs = ("N", _attr_N, "T", _attr_T, "axis", axis)
  _result = _execute.execute(b"Pack", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Pack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Pad_T = TypeVar("TV_Pad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Pad_Tpaddings = TypeVar("TV_Pad_Tpaddings", _atypes.Int32, _atypes.Int64)

def pad(input: Annotated[Any, TV_Pad_T], paddings: Annotated[Any, TV_Pad_Tpaddings], name=None) -> Annotated[Any, TV_Pad_T]:
  r"""Pads a tensor with zeros.

  This operation pads a `input` with zeros according to the `paddings` you
  specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
  rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many zeros to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
  in that dimension.

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```
  # 't' is [[1, 1], [2, 2]]
  # 'paddings' is [[1, 1], [2, 2]]
  # rank of 't' is 2
  pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                        [0, 0, 1, 1, 0, 0]
                        [0, 0, 2, 2, 0, 0]
                        [0, 0, 0, 0, 0, 0]]
  ```

  Args:
    input: A `Tensor`.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Pad", name, input, paddings)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return pad_eager_fallback(
          input, paddings, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Pad", input=input, paddings=paddings, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tpaddings",
              _op._get_attr_type("Tpaddings"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Pad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Pad = tf_export("raw_ops.Pad")(_ops.to_raw_op(pad))


def pad_eager_fallback(input: Annotated[Any, TV_Pad_T], paddings: Annotated[Any, TV_Pad_Tpaddings], name, ctx) -> Annotated[Any, TV_Pad_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, paddings]
  _attrs = ("T", _attr_T, "Tpaddings", _attr_Tpaddings)
  _result = _execute.execute(b"Pad", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Pad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PadV2_T = TypeVar("TV_PadV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_PadV2_Tpaddings = TypeVar("TV_PadV2_Tpaddings", _atypes.Int32, _atypes.Int64)

def pad_v2(input: Annotated[Any, TV_PadV2_T], paddings: Annotated[Any, TV_PadV2_Tpaddings], constant_values: Annotated[Any, TV_PadV2_T], name=None) -> Annotated[Any, TV_PadV2_T]:
  r"""Pads a tensor.

  This operation pads `input` according to the `paddings` and `constant_values`
  you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
  the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many padding values to add before the contents of `input` in that dimension,
  and `paddings[D, 1]` indicates how many padding values to add after the contents
  of `input` in that dimension. `constant_values` is a scalar tensor of the same
  type as `input` that indicates the value to use for padding `input`.

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```
  # 't' is [[1, 1], [2, 2]]
  # 'paddings' is [[1, 1], [2, 2]]
  # 'constant_values' is 0
  # rank of 't' is 2
  pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                        [0, 0, 1, 1, 0, 0]
                        [0, 0, 2, 2, 0, 0]
                        [0, 0, 0, 0, 0, 0]]
  ```

  Args:
    input: A `Tensor`.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    constant_values: A `Tensor`. Must have the same type as `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PadV2", name, input, paddings, constant_values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return pad_v2_eager_fallback(
          input, paddings, constant_values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PadV2", input=input, paddings=paddings,
                 constant_values=constant_values, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tpaddings",
              _op._get_attr_type("Tpaddings"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PadV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PadV2 = tf_export("raw_ops.PadV2")(_ops.to_raw_op(pad_v2))


def pad_v2_eager_fallback(input: Annotated[Any, TV_PadV2_T], paddings: Annotated[Any, TV_PadV2_Tpaddings], constant_values: Annotated[Any, TV_PadV2_T], name, ctx) -> Annotated[Any, TV_PadV2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, constant_values], ctx, [])
  (input, constant_values) = _inputs_T
  _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, paddings, constant_values]
  _attrs = ("T", _attr_T, "Tpaddings", _attr_Tpaddings)
  _result = _execute.execute(b"PadV2", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PadV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ParallelConcat_T = TypeVar("TV_ParallelConcat_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def parallel_concat(values: Annotated[List[Any], TV_ParallelConcat_T], shape, name=None) -> Annotated[Any, TV_ParallelConcat_T]:
  r"""Concatenates a list of `N` tensors along the first dimension.

  The input tensors are all required to have size 1 in the first dimension.

  For example:

  ```
  # 'x' is [[1, 4]]
  # 'y' is [[2, 5]]
  # 'z' is [[3, 6]]
  parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
  ```

  The difference between concat and parallel_concat is that concat requires all
  of the inputs be computed before the operation will begin but doesn't require
  that the input shapes be known during graph construction.  Parallel concat
  will copy pieces of the input into the output as they become available, in
  some situations this can provide a performance benefit.

  Args:
    values: A list of at least 1 `Tensor` objects with the same type.
      Tensors to be concatenated. All must have size 1 in the first dimension
      and same shape.
    shape: A `tf.TensorShape` or list of `ints`.
      the final shape of the result; should be equal to the shapes of any input
      but with the number of input values in the first dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelConcat", name, values, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_concat_eager_fallback(
          values, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'parallel_concat' Op, not %r." % values)
  _attr_N = len(values)
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelConcat", values=values, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "shape", _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelConcat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelConcat = tf_export("raw_ops.ParallelConcat")(_ops.to_raw_op(parallel_concat))


def parallel_concat_eager_fallback(values: Annotated[List[Any], TV_ParallelConcat_T], shape, name, ctx) -> Annotated[Any, TV_ParallelConcat_T]:
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'parallel_concat' Op, not %r." % values)
  _attr_N = len(values)
  shape = _execute.make_shape(shape, "shape")
  _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
  _inputs_flat = list(values)
  _attrs = ("N", _attr_N, "T", _attr_T, "shape", shape)
  _result = _execute.execute(b"ParallelConcat", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelConcat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Placeholder_dtype = TypeVar("TV_Placeholder_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def placeholder(dtype: TV_Placeholder_dtype, shape=None, name=None) -> Annotated[Any, TV_Placeholder_dtype]:
  r"""A placeholder op for a value that will be fed into the computation.

  N.B. This operation will fail with an error if it is executed. It is
  intended as a way to represent a value that will always be fed, and to
  provide attrs that enable the fed value to be checked at runtime.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      (Optional) The shape of the tensor. If the shape has 0 dimensions, the
      shape is unconstrained.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Placeholder", name, "dtype", dtype, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return placeholder_eager_fallback(
          dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if shape is None:
    shape = None
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Placeholder", dtype=dtype, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Placeholder", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Placeholder = tf_export("raw_ops.Placeholder")(_ops.to_raw_op(placeholder))


def placeholder_eager_fallback(dtype: TV_Placeholder_dtype, shape, name, ctx) -> Annotated[Any, TV_Placeholder_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if shape is None:
    shape = None
  shape = _execute.make_shape(shape, "shape")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape)
  _result = _execute.execute(b"Placeholder", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Placeholder", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PlaceholderV2_dtype = TypeVar("TV_PlaceholderV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def placeholder_v2(dtype: TV_PlaceholderV2_dtype, shape, name=None) -> Annotated[Any, TV_PlaceholderV2_dtype]:
  r"""A placeholder op for a value that will be fed into the computation.

  N.B. This operation will fail with an error if it is executed. It is
  intended as a way to represent a value that will always be fed, and to
  provide attrs that enable the fed value to be checked at runtime.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`.
      The shape of the tensor. The shape can be any partially-specified
      shape.  To be unconstrained, pass in a shape with unknown rank.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PlaceholderV2", name, "dtype", dtype, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return placeholder_v2_eager_fallback(
          dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PlaceholderV2", dtype=dtype, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PlaceholderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PlaceholderV2 = tf_export("raw_ops.PlaceholderV2")(_ops.to_raw_op(placeholder_v2))


def placeholder_v2_eager_fallback(dtype: TV_PlaceholderV2_dtype, shape, name, ctx) -> Annotated[Any, TV_PlaceholderV2_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape)
  _result = _execute.execute(b"PlaceholderV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PlaceholderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PlaceholderWithDefault_dtype = TypeVar("TV_PlaceholderWithDefault_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def placeholder_with_default(input: Annotated[Any, TV_PlaceholderWithDefault_dtype], shape, name=None) -> Annotated[Any, TV_PlaceholderWithDefault_dtype]:
  r"""A placeholder op that passes through `input` when its output is not fed.

  Args:
    input: A `Tensor`. The default value to produce when `output` is not fed.
    shape: A `tf.TensorShape` or list of `ints`.
      The (possibly partial) shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PlaceholderWithDefault", name, input, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return placeholder_with_default_eager_fallback(
          input, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PlaceholderWithDefault", input=input, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PlaceholderWithDefault", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PlaceholderWithDefault = tf_export("raw_ops.PlaceholderWithDefault")(_ops.to_raw_op(placeholder_with_default))


def placeholder_with_default_eager_fallback(input: Annotated[Any, TV_PlaceholderWithDefault_dtype], shape, name, ctx) -> Annotated[Any, TV_PlaceholderWithDefault_dtype]:
  shape = _execute.make_shape(shape, "shape")
  _attr_dtype, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("dtype", _attr_dtype, "shape", shape)
  _result = _execute.execute(b"PlaceholderWithDefault", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PlaceholderWithDefault", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PreventGradient_T = TypeVar("TV_PreventGradient_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def prevent_gradient(input: Annotated[Any, TV_PreventGradient_T], message:str="", name=None) -> Annotated[Any, TV_PreventGradient_T]:
  r"""An identity op that triggers an error if a gradient is requested.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, the TensorFlow gradient system
  will return an error when trying to lookup the gradient of this op,
  because no gradient must ever be registered for this function.  This
  op exists to prevent subtle bugs from silently returning unimplemented
  gradients in some corner cases.

  Args:
    input: A `Tensor`. any tensor.
    message: An optional `string`. Defaults to `""`.
      Will be printed in the error when anyone tries to differentiate
      this operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PreventGradient", name, input, "message", message)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return prevent_gradient_eager_fallback(
          input, message=message, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if message is None:
    message = ""
  message = _execute.make_str(message, "message")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PreventGradient", input=input, message=message, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "message",
              _op.get_attr("message"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PreventGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PreventGradient = tf_export("raw_ops.PreventGradient")(_ops.to_raw_op(prevent_gradient))


def prevent_gradient_eager_fallback(input: Annotated[Any, TV_PreventGradient_T], message: str, name, ctx) -> Annotated[Any, TV_PreventGradient_T]:
  if message is None:
    message = ""
  message = _execute.make_str(message, "message")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "message", message)
  _result = _execute.execute(b"PreventGradient", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PreventGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_QuantizeAndDequantize_T = TypeVar("TV_QuantizeAndDequantize_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def quantize_and_dequantize(input: Annotated[Any, TV_QuantizeAndDequantize_T], signed_input:bool=True, num_bits:int=8, range_given:bool=False, input_min:float=0, input_max:float=0, name=None) -> Annotated[Any, TV_QuantizeAndDequantize_T]:
  r"""Use QuantizeAndDequantizeV2 instead.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    signed_input: An optional `bool`. Defaults to `True`.
    num_bits: An optional `int`. Defaults to `8`.
    range_given: An optional `bool`. Defaults to `False`.
    input_min: An optional `float`. Defaults to `0`.
    input_max: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeAndDequantize", name, input, "signed_input",
        signed_input, "num_bits", num_bits, "range_given", range_given,
        "input_min", input_min, "input_max", input_max)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_and_dequantize_eager_fallback(
          input, signed_input=signed_input, num_bits=num_bits,
          range_given=range_given, input_min=input_min, input_max=input_max,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if range_given is None:
    range_given = False
  range_given = _execute.make_bool(range_given, "range_given")
  if input_min is None:
    input_min = 0
  input_min = _execute.make_float(input_min, "input_min")
  if input_max is None:
    input_max = 0
  input_max = _execute.make_float(input_max, "input_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeAndDequantize", input=input, signed_input=signed_input,
                                 num_bits=num_bits, range_given=range_given,
                                 input_min=input_min, input_max=input_max,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("signed_input", _op._get_attr_bool("signed_input"), "num_bits",
              _op._get_attr_int("num_bits"), "range_given",
              _op._get_attr_bool("range_given"), "input_min",
              _op.get_attr("input_min"), "input_max",
              _op.get_attr("input_max"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeAndDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QuantizeAndDequantize = tf_export("raw_ops.QuantizeAndDequantize")(_ops.to_raw_op(quantize_and_dequantize))


def quantize_and_dequantize_eager_fallback(input: Annotated[Any, TV_QuantizeAndDequantize_T], signed_input: bool, num_bits: int, range_given: bool, input_min: float, input_max: float, name, ctx) -> Annotated[Any, TV_QuantizeAndDequantize_T]:
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if range_given is None:
    range_given = False
  range_given = _execute.make_bool(range_given, "range_given")
  if input_min is None:
    input_min = 0
  input_min = _execute.make_float(input_min, "input_min")
  if input_max is None:
    input_max = 0
  input_max = _execute.make_float(input_max, "input_max")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [input]
  _attrs = ("signed_input", signed_input, "num_bits", num_bits, "range_given",
  range_given, "input_min", input_min, "input_max", input_max, "T", _attr_T)
  _result = _execute.execute(b"QuantizeAndDequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeAndDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_QuantizeAndDequantizeV2_T = TypeVar("TV_QuantizeAndDequantizeV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def quantize_and_dequantize_v2(input: Annotated[Any, TV_QuantizeAndDequantizeV2_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV2_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV2_T], signed_input:bool=True, num_bits:int=8, range_given:bool=False, round_mode:str="HALF_TO_EVEN", narrow_range:bool=False, axis:int=-1, name=None) -> Annotated[Any, TV_QuantizeAndDequantizeV2_T]:
  r"""Quantizes then dequantizes a tensor.

  This op simulates the precision loss from the quantized forward pass by:

  1. Quantizing the tensor to fixed point numbers, which should match the target
     quantization method when it is used in inference.
  2. Dequantizing it back to floating point numbers for the following ops, most
     likely matmul.

  There are different ways to quantize. This version uses only scaling, so 0.0
  maps to 0.

  From the specified 'num_bits' in the quantized output type, it determines
  minimum and maximum representable quantized values.

  e.g.

  *   [-128, 127] for signed, num_bits = 8, or
  *   [0, 255] for unsigned, num_bits = 8.

  If range_given == False, the initial input_min, input_max will be determined
  automatically as the minimum and maximum values in the input tensor, otherwise
  the specified values of input_min, input_max are used.

  Note: If the input_min, input_max are specified, they do not need to equal the
  actual minimum and maximum values in the tensor. e.g. in some cases it may be
  beneficial to specify these values such that the low probability extremes of the
  input distribution are clipped.

  This op determines the maximum scale_factor that would map the initial
  [input_min, input_max] range to a range that lies within the representable
  quantized range.

  It determines the scale from one of input_min and input_max, then updates the
  other one to maximize the representable range.

  e.g.

  *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
      5.0]: it would use a scale_factor of -128 / -10.0 = 12.8 In this case, it
      would update input_max to be 127 / 12.8 = 9.921875
  *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
      10.0]: it would use a scale_factor of 127 / 10.0 = 12.7 In this case, it
      would update input_min to be 128.0 / 12.7 = -10.07874
  *   if the output is unsigned, input_min is forced to be 0, and only the
      specified input_max is used.

  After determining the scale_factor and updating the input range, it applies the
  following to each value in the 'input' tensor.

  output = round(clamp(value, input_min, input_max) * scale_factor) / scale_factor.

  The above round function rounds the value based on the given round_mode.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      Tensor to quantize and then dequantize.
    input_min: A `Tensor`. Must have the same type as `input`.
      If `range_given == True`, this specifies the minimum input value that needs to
      be represented, otherwise it is determined from the min value of the `input`
      tensor.
    input_max: A `Tensor`. Must have the same type as `input`.
      If `range_given == True`, this specifies the maximum input value that needs to
      be represented, otherwise it is determined from the max value of the `input`
      tensor.
    signed_input: An optional `bool`. Defaults to `True`.
      Whether the quantization is signed or unsigned. (actually this parameter should
      have been called <b>`signed_output`</b>)
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization.
    range_given: An optional `bool`. Defaults to `False`.
      Whether the range is given or should be determined from the `input` tensor.
    round_mode: An optional `string` from: `"HALF_TO_EVEN", "HALF_UP"`. Defaults to `"HALF_TO_EVEN"`.
      The 'round_mode' attribute controls which rounding tie-breaking algorithm is
      used when rounding float values to their quantized equivalents. The following
      rounding modes are currently supported:

      *   HALF_TO_EVEN: this is the default round_mode.
      *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
          rounds up to -7.
    narrow_range: An optional `bool`. Defaults to `False`.
      If True, then the absolute value of the quantized minimum value is the same as
      the quantized maximum value, instead of 1 greater.
      i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    axis: An optional `int`. Defaults to `-1`.
      If specified, this axis is treated as a channel or slice axis, and a separate
      quantization range is used for each channel or slice along this axis.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeAndDequantizeV2", name, input, input_min, input_max,
        "signed_input", signed_input, "num_bits", num_bits, "range_given",
        range_given, "round_mode", round_mode, "narrow_range", narrow_range,
        "axis", axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_and_dequantize_v2_eager_fallback(
          input, input_min, input_max, signed_input=signed_input,
          num_bits=num_bits, range_given=range_given, round_mode=round_mode,
          narrow_range=narrow_range, axis=axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if range_given is None:
    range_given = False
  range_given = _execute.make_bool(range_given, "range_given")
  if round_mode is None:
    round_mode = "HALF_TO_EVEN"
  round_mode = _execute.make_str(round_mode, "round_mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeAndDequantizeV2", input=input, input_min=input_min,
                                   input_max=input_max,
                                   signed_input=signed_input,
                                   num_bits=num_bits, range_given=range_given,
                                   round_mode=round_mode,
                                   narrow_range=narrow_range, axis=axis,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("signed_input", _op._get_attr_bool("signed_input"), "num_bits",
              _op._get_attr_int("num_bits"), "range_given",
              _op._get_attr_bool("range_given"), "T", _op._get_attr_type("T"),
              "round_mode", _op.get_attr("round_mode"), "narrow_range",
              _op._get_attr_bool("narrow_range"), "axis",
              _op._get_attr_int("axis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeAndDequantizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QuantizeAndDequantizeV2 = tf_export("raw_ops.QuantizeAndDequantizeV2")(_ops.to_raw_op(quantize_and_dequantize_v2))


def quantize_and_dequantize_v2_eager_fallback(input: Annotated[Any, TV_QuantizeAndDequantizeV2_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV2_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV2_T], signed_input: bool, num_bits: int, range_given: bool, round_mode: str, narrow_range: bool, axis: int, name, ctx) -> Annotated[Any, TV_QuantizeAndDequantizeV2_T]:
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if range_given is None:
    range_given = False
  range_given = _execute.make_bool(range_given, "range_given")
  if round_mode is None:
    round_mode = "HALF_TO_EVEN"
  round_mode = _execute.make_str(round_mode, "round_mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_min, input_max], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_min, input_max) = _inputs_T
  _inputs_flat = [input, input_min, input_max]
  _attrs = ("signed_input", signed_input, "num_bits", num_bits, "range_given",
  range_given, "T", _attr_T, "round_mode", round_mode, "narrow_range",
  narrow_range, "axis", axis)
  _result = _execute.execute(b"QuantizeAndDequantizeV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeAndDequantizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_QuantizeAndDequantizeV3_T = TypeVar("TV_QuantizeAndDequantizeV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def quantize_and_dequantize_v3(input: Annotated[Any, TV_QuantizeAndDequantizeV3_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV3_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV3_T], num_bits: Annotated[Any, _atypes.Int32], signed_input:bool=True, range_given:bool=True, narrow_range:bool=False, axis:int=-1, name=None) -> Annotated[Any, TV_QuantizeAndDequantizeV3_T]:
  r"""Quantizes then dequantizes a tensor.

  This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
  tensor, so its value can change during training.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_min: A `Tensor`. Must have the same type as `input`.
    input_max: A `Tensor`. Must have the same type as `input`.
    num_bits: A `Tensor` of type `int32`.
    signed_input: An optional `bool`. Defaults to `True`.
    range_given: An optional `bool`. Defaults to `True`.
    narrow_range: An optional `bool`. Defaults to `False`.
    axis: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeAndDequantizeV3", name, input, input_min, input_max,
        num_bits, "signed_input", signed_input, "range_given", range_given,
        "narrow_range", narrow_range, "axis", axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_and_dequantize_v3_eager_fallback(
          input, input_min, input_max, num_bits, signed_input=signed_input,
          range_given=range_given, narrow_range=narrow_range, axis=axis,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if range_given is None:
    range_given = True
  range_given = _execute.make_bool(range_given, "range_given")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeAndDequantizeV3", input=input, input_min=input_min,
                                   input_max=input_max, num_bits=num_bits,
                                   signed_input=signed_input,
                                   range_given=range_given,
                                   narrow_range=narrow_range, axis=axis,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("signed_input", _op._get_attr_bool("signed_input"),
              "range_given", _op._get_attr_bool("range_given"), "T",
              _op._get_attr_type("T"), "narrow_range",
              _op._get_attr_bool("narrow_range"), "axis",
              _op._get_attr_int("axis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeAndDequantizeV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QuantizeAndDequantizeV3 = tf_export("raw_ops.QuantizeAndDequantizeV3")(_ops.to_raw_op(quantize_and_dequantize_v3))


def quantize_and_dequantize_v3_eager_fallback(input: Annotated[Any, TV_QuantizeAndDequantizeV3_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV3_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV3_T], num_bits: Annotated[Any, _atypes.Int32], signed_input: bool, range_given: bool, narrow_range: bool, axis: int, name, ctx) -> Annotated[Any, TV_QuantizeAndDequantizeV3_T]:
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if range_given is None:
    range_given = True
  range_given = _execute.make_bool(range_given, "range_given")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_min, input_max], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_min, input_max) = _inputs_T
  num_bits = _ops.convert_to_tensor(num_bits, _dtypes.int32)
  _inputs_flat = [input, input_min, input_max, num_bits]
  _attrs = ("signed_input", signed_input, "range_given", range_given, "T",
  _attr_T, "narrow_range", narrow_range, "axis", axis)
  _result = _execute.execute(b"QuantizeAndDequantizeV3", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeAndDequantizeV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_QuantizeAndDequantizeV4_T = TypeVar("TV_QuantizeAndDequantizeV4_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def quantize_and_dequantize_v4(input: Annotated[Any, TV_QuantizeAndDequantizeV4_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV4_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV4_T], signed_input:bool=True, num_bits:int=8, range_given:bool=False, round_mode:str="HALF_TO_EVEN", narrow_range:bool=False, axis:int=-1, name=None) -> Annotated[Any, TV_QuantizeAndDequantizeV4_T]:
  r"""Quantizes then dequantizes a tensor.

  This is almost identical to QuantizeAndDequantizeV2, except that it returns a
  gradient of 1 for inputs that are within the quantization range, or 0 otherwise.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      Tensor to quantize and then dequantize.
    input_min: A `Tensor`. Must have the same type as `input`.
      If `range_given == True`, this specifies the minimum input value that needs to
      be represented, otherwise it is determined from the min value of the `input`
      tensor.
    input_max: A `Tensor`. Must have the same type as `input`.
      If `range_given == True`, this specifies the maximum input value that needs to
      be represented, otherwise it is determined from the max value of the `input`
      tensor.
    signed_input: An optional `bool`. Defaults to `True`.
      Whether the quantization is signed or unsigned. (actually this parameter should
      have been called <b>`signed_output`</b>)
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization.
    range_given: An optional `bool`. Defaults to `False`.
      Whether the range is given or should be determined from the `input` tensor.
    round_mode: An optional `string` from: `"HALF_TO_EVEN", "HALF_UP"`. Defaults to `"HALF_TO_EVEN"`.
      The 'round_mode' attribute controls which rounding tie-breaking algorithm is
      used when rounding float values to their quantized equivalents. The following
      rounding modes are currently supported:

      *   HALF_TO_EVEN: this is the default round_mode.
      *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
          rounds up to -7.
    narrow_range: An optional `bool`. Defaults to `False`.
      If True, then the absolute value of the quantized minimum value is the same as
      the quantized maximum value, instead of 1 greater.
      i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    axis: An optional `int`. Defaults to `-1`.
      If specified, this axis is treated as a channel or slice axis, and a separate
      quantization range is used for each channel or slice along this axis.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeAndDequantizeV4", name, input, input_min, input_max,
        "signed_input", signed_input, "num_bits", num_bits, "range_given",
        range_given, "round_mode", round_mode, "narrow_range", narrow_range,
        "axis", axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_and_dequantize_v4_eager_fallback(
          input, input_min, input_max, signed_input=signed_input,
          num_bits=num_bits, range_given=range_given, round_mode=round_mode,
          narrow_range=narrow_range, axis=axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if range_given is None:
    range_given = False
  range_given = _execute.make_bool(range_given, "range_given")
  if round_mode is None:
    round_mode = "HALF_TO_EVEN"
  round_mode = _execute.make_str(round_mode, "round_mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeAndDequantizeV4", input=input, input_min=input_min,
                                   input_max=input_max,
                                   signed_input=signed_input,
                                   num_bits=num_bits, range_given=range_given,
                                   round_mode=round_mode,
                                   narrow_range=narrow_range, axis=axis,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("signed_input", _op._get_attr_bool("signed_input"), "num_bits",
              _op._get_attr_int("num_bits"), "range_given",
              _op._get_attr_bool("range_given"), "T", _op._get_attr_type("T"),
              "round_mode", _op.get_attr("round_mode"), "narrow_range",
              _op._get_attr_bool("narrow_range"), "axis",
              _op._get_attr_int("axis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeAndDequantizeV4", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QuantizeAndDequantizeV4 = tf_export("raw_ops.QuantizeAndDequantizeV4")(_ops.to_raw_op(quantize_and_dequantize_v4))


def quantize_and_dequantize_v4_eager_fallback(input: Annotated[Any, TV_QuantizeAndDequantizeV4_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV4_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV4_T], signed_input: bool, num_bits: int, range_given: bool, round_mode: str, narrow_range: bool, axis: int, name, ctx) -> Annotated[Any, TV_QuantizeAndDequantizeV4_T]:
  if signed_input is None:
    signed_input = True
  signed_input = _execute.make_bool(signed_input, "signed_input")
  if num_bits is None:
    num_bits = 8
  num_bits = _execute.make_int(num_bits, "num_bits")
  if range_given is None:
    range_given = False
  range_given = _execute.make_bool(range_given, "range_given")
  if round_mode is None:
    round_mode = "HALF_TO_EVEN"
  round_mode = _execute.make_str(round_mode, "round_mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_min, input_max], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_min, input_max) = _inputs_T
  _inputs_flat = [input, input_min, input_max]
  _attrs = ("signed_input", signed_input, "num_bits", num_bits, "range_given",
  range_given, "T", _attr_T, "round_mode", round_mode, "narrow_range",
  narrow_range, "axis", axis)
  _result = _execute.execute(b"QuantizeAndDequantizeV4", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeAndDequantizeV4", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_QuantizeAndDequantizeV4GradOutput = collections.namedtuple(
    "QuantizeAndDequantizeV4Grad",
    ["input_backprop", "input_min_backprop", "input_max_backprop"])


TV_QuantizeAndDequantizeV4Grad_T = TypeVar("TV_QuantizeAndDequantizeV4Grad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def quantize_and_dequantize_v4_grad(gradients: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], input: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], axis:int=-1, name=None):
  r"""Returns the gradient of `QuantizeAndDequantizeV4`.

  Returns a gradient of 1 for inputs that are within the quantization range,
  or 0 otherwise.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input: A `Tensor`. Must have the same type as `gradients`.
    input_min: A `Tensor`. Must have the same type as `gradients`.
    input_max: A `Tensor`. Must have the same type as `gradients`.
    axis: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_min_backprop, input_max_backprop).

    input_backprop: A `Tensor`. Has the same type as `gradients`.
    input_min_backprop: A `Tensor`. Has the same type as `gradients`.
    input_max_backprop: A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeAndDequantizeV4Grad", name, gradients, input,
        input_min, input_max, "axis", axis)
      _result = _QuantizeAndDequantizeV4GradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_and_dequantize_v4_grad_eager_fallback(
          gradients, input, input_min, input_max, axis=axis, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeAndDequantizeV4Grad", gradients=gradients, input=input,
                                       input_min=input_min,
                                       input_max=input_max, axis=axis,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "axis", _op._get_attr_int("axis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeAndDequantizeV4Grad", _inputs_flat, _attrs, _result)
  _result = _QuantizeAndDequantizeV4GradOutput._make(_result)
  return _result

QuantizeAndDequantizeV4Grad = tf_export("raw_ops.QuantizeAndDequantizeV4Grad")(_ops.to_raw_op(quantize_and_dequantize_v4_grad))


def quantize_and_dequantize_v4_grad_eager_fallback(gradients: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], input: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], input_min: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], input_max: Annotated[Any, TV_QuantizeAndDequantizeV4Grad_T], axis: int, name, ctx):
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, input, input_min, input_max], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (gradients, input, input_min, input_max) = _inputs_T
  _inputs_flat = [gradients, input, input_min, input_max]
  _attrs = ("T", _attr_T, "axis", axis)
  _result = _execute.execute(b"QuantizeAndDequantizeV4Grad", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeAndDequantizeV4Grad", _inputs_flat, _attrs, _result)
  _result = _QuantizeAndDequantizeV4GradOutput._make(_result)
  return _result

_QuantizeV2Output = collections.namedtuple(
    "QuantizeV2",
    ["output", "output_min", "output_max"])


TV_QuantizeV2_T = TypeVar("TV_QuantizeV2_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantize_v2(input: Annotated[Any, _atypes.Float32], min_range: Annotated[Any, _atypes.Float32], max_range: Annotated[Any, _atypes.Float32], T: TV_QuantizeV2_T, mode:str="MIN_COMBINED", round_mode:str="HALF_AWAY_FROM_ZERO", narrow_range:bool=False, axis:int=-1, ensure_minimum_range:float=0.01, name=None):
  r"""Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.  The
  'round_mode' attribute controls which rounding tie-breaking algorithm is used
  when rounding float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
  if T == qint8: out[i] -= (range(T) + 1) / 2.0
  ```

  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  Assume the input is type float and has a possible range of [0.0, 6.0] and the
  output type is quint8 ([0, 255]). The min_range and max_range values should be
  specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
  value of the input by 255/6 and cast to quint8.

  If the output type was qint8 ([-128, 127]), the operation will additionally
  subtract each value by 128 prior to casting, so that the range of values aligns
  with the range of qint8.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```
  num_discrete_values = 1 << (# of bits in T)
  range_adjust = num_discrete_values / (num_discrete_values - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = num_discrete_values / range
  quantized = round(input * range_scale) - round(range_min * range_scale) +
    numeric_limits<T>::min()
  quantized = max(quantized, numeric_limits<T>::min())
  quantized = min(quantized, numeric_limits<T>::max())
  ```

  The biggest difference between this and MIN_COMBINED is that the minimum range
  is rounded first, before it's subtracted from the rounded value. With
  MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
  and dequantizing will introduce a larger and larger error.

  *SCALED mode Example*

  `SCALED` mode matches the quantization approach used in
  `QuantizeAndDequantize{V2|V3}`.

  If the mode is `SCALED`, the quantization is performed by multiplying each
  input value by a scaling_factor.
  The scaling_factor is determined from `min_range` and `max_range` to be as large
  as possible such that the range from `min_range` to `max_range` is representable
  within values of type T.

  ```c++

    const int min_T = std::numeric_limits<T>::min();
    const int max_T = std::numeric_limits<T>::max();
    const float max_float = std::numeric_limits<float>::max();

    const float scale_factor_from_min_side =
        (min_T * min_range > 0) ? min_T / min_range : max_float;
    const float scale_factor_from_max_side =
        (max_T * max_range > 0) ? max_T / max_range : max_float;

    const float scale_factor = std::min(scale_factor_from_min_side,
                                        scale_factor_from_max_side);
  ```

  We next use the scale_factor to adjust min_range and max_range as follows:

  ```c++
        min_range = min_T / scale_factor;
        max_range = max_T / scale_factor;
  ```


  e.g. if T = qint8, and initially min_range = -10, and max_range = 9, we would
  compare -128/-10.0 = 12.8 to 127/9.0 = 14.11, and set scaling_factor = 12.8
  In this case, min_range would remain -10, but max_range would be adjusted to
  127 / 12.8 = 9.921875

  So we will quantize input values in the range (-10, 9.921875) to (-128, 127).

  The input tensor can now be quantized by clipping values to the range
  `min_range` to `max_range`, then multiplying by scale_factor as follows:

  ```c++
  result = round(min(max_range, max(min_range, input)) * scale_factor)
  ```

  The adjusted `min_range` and `max_range` are returned as outputs 2 and 3 of
  this operation. These outputs should be used as the range for any further
  calculations.


  *narrow_range (bool) attribute*

  If true, we do not use the minimum quantized value.
  i.e. for int8 the quantized output, it would be restricted to the range
  -127..127 instead of the full -128..127 range.
  This is provided for compatibility with certain inference backends.
  (Only applies to SCALED mode)


  *axis (int) attribute*

  An optional `axis` attribute can specify a dimension index of the input tensor,
  such that quantization ranges will be calculated and applied separately for each
  slice of the tensor along that dimension. This is useful for per-channel
  quantization.

  If axis is specified, min_range and max_range

  if `axis`=None, per-tensor quantization is performed as normal.


  *ensure_minimum_range (float) attribute*

  Ensures the minimum quantization range is at least this value.
  The legacy default value for this is 0.01, but it is strongly suggested to
  set it to 0 for new uses.

  Args:
    input: A `Tensor` of type `float32`.
    min_range: A `Tensor` of type `float32`.
      The minimum value of the quantization range. This value may be adjusted by the
      op depending on other parameters. The adjusted value is written to `output_min`.
      If the `axis` attribute is specified, this must be a 1-D tensor whose size
      matches the `axis` dimension of the input and output tensors.
    max_range: A `Tensor` of type `float32`.
      The maximum value of the quantization range. This value may be adjusted by the
      op depending on other parameters. The adjusted value is written to `output_max`.
      If the `axis` attribute is specified, this must be a 1-D tensor whose size
      matches the `axis` dimension of the input and output tensors.
    T: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    mode: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST", "SCALED"`. Defaults to `"MIN_COMBINED"`.
    round_mode: An optional `string` from: `"HALF_AWAY_FROM_ZERO", "HALF_TO_EVEN"`. Defaults to `"HALF_AWAY_FROM_ZERO"`.
    narrow_range: An optional `bool`. Defaults to `False`.
    axis: An optional `int`. Defaults to `-1`.
    ensure_minimum_range: An optional `float`. Defaults to `0.01`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `T`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeV2", name, input, min_range, max_range, "T", T, "mode",
        mode, "round_mode", round_mode, "narrow_range", narrow_range, "axis",
        axis, "ensure_minimum_range", ensure_minimum_range)
      _result = _QuantizeV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_v2_eager_fallback(
          input, min_range, max_range, T=T, mode=mode, round_mode=round_mode,
          narrow_range=narrow_range, axis=axis,
          ensure_minimum_range=ensure_minimum_range, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  if mode is None:
    mode = "MIN_COMBINED"
  mode = _execute.make_str(mode, "mode")
  if round_mode is None:
    round_mode = "HALF_AWAY_FROM_ZERO"
  round_mode = _execute.make_str(round_mode, "round_mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  if ensure_minimum_range is None:
    ensure_minimum_range = 0.01
  ensure_minimum_range = _execute.make_float(ensure_minimum_range, "ensure_minimum_range")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeV2", input=input, min_range=min_range, max_range=max_range,
                      T=T, mode=mode, round_mode=round_mode,
                      narrow_range=narrow_range, axis=axis,
                      ensure_minimum_range=ensure_minimum_range, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "mode", _op.get_attr("mode"),
              "round_mode", _op.get_attr("round_mode"), "narrow_range",
              _op._get_attr_bool("narrow_range"), "axis",
              _op._get_attr_int("axis"), "ensure_minimum_range",
              _op.get_attr("ensure_minimum_range"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeV2", _inputs_flat, _attrs, _result)
  _result = _QuantizeV2Output._make(_result)
  return _result

QuantizeV2 = tf_export("raw_ops.QuantizeV2")(_ops.to_raw_op(quantize_v2))


def quantize_v2_eager_fallback(input: Annotated[Any, _atypes.Float32], min_range: Annotated[Any, _atypes.Float32], max_range: Annotated[Any, _atypes.Float32], T: TV_QuantizeV2_T, mode: str, round_mode: str, narrow_range: bool, axis: int, ensure_minimum_range: float, name, ctx):
  T = _execute.make_type(T, "T")
  if mode is None:
    mode = "MIN_COMBINED"
  mode = _execute.make_str(mode, "mode")
  if round_mode is None:
    round_mode = "HALF_AWAY_FROM_ZERO"
  round_mode = _execute.make_str(round_mode, "round_mode")
  if narrow_range is None:
    narrow_range = False
  narrow_range = _execute.make_bool(narrow_range, "narrow_range")
  if axis is None:
    axis = -1
  axis = _execute.make_int(axis, "axis")
  if ensure_minimum_range is None:
    ensure_minimum_range = 0.01
  ensure_minimum_range = _execute.make_float(ensure_minimum_range, "ensure_minimum_range")
  input = _ops.convert_to_tensor(input, _dtypes.float32)
  min_range = _ops.convert_to_tensor(min_range, _dtypes.float32)
  max_range = _ops.convert_to_tensor(max_range, _dtypes.float32)
  _inputs_flat = [input, min_range, max_range]
  _attrs = ("T", T, "mode", mode, "round_mode", round_mode, "narrow_range",
  narrow_range, "axis", axis, "ensure_minimum_range", ensure_minimum_range)
  _result = _execute.execute(b"QuantizeV2", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeV2", _inputs_flat, _attrs, _result)
  _result = _QuantizeV2Output._make(_result)
  return _result

_QuantizedConcatOutput = collections.namedtuple(
    "QuantizedConcat",
    ["output", "output_min", "output_max"])


TV_QuantizedConcat_T = TypeVar("TV_QuantizedConcat_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.quantized_concat', v1=['quantization.quantized_concat', 'quantized_concat'])
@deprecated_endpoints('quantized_concat')
def quantized_concat(concat_dim: Annotated[Any, _atypes.Int32], values: Annotated[List[Any], TV_QuantizedConcat_T], input_mins: Annotated[List[Any], _atypes.Float32], input_maxes: Annotated[List[Any], _atypes.Float32], name=None):
  r"""Concatenates quantized tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects with the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    input_mins: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The minimum scalar values for each of the input tensors.
    input_maxes: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The maximum scalar values for each of the input tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor`. Has the same type as `values`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConcat", name, concat_dim, values, input_mins,
        input_maxes)
      _result = _QuantizedConcatOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_quantized_concat(
          (concat_dim, values, input_mins, input_maxes, name,), None)
      if _result is not NotImplemented:
        return _result
      return quantized_concat_eager_fallback(
          concat_dim, values, input_mins, input_maxes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            quantized_concat, (), dict(concat_dim=concat_dim, values=values,
                                       input_mins=input_mins,
                                       input_maxes=input_maxes, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_quantized_concat(
        (concat_dim, values, input_mins, input_maxes, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'quantized_concat' Op, not %r." % values)
  _attr_N = len(values)
  if not isinstance(input_mins, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_mins' argument to "
        "'quantized_concat' Op, not %r." % input_mins)
  if len(input_mins) != _attr_N:
    raise ValueError(
        "List argument 'input_mins' to 'quantized_concat' Op with length %d "
        "must match length %d of argument 'values'." %
        (len(input_mins), _attr_N))
  if not isinstance(input_maxes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_maxes' argument to "
        "'quantized_concat' Op, not %r." % input_maxes)
  if len(input_maxes) != _attr_N:
    raise ValueError(
        "List argument 'input_maxes' to 'quantized_concat' Op with length %d "
        "must match length %d of argument 'values'." %
        (len(input_maxes), _attr_N))
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConcat", concat_dim=concat_dim, values=values,
                           input_mins=input_mins, input_maxes=input_maxes,
                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          quantized_concat, (), dict(concat_dim=concat_dim, values=values,
                                     input_mins=input_mins,
                                     input_maxes=input_maxes, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConcat", _inputs_flat, _attrs, _result)
  _result = _QuantizedConcatOutput._make(_result)
  return _result

QuantizedConcat = tf_export("raw_ops.QuantizedConcat")(_ops.to_raw_op(quantized_concat))
_dispatcher_for_quantized_concat = quantized_concat._tf_type_based_dispatcher.Dispatch


def quantized_concat_eager_fallback(concat_dim: Annotated[Any, _atypes.Int32], values: Annotated[List[Any], TV_QuantizedConcat_T], input_mins: Annotated[List[Any], _atypes.Float32], input_maxes: Annotated[List[Any], _atypes.Float32], name, ctx):
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'quantized_concat' Op, not %r." % values)
  _attr_N = len(values)
  if not isinstance(input_mins, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_mins' argument to "
        "'quantized_concat' Op, not %r." % input_mins)
  if len(input_mins) != _attr_N:
    raise ValueError(
        "List argument 'input_mins' to 'quantized_concat' Op with length %d "
        "must match length %d of argument 'values'." %
        (len(input_mins), _attr_N))
  if not isinstance(input_maxes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_maxes' argument to "
        "'quantized_concat' Op, not %r." % input_maxes)
  if len(input_maxes) != _attr_N:
    raise ValueError(
        "List argument 'input_maxes' to 'quantized_concat' Op with length %d "
        "must match length %d of argument 'values'." %
        (len(input_maxes), _attr_N))
  _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
  concat_dim = _ops.convert_to_tensor(concat_dim, _dtypes.int32)
  input_mins = _ops.convert_n_to_tensor(input_mins, _dtypes.float32)
  input_maxes = _ops.convert_n_to_tensor(input_maxes, _dtypes.float32)
  _inputs_flat = [concat_dim] + list(values) + list(input_mins) + list(input_maxes)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"QuantizedConcat", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConcat", _inputs_flat, _attrs, _result)
  _result = _QuantizedConcatOutput._make(_result)
  return _result

_QuantizedInstanceNormOutput = collections.namedtuple(
    "QuantizedInstanceNorm",
    ["y", "y_min", "y_max"])


TV_QuantizedInstanceNorm_T = TypeVar("TV_QuantizedInstanceNorm_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_instance_norm(x: Annotated[Any, TV_QuantizedInstanceNorm_T], x_min: Annotated[Any, _atypes.Float32], x_max: Annotated[Any, _atypes.Float32], output_range_given:bool=False, given_y_min:float=0, given_y_max:float=0, variance_epsilon:float=1e-05, min_separation:float=0.001, name=None):
  r"""Quantized Instance normalization.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 4D input Tensor.
    x_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    x_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    output_range_given: An optional `bool`. Defaults to `False`.
      If True, `given_y_min` and `given_y_min`
      and `given_y_max` are used as the output range. Otherwise,
      the implementation computes the output range.
    given_y_min: An optional `float`. Defaults to `0`.
      Output in `y_min` if `output_range_given` is True.
    given_y_max: An optional `float`. Defaults to `0`.
      Output in `y_max` if `output_range_given` is True.
    variance_epsilon: An optional `float`. Defaults to `1e-05`.
      A small float number to avoid dividing by 0.
    min_separation: An optional `float`. Defaults to `0.001`.
      Minimum value of `y_max - y_min`
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, y_min, y_max).

    y: A `Tensor`. Has the same type as `x`.
    y_min: A `Tensor` of type `float32`.
    y_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedInstanceNorm", name, x, x_min, x_max,
        "output_range_given", output_range_given, "given_y_min", given_y_min,
        "given_y_max", given_y_max, "variance_epsilon", variance_epsilon,
        "min_separation", min_separation)
      _result = _QuantizedInstanceNormOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_instance_norm_eager_fallback(
          x, x_min, x_max, output_range_given=output_range_given,
          given_y_min=given_y_min, given_y_max=given_y_max,
          variance_epsilon=variance_epsilon, min_separation=min_separation,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_range_given is None:
    output_range_given = False
  output_range_given = _execute.make_bool(output_range_given, "output_range_given")
  if given_y_min is None:
    given_y_min = 0
  given_y_min = _execute.make_float(given_y_min, "given_y_min")
  if given_y_max is None:
    given_y_max = 0
  given_y_max = _execute.make_float(given_y_max, "given_y_max")
  if variance_epsilon is None:
    variance_epsilon = 1e-05
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  if min_separation is None:
    min_separation = 0.001
  min_separation = _execute.make_float(min_separation, "min_separation")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedInstanceNorm", x=x, x_min=x_min, x_max=x_max,
                                 output_range_given=output_range_given,
                                 given_y_min=given_y_min,
                                 given_y_max=given_y_max,
                                 variance_epsilon=variance_epsilon,
                                 min_separation=min_separation, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "output_range_given",
              _op._get_attr_bool("output_range_given"), "given_y_min",
              _op.get_attr("given_y_min"), "given_y_max",
              _op.get_attr("given_y_max"), "variance_epsilon",
              _op.get_attr("variance_epsilon"), "min_separation",
              _op.get_attr("min_separation"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedInstanceNorm", _inputs_flat, _attrs, _result)
  _result = _QuantizedInstanceNormOutput._make(_result)
  return _result

QuantizedInstanceNorm = tf_export("raw_ops.QuantizedInstanceNorm")(_ops.to_raw_op(quantized_instance_norm))


def quantized_instance_norm_eager_fallback(x: Annotated[Any, TV_QuantizedInstanceNorm_T], x_min: Annotated[Any, _atypes.Float32], x_max: Annotated[Any, _atypes.Float32], output_range_given: bool, given_y_min: float, given_y_max: float, variance_epsilon: float, min_separation: float, name, ctx):
  if output_range_given is None:
    output_range_given = False
  output_range_given = _execute.make_bool(output_range_given, "output_range_given")
  if given_y_min is None:
    given_y_min = 0
  given_y_min = _execute.make_float(given_y_min, "given_y_min")
  if given_y_max is None:
    given_y_max = 0
  given_y_max = _execute.make_float(given_y_max, "given_y_max")
  if variance_epsilon is None:
    variance_epsilon = 1e-05
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  if min_separation is None:
    min_separation = 0.001
  min_separation = _execute.make_float(min_separation, "min_separation")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  x_min = _ops.convert_to_tensor(x_min, _dtypes.float32)
  x_max = _ops.convert_to_tensor(x_max, _dtypes.float32)
  _inputs_flat = [x, x_min, x_max]
  _attrs = ("T", _attr_T, "output_range_given", output_range_given,
  "given_y_min", given_y_min, "given_y_max", given_y_max, "variance_epsilon",
  variance_epsilon, "min_separation", min_separation)
  _result = _execute.execute(b"QuantizedInstanceNorm", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedInstanceNorm", _inputs_flat, _attrs, _result)
  _result = _QuantizedInstanceNormOutput._make(_result)
  return _result

_QuantizedReshapeOutput = collections.namedtuple(
    "QuantizedReshape",
    ["output", "output_min", "output_max"])


TV_QuantizedReshape_T = TypeVar("TV_QuantizedReshape_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_QuantizedReshape_Tshape = TypeVar("TV_QuantizedReshape_Tshape", _atypes.Int32, _atypes.Int64)

def quantized_reshape(tensor: Annotated[Any, TV_QuantizedReshape_T], shape: Annotated[Any, TV_QuantizedReshape_Tshape], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], name=None):
  r"""Reshapes a quantized tensor as per the Reshape op.

  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Defines the shape of the output tensor.
    input_min: A `Tensor` of type `float32`. The minimum value of the input.
    input_max: A `Tensor` of type `float32`. The maximum value of the input.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor`. Has the same type as `tensor`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedReshape", name, tensor, shape, input_min, input_max)
      _result = _QuantizedReshapeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_reshape_eager_fallback(
          tensor, shape, input_min, input_max, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedReshape", tensor=tensor, shape=shape, input_min=input_min,
                            input_max=input_max, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedReshape", _inputs_flat, _attrs, _result)
  _result = _QuantizedReshapeOutput._make(_result)
  return _result

QuantizedReshape = tf_export("raw_ops.QuantizedReshape")(_ops.to_raw_op(quantized_reshape))


def quantized_reshape_eager_fallback(tensor: Annotated[Any, TV_QuantizedReshape_T], shape: Annotated[Any, TV_QuantizedReshape_Tshape], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], name, ctx):
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
  input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
  _inputs_flat = [tensor, shape, input_min, input_max]
  _attrs = ("T", _attr_T, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"QuantizedReshape", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedReshape", _inputs_flat, _attrs, _result)
  _result = _QuantizedReshapeOutput._make(_result)
  return _result


TV_Rank_T = TypeVar("TV_Rank_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def rank(input: Annotated[Any, TV_Rank_T], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Returns the rank of a tensor.

  This operation returns an integer representing the rank of `input`.

  For example:

  ```
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  # shape of tensor 't' is [2, 2, 3]
  rank(t) ==> 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
  of a tensor is the number of indices required to uniquely select each element
  of the tensor. Rank is also known as "order", "degree", or "ndims."

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Rank", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rank_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Rank", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Rank", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Rank = tf_export("raw_ops.Rank")(_ops.to_raw_op(rank))


def rank_eager_fallback(input: Annotated[Any, TV_Rank_T], name, ctx) -> Annotated[Any, _atypes.Int32]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Rank", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Rank", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RefIdentity_T = TypeVar("TV_RefIdentity_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def ref_identity(input: Annotated[Any, TV_RefIdentity_T], name=None) -> Annotated[Any, TV_RefIdentity_T]:
  r"""Return the same ref tensor as the input ref tensor.

  Args:
    input: A mutable `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_identity op does not support eager execution. Arg 'output' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefIdentity", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RefIdentity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RefIdentity = tf_export("raw_ops.RefIdentity")(_ops.to_raw_op(ref_identity))


def ref_identity_eager_fallback(input: Annotated[Any, TV_RefIdentity_T], name, ctx) -> Annotated[Any, TV_RefIdentity_T]:
  raise RuntimeError("ref_identity op does not support eager execution. Arg 'output' is a ref.")

TV_Reshape_T = TypeVar("TV_Reshape_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Reshape_Tshape = TypeVar("TV_Reshape_Tshape", _atypes.Int32, _atypes.Int64)

def reshape(tensor: Annotated[Any, TV_Reshape_T], shape: Annotated[Any, TV_Reshape_Tshape], name=None) -> Annotated[Any, TV_Reshape_T]:
  r"""Reshapes a tensor.

  Given `tensor`, this operation returns a tensor that has the same values
  as `tensor` with shape `shape`.

  If one component of 1-D tensor `shape` is the special value -1, the size of that
  dimension is computed so that the total size remains constant.  In particular, a
  `shape` of `[-1]` flattens into 1-D.  At most one component of `shape` may be
  unknown.

  The `shape` must be 1-D and the operation returns a tensor with shape
  `shape` filled with the values of `tensor`. In this case, the number of elements
  implied by `shape` must be the same as the number of elements in `tensor`.

  It is an error if `shape` is not 1-D.

  For example:

  ```
  # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # tensor 't' has shape [9]
  reshape(t, [3, 3]) ==> [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]

  # tensor 't' is [[[1, 1], [2, 2]],
  #                [[3, 3], [4, 4]]]
  # tensor 't' has shape [2, 2, 2]
  reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                          [3, 3, 4, 4]]

  # tensor 't' is [[[1, 1, 1],
  #                 [2, 2, 2]],
  #                [[3, 3, 3],
  #                 [4, 4, 4]],
  #                [[5, 5, 5],
  #                 [6, 6, 6]]]
  # tensor 't' has shape [3, 2, 3]
  # pass '[-1]' to flatten 't'
  reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

  # -1 can also be used to infer the shape

  # -1 is inferred to be 9:
  reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 2:
  reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 3:
  reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]],
                               [[4, 4, 4],
                                [5, 5, 5],
                                [6, 6, 6]]]

  # tensor 't' is [7]
  # shape `[]` reshapes to a scalar
  reshape(t, []) ==> 7
  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Defines the shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Reshape", name, tensor, shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reshape_eager_fallback(
          tensor, shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Reshape", tensor=tensor, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Reshape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Reshape = tf_export("raw_ops.Reshape")(_ops.to_raw_op(reshape))


def reshape_eager_fallback(tensor: Annotated[Any, TV_Reshape_T], shape: Annotated[Any, TV_Reshape_Tshape], name, ctx) -> Annotated[Any, TV_Reshape_T]:
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [tensor, shape]
  _attrs = ("T", _attr_T, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"Reshape", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Reshape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResourceStridedSliceAssign_T = TypeVar("TV_ResourceStridedSliceAssign_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ResourceStridedSliceAssign_Index = TypeVar("TV_ResourceStridedSliceAssign_Index", _atypes.Int32, _atypes.Int64)

def resource_strided_slice_assign(ref: Annotated[Any, _atypes.Resource], begin: Annotated[Any, TV_ResourceStridedSliceAssign_Index], end: Annotated[Any, TV_ResourceStridedSliceAssign_Index], strides: Annotated[Any, TV_ResourceStridedSliceAssign_Index], value: Annotated[Any, TV_ResourceStridedSliceAssign_T], begin_mask:int=0, end_mask:int=0, ellipsis_mask:int=0, new_axis_mask:int=0, shrink_axis_mask:int=0, name=None):
  r"""Assign `value` to the sliced l-value reference of `ref`.

  The values of `value` are assigned to the positions in the variable
  `ref` that are selected by the slice parameters. The slice parameters
  `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.

  NOTE this op currently does not support broadcasting and so `value`'s
  shape must be exactly the shape produced by the slice of `ref`.

  Args:
    ref: A `Tensor` of type `resource`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    end: A `Tensor`. Must have the same type as `begin`.
    strides: A `Tensor`. Must have the same type as `begin`.
    value: A `Tensor`.
    begin_mask: An optional `int`. Defaults to `0`.
    end_mask: An optional `int`. Defaults to `0`.
    ellipsis_mask: An optional `int`. Defaults to `0`.
    new_axis_mask: An optional `int`. Defaults to `0`.
    shrink_axis_mask: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceStridedSliceAssign", name, ref, begin, end, strides,
        value, "begin_mask", begin_mask, "end_mask", end_mask,
        "ellipsis_mask", ellipsis_mask, "new_axis_mask", new_axis_mask,
        "shrink_axis_mask", shrink_axis_mask)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_strided_slice_assign_eager_fallback(
          ref, begin, end, strides, value, begin_mask=begin_mask,
          end_mask=end_mask, ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceStridedSliceAssign", ref=ref, begin=begin, end=end,
                                      strides=strides, value=value,
                                      begin_mask=begin_mask,
                                      end_mask=end_mask,
                                      ellipsis_mask=ellipsis_mask,
                                      new_axis_mask=new_axis_mask,
                                      shrink_axis_mask=shrink_axis_mask,
                                      name=name)
  return _op
ResourceStridedSliceAssign = tf_export("raw_ops.ResourceStridedSliceAssign")(_ops.to_raw_op(resource_strided_slice_assign))


def resource_strided_slice_assign_eager_fallback(ref: Annotated[Any, _atypes.Resource], begin: Annotated[Any, TV_ResourceStridedSliceAssign_Index], end: Annotated[Any, TV_ResourceStridedSliceAssign_Index], strides: Annotated[Any, TV_ResourceStridedSliceAssign_Index], value: Annotated[Any, TV_ResourceStridedSliceAssign_T], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx):
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _attr_Index, _inputs_Index = _execute.args_to_matching_eager([begin, end, strides], ctx, [_dtypes.int32, _dtypes.int64, ])
  (begin, end, strides) = _inputs_Index
  ref = _ops.convert_to_tensor(ref, _dtypes.resource)
  _inputs_flat = [ref, begin, end, strides, value]
  _attrs = ("T", _attr_T, "Index", _attr_Index, "begin_mask", begin_mask,
  "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask",
  new_axis_mask, "shrink_axis_mask", shrink_axis_mask)
  _result = _execute.execute(b"ResourceStridedSliceAssign", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_Reverse_T = TypeVar("TV_Reverse_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def reverse(tensor: Annotated[Any, TV_Reverse_T], dims: Annotated[Any, _atypes.Bool], name=None) -> Annotated[Any, TV_Reverse_T]:
  r"""Reverses specific dimensions of a tensor.

  Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
  of `tensor`, this operation reverses each dimension i of `tensor` where
  `dims[i]` is `True`.

  `tensor` can have up to 8 dimensions. The number of dimensions
  of `tensor` must equal the number of elements in `dims`. In other words:

  `rank(tensor) = size(dims)`

  For example:

  ```
  # tensor 't' is [[[[ 0,  1,  2,  3],
  #                  [ 4,  5,  6,  7],
  #                  [ 8,  9, 10, 11]],
  #                 [[12, 13, 14, 15],
  #                  [16, 17, 18, 19],
  #                  [20, 21, 22, 23]]]]
  # tensor 't' shape is [1, 2, 3, 4]

  # 'dims' is [False, False, False, True]
  reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                          [ 7,  6,  5,  4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

  # 'dims' is [False, True, False, False]
  reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]]]]

  # 'dims' is [False, False, True, False]
  reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `bool`, `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`, `string`.
      Up to 8-D.
    dims: A `Tensor` of type `bool`. 1-D. The dimensions to reverse.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Reverse", name, tensor, dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reverse_eager_fallback(
          tensor, dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Reverse", tensor=tensor, dims=dims, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Reverse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Reverse = tf_export("raw_ops.Reverse")(_ops.to_raw_op(reverse))


def reverse_eager_fallback(tensor: Annotated[Any, TV_Reverse_T], dims: Annotated[Any, _atypes.Bool], name, ctx) -> Annotated[Any, TV_Reverse_T]:
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.uint32, _dtypes.int32, _dtypes.uint64, _dtypes.int64, _dtypes.bool, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, _dtypes.string, ])
  dims = _ops.convert_to_tensor(dims, _dtypes.bool)
  _inputs_flat = [tensor, dims]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Reverse", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Reverse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ReverseSequence_T = TypeVar("TV_ReverseSequence_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ReverseSequence_Tlen = TypeVar("TV_ReverseSequence_Tlen", _atypes.Int32, _atypes.Int64)

def reverse_sequence(input: Annotated[Any, TV_ReverseSequence_T], seq_lengths: Annotated[Any, TV_ReverseSequence_Tlen], seq_dim: int, batch_dim:int=0, name=None) -> Annotated[Any, TV_ReverseSequence_T]:
  r"""Reverses variable length slices.

  This op first slices `input` along the dimension `batch_dim`, and for each
  slice `i`, reverses the first `seq_lengths[i]` elements along
  the dimension `seq_dim`.

  The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
  and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

  The output slice `i` along dimension `batch_dim` is then given by input
  slice `i`, with the first `seq_lengths[i]` slices along dimension
  `seq_dim` reversed.

  For example:

  ```
  # Given this:
  batch_dim = 0
  seq_dim = 1
  input.dims = (4, 8, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
  output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
  output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
  output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

  # while entries past seq_lens are copied through:
  output[0, 7:, :, ...] = input[0, 7:, :, ...]
  output[1, 2:, :, ...] = input[1, 2:, :, ...]
  output[2, 3:, :, ...] = input[2, 3:, :, ...]
  output[3, 2:, :, ...] = input[3, 2:, :, ...]
  ```

  In contrast, if:

  ```
  # Given this:
  batch_dim = 2
  seq_dim = 0
  input.dims = (8, ?, 4, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
  output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
  output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
  output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

  # while entries past seq_lens are copied through:
  output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
  output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
  output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
  output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
  ```

  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with length `input.dims(batch_dim)` and
      `max(seq_lengths) <= input.dims(seq_dim)`
    seq_dim: An `int`. The dimension which is partially reversed.
    batch_dim: An optional `int`. Defaults to `0`.
      The dimension along which reversal is performed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReverseSequence", name, input, seq_lengths, "seq_dim", seq_dim,
        "batch_dim", batch_dim)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reverse_sequence_eager_fallback(
          input, seq_lengths, seq_dim=seq_dim, batch_dim=batch_dim, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  seq_dim = _execute.make_int(seq_dim, "seq_dim")
  if batch_dim is None:
    batch_dim = 0
  batch_dim = _execute.make_int(batch_dim, "batch_dim")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReverseSequence", input=input, seq_lengths=seq_lengths,
                           seq_dim=seq_dim, batch_dim=batch_dim, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("seq_dim", _op._get_attr_int("seq_dim"), "batch_dim",
              _op._get_attr_int("batch_dim"), "T", _op._get_attr_type("T"),
              "Tlen", _op._get_attr_type("Tlen"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReverseSequence", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReverseSequence = tf_export("raw_ops.ReverseSequence")(_ops.to_raw_op(reverse_sequence))


def reverse_sequence_eager_fallback(input: Annotated[Any, TV_ReverseSequence_T], seq_lengths: Annotated[Any, TV_ReverseSequence_Tlen], seq_dim: int, batch_dim: int, name, ctx) -> Annotated[Any, TV_ReverseSequence_T]:
  seq_dim = _execute.make_int(seq_dim, "seq_dim")
  if batch_dim is None:
    batch_dim = 0
  batch_dim = _execute.make_int(batch_dim, "batch_dim")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tlen, (seq_lengths,) = _execute.args_to_matching_eager([seq_lengths], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [input, seq_lengths]
  _attrs = ("seq_dim", seq_dim, "batch_dim", batch_dim, "T", _attr_T, "Tlen",
  _attr_Tlen)
  _result = _execute.execute(b"ReverseSequence", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReverseSequence", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ReverseV2_Tidx = TypeVar("TV_ReverseV2_Tidx", _atypes.Int32, _atypes.Int64)
TV_ReverseV2_T = TypeVar("TV_ReverseV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('reverse', v1=['reverse', 'manip.reverse', 'reverse_v2'])
@deprecated_endpoints('manip.reverse', 'reverse_v2')
def reverse_v2(tensor: Annotated[Any, TV_ReverseV2_T], axis: Annotated[Any, TV_ReverseV2_Tidx], name=None) -> Annotated[Any, TV_ReverseV2_T]:
  r"""Reverses specific dimensions of a tensor.

  Given a `tensor`, and a `int32` tensor `axis` representing the set of
  dimensions of `tensor` to reverse. This operation reverses each dimension
  `i` for which there exists `j` s.t. `axis[j] == i`.

  `tensor` can have up to 8 dimensions. The number of dimensions specified
  in `axis` may be 0 or more entries. If an index is specified more than
  once, a InvalidArgument error is raised.

  For example:

  ```
  # tensor 't' is [[[[ 0,  1,  2,  3],
  #                  [ 4,  5,  6,  7],
  #                  [ 8,  9, 10, 11]],
  #                 [[12, 13, 14, 15],
  #                  [16, 17, 18, 19],
  #                  [20, 21, 22, 23]]]]
  # tensor 't' shape is [1, 2, 3, 4]

  # 'dims' is [3] or 'dims' is [-1]
  reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                          [ 7,  6,  5,  4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

  # 'dims' is '[1]' (or 'dims' is '[-3]')
  reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]]]]

  # 'dims' is '[2]' (or 'dims' is '[-2]')
  reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `uint16`, `int16`, `int32`, `uint32`, `int64`, `uint64`, `bool`, `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`, `string`.
      Up to 8-D.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D. The indices of the dimensions to reverse. Must be in the range
      `[-rank(tensor), rank(tensor))`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReverseV2", name, tensor, axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_reverse_v2(
          (tensor, axis, name,), None)
      if _result is not NotImplemented:
        return _result
      return reverse_v2_eager_fallback(
          tensor, axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            reverse_v2, (), dict(tensor=tensor, axis=axis, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_reverse_v2(
        (tensor, axis, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReverseV2", tensor=tensor, axis=axis, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          reverse_v2, (), dict(tensor=tensor, axis=axis, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tidx", _op._get_attr_type("Tidx"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReverseV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReverseV2 = tf_export("raw_ops.ReverseV2")(_ops.to_raw_op(reverse_v2))
_dispatcher_for_reverse_v2 = reverse_v2._tf_type_based_dispatcher.Dispatch


def reverse_v2_eager_fallback(tensor: Annotated[Any, TV_ReverseV2_T], axis: Annotated[Any, TV_ReverseV2_Tidx], name, ctx) -> Annotated[Any, TV_ReverseV2_T]:
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.uint32, _dtypes.int64, _dtypes.uint64, _dtypes.bool, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, _dtypes.string, ])
  _inputs_flat = [tensor, axis]
  _attrs = ("Tidx", _attr_Tidx, "T", _attr_T)
  _result = _execute.execute(b"ReverseV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReverseV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ScatterNd_T = TypeVar("TV_ScatterNd_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ScatterNd_Tindices = TypeVar("TV_ScatterNd_Tindices", _atypes.Int16, _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('scatter_nd', v1=['scatter_nd', 'manip.scatter_nd'])
@deprecated_endpoints('manip.scatter_nd')
def scatter_nd(indices: Annotated[Any, TV_ScatterNd_Tindices], updates: Annotated[Any, TV_ScatterNd_T], shape: Annotated[Any, TV_ScatterNd_Tindices], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_ScatterNd_T]:
  r"""Scatters `updates` into a tensor of shape `shape` according to `indices`.

  Scatter sparse `updates` according to individual values at the specified
  `indices`. This op returns an output tensor with the `shape` you specify. This
  op is the inverse of the `tf.gather_nd` operator which extracts values or slices
  from a given tensor.

  This operation is similar to `tf.tensor_scatter_nd_add`, except that the tensor
  is zero-initialized. Calling `tf.scatter_nd(indices, updates, shape)`
  is identical to calling
  `tf.tensor_scatter_nd_add(tf.zeros(shape, updates.dtype), indices, updates)`

  If `indices` contains duplicates, the associated `updates` are accumulated
  (summed) into the output tensor.

  **WARNING**: For floating-point data types, the output may be nondeterministic.
  This is because the order in which the updates are applied is nondeterministic
  and when floating-point numbers are added in different orders the resulting
  numerical approximation error can be slightly different. However, the output
  will be deterministic if op determinism is enabled via
  `tf.config.experimental.enable_op_determinism`.

  `indices` is an integer tensor containing indices into the output tensor. The
  last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices of elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.

  `updates` is a tensor with shape:

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of the scatter op is to insert individual elements in
  a tensor by index. Consider an example where you want to insert 4 scattered
  elements in a rank-1 tensor with 8 elements.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      shape = tf.constant([8])
      scatter = tf.scatter_nd(indices, updates, shape)
      print(scatter)
  ```

  The resulting tensor would look like this:

      [0, 11, 0, 10, 9, 0, 0, 12]

  You can also insert entire slices of a higher rank tensor all at once. For
  example, you can insert two slices in the first dimension of a rank-3 tensor
  with two matrices of new values.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[1], [3]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      shape = tf.constant([4, 4, 4])
      scatter = tf.scatter_nd(indices, updates, shape)
      print(scatter)
  ```

  The resulting tensor would look like this:

      [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
       [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
       [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]

  If `indices` contains any out-of-bound indices, depending on
  `bad_indices_policy`, the op will either return an error or ignore the
  out-of-bound indices. `bad_indices_policy` can be one of the following values:
  1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
     historically on CPU and GPU we handle errors in different ways, and for
     backward compatibility we keep the default behavior.
  2. "ERROR": raises error; GPU does not support this value.
  3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.

  Args:
    indices: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      Tensor of indices.
    updates: A `Tensor`. Values to scatter into the output tensor.
    shape: A `Tensor`. Must have the same type as `indices`.
      1-D. The shape of the output tensor.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `updates`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ScatterNd", name, indices, updates, shape,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_scatter_nd(
          (indices, updates, shape, bad_indices_policy, name,), None)
      if _result is not NotImplemented:
        return _result
      return scatter_nd_eager_fallback(
          indices, updates, shape, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            scatter_nd, (), dict(indices=indices, updates=updates,
                                 shape=shape,
                                 bad_indices_policy=bad_indices_policy,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_scatter_nd(
        (indices, updates, shape, bad_indices_policy, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ScatterNd", indices=indices, updates=updates, shape=shape,
                     bad_indices_policy=bad_indices_policy, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          scatter_nd, (), dict(indices=indices, updates=updates, shape=shape,
                               bad_indices_policy=bad_indices_policy,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ScatterNd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ScatterNd = tf_export("raw_ops.ScatterNd")(_ops.to_raw_op(scatter_nd))
_dispatcher_for_scatter_nd = scatter_nd._tf_type_based_dispatcher.Dispatch


def scatter_nd_eager_fallback(indices: Annotated[Any, TV_ScatterNd_Tindices], updates: Annotated[Any, TV_ScatterNd_T], shape: Annotated[Any, TV_ScatterNd_Tindices], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_ScatterNd_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, (updates,) = _execute.args_to_matching_eager([updates], ctx, [])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([indices, shape], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  (indices, shape) = _inputs_Tindices
  _inputs_flat = [indices, updates, shape]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"ScatterNd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ScatterNd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ScatterNdNonAliasingAdd_T = TypeVar("TV_ScatterNdNonAliasingAdd_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ScatterNdNonAliasingAdd_Tindices = TypeVar("TV_ScatterNdNonAliasingAdd_Tindices", _atypes.Int32, _atypes.Int64)

def scatter_nd_non_aliasing_add(input: Annotated[Any, TV_ScatterNdNonAliasingAdd_T], indices: Annotated[Any, TV_ScatterNdNonAliasingAdd_Tindices], updates: Annotated[Any, TV_ScatterNdNonAliasingAdd_T], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_ScatterNdNonAliasingAdd_T]:
  r"""Applies sparse addition to `input` using individual values or slices

  from `updates` according to indices `indices`.  The updates are non-aliasing:
  `input` is only modified in-place if no other operations will use it.
  Otherwise, a copy of `input` is made.  This operation has a gradient with
  respect to both `input` and `updates`.

  `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

  `indices` must be integer tensor, containing indices into `input`.
  It must be shape \\([d_0, ..., d_{Q-2}, K]\\) where `0 < K <= P`.

  The innermost dimension of `indices` (with length `K`) corresponds to
  indices into elements (if `K = P`) or `(P-K)`-dimensional slices
  (if `K < P`) along the `K`th dimension of `input`.

  `updates` is `Tensor` of rank `Q-1+P-K` with shape:

  $$[d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].$$

  For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
  elements. In Python, that addition would look like this:

      input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
      with tf.Session() as sess:
        print(sess.run(output))

  The resulting value `output` would look like this:

      [1, 13, 3, 14, 14, 6, 7, 20]

  See `tf.scatter_nd` for more details about how to make updates to slices.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      A Tensor.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into `input`.
    updates: A `Tensor`. Must have the same type as `input`.
      A Tensor. Must have the same type as ref. A tensor of updated values
      to add to `input`.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ScatterNdNonAliasingAdd", name, input, indices, updates,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return scatter_nd_non_aliasing_add_eager_fallback(
          input, indices, updates, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ScatterNdNonAliasingAdd", input=input, indices=indices,
                                   updates=updates,
                                   bad_indices_policy=bad_indices_policy,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ScatterNdNonAliasingAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ScatterNdNonAliasingAdd = tf_export("raw_ops.ScatterNdNonAliasingAdd")(_ops.to_raw_op(scatter_nd_non_aliasing_add))


def scatter_nd_non_aliasing_add_eager_fallback(input: Annotated[Any, TV_ScatterNdNonAliasingAdd_T], indices: Annotated[Any, TV_ScatterNdNonAliasingAdd_Tindices], updates: Annotated[Any, TV_ScatterNdNonAliasingAdd_T], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_ScatterNdNonAliasingAdd_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, updates], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (input, updates) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input, indices, updates]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"ScatterNdNonAliasingAdd", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ScatterNdNonAliasingAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Shape_T = TypeVar("TV_Shape_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Shape_out_type = TypeVar("TV_Shape_out_type", _atypes.Int32, _atypes.Int64)

def shape(input: Annotated[Any, TV_Shape_T], out_type:TV_Shape_out_type=_dtypes.int32, name=None) -> Annotated[Any, TV_Shape_out_type]:
  r"""Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```

  Args:
    input: A `Tensor`.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Shape", name, input, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shape_eager_fallback(
          input, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Shape", input=input, out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Shape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Shape = tf_export("raw_ops.Shape")(_ops.to_raw_op(shape))


def shape_eager_fallback(input: Annotated[Any, TV_Shape_T], out_type: TV_Shape_out_type, name, ctx) -> Annotated[Any, TV_Shape_out_type]:
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"Shape", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Shape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ShapeN_T = TypeVar("TV_ShapeN_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_ShapeN_out_type = TypeVar("TV_ShapeN_out_type", _atypes.Int32, _atypes.Int64)

def shape_n(input: Annotated[List[Any], TV_ShapeN_T], out_type:TV_ShapeN_out_type=_dtypes.int32, name=None):
  r"""Returns shape of tensors.

  This operation returns N 1-D integer tensors representing shape of `input[i]s`.

  Args:
    input: A list of at least 1 `Tensor` objects with the same type.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `input` of `Tensor` objects with type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShapeN", name, input, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shape_n_eager_fallback(
          input, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'shape_n' Op, not %r." % input)
  _attr_N = len(input)
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShapeN", input=input, out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "out_type", _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShapeN", _inputs_flat, _attrs, _result)
  return _result

ShapeN = tf_export("raw_ops.ShapeN")(_ops.to_raw_op(shape_n))


def shape_n_eager_fallback(input: Annotated[List[Any], TV_ShapeN_T], out_type: TV_ShapeN_out_type, name, ctx):
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'shape_n' Op, not %r." % input)
  _attr_N = len(input)
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, input = _execute.args_to_matching_eager(list(input), ctx, [])
  _inputs_flat = list(input)
  _attrs = ("N", _attr_N, "T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"ShapeN", _attr_N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShapeN", _inputs_flat, _attrs, _result)
  return _result


TV_Size_T = TypeVar("TV_Size_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Size_out_type = TypeVar("TV_Size_out_type", _atypes.Int32, _atypes.Int64)

def size(input: Annotated[Any, TV_Size_T], out_type:TV_Size_out_type=_dtypes.int32, name=None) -> Annotated[Any, TV_Size_out_type]:
  r"""Returns the size of a tensor.

  This operation returns an integer representing the number of elements in
  `input`.

  For example:

  ```
  # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  size(t) ==> 12
  ```

  Args:
    input: A `Tensor`.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Size", name, input, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return size_eager_fallback(
          input, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Size", input=input, out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Size", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Size = tf_export("raw_ops.Size")(_ops.to_raw_op(size))


def size_eager_fallback(input: Annotated[Any, TV_Size_T], out_type: TV_Size_out_type, name, ctx) -> Annotated[Any, TV_Size_out_type]:
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"Size", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Size", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Slice_T = TypeVar("TV_Slice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Slice_Index = TypeVar("TV_Slice_Index", _atypes.Int32, _atypes.Int64)

def _slice(input: Annotated[Any, TV_Slice_T], begin: Annotated[Any, TV_Slice_Index], size: Annotated[Any, TV_Slice_Index], name=None) -> Annotated[Any, TV_Slice_T]:
  r"""Return a slice from 'input'.

  The output tensor is a tensor with dimensions described by 'size'
  whose values are extracted from 'input' starting at the offsets in
  'begin'.

  *Requirements*:
    0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

  Args:
    input: A `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      begin[i] specifies the offset into the 'i'th dimension of
      'input' to slice from.
    size: A `Tensor`. Must have the same type as `begin`.
      size[i] specifies the number of elements of the 'i'th dimension
      of 'input' to slice. If size[i] is -1, all remaining elements in dimension
      i are included in the slice (i.e. this is equivalent to setting
      size[i] = input.dim_size(i) - begin[i]).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Slice", name, input, begin, size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _slice_eager_fallback(
          input, begin, size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Slice", input=input, begin=begin, size=size, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Index",
              _op._get_attr_type("Index"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Slice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Slice = tf_export("raw_ops.Slice")(_ops.to_raw_op(_slice))


def _slice_eager_fallback(input: Annotated[Any, TV_Slice_T], begin: Annotated[Any, TV_Slice_Index], size: Annotated[Any, TV_Slice_Index], name, ctx) -> Annotated[Any, TV_Slice_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Index, _inputs_Index = _execute.args_to_matching_eager([begin, size], ctx, [_dtypes.int32, _dtypes.int64, ])
  (begin, size) = _inputs_Index
  _inputs_flat = [input, begin, size]
  _attrs = ("T", _attr_T, "Index", _attr_Index)
  _result = _execute.execute(b"Slice", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Slice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Snapshot_T = TypeVar("TV_Snapshot_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def snapshot(input: Annotated[Any, TV_Snapshot_T], name=None) -> Annotated[Any, TV_Snapshot_T]:
  r"""Returns a copy of the input tensor.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Snapshot", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return snapshot_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Snapshot", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Snapshot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Snapshot = tf_export("raw_ops.Snapshot")(_ops.to_raw_op(snapshot))


def snapshot_eager_fallback(input: Annotated[Any, TV_Snapshot_T], name, ctx) -> Annotated[Any, TV_Snapshot_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Snapshot", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Snapshot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SpaceToBatch_T = TypeVar("TV_SpaceToBatch_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SpaceToBatch_Tpaddings = TypeVar("TV_SpaceToBatch_Tpaddings", _atypes.Int32, _atypes.Int64)

def space_to_batch(input: Annotated[Any, TV_SpaceToBatch_T], paddings: Annotated[Any, TV_SpaceToBatch_Tpaddings], block_size: int, name=None) -> Annotated[Any, TV_SpaceToBatch_T]:
  r"""SpaceToBatch for 4-D tensors of type T.

  This is a legacy version of the more general SpaceToBatchND.

  Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
  More specifically, this op outputs a copy of the input tensor where values from
  the `height` and `width` dimensions are moved to the `batch` dimension. After
  the zero-padding, both `height` and `width` of the input must be divisible by the
  block size.

  The attr `block_size` must be greater than one. It indicates the block size.

    * Non-overlapping blocks of size `block_size x block size` in the height and
      width dimensions are rearranged into the batch dimension at each location.
    * The batch of the output tensor is `batch * block_size * block_size`.
    * Both height_pad and width_pad must be divisible by block_size.

  The shape of the output will be:

      [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
       depth]

  Some examples:

  (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:

  ```
  x = [[[[1], [2]], [[3], [4]]]]
  ```

  The output tensor has shape `[4, 1, 1, 1]` and value:

  ```
  [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
  ```

  (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:

  ```
  x = [[[[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]]]
  ```

  The output tensor has shape `[4, 1, 1, 3]` and value:

  ```
  [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
  ```

  (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:

  ```
  x = [[[[1],   [2],  [3],  [4]],
        [[5],   [6],  [7],  [8]],
        [[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]]]
  ```

  The output tensor has shape `[4, 2, 2, 1]` and value:

  ```
  x = [[[[1], [3]], [[9], [11]]],
       [[[2], [4]], [[10], [12]]],
       [[[5], [7]], [[13], [15]]],
       [[[6], [8]], [[14], [16]]]]
  ```

  (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:

  ```
  x = [[[[1],   [2],  [3],  [4]],
        [[5],   [6],  [7],  [8]]],
       [[[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]]]
  ```

  The output tensor has shape `[8, 1, 2, 1]` and value:

  ```
  x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
       [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
  ```

  Among others, this operation is useful for reducing atrous convolution into
  regular convolution.

  Args:
    input: A `Tensor`. 4-D with shape `[batch, height, width, depth]`.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
        the padding of the input with zeros across the spatial dimensions as follows:

            paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

        The effective spatial dimensions of the zero-padded input tensor will be:

            height_pad = pad_top + height + pad_bottom
            width_pad = pad_left + width + pad_right
    block_size: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SpaceToBatch", name, input, paddings, "block_size", block_size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return space_to_batch_eager_fallback(
          input, paddings, block_size=block_size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  block_size = _execute.make_int(block_size, "block_size")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SpaceToBatch", input=input, paddings=paddings, block_size=block_size,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tpaddings",
              _op._get_attr_type("Tpaddings"), "block_size",
              _op._get_attr_int("block_size"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SpaceToBatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SpaceToBatch = tf_export("raw_ops.SpaceToBatch")(_ops.to_raw_op(space_to_batch))


def space_to_batch_eager_fallback(input: Annotated[Any, TV_SpaceToBatch_T], paddings: Annotated[Any, TV_SpaceToBatch_Tpaddings], block_size: int, name, ctx) -> Annotated[Any, TV_SpaceToBatch_T]:
  block_size = _execute.make_int(block_size, "block_size")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, paddings]
  _attrs = ("T", _attr_T, "Tpaddings", _attr_Tpaddings, "block_size",
  block_size)
  _result = _execute.execute(b"SpaceToBatch", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SpaceToBatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SpaceToBatchND_T = TypeVar("TV_SpaceToBatchND_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SpaceToBatchND_Tblock_shape = TypeVar("TV_SpaceToBatchND_Tblock_shape", _atypes.Int32, _atypes.Int64)
TV_SpaceToBatchND_Tpaddings = TypeVar("TV_SpaceToBatchND_Tpaddings", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('space_to_batch_nd', v1=['space_to_batch_nd', 'manip.space_to_batch_nd'])
@deprecated_endpoints('manip.space_to_batch_nd')
def space_to_batch_nd(input: Annotated[Any, TV_SpaceToBatchND_T], block_shape: Annotated[Any, TV_SpaceToBatchND_Tblock_shape], paddings: Annotated[Any, TV_SpaceToBatchND_Tpaddings], name=None) -> Annotated[Any, TV_SpaceToBatchND_T]:
  r"""SpaceToBatch for N-D tensors of type T.

  This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
  grid of blocks of shape `block_shape`, and interleaves these blocks with the
  "batch" dimension (0) such that in the output, the spatial dimensions
  `[1, ..., M]` correspond to the position within the grid, and the batch
  dimension combines both the position within a spatial block and the original
  batch position.  Prior to division into blocks, the spatial dimensions of the
  input are optionally zero padded according to `paddings`. See below for a
  precise description.

  This operation is equivalent to the following steps:

  1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
     input according to `paddings` to produce `padded` of shape `padded_shape`.

  2. Reshape `padded` to `reshaped_padded` of shape:

       [batch] +
       [padded_shape[1] / block_shape[0],
         block_shape[0],
        ...,
        padded_shape[M] / block_shape[M-1],
        block_shape[M-1]] +
       remaining_shape

  3. Permute dimensions of `reshaped_padded` to produce
     `permuted_reshaped_padded` of shape:

       block_shape +
       [batch] +
       [padded_shape[1] / block_shape[0],
        ...,
        padded_shape[M] / block_shape[M-1]] +
       remaining_shape

  4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
     dimension, producing an output tensor of shape:

       [batch * prod(block_shape)] +
       [padded_shape[1] / block_shape[0],
        ...,
        padded_shape[M] / block_shape[M-1]] +
       remaining_shape

  Some examples:

  (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
      `paddings = [[0, 0], [0, 0]]`:

  ```
  x = [[[[1], [2]], [[3], [4]]]]
  ```

  The output tensor has shape `[4, 1, 1, 1]` and value:

  ```
  [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
  ```

  (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
      `paddings = [[0, 0], [0, 0]]`:

  ```
  x = [[[[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]]]
  ```

  The output tensor has shape `[4, 1, 1, 3]` and value:

  ```
  [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
  ```

  (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
      `paddings = [[0, 0], [0, 0]]`:

  ```
  x = [[[[1],   [2],  [3],  [4]],
        [[5],   [6],  [7],  [8]],
        [[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]]]
  ```

  The output tensor has shape `[4, 2, 2, 1]` and value:

  ```
  x = [[[[1], [3]], [[9], [11]]],
       [[[2], [4]], [[10], [12]]],
       [[[5], [7]], [[13], [15]]],
       [[[6], [8]], [[14], [16]]]]
  ```

  (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
      paddings = `[[0, 0], [2, 0]]`:

  ```
  x = [[[[1],   [2],  [3],  [4]],
        [[5],   [6],  [7],  [8]]],
       [[[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]]]
  ```

  The output tensor has shape `[8, 1, 3, 1]` and value:

  ```
  x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
       [[[0], [2], [4]]], [[[0], [10], [12]]],
       [[[0], [5], [7]]], [[[0], [13], [15]]],
       [[[0], [6], [8]]], [[[0], [14], [16]]]]
  ```

  Among others, this operation is useful for reducing atrous convolution into
  regular convolution.

  Args:
    input: A `Tensor`.
      N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
      where spatial_shape has `M` dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with shape `[M]`, all values must be >= 1.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D with shape `[M, 2]`, all values must be >= 0.
        `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
        `i + 1`, which corresponds to spatial dimension `i`.  It is required that
        `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SpaceToBatchND", name, input, block_shape, paddings)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_space_to_batch_nd(
          (input, block_shape, paddings, name,), None)
      if _result is not NotImplemented:
        return _result
      return space_to_batch_nd_eager_fallback(
          input, block_shape, paddings, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            space_to_batch_nd, (), dict(input=input, block_shape=block_shape,
                                        paddings=paddings, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_space_to_batch_nd(
        (input, block_shape, paddings, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SpaceToBatchND", input=input, block_shape=block_shape,
                          paddings=paddings, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          space_to_batch_nd, (), dict(input=input, block_shape=block_shape,
                                      paddings=paddings, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tblock_shape",
              _op._get_attr_type("Tblock_shape"), "Tpaddings",
              _op._get_attr_type("Tpaddings"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SpaceToBatchND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SpaceToBatchND = tf_export("raw_ops.SpaceToBatchND")(_ops.to_raw_op(space_to_batch_nd))
_dispatcher_for_space_to_batch_nd = space_to_batch_nd._tf_type_based_dispatcher.Dispatch


def space_to_batch_nd_eager_fallback(input: Annotated[Any, TV_SpaceToBatchND_T], block_shape: Annotated[Any, TV_SpaceToBatchND_Tblock_shape], paddings: Annotated[Any, TV_SpaceToBatchND_Tpaddings], name, ctx) -> Annotated[Any, TV_SpaceToBatchND_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tblock_shape, (block_shape,) = _execute.args_to_matching_eager([block_shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, block_shape, paddings]
  _attrs = ("T", _attr_T, "Tblock_shape", _attr_Tblock_shape, "Tpaddings",
  _attr_Tpaddings)
  _result = _execute.execute(b"SpaceToBatchND", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SpaceToBatchND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SpaceToDepth_T = TypeVar("TV_SpaceToDepth_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def space_to_depth(input: Annotated[Any, TV_SpaceToDepth_T], block_size: int, data_format:str="NHWC", name=None) -> Annotated[Any, TV_SpaceToDepth_T]:
  r"""SpaceToDepth for tensors of type T.

  Rearranges blocks of spatial data, into depth. More specifically,
  this op outputs a copy of the input tensor where values from the `height`
  and `width` dimensions are moved to the `depth` dimension.
  The attr `block_size` indicates the input block size.

    * Non-overlapping blocks of size `block_size x block size` are rearranged
      into depth at each location.
    * The depth of the output tensor is `block_size * block_size * input_depth`.
    * The Y, X coordinates within each block of the input become the high order
      component of the output channel index.
    * The input tensor's height and width must be divisible by block_size.

  The `data_format` attr specifies the layout of the input and output tensors
  with the following options:
    "NHWC": `[ batch, height, width, channels ]`
    "NCHW": `[ batch, channels, height, width ]`
    "NCHW_VECT_C":
        `qint8 [ batch, channels / 4, height, width, 4 ]`

  It is useful to consider the operation as transforming a 6-D Tensor.
  e.g. for data_format = NHWC,
       Each element in the input tensor can be specified via 6 coordinates,
       ordered by decreasing memory layout significance as:
       n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
                          within the output image, bX, bY means coordinates
                          within the input block, iC means input channels).
       The output would be a transpose to the following layout:
       n,oY,oX,bY,bX,iC

  This operation is useful for resizing the activations between convolutions
  (but keeping all data), e.g. instead of pooling. It is also useful for training
  purely convolutional models.

  For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
  block_size = 2:

  ```
  x = [[[[1], [2]],
        [[3], [4]]]]
  ```

  This operation will output a tensor of shape `[1, 1, 1, 4]`:

  ```
  [[[[1, 2, 3, 4]]]]
  ```

  Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
  the corresponding output will have a single element (i.e. width and height are
  both 1) and will have a depth of 4 channels (1 * block_size * block_size).
  The output element shape is `[1, 1, 4]`.

  For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

  ```
  x = [[[[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]]]
  ```

  This operation, for block_size of 2, will return the following tensor of shape
  `[1, 1, 1, 12]`

  ```
  [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
  ```

  Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

  ```
  x = [[[[1],   [2],  [5],  [6]],
        [[3],   [4],  [7],  [8]],
        [[9],  [10], [13],  [14]],
        [[11], [12], [15],  [16]]]]
  ```

  the operator will return the following tensor of shape `[1 2 2 4]`:

  ```
  x = [[[[1, 2, 3, 4],
         [5, 6, 7, 8]],
        [[9, 10, 11, 12],
         [13, 14, 15, 16]]]]
  ```

  Args:
    input: A `Tensor`.
    block_size: An `int` that is `>= 2`. The size of the spatial block.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NCHW_VECT_C"`. Defaults to `"NHWC"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SpaceToDepth", name, input, "block_size", block_size,
        "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return space_to_depth_eager_fallback(
          input, block_size=block_size, data_format=data_format, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  block_size = _execute.make_int(block_size, "block_size")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SpaceToDepth", input=input, block_size=block_size,
                        data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "block_size",
              _op._get_attr_int("block_size"), "data_format",
              _op.get_attr("data_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SpaceToDepth", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SpaceToDepth = tf_export("raw_ops.SpaceToDepth")(_ops.to_raw_op(space_to_depth))


def space_to_depth_eager_fallback(input: Annotated[Any, TV_SpaceToDepth_T], block_size: int, data_format: str, name, ctx) -> Annotated[Any, TV_SpaceToDepth_T]:
  block_size = _execute.make_int(block_size, "block_size")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "block_size", block_size, "data_format",
  data_format)
  _result = _execute.execute(b"SpaceToDepth", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SpaceToDepth", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Split_T = TypeVar("TV_Split_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def split(axis: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_Split_T], num_split: int, name=None):
  r"""Splits a tensor into `num_split` tensors along one dimension.

  Args:
    axis: A `Tensor` of type `int32`.
      0-D.  The dimension along which to split.  Must be in the range
      `[-rank(value), rank(value))`.
    value: A `Tensor`. The tensor to split.
    num_split: An `int` that is `>= 1`.
      The number of ways to split.  Must evenly divide
      `value.shape[split_dim]`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_split` `Tensor` objects with the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Split", name, axis, value, "num_split", num_split)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return split_eager_fallback(
          axis, value, num_split=num_split, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_split = _execute.make_int(num_split, "num_split")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Split", split_dim=axis, value=value, num_split=num_split, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_split", _op._get_attr_int("num_split"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Split", _inputs_flat, _attrs, _result)
  return _result

Split = tf_export("raw_ops.Split")(_ops.to_raw_op(split))


def split_eager_fallback(axis: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_Split_T], num_split: int, name, ctx):
  num_split = _execute.make_int(num_split, "num_split")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  axis = _ops.convert_to_tensor(axis, _dtypes.int32)
  _inputs_flat = [axis, value]
  _attrs = ("num_split", num_split, "T", _attr_T)
  _result = _execute.execute(b"Split", num_split, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Split", _inputs_flat, _attrs, _result)
  return _result


TV_SplitV_T = TypeVar("TV_SplitV_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SplitV_Tlen = TypeVar("TV_SplitV_Tlen", _atypes.Int32, _atypes.Int64, _atypes.Int8)

def split_v(value: Annotated[Any, TV_SplitV_T], size_splits: Annotated[Any, TV_SplitV_Tlen], axis: Annotated[Any, _atypes.Int32], num_split: int, name=None):
  r"""Splits a tensor into `num_split` tensors along one dimension.

  Args:
    value: A `Tensor`. The tensor to split.
    size_splits: A `Tensor`. Must be one of the following types: `int8`, `int32`, `int64`.
      list containing the sizes of each output tensor along the split
      dimension. Must sum to the dimension of value along split_dim.
      Can contain one -1 indicating that dimension is to be inferred.
    axis: A `Tensor` of type `int32`.
      0-D.  The dimension along which to split.  Must be in the range
      `[-rank(value), rank(value))`.
    num_split: An `int` that is `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_split` `Tensor` objects with the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SplitV", name, value, size_splits, axis, "num_split",
        num_split)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return split_v_eager_fallback(
          value, size_splits, axis, num_split=num_split, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_split = _execute.make_int(num_split, "num_split")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SplitV", value=value, size_splits=size_splits, split_dim=axis,
                  num_split=num_split, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_split", _op._get_attr_int("num_split"), "T",
              _op._get_attr_type("T"), "Tlen", _op._get_attr_type("Tlen"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SplitV", _inputs_flat, _attrs, _result)
  return _result

SplitV = tf_export("raw_ops.SplitV")(_ops.to_raw_op(split_v))


def split_v_eager_fallback(value: Annotated[Any, TV_SplitV_T], size_splits: Annotated[Any, TV_SplitV_Tlen], axis: Annotated[Any, _atypes.Int32], num_split: int, name, ctx):
  num_split = _execute.make_int(num_split, "num_split")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _attr_Tlen, (size_splits,) = _execute.args_to_matching_eager([size_splits], ctx, [_dtypes.int8, _dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  axis = _ops.convert_to_tensor(axis, _dtypes.int32)
  _inputs_flat = [value, size_splits, axis]
  _attrs = ("num_split", num_split, "T", _attr_T, "Tlen", _attr_Tlen)
  _result = _execute.execute(b"SplitV", num_split, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SplitV", _inputs_flat, _attrs, _result)
  return _result


TV_Squeeze_T = TypeVar("TV_Squeeze_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def squeeze(input: Annotated[Any, TV_Squeeze_T], axis=[], name=None) -> Annotated[Any, TV_Squeeze_T]:
  r"""Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `axis`.

  For example:

  ```
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t)) ==> [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
  ```

  Args:
    input: A `Tensor`. The `input` to squeeze.
    axis: An optional list of `ints`. Defaults to `[]`.
      If specified, only squeezes the dimensions listed. The dimension
      index starts at 0. It is an error to squeeze a dimension that is not 1. Must
      be in the range `[-rank(input), rank(input))`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Squeeze", name, input, "squeeze_dims", axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return squeeze_eager_fallback(
          input, axis=axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if axis is None:
    axis = []
  if not isinstance(axis, (list, tuple)):
    raise TypeError(
        "Expected list for 'axis' argument to "
        "'squeeze' Op, not %r." % axis)
  axis = [_execute.make_int(_i, "axis") for _i in axis]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Squeeze", input=input, squeeze_dims=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "squeeze_dims",
              _op.get_attr("squeeze_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Squeeze", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Squeeze = tf_export("raw_ops.Squeeze")(_ops.to_raw_op(squeeze))


def squeeze_eager_fallback(input: Annotated[Any, TV_Squeeze_T], axis, name, ctx) -> Annotated[Any, TV_Squeeze_T]:
  if axis is None:
    axis = []
  if not isinstance(axis, (list, tuple)):
    raise TypeError(
        "Expected list for 'axis' argument to "
        "'squeeze' Op, not %r." % axis)
  axis = [_execute.make_int(_i, "axis") for _i in axis]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "squeeze_dims", axis)
  _result = _execute.execute(b"Squeeze", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Squeeze", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StopGradient_T = TypeVar("TV_StopGradient_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stop_gradient(input: Annotated[Any, TV_StopGradient_T], name=None) -> Annotated[Any, TV_StopGradient_T]:
  r"""Stops gradient computation.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, this op prevents the contribution of
  its inputs to be taken into account.  Normally, the gradient generator adds ops
  to a graph to compute the derivatives of a specified 'loss' by recursively
  finding out inputs that contributed to its computation.  If you insert this op
  in the graph it inputs are masked from the gradient generator.  They are not
  taken into account for computing gradients.

  This is useful any time you want to compute a value with TensorFlow but need
  to pretend that the value was a constant. For example, the softmax function
  for a vector x can be written as

  ```python

    def softmax(x):
      numerator = tf.exp(x)
      denominator = tf.reduce_sum(numerator)
      return numerator / denominator
  ```

  This however is susceptible to overflow if the values in x are large. An
  alternative more stable way is to subtract the maximum of x from each of the
  values.

  ```python

    def stable_softmax(x):
      z = x - tf.reduce_max(x)
      numerator = tf.exp(z)
      denominator = tf.reduce_sum(numerator)
      return numerator / denominator
  ```

  However, when we backprop through the softmax to x, we dont want to backprop
  through the `tf.reduce_max(x)` (if the max values are not unique then the
  gradient could flow to the wrong input) calculation and treat that as a
  constant. Therefore, we should write this out as

  ```python

    def stable_softmax(x):
      z = x - tf.stop_gradient(tf.reduce_max(x))
      numerator = tf.exp(z)
      denominator = tf.reduce_sum(numerator)
      return numerator / denominator
  ```

  Some other examples include:

  *  The *EM* algorithm where the *M-step* should not involve backpropagation
     through the output of the *E-step*.
  *  Contrastive divergence training of Boltzmann machines where, when
     differentiating the energy function, the training must not backpropagate
     through the graph that generated the samples from the model.
  *  Adversarial training, where no backprop should happen through the adversarial
     example generation process.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StopGradient", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stop_gradient_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StopGradient", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StopGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StopGradient = tf_export("raw_ops.StopGradient")(_ops.to_raw_op(stop_gradient))


def stop_gradient_eager_fallback(input: Annotated[Any, TV_StopGradient_T], name, ctx) -> Annotated[Any, TV_StopGradient_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"StopGradient", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StopGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StridedSlice_T = TypeVar("TV_StridedSlice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StridedSlice_Index = TypeVar("TV_StridedSlice_Index", _atypes.Int16, _atypes.Int32, _atypes.Int64)

def strided_slice(input: Annotated[Any, TV_StridedSlice_T], begin: Annotated[Any, TV_StridedSlice_Index], end: Annotated[Any, TV_StridedSlice_Index], strides: Annotated[Any, TV_StridedSlice_Index], begin_mask:int=0, end_mask:int=0, ellipsis_mask:int=0, new_axis_mask:int=0, shrink_axis_mask:int=0, name=None) -> Annotated[Any, TV_StridedSlice_T]:
  r"""Return a strided slice from `input`.

  Note, most python users will want to use the Python `Tensor.__getitem__`
  or `Variable.__getitem__` rather than this op directly.

  The goal of this op is to produce a new tensor with a subset of
  the elements from the `n` dimensional `input` tensor. The subset is chosen using
  a sequence of `m` sparse range specifications encoded into the arguments
  of this function. Note, in some cases
  `m` could be equal to `n`, but this need not be the case. Each
  range specification entry can be one of the following:

  - An ellipsis (...). Ellipses are used to imply zero or more
    dimensions of full-dimension selection and are produced using
    `ellipsis_mask`. For example, `foo[...]` is the identity slice.

  - A new axis. This is used to insert a new shape=1 dimension and is
    produced using `new_axis_mask`. For example, `foo[:, ...]` where
    `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.


  - A range `begin:end:stride`. This is used to specify how much to choose from
    a given dimension. `stride` can be any integer but 0.  `begin` is an integer
    which represents the index of the first value to select while `end` represents
    the index of the last value to select. The number of values selected in each
    dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
    `begin` and `end` can be negative where `-1` is the last element, `-2` is
    the second to last. `begin_mask` controls whether to replace the explicitly
    given `begin` with an implicit effective value of `0` if `stride > 0` and
    `-1` if `stride < 0`. `end_mask` is analogous but produces the number
    required to create the largest open interval. For example, given a shape
    `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
    not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
    and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
    first dimension of a tensor while dropping the last two (in the original
    order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.

  - A single index. This is used to keep only elements that have a given
    index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
    shape `(6,)` tensor. This is encoded in `begin` and `end` and
    `shrink_axis_mask`.

  Each conceptual range specification is encoded in the op's argument. This
  encoding is best understand by considering a non-trivial example. In
  particular,
  `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as

  ```
  begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
  end = [2, 4, x, x, -3, x]
  strides = [1, 1, x, x, -1, 1]
  begin_mask = 1<<4 | 1<<5 = 48
  end_mask = 1<<5 = 32
  ellipsis_mask = 1<<3 = 8
  new_axis_mask = 1<<2 = 4
  shrink_axis_mask = 1<<0 = 1
  ```

  In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
  the slice becomes (2, 1, 5, 5, 2, 5).
  Let us walk step by step through each argument specification.

  1.  The first argument in the example slice is turned into `begin = 1` and
  `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
  also set the appropriate bit in `shrink_axis_mask`.

  2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
  zero bits contributed.

  3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
  dimension in the final shape. Dummy values are contributed to begin,
  end and stride, while the new_axis_mask bit is set.

  4. `...` grab the full ranges from as many dimensions as needed to
  fully specify a slice for every dimension of the input shape.

  5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
  with a dimension that has shape `s` is converted to a positive index
  `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
  is done internally so begin, end and strides receive x, -3, and -1.
  The appropriate begin_mask bit is set to indicate the start range is the
  full range (ignoring the x).

  6. `:` indicates that the entire contents of the corresponding dimension
  is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
  receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
  `end_mask` are also set.

  *Requirements*:
    `0 != strides[i] for i in [0, m)`
    `ellipsis_mask must be a power of two (only one ellipsis)`

  Args:
    input: A `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      `begin[k]` specifies the offset into the `k`th range specification.
      The exact dimension this corresponds to will be determined by context.
      Out-of-bounds values will be silently clamped. If the `k`th bit of
      `begin_mask` then `begin[k]` is ignored and the full range of the
      appropriate dimension is used instead. Negative values causes indexing
      to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
    end: A `Tensor`. Must have the same type as `begin`.
      `end[i]` is like `begin` with the exception that `end_mask` is
      used to determine full ranges.
    strides: A `Tensor`. Must have the same type as `begin`.
      `strides[i]` specifies the increment in the `i`th specification
      after extracting a given element. Negative indices will reverse
      the original order. Out or range values are
      clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
    begin_mask: An optional `int`. Defaults to `0`.
      a bitmask where a bit i being 1 means to ignore the begin
      value and instead use the largest interval possible. At runtime
      begin[i] will be replaced with `[0, n-1)` if `stride[i] > 0` or
      `[-1, n-1]` if `stride[i] < 0`
    end_mask: An optional `int`. Defaults to `0`. analogous to `begin_mask`
    ellipsis_mask: An optional `int`. Defaults to `0`.
      a bitmask where bit `i` being 1 means the `i`th
      position is actually an ellipsis. One bit at most can be 1.
      If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
      is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
      implicitly creates as many range specifications as necessary to fully
      specify the sliced range for every dimension. For example for a 4-dimensional
      tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
    new_axis_mask: An optional `int`. Defaults to `0`.
      a bitmask where bit `i` being 1 means the `i`th
      specification creates a new shape 1 dimension. For example
      `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
    shrink_axis_mask: An optional `int`. Defaults to `0`.
      a bitmask where bit `i` implies that the `i`th
      specification should shrink the dimensionality. begin and end
      must imply a slice of size 1 in the dimension. For example in
      python one might do `foo[:, 3, :]` which would result in
      `shrink_axis_mask` being 2.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StridedSlice", name, input, begin, end, strides, "begin_mask",
        begin_mask, "end_mask", end_mask, "ellipsis_mask", ellipsis_mask,
        "new_axis_mask", new_axis_mask, "shrink_axis_mask", shrink_axis_mask)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return strided_slice_eager_fallback(
          input, begin, end, strides, begin_mask=begin_mask,
          end_mask=end_mask, ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StridedSlice", input=input, begin=begin, end=end, strides=strides,
                        begin_mask=begin_mask, end_mask=end_mask,
                        ellipsis_mask=ellipsis_mask,
                        new_axis_mask=new_axis_mask,
                        shrink_axis_mask=shrink_axis_mask, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Index",
              _op._get_attr_type("Index"), "begin_mask",
              _op._get_attr_int("begin_mask"), "end_mask",
              _op._get_attr_int("end_mask"), "ellipsis_mask",
              _op._get_attr_int("ellipsis_mask"), "new_axis_mask",
              _op._get_attr_int("new_axis_mask"), "shrink_axis_mask",
              _op._get_attr_int("shrink_axis_mask"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StridedSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StridedSlice = tf_export("raw_ops.StridedSlice")(_ops.to_raw_op(strided_slice))


def strided_slice_eager_fallback(input: Annotated[Any, TV_StridedSlice_T], begin: Annotated[Any, TV_StridedSlice_Index], end: Annotated[Any, TV_StridedSlice_Index], strides: Annotated[Any, TV_StridedSlice_Index], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx) -> Annotated[Any, TV_StridedSlice_T]:
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Index, _inputs_Index = _execute.args_to_matching_eager([begin, end, strides], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  (begin, end, strides) = _inputs_Index
  _inputs_flat = [input, begin, end, strides]
  _attrs = ("T", _attr_T, "Index", _attr_Index, "begin_mask", begin_mask,
  "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask",
  new_axis_mask, "shrink_axis_mask", shrink_axis_mask)
  _result = _execute.execute(b"StridedSlice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StridedSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StridedSliceAssign_T = TypeVar("TV_StridedSliceAssign_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StridedSliceAssign_Index = TypeVar("TV_StridedSliceAssign_Index", _atypes.Int32, _atypes.Int64)

def strided_slice_assign(ref: Annotated[Any, TV_StridedSliceAssign_T], begin: Annotated[Any, TV_StridedSliceAssign_Index], end: Annotated[Any, TV_StridedSliceAssign_Index], strides: Annotated[Any, TV_StridedSliceAssign_Index], value: Annotated[Any, TV_StridedSliceAssign_T], begin_mask:int=0, end_mask:int=0, ellipsis_mask:int=0, new_axis_mask:int=0, shrink_axis_mask:int=0, name=None) -> Annotated[Any, TV_StridedSliceAssign_T]:
  r"""Assign `value` to the sliced l-value reference of `ref`.

  The values of `value` are assigned to the positions in the variable
  `ref` that are selected by the slice parameters. The slice parameters
  `begin`, `end`, `strides`, etc. work exactly as in `StridedSlice`.

  NOTE this op currently does not support broadcasting and so `value`'s
  shape must be exactly the shape produced by the slice of `ref`.

  Args:
    ref: A mutable `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    end: A `Tensor`. Must have the same type as `begin`.
    strides: A `Tensor`. Must have the same type as `begin`.
    value: A `Tensor`. Must have the same type as `ref`.
    begin_mask: An optional `int`. Defaults to `0`.
    end_mask: An optional `int`. Defaults to `0`.
    ellipsis_mask: An optional `int`. Defaults to `0`.
    new_axis_mask: An optional `int`. Defaults to `0`.
    shrink_axis_mask: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `ref`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("strided_slice_assign op does not support eager execution. Arg 'output_ref' is a ref.")
  # Add nodes to the TensorFlow graph.
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StridedSliceAssign", ref=ref, begin=begin, end=end, strides=strides,
                              value=value, begin_mask=begin_mask,
                              end_mask=end_mask, ellipsis_mask=ellipsis_mask,
                              new_axis_mask=new_axis_mask,
                              shrink_axis_mask=shrink_axis_mask, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Index",
              _op._get_attr_type("Index"), "begin_mask",
              _op._get_attr_int("begin_mask"), "end_mask",
              _op._get_attr_int("end_mask"), "ellipsis_mask",
              _op._get_attr_int("ellipsis_mask"), "new_axis_mask",
              _op._get_attr_int("new_axis_mask"), "shrink_axis_mask",
              _op._get_attr_int("shrink_axis_mask"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StridedSliceAssign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StridedSliceAssign = tf_export("raw_ops.StridedSliceAssign")(_ops.to_raw_op(strided_slice_assign))


def strided_slice_assign_eager_fallback(ref: Annotated[Any, TV_StridedSliceAssign_T], begin: Annotated[Any, TV_StridedSliceAssign_Index], end: Annotated[Any, TV_StridedSliceAssign_Index], strides: Annotated[Any, TV_StridedSliceAssign_Index], value: Annotated[Any, TV_StridedSliceAssign_T], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx) -> Annotated[Any, TV_StridedSliceAssign_T]:
  raise RuntimeError("strided_slice_assign op does not support eager execution. Arg 'output_ref' is a ref.")

TV_StridedSliceGrad_T = TypeVar("TV_StridedSliceGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StridedSliceGrad_Index = TypeVar("TV_StridedSliceGrad_Index", _atypes.Int32, _atypes.Int64)

def strided_slice_grad(shape: Annotated[Any, TV_StridedSliceGrad_Index], begin: Annotated[Any, TV_StridedSliceGrad_Index], end: Annotated[Any, TV_StridedSliceGrad_Index], strides: Annotated[Any, TV_StridedSliceGrad_Index], dy: Annotated[Any, TV_StridedSliceGrad_T], begin_mask:int=0, end_mask:int=0, ellipsis_mask:int=0, new_axis_mask:int=0, shrink_axis_mask:int=0, name=None) -> Annotated[Any, TV_StridedSliceGrad_T]:
  r"""Returns the gradient of `StridedSlice`.

  Since `StridedSlice` cuts out pieces of its `input` which is size
  `shape`, its gradient will have the same shape (which is passed here
  as `shape`). The gradient will be zero in any element that the slice
  does not select.

  Arguments are the same as StridedSliceGrad with the exception that
  `dy` is the input gradient to be propagated and `shape` is the
  shape of `StridedSlice`'s `input`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    begin: A `Tensor`. Must have the same type as `shape`.
    end: A `Tensor`. Must have the same type as `shape`.
    strides: A `Tensor`. Must have the same type as `shape`.
    dy: A `Tensor`.
    begin_mask: An optional `int`. Defaults to `0`.
    end_mask: An optional `int`. Defaults to `0`.
    ellipsis_mask: An optional `int`. Defaults to `0`.
    new_axis_mask: An optional `int`. Defaults to `0`.
    shrink_axis_mask: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `dy`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StridedSliceGrad", name, shape, begin, end, strides, dy,
        "begin_mask", begin_mask, "end_mask", end_mask, "ellipsis_mask",
        ellipsis_mask, "new_axis_mask", new_axis_mask, "shrink_axis_mask",
        shrink_axis_mask)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return strided_slice_grad_eager_fallback(
          shape, begin, end, strides, dy, begin_mask=begin_mask,
          end_mask=end_mask, ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StridedSliceGrad", shape=shape, begin=begin, end=end,
                            strides=strides, dy=dy, begin_mask=begin_mask,
                            end_mask=end_mask, ellipsis_mask=ellipsis_mask,
                            new_axis_mask=new_axis_mask,
                            shrink_axis_mask=shrink_axis_mask, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Index",
              _op._get_attr_type("Index"), "begin_mask",
              _op._get_attr_int("begin_mask"), "end_mask",
              _op._get_attr_int("end_mask"), "ellipsis_mask",
              _op._get_attr_int("ellipsis_mask"), "new_axis_mask",
              _op._get_attr_int("new_axis_mask"), "shrink_axis_mask",
              _op._get_attr_int("shrink_axis_mask"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StridedSliceGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StridedSliceGrad = tf_export("raw_ops.StridedSliceGrad")(_ops.to_raw_op(strided_slice_grad))


def strided_slice_grad_eager_fallback(shape: Annotated[Any, TV_StridedSliceGrad_Index], begin: Annotated[Any, TV_StridedSliceGrad_Index], end: Annotated[Any, TV_StridedSliceGrad_Index], strides: Annotated[Any, TV_StridedSliceGrad_Index], dy: Annotated[Any, TV_StridedSliceGrad_T], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx) -> Annotated[Any, TV_StridedSliceGrad_T]:
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _attr_T, (dy,) = _execute.args_to_matching_eager([dy], ctx, [])
  _attr_Index, _inputs_Index = _execute.args_to_matching_eager([shape, begin, end, strides], ctx, [_dtypes.int32, _dtypes.int64, ])
  (shape, begin, end, strides) = _inputs_Index
  _inputs_flat = [shape, begin, end, strides, dy]
  _attrs = ("T", _attr_T, "Index", _attr_Index, "begin_mask", begin_mask,
  "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask",
  new_axis_mask, "shrink_axis_mask", shrink_axis_mask)
  _result = _execute.execute(b"StridedSliceGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StridedSliceGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorScatterAdd_T = TypeVar("TV_TensorScatterAdd_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorScatterAdd_Tindices = TypeVar("TV_TensorScatterAdd_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_scatter_nd_add', v1=['tensor_scatter_nd_add', 'tensor_scatter_add'])
@deprecated_endpoints('tensor_scatter_add')
def tensor_scatter_add(tensor: Annotated[Any, TV_TensorScatterAdd_T], indices: Annotated[Any, TV_TensorScatterAdd_Tindices], updates: Annotated[Any, TV_TensorScatterAdd_T], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_TensorScatterAdd_T]:
  r"""Adds sparse `updates` to an existing tensor according to `indices`.

  This operation creates a new tensor by adding sparse `updates` to the passed
  in `tensor`.
  This operation is very similar to `tf.compat.v1.scatter_nd_add`, except that the
  updates are added onto an existing tensor (as opposed to a variable). If the
  memory for the existing tensor cannot be re-used, a copy is made and updated.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `tensor.shape`.  The last dimension of `indices` can be at most the rank of
  `tensor.shape`:

  ```
  indices.shape[-1] <= tensor.shape.rank
  ```

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = tensor.shape.rank`) or slices
  (if `indices.shape[-1] < tensor.shape.rank`) along dimension
  `indices.shape[-1]` of `tensor.shape`.  `updates` is a tensor with shape

  ```
  indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
  ```

  The simplest form of `tensor_scatter_nd_add` is to add individual elements to a
  tensor by index. For example, say we want to add 4 elements in a rank-1
  tensor with 8 elements.

  In Python, this scatter add operation would look like this:

  >>> indices = tf.constant([[4], [3], [1], [7]])
  >>> updates = tf.constant([9, 10, 11, 12])
  >>> tensor = tf.ones([8], dtype=tf.int32)
  >>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
  >>> updated
  <tf.Tensor: shape=(8,), dtype=int32,
  numpy=array([ 1, 12,  1, 11, 10,  1,  1, 13], dtype=int32)>

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  In Python, this scatter add operation would look like this:

  >>> indices = tf.constant([[0], [2]])
  >>> updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
  ...                         [7, 7, 7, 7], [8, 8, 8, 8]],
  ...                        [[5, 5, 5, 5], [6, 6, 6, 6],
  ...                         [7, 7, 7, 7], [8, 8, 8, 8]]])
  >>> tensor = tf.ones([4, 4, 4],dtype=tf.int32)
  >>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
  >>> updated
  <tf.Tensor: shape=(4, 4, 4), dtype=int32,
  numpy=array([[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
               [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
               [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
               [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=int32)>


  If `indices` contains any out-of-bound indices, depending on
  `bad_indices_policy`, the op will either return an error or ignore the
  out-of-bound indices. `bad_indices_policy` can be one of the following values:
  1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
     historically on CPU and GPU we handle errors in different ways, and for
     backward compatibility we keep the default behavior.
  2. "ERROR": raises error; GPU does not support this value.
  3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorScatterAdd", name, tensor, indices, updates,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_scatter_add(
          (tensor, indices, updates, bad_indices_policy, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_scatter_add_eager_fallback(
          tensor, indices, updates, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_scatter_add, (), dict(tensor=tensor, indices=indices,
                                         updates=updates,
                                         bad_indices_policy=bad_indices_policy,
                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_scatter_add(
        (tensor, indices, updates, bad_indices_policy, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorScatterAdd", tensor=tensor, indices=indices, updates=updates,
                            bad_indices_policy=bad_indices_policy, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_scatter_add, (), dict(tensor=tensor, indices=indices,
                                       updates=updates,
                                       bad_indices_policy=bad_indices_policy,
                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorScatterAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorScatterAdd = tf_export("raw_ops.TensorScatterAdd")(_ops.to_raw_op(tensor_scatter_add))
_dispatcher_for_tensor_scatter_add = tensor_scatter_add._tf_type_based_dispatcher.Dispatch


def tensor_scatter_add_eager_fallback(tensor: Annotated[Any, TV_TensorScatterAdd_T], indices: Annotated[Any, TV_TensorScatterAdd_Tindices], updates: Annotated[Any, TV_TensorScatterAdd_T], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_TensorScatterAdd_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([tensor, updates], ctx, [])
  (tensor, updates) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [tensor, indices, updates]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"TensorScatterAdd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorScatterAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorScatterMax_T = TypeVar("TV_TensorScatterMax_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorScatterMax_Tindices = TypeVar("TV_TensorScatterMax_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_scatter_nd_max')
def tensor_scatter_max(tensor: Annotated[Any, TV_TensorScatterMax_T], indices: Annotated[Any, TV_TensorScatterMax_Tindices], updates: Annotated[Any, TV_TensorScatterMax_T], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_TensorScatterMax_T]:
  r"""Apply a sparse update to a tensor taking the element-wise maximum.

  Returns a new tensor copied from `tensor` whose values are element-wise maximum between
  tensor and updates according to the indices.

  >>> tensor = [0, 0, 0, 0, 0, 0, 0, 0]
  >>> indices = [[1], [4], [5]]
  >>> updates = [1, -1, 1]
  >>> tf.tensor_scatter_nd_max(tensor, indices, updates).numpy()
  array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int32)

  Refer to `tf.tensor_scatter_nd_update` for more details.

  Args:
    tensor: A `Tensor`. Tensor to update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorScatterMax", name, tensor, indices, updates,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_scatter_max(
          (tensor, indices, updates, bad_indices_policy, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_scatter_max_eager_fallback(
          tensor, indices, updates, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_scatter_max, (), dict(tensor=tensor, indices=indices,
                                         updates=updates,
                                         bad_indices_policy=bad_indices_policy,
                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_scatter_max(
        (tensor, indices, updates, bad_indices_policy, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorScatterMax", tensor=tensor, indices=indices, updates=updates,
                            bad_indices_policy=bad_indices_policy, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_scatter_max, (), dict(tensor=tensor, indices=indices,
                                       updates=updates,
                                       bad_indices_policy=bad_indices_policy,
                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorScatterMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorScatterMax = tf_export("raw_ops.TensorScatterMax")(_ops.to_raw_op(tensor_scatter_max))
_dispatcher_for_tensor_scatter_max = tensor_scatter_max._tf_type_based_dispatcher.Dispatch


def tensor_scatter_max_eager_fallback(tensor: Annotated[Any, TV_TensorScatterMax_T], indices: Annotated[Any, TV_TensorScatterMax_Tindices], updates: Annotated[Any, TV_TensorScatterMax_T], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_TensorScatterMax_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([tensor, updates], ctx, [])
  (tensor, updates) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [tensor, indices, updates]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"TensorScatterMax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorScatterMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorScatterMin_T = TypeVar("TV_TensorScatterMin_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorScatterMin_Tindices = TypeVar("TV_TensorScatterMin_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_scatter_nd_min')
def tensor_scatter_min(tensor: Annotated[Any, TV_TensorScatterMin_T], indices: Annotated[Any, TV_TensorScatterMin_Tindices], updates: Annotated[Any, TV_TensorScatterMin_T], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_TensorScatterMin_T]:
  r"""TODO: add doc.

  Args:
    tensor: A `Tensor`. Tensor to update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorScatterMin", name, tensor, indices, updates,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_scatter_min(
          (tensor, indices, updates, bad_indices_policy, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_scatter_min_eager_fallback(
          tensor, indices, updates, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_scatter_min, (), dict(tensor=tensor, indices=indices,
                                         updates=updates,
                                         bad_indices_policy=bad_indices_policy,
                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_scatter_min(
        (tensor, indices, updates, bad_indices_policy, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorScatterMin", tensor=tensor, indices=indices, updates=updates,
                            bad_indices_policy=bad_indices_policy, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_scatter_min, (), dict(tensor=tensor, indices=indices,
                                       updates=updates,
                                       bad_indices_policy=bad_indices_policy,
                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorScatterMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorScatterMin = tf_export("raw_ops.TensorScatterMin")(_ops.to_raw_op(tensor_scatter_min))
_dispatcher_for_tensor_scatter_min = tensor_scatter_min._tf_type_based_dispatcher.Dispatch


def tensor_scatter_min_eager_fallback(tensor: Annotated[Any, TV_TensorScatterMin_T], indices: Annotated[Any, TV_TensorScatterMin_Tindices], updates: Annotated[Any, TV_TensorScatterMin_T], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_TensorScatterMin_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([tensor, updates], ctx, [])
  (tensor, updates) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [tensor, indices, updates]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"TensorScatterMin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorScatterMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorScatterSub_T = TypeVar("TV_TensorScatterSub_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorScatterSub_Tindices = TypeVar("TV_TensorScatterSub_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_scatter_nd_sub', v1=['tensor_scatter_nd_sub', 'tensor_scatter_sub'])
@deprecated_endpoints('tensor_scatter_sub')
def tensor_scatter_sub(tensor: Annotated[Any, TV_TensorScatterSub_T], indices: Annotated[Any, TV_TensorScatterSub_Tindices], updates: Annotated[Any, TV_TensorScatterSub_T], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_TensorScatterSub_T]:
  r"""Subtracts sparse `updates` from an existing tensor according to `indices`.

  This operation creates a new tensor by subtracting sparse `updates` from the
  passed in `tensor`.
  This operation is very similar to `tf.scatter_nd_sub`, except that the updates
  are subtracted from an existing tensor (as opposed to a variable). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.  The last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.  `updates` is a tensor with shape

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of tensor_scatter_sub is to subtract individual elements
  from a tensor by index. For example, say we want to insert 4 scattered elements
  in a rank-1 tensor with 8 elements.

  In Python, this scatter subtract operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      tensor = tf.ones([8], dtype=tf.int32)
      updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
      print(updated)
  ```

  The resulting tensor would look like this:

      [1, -10, 1, -9, -8, 1, 1, -11]

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  In Python, this scatter add operation would look like this:

  ```python
      indices = tf.constant([[0], [2]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      tensor = tf.ones([4, 4, 4],dtype=tf.int32)
      updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
      print(updated)
  ```

  The resulting tensor would look like this:

      [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
       [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorScatterSub", name, tensor, indices, updates,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_scatter_sub(
          (tensor, indices, updates, bad_indices_policy, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_scatter_sub_eager_fallback(
          tensor, indices, updates, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_scatter_sub, (), dict(tensor=tensor, indices=indices,
                                         updates=updates,
                                         bad_indices_policy=bad_indices_policy,
                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_scatter_sub(
        (tensor, indices, updates, bad_indices_policy, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorScatterSub", tensor=tensor, indices=indices, updates=updates,
                            bad_indices_policy=bad_indices_policy, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_scatter_sub, (), dict(tensor=tensor, indices=indices,
                                       updates=updates,
                                       bad_indices_policy=bad_indices_policy,
                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorScatterSub", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorScatterSub = tf_export("raw_ops.TensorScatterSub")(_ops.to_raw_op(tensor_scatter_sub))
_dispatcher_for_tensor_scatter_sub = tensor_scatter_sub._tf_type_based_dispatcher.Dispatch


def tensor_scatter_sub_eager_fallback(tensor: Annotated[Any, TV_TensorScatterSub_T], indices: Annotated[Any, TV_TensorScatterSub_Tindices], updates: Annotated[Any, TV_TensorScatterSub_T], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_TensorScatterSub_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([tensor, updates], ctx, [])
  (tensor, updates) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [tensor, indices, updates]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"TensorScatterSub", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorScatterSub", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorScatterUpdate_T = TypeVar("TV_TensorScatterUpdate_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorScatterUpdate_Tindices = TypeVar("TV_TensorScatterUpdate_Tindices", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.UInt16)

def tensor_scatter_update(tensor: Annotated[Any, TV_TensorScatterUpdate_T], indices: Annotated[Any, TV_TensorScatterUpdate_Tindices], updates: Annotated[Any, TV_TensorScatterUpdate_T], bad_indices_policy:str="", name=None) -> Annotated[Any, TV_TensorScatterUpdate_T]:
  r"""Scatter `updates` into an existing tensor according to `indices`.

  This operation creates a new tensor by applying sparse `updates` to the passed
  in `tensor`.
  This operation is very similar to `tf.scatter_nd`, except that the updates are
  scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  If `indices` contains duplicates, then we pick the last update for the index.

  **WARNING**: There are some GPU specific semantics for this operation.
  - If an out of bound index is found, the index is ignored.
  - The order in which updates are applied is nondeterministic, so the output
  will be nondeterministic if `indices` contains duplicates.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.

  * `indices` must have at least 2 axes: `(num_updates, index_depth)`.
  * The last axis of `indices` is how deep to index into `tensor` so  this index
    depth must be less than the rank of `tensor`: `indices.shape[-1] <= tensor.ndim`

  if `indices.shape[-1] = tensor.rank` this Op indexes and updates scalar elements.
  if `indices.shape[-1] < tensor.rank` it indexes and updates slices of the input
  `tensor`.

  Each `update` has a rank of `tensor.rank - indices.shape[-1]`.
  The overall shape of `updates` is:

  ```
  indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
  ```

  If `indices` contains any out-of-bound indices, depending on
  `bad_indices_policy`, the op will either return an error or ignore the
  out-of-bound indices. `bad_indices_policy` can be one of the following values:
  1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
     historically on CPU and GPU we handle errors in different ways, and for
     backward compatibility we keep the default behavior.
  2. "ERROR": raises error; GPU does not support this value.
  3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.

  For usage examples see the python [tf.tensor_scatter_nd_update](
  https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`, `uint16`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    bad_indices_policy: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorScatterUpdate", name, tensor, indices, updates,
        "bad_indices_policy", bad_indices_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_scatter_update_eager_fallback(
          tensor, indices, updates, bad_indices_policy=bad_indices_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorScatterUpdate", tensor=tensor, indices=indices,
                               updates=updates,
                               bad_indices_policy=bad_indices_policy,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "bad_indices_policy",
              _op.get_attr("bad_indices_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorScatterUpdate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorScatterUpdate = tf_export("raw_ops.TensorScatterUpdate")(_ops.to_raw_op(tensor_scatter_update))


def tensor_scatter_update_eager_fallback(tensor: Annotated[Any, TV_TensorScatterUpdate_T], indices: Annotated[Any, TV_TensorScatterUpdate_Tindices], updates: Annotated[Any, TV_TensorScatterUpdate_T], bad_indices_policy: str, name, ctx) -> Annotated[Any, TV_TensorScatterUpdate_T]:
  if bad_indices_policy is None:
    bad_indices_policy = ""
  bad_indices_policy = _execute.make_str(bad_indices_policy, "bad_indices_policy")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([tensor, updates], ctx, [])
  (tensor, updates) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint16, ])
  _inputs_flat = [tensor, indices, updates]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "bad_indices_policy",
  bad_indices_policy)
  _result = _execute.execute(b"TensorScatterUpdate", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorScatterUpdate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorStridedSliceUpdate_T = TypeVar("TV_TensorStridedSliceUpdate_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorStridedSliceUpdate_Index = TypeVar("TV_TensorStridedSliceUpdate_Index", _atypes.Int32, _atypes.Int64)

def tensor_strided_slice_update(input: Annotated[Any, TV_TensorStridedSliceUpdate_T], begin: Annotated[Any, TV_TensorStridedSliceUpdate_Index], end: Annotated[Any, TV_TensorStridedSliceUpdate_Index], strides: Annotated[Any, TV_TensorStridedSliceUpdate_Index], value: Annotated[Any, TV_TensorStridedSliceUpdate_T], begin_mask:int=0, end_mask:int=0, ellipsis_mask:int=0, new_axis_mask:int=0, shrink_axis_mask:int=0, name=None) -> Annotated[Any, TV_TensorStridedSliceUpdate_T]:
  r"""Assign `value` to the sliced l-value reference of `input`.

  The values of `value` are assigned to the positions in the tensor `input` that
  are selected by the slice parameters. The slice parameters `begin` `end`
  `strides` etc. work exactly as in `StridedSlice`.

  NOTE this op currently does not support broadcasting and so `value`'s shape
  must be exactly the shape produced by the slice of `input`.

  Args:
    input: A `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    end: A `Tensor`. Must have the same type as `begin`.
    strides: A `Tensor`. Must have the same type as `begin`.
    value: A `Tensor`. Must have the same type as `input`.
    begin_mask: An optional `int`. Defaults to `0`.
    end_mask: An optional `int`. Defaults to `0`.
    ellipsis_mask: An optional `int`. Defaults to `0`.
    new_axis_mask: An optional `int`. Defaults to `0`.
    shrink_axis_mask: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorStridedSliceUpdate", name, input, begin, end, strides,
        value, "begin_mask", begin_mask, "end_mask", end_mask,
        "ellipsis_mask", ellipsis_mask, "new_axis_mask", new_axis_mask,
        "shrink_axis_mask", shrink_axis_mask)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_strided_slice_update_eager_fallback(
          input, begin, end, strides, value, begin_mask=begin_mask,
          end_mask=end_mask, ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorStridedSliceUpdate", input=input, begin=begin, end=end,
                                    strides=strides, value=value,
                                    begin_mask=begin_mask, end_mask=end_mask,
                                    ellipsis_mask=ellipsis_mask,
                                    new_axis_mask=new_axis_mask,
                                    shrink_axis_mask=shrink_axis_mask,
                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Index",
              _op._get_attr_type("Index"), "begin_mask",
              _op._get_attr_int("begin_mask"), "end_mask",
              _op._get_attr_int("end_mask"), "ellipsis_mask",
              _op._get_attr_int("ellipsis_mask"), "new_axis_mask",
              _op._get_attr_int("new_axis_mask"), "shrink_axis_mask",
              _op._get_attr_int("shrink_axis_mask"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorStridedSliceUpdate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorStridedSliceUpdate = tf_export("raw_ops.TensorStridedSliceUpdate")(_ops.to_raw_op(tensor_strided_slice_update))


def tensor_strided_slice_update_eager_fallback(input: Annotated[Any, TV_TensorStridedSliceUpdate_T], begin: Annotated[Any, TV_TensorStridedSliceUpdate_Index], end: Annotated[Any, TV_TensorStridedSliceUpdate_Index], strides: Annotated[Any, TV_TensorStridedSliceUpdate_Index], value: Annotated[Any, TV_TensorStridedSliceUpdate_T], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx) -> Annotated[Any, TV_TensorStridedSliceUpdate_T]:
  if begin_mask is None:
    begin_mask = 0
  begin_mask = _execute.make_int(begin_mask, "begin_mask")
  if end_mask is None:
    end_mask = 0
  end_mask = _execute.make_int(end_mask, "end_mask")
  if ellipsis_mask is None:
    ellipsis_mask = 0
  ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
  if new_axis_mask is None:
    new_axis_mask = 0
  new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
  if shrink_axis_mask is None:
    shrink_axis_mask = 0
  shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, value], ctx, [])
  (input, value) = _inputs_T
  _attr_Index, _inputs_Index = _execute.args_to_matching_eager([begin, end, strides], ctx, [_dtypes.int32, _dtypes.int64, ])
  (begin, end, strides) = _inputs_Index
  _inputs_flat = [input, begin, end, strides, value]
  _attrs = ("T", _attr_T, "Index", _attr_Index, "begin_mask", begin_mask,
  "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask",
  new_axis_mask, "shrink_axis_mask", shrink_axis_mask)
  _result = _execute.execute(b"TensorStridedSliceUpdate", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorStridedSliceUpdate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Tile_T = TypeVar("TV_Tile_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Tile_Tmultiples = TypeVar("TV_Tile_Tmultiples", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tile', v1=['tile', 'manip.tile'])
@deprecated_endpoints('manip.tile')
def tile(input: Annotated[Any, TV_Tile_T], multiples: Annotated[Any, TV_Tile_Tmultiples], name=None) -> Annotated[Any, TV_Tile_T]:
  r"""Constructs a tensor by tiling a given tensor.

  This operation creates a new tensor by replicating `input` `multiples` times.
  The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
  and the values of `input` are replicated `multiples[i]` times along the 'i'th
  dimension. For example, tiling `[a b c d]` by `[2]` produces
  `[a b c d a b c d]`.

  >>> a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
  >>> b = tf.constant([1,2], tf.int32)
  >>> tf.tile(a, b)
  <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
  array([[1, 2, 3, 1, 2, 3],
         [4, 5, 6, 4, 5, 6]], dtype=int32)>
  >>> c = tf.constant([2,1], tf.int32)
  >>> tf.tile(a, c)
  <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
  array([[1, 2, 3],
         [4, 5, 6],
         [1, 2, 3],
         [4, 5, 6]], dtype=int32)>
  >>> d = tf.constant([2,2], tf.int32)
  >>> tf.tile(a, d)
  <tf.Tensor: shape=(4, 6), dtype=int32, numpy=
  array([[1, 2, 3, 1, 2, 3],
         [4, 5, 6, 4, 5, 6],
         [1, 2, 3, 1, 2, 3],
         [4, 5, 6, 4, 5, 6]], dtype=int32)>

  Args:
    input: A `Tensor`. Can be of any rank.
    multiples: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D. Length must be the same as the number of dimensions in `input`
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Tile", name, input, multiples)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tile(
          (input, multiples, name,), None)
      if _result is not NotImplemented:
        return _result
      return tile_eager_fallback(
          input, multiples, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tile, (), dict(input=input, multiples=multiples, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tile(
        (input, multiples, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Tile", input=input, multiples=multiples, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tile, (), dict(input=input, multiples=multiples, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tmultiples",
              _op._get_attr_type("Tmultiples"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Tile", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Tile = tf_export("raw_ops.Tile")(_ops.to_raw_op(tile))
_dispatcher_for_tile = tile._tf_type_based_dispatcher.Dispatch


def tile_eager_fallback(input: Annotated[Any, TV_Tile_T], multiples: Annotated[Any, TV_Tile_Tmultiples], name, ctx) -> Annotated[Any, TV_Tile_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tmultiples, (multiples,) = _execute.args_to_matching_eager([multiples], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, multiples]
  _attrs = ("T", _attr_T, "Tmultiples", _attr_Tmultiples)
  _result = _execute.execute(b"Tile", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Tile", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TileGrad_T = TypeVar("TV_TileGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tile_grad(input: Annotated[Any, TV_TileGrad_T], multiples: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_TileGrad_T]:
  r"""Returns the gradient of `Tile`.

  Since `Tile` takes an input and repeats the input `multiples` times
  along each dimension, `TileGrad` takes in `multiples` and aggregates
  each repeated tile of `input` into `output`.

  Args:
    input: A `Tensor`.
    multiples: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TileGrad", name, input, multiples)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tile_grad_eager_fallback(
          input, multiples, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TileGrad", input=input, multiples=multiples, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TileGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TileGrad = tf_export("raw_ops.TileGrad")(_ops.to_raw_op(tile_grad))


def tile_grad_eager_fallback(input: Annotated[Any, TV_TileGrad_T], multiples: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_TileGrad_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  multiples = _ops.convert_to_tensor(multiples, _dtypes.int32)
  _inputs_flat = [input, multiples]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TileGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TileGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Transpose_T = TypeVar("TV_Transpose_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Transpose_Tperm = TypeVar("TV_Transpose_Tperm", _atypes.Int32, _atypes.Int64)

def transpose(x: Annotated[Any, TV_Transpose_T], perm: Annotated[Any, TV_Transpose_Tperm], name=None) -> Annotated[Any, TV_Transpose_T]:
  r"""Shuffle dimensions of x according to a permutation.

  The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

  Args:
    x: A `Tensor`.
    perm: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Transpose", name, x, perm)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return transpose_eager_fallback(
          x, perm, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Transpose", x=x, perm=perm, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tperm",
              _op._get_attr_type("Tperm"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Transpose", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Transpose = tf_export("raw_ops.Transpose")(_ops.to_raw_op(transpose))


def transpose_eager_fallback(x: Annotated[Any, TV_Transpose_T], perm: Annotated[Any, TV_Transpose_Tperm], name, ctx) -> Annotated[Any, TV_Transpose_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _attr_Tperm, (perm,) = _execute.args_to_matching_eager([perm], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x, perm]
  _attrs = ("T", _attr_T, "Tperm", _attr_Tperm)
  _result = _execute.execute(b"Transpose", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Transpose", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_UniqueOutput = collections.namedtuple(
    "Unique",
    ["y", "idx"])


TV_Unique_T = TypeVar("TV_Unique_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Unique_out_idx = TypeVar("TV_Unique_out_idx", _atypes.Int32, _atypes.Int64)

def unique(x: Annotated[Any, TV_Unique_T], out_idx:TV_Unique_out_idx=_dtypes.int32, name=None):
  r"""Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`; `x` does not need to be sorted.
  This operation also returns a tensor `idx` the same size as `x` that contains
  the index of each value of `x` in the unique output `y`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  Examples:

  ```
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx = unique(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  ```

  ```
  # tensor 'x' is [4, 5, 1, 2, 3, 3, 4, 5]
  y, idx = unique(x)
  y ==> [4, 5, 1, 2, 3]
  idx ==> [0, 1, 2, 3, 4, 4, 0, 1]
  ```

  Args:
    x: A `Tensor`. 1-D.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Unique", name, x, "out_idx", out_idx)
      _result = _UniqueOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unique_eager_fallback(
          x, out_idx=out_idx, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Unique", x=x, out_idx=out_idx, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_idx",
              _op._get_attr_type("out_idx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Unique", _inputs_flat, _attrs, _result)
  _result = _UniqueOutput._make(_result)
  return _result

Unique = tf_export("raw_ops.Unique")(_ops.to_raw_op(unique))


def unique_eager_fallback(x: Annotated[Any, TV_Unique_T], out_idx: TV_Unique_out_idx, name, ctx):
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T, "out_idx", out_idx)
  _result = _execute.execute(b"Unique", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Unique", _inputs_flat, _attrs, _result)
  _result = _UniqueOutput._make(_result)
  return _result

_UniqueV2Output = collections.namedtuple(
    "UniqueV2",
    ["y", "idx"])


TV_UniqueV2_T = TypeVar("TV_UniqueV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_UniqueV2_Taxis = TypeVar("TV_UniqueV2_Taxis", _atypes.Int32, _atypes.Int64)
TV_UniqueV2_out_idx = TypeVar("TV_UniqueV2_out_idx", _atypes.Int32, _atypes.Int64)

def unique_v2(x: Annotated[Any, TV_UniqueV2_T], axis: Annotated[Any, TV_UniqueV2_Taxis], out_idx:TV_UniqueV2_out_idx=_dtypes.int32, name=None):
  r"""Finds unique elements along an axis of a tensor.

  This operation either returns a tensor `y` containing unique elements
  along the `axis` of a tensor. The returned unique elements is sorted
  in the same order as they occur along `axis` in `x`.
  This operation also returns a tensor `idx` that is the same size as
  the number of the elements in `x` along the `axis` dimension. It
  contains the index in the unique output `y`.
  In other words, for an `1-D` tensor `x` with `axis = None:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx = unique(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  ```

  For an `2-D` tensor `x` with `axis = 0`:

  ```
  # tensor 'x' is [[1, 0, 0],
  #                [1, 0, 0],
  #                [2, 0, 0]]
  y, idx = unique(x, axis=0)
  y ==> [[1, 0, 0],
         [2, 0, 0]]
  idx ==> [0, 0, 1]
  ```

  For an `2-D` tensor `x` with `axis = 1`:

  ```
  # tensor 'x' is [[1, 0, 0],
  #                [1, 0, 0],
  #                [2, 0, 0]]
  y, idx = unique(x, axis=1)
  y ==> [[1, 0],
         [1, 0],
         [2, 0]]
  idx ==> [0, 1, 1]
  ```

  Args:
    x: A `Tensor`. A `Tensor`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: None). The axis of the Tensor to
      find the unique elements.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniqueV2", name, x, axis, "out_idx", out_idx)
      _result = _UniqueV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unique_v2_eager_fallback(
          x, axis, out_idx=out_idx, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniqueV2", x=x, axis=axis, out_idx=out_idx, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Taxis",
              _op._get_attr_type("Taxis"), "out_idx",
              _op._get_attr_type("out_idx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniqueV2", _inputs_flat, _attrs, _result)
  _result = _UniqueV2Output._make(_result)
  return _result

UniqueV2 = tf_export("raw_ops.UniqueV2")(_ops.to_raw_op(unique_v2))


def unique_v2_eager_fallback(x: Annotated[Any, TV_UniqueV2_T], axis: Annotated[Any, TV_UniqueV2_Taxis], out_idx: TV_UniqueV2_out_idx, name, ctx):
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _attr_Taxis, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [x, axis]
  _attrs = ("T", _attr_T, "Taxis", _attr_Taxis, "out_idx", out_idx)
  _result = _execute.execute(b"UniqueV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniqueV2", _inputs_flat, _attrs, _result)
  _result = _UniqueV2Output._make(_result)
  return _result

_UniqueWithCountsOutput = collections.namedtuple(
    "UniqueWithCounts",
    ["y", "idx", "count"])


TV_UniqueWithCounts_T = TypeVar("TV_UniqueWithCounts_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_UniqueWithCounts_out_idx = TypeVar("TV_UniqueWithCounts_out_idx", _atypes.Int32, _atypes.Int64)

def unique_with_counts(x: Annotated[Any, TV_UniqueWithCounts_T], out_idx:TV_UniqueWithCounts_out_idx=_dtypes.int32, name=None):
  r"""Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. Finally, it returns a third tensor `count` that
  contains the count of each element of `y` in `x`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx, count = unique_with_counts(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  count ==> [2, 1, 3, 1, 2]
  ```

  Args:
    x: A `Tensor`. 1-D.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx, count).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
    count: A `Tensor` of type `out_idx`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniqueWithCounts", name, x, "out_idx", out_idx)
      _result = _UniqueWithCountsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unique_with_counts_eager_fallback(
          x, out_idx=out_idx, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniqueWithCounts", x=x, out_idx=out_idx, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_idx",
              _op._get_attr_type("out_idx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniqueWithCounts", _inputs_flat, _attrs, _result)
  _result = _UniqueWithCountsOutput._make(_result)
  return _result

UniqueWithCounts = tf_export("raw_ops.UniqueWithCounts")(_ops.to_raw_op(unique_with_counts))


def unique_with_counts_eager_fallback(x: Annotated[Any, TV_UniqueWithCounts_T], out_idx: TV_UniqueWithCounts_out_idx, name, ctx):
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T, "out_idx", out_idx)
  _result = _execute.execute(b"UniqueWithCounts", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniqueWithCounts", _inputs_flat, _attrs, _result)
  _result = _UniqueWithCountsOutput._make(_result)
  return _result

_UniqueWithCountsV2Output = collections.namedtuple(
    "UniqueWithCountsV2",
    ["y", "idx", "count"])


TV_UniqueWithCountsV2_T = TypeVar("TV_UniqueWithCountsV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_UniqueWithCountsV2_Taxis = TypeVar("TV_UniqueWithCountsV2_Taxis", _atypes.Int32, _atypes.Int64)
TV_UniqueWithCountsV2_out_idx = TypeVar("TV_UniqueWithCountsV2_out_idx", _atypes.Int32, _atypes.Int64)

def unique_with_counts_v2(x: Annotated[Any, TV_UniqueWithCountsV2_T], axis: Annotated[Any, TV_UniqueWithCountsV2_Taxis], out_idx:TV_UniqueWithCountsV2_out_idx=_dtypes.int32, name=None):
  r"""Finds unique elements along an axis of a tensor.

  This operation either returns a tensor `y` containing unique elements
  along the `axis` of a tensor. The returned unique elements is sorted
  in the same order as they occur along `axis` in `x`.
  This operation also returns a tensor `idx` and a tensor `count`
  that are the same size as the number of the elements in `x` along the
  `axis` dimension. The `idx` contains the index in the unique output `y`
  and the `count` contains the count in the unique output `y`.
  In other words, for an `1-D` tensor `x` with `axis = None:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```
  x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
  y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis = [0])
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  count ==> [2, 1, 3, 1, 2]
  ```

  For a `2-D` tensor `x` with `axis = 0`:

  ```
  x = tf.constant([[1, 0, 0],
                  [1, 0, 0],
                  [2, 0, 0]])
  y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[0])
  y ==> [[1, 0, 0],
         [2, 0, 0]]
  idx ==> [0, 0, 1]
  count ==> [2, 1]
  ```

  For a `2-D` tensor `x` with `axis = 1`:

  ```
  x = tf.constant([[1, 0, 0],
                  [1, 0, 0],
                  [2, 0, 0]])
  y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[1])
  y ==> [[1, 0],
         [1, 0],
         [2, 0]]
  idx ==> [0, 1, 1]
  count ==> [1, 2]
  ```

  Args:
    x: A `Tensor`. A `Tensor`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: None). The axis of the Tensor to
      find the unique elements.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx, count).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
    count: A `Tensor` of type `out_idx`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniqueWithCountsV2", name, x, axis, "out_idx", out_idx)
      _result = _UniqueWithCountsV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unique_with_counts_v2_eager_fallback(
          x, axis, out_idx=out_idx, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniqueWithCountsV2", x=x, axis=axis, out_idx=out_idx, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Taxis",
              _op._get_attr_type("Taxis"), "out_idx",
              _op._get_attr_type("out_idx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniqueWithCountsV2", _inputs_flat, _attrs, _result)
  _result = _UniqueWithCountsV2Output._make(_result)
  return _result

UniqueWithCountsV2 = tf_export("raw_ops.UniqueWithCountsV2")(_ops.to_raw_op(unique_with_counts_v2))


def unique_with_counts_v2_eager_fallback(x: Annotated[Any, TV_UniqueWithCountsV2_T], axis: Annotated[Any, TV_UniqueWithCountsV2_Taxis], out_idx: TV_UniqueWithCountsV2_out_idx, name, ctx):
  if out_idx is None:
    out_idx = _dtypes.int32
  out_idx = _execute.make_type(out_idx, "out_idx")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _attr_Taxis, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [x, axis]
  _attrs = ("T", _attr_T, "Taxis", _attr_Taxis, "out_idx", out_idx)
  _result = _execute.execute(b"UniqueWithCountsV2", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniqueWithCountsV2", _inputs_flat, _attrs, _result)
  _result = _UniqueWithCountsV2Output._make(_result)
  return _result


TV_Unpack_T = TypeVar("TV_Unpack_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def unpack(value: Annotated[Any, TV_Unpack_T], num: int, axis:int=0, name=None):
  r"""Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
  For example, given a tensor of shape `(A, B, C, D)`;

  If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
    and each tensor in `output` will have shape `(B, C, D)`. (Note that the
    dimension unpacked along is gone, unlike `split`).

  If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
    and each tensor in `output` will have shape `(A, C, D)`.
  Etc.

  This is the opposite of `pack`.

  Args:
    value: A `Tensor`.
      1-D or higher, with `axis` dimension size equal to `num`.
    num: An `int` that is `>= 0`.
    axis: An optional `int`. Defaults to `0`.
      Dimension along which to unpack.  Negative values wrap around, so the
      valid range is `[-R, R)`.
    name: A name for the operation (optional).

  Returns:
    A list of `num` `Tensor` objects with the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Unpack", name, value, "num", num, "axis", axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unpack_eager_fallback(
          value, num=num, axis=axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num = _execute.make_int(num, "num")
  if axis is None:
    axis = 0
  axis = _execute.make_int(axis, "axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Unpack", value=value, num=num, axis=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num", _op._get_attr_int("num"), "T", _op._get_attr_type("T"),
              "axis", _op._get_attr_int("axis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Unpack", _inputs_flat, _attrs, _result)
  return _result

Unpack = tf_export("raw_ops.Unpack")(_ops.to_raw_op(unpack))


def unpack_eager_fallback(value: Annotated[Any, TV_Unpack_T], num: int, axis: int, name, ctx):
  num = _execute.make_int(num, "num")
  if axis is None:
    axis = 0
  axis = _execute.make_int(axis, "axis")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _inputs_flat = [value]
  _attrs = ("num", num, "T", _attr_T, "axis", axis)
  _result = _execute.execute(b"Unpack", num, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Unpack", _inputs_flat, _attrs, _result)
  return _result


TV_UnravelIndex_Tidx = TypeVar("TV_UnravelIndex_Tidx", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('unravel_index')
def unravel_index(indices: Annotated[Any, TV_UnravelIndex_Tidx], dims: Annotated[Any, TV_UnravelIndex_Tidx], name=None) -> Annotated[Any, TV_UnravelIndex_Tidx]:
  r"""Converts an array of flat indices into a tuple of coordinate arrays.

  
  Example:

  ```
  y = tf.unravel_index(indices=[2, 5, 7], dims=[3, 3])
  # 'dims' represent a hypothetical (3, 3) tensor of indices:
  # [[0, 1, *2*],
  #  [3, 4, *5*],
  #  [6, *7*, 8]]
  # For each entry from 'indices', this operation returns
  # its coordinates (marked with '*'), such as
  # 2 ==> (0, 2)
  # 5 ==> (1, 2)
  # 7 ==> (2, 1)
  y ==> [[0, 1, 2], [2, 2, 1]]
  ```

  @compatibility(numpy)
  Equivalent to np.unravel_index
  @end_compatibility

  Args:
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An 0-D or 1-D `int` Tensor whose elements are indices into the
      flattened version of an array of dimensions dims.
    dims: A `Tensor`. Must have the same type as `indices`.
      An 1-D `int` Tensor. The shape of the array to use for unraveling
      indices.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `indices`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnravelIndex", name, indices, dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unravel_index(
          (indices, dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return unravel_index_eager_fallback(
          indices, dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unravel_index, (), dict(indices=indices, dims=dims, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unravel_index(
        (indices, dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnravelIndex", indices=indices, dims=dims, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unravel_index, (), dict(indices=indices, dims=dims, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnravelIndex", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnravelIndex = tf_export("raw_ops.UnravelIndex")(_ops.to_raw_op(unravel_index))
_dispatcher_for_unravel_index = unravel_index._tf_type_based_dispatcher.Dispatch


def unravel_index_eager_fallback(indices: Annotated[Any, TV_UnravelIndex_Tidx], dims: Annotated[Any, TV_UnravelIndex_Tidx], name, ctx) -> Annotated[Any, TV_UnravelIndex_Tidx]:
  _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([indices, dims], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  (indices, dims) = _inputs_Tidx
  _inputs_flat = [indices, dims]
  _attrs = ("Tidx", _attr_Tidx)
  _result = _execute.execute(b"UnravelIndex", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnravelIndex", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UpperBound_T = TypeVar("TV_UpperBound_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_UpperBound_out_type = TypeVar("TV_UpperBound_out_type", _atypes.Int32, _atypes.Int64)

def upper_bound(sorted_inputs: Annotated[Any, TV_UpperBound_T], values: Annotated[Any, TV_UpperBound_T], out_type:TV_UpperBound_out_type=_dtypes.int32, name=None) -> Annotated[Any, TV_UpperBound_out_type]:
  r"""Applies upper_bound(sorted_search_values, values) along each row.

  Each set of rows with the same index in (sorted_inputs, values) is treated
  independently.  The resulting row is the equivalent of calling
  `np.searchsorted(sorted_inputs, values, side='right')`.

  The result is not a global index to the entire
  `Tensor`, but rather just the index in the last dimension.

  A 2-D example:
    sorted_sequence = [[0, 3, 9, 9, 10],
                       [1, 2, 3, 4, 5]]
    values = [[2, 4, 9],
              [0, 2, 6]]

    result = UpperBound(sorted_sequence, values)

    result == [[1, 2, 4],
               [0, 2, 5]]

  Args:
    sorted_inputs: A `Tensor`. 2-D Tensor where each row is ordered.
    values: A `Tensor`. Must have the same type as `sorted_inputs`.
      2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
      the values that will be searched for in `sorted_search_values`.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UpperBound", name, sorted_inputs, values, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return upper_bound_eager_fallback(
          sorted_inputs, values, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UpperBound", sorted_inputs=sorted_inputs, values=values,
                      out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UpperBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UpperBound = tf_export("raw_ops.UpperBound")(_ops.to_raw_op(upper_bound))


def upper_bound_eager_fallback(sorted_inputs: Annotated[Any, TV_UpperBound_T], values: Annotated[Any, TV_UpperBound_T], out_type: TV_UpperBound_out_type, name, ctx) -> Annotated[Any, TV_UpperBound_out_type]:
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([sorted_inputs, values], ctx, [])
  (sorted_inputs, values) = _inputs_T
  _inputs_flat = [sorted_inputs, values]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"UpperBound", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UpperBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Where_T = TypeVar("TV_Where_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def where(condition: Annotated[Any, TV_Where_T], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns locations of nonzero / true values in a tensor.

  This operation returns the coordinates of true elements in `condition`. The
  coordinates are returned in a 2-D tensor where the first dimension (rows)
  represents the number of true elements, and the second dimension (columns)
  represents the coordinates of the true elements. Keep in mind, the shape of
  the output tensor can vary depending on how many true values there are in
  `condition`. Indices are output in row-major order.

  For example:

  ```
  # 'input' tensor is [[True, False]
  #                    [True, False]]
  # 'input' has two true values, so output has two coordinates.
  # 'input' has rank of 2, so coordinates have two indices.
  where(input) ==> [[0, 0],
                    [1, 0]]

  # `condition` tensor is [[[True, False]
  #                     [True, False]]
  #                    [[False, True]
  #                     [False, True]]
  #                    [[False, False]
  #                     [False, True]]]
  # 'input' has 5 true values, so output has 5 coordinates.
  # 'input' has rank of 3, so coordinates have three indices.
  where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]

  # `condition` tensor is [[[1.5,  0.0]
  #                     [-0.5, 0.0]]
  #                    [[0.0,  0.25]
  #                     [0.0,  0.75]]
  #                    [[0.0,  0.0]
  #                     [0.0,  0.01]]]
  # 'input' has 5 nonzero values, so output has 5 coordinates.
  # 'input' has rank of 3, so coordinates have three indices.
  where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]

  # `condition` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
  #                     [0.0 + 0.5j, 0.0  + 0.0j]]
  #                    [[0.0 + 0.0j, 0.25 + 1.5j]
  #                     [0.0 + 0.0j, 0.75 + 0.0j]]
  #                    [[0.0 + 0.0j, 0.0  + 0.0j]
  #                     [0.0 + 0.0j, 0.01 + 0.0j]]]
  # 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
  # 'input' has rank of 3, so coordinates have three indices.
  where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]
  ```

  Args:
    condition: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Where", name, condition)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return where_eager_fallback(
          condition, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Where", input=condition, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Where", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Where = tf_export("raw_ops.Where")(_ops.to_raw_op(where))


def where_eager_fallback(condition: Annotated[Any, TV_Where_T], name, ctx) -> Annotated[Any, _atypes.Int64]:
  _attr_T, (condition,) = _execute.args_to_matching_eager([condition], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ], _dtypes.bool)
  _inputs_flat = [condition]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Where", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Where", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ZerosLike_T = TypeVar("TV_ZerosLike_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def zeros_like(x: Annotated[Any, TV_ZerosLike_T], name=None) -> Annotated[Any, TV_ZerosLike_T]:
  r"""Returns a tensor of zeros with the same shape and type as x.

  Args:
    x: A `Tensor`. a tensor of type T.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ZerosLike", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return zeros_like_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ZerosLike", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ZerosLike", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ZerosLike = tf_export("raw_ops.ZerosLike")(_ops.to_raw_op(zeros_like))


def zeros_like_eager_fallback(x: Annotated[Any, TV_ZerosLike_T], name, ctx) -> Annotated[Any, TV_ZerosLike_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"ZerosLike", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ZerosLike", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


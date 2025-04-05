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
_BlockLSTMOutput = collections.namedtuple(
    "BlockLSTM",
    ["i", "cs", "f", "o", "ci", "co", "h"])


TV_BlockLSTM_T = TypeVar("TV_BlockLSTM_T", _atypes.Float32, _atypes.Half)

def block_lstm(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTM_T], cs_prev: Annotated[Any, TV_BlockLSTM_T], h_prev: Annotated[Any, TV_BlockLSTM_T], w: Annotated[Any, TV_BlockLSTM_T], wci: Annotated[Any, TV_BlockLSTM_T], wcf: Annotated[Any, TV_BlockLSTM_T], wco: Annotated[Any, TV_BlockLSTM_T], b: Annotated[Any, TV_BlockLSTM_T], forget_bias:float=1, cell_clip:float=3, use_peephole:bool=False, name=None):
  r"""Computes the LSTM cell forward propagation for all the time steps.

  This is equivalent to applying LSTMBlockCell in a loop, like so:

  ```python
  for x1 in unpack(x):
    i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
      x1, cs_prev, h_prev, w, wci, wcf, wco, b)
    cs_prev = cs1
    h_prev = h1
    i.append(i1)
    cs.append(cs1)
    f.append(f1)
    o.append(o1)
    ci.append(ci1)
    co.append(co1)
    h.append(h1)
  return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
  ```

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).

    i: A `Tensor`. Has the same type as `x`.
    cs: A `Tensor`. Has the same type as `x`.
    f: A `Tensor`. Has the same type as `x`.
    o: A `Tensor`. Has the same type as `x`.
    ci: A `Tensor`. Has the same type as `x`.
    co: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BlockLSTM", name, seq_len_max, x, cs_prev, h_prev, w, wci, wcf,
        wco, b, "forget_bias", forget_bias, "cell_clip", cell_clip,
        "use_peephole", use_peephole)
      _result = _BlockLSTMOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return block_lstm_eager_fallback(
          seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b,
          forget_bias=forget_bias, cell_clip=cell_clip,
          use_peephole=use_peephole, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if forget_bias is None:
    forget_bias = 1
  forget_bias = _execute.make_float(forget_bias, "forget_bias")
  if cell_clip is None:
    cell_clip = 3
  cell_clip = _execute.make_float(cell_clip, "cell_clip")
  if use_peephole is None:
    use_peephole = False
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BlockLSTM", seq_len_max=seq_len_max, x=x, cs_prev=cs_prev,
                     h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b,
                     forget_bias=forget_bias, cell_clip=cell_clip,
                     use_peephole=use_peephole, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("forget_bias", _op.get_attr("forget_bias"), "cell_clip",
              _op.get_attr("cell_clip"), "use_peephole",
              _op._get_attr_bool("use_peephole"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BlockLSTM", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMOutput._make(_result)
  return _result

BlockLSTM = tf_export("raw_ops.BlockLSTM")(_ops.to_raw_op(block_lstm))


def block_lstm_eager_fallback(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTM_T], cs_prev: Annotated[Any, TV_BlockLSTM_T], h_prev: Annotated[Any, TV_BlockLSTM_T], w: Annotated[Any, TV_BlockLSTM_T], wci: Annotated[Any, TV_BlockLSTM_T], wcf: Annotated[Any, TV_BlockLSTM_T], wco: Annotated[Any, TV_BlockLSTM_T], b: Annotated[Any, TV_BlockLSTM_T], forget_bias: float, cell_clip: float, use_peephole: bool, name, ctx):
  if forget_bias is None:
    forget_bias = 1
  forget_bias = _execute.make_float(forget_bias, "forget_bias")
  if cell_clip is None:
    cell_clip = 3
  cell_clip = _execute.make_float(cell_clip, "cell_clip")
  if use_peephole is None:
    use_peephole = False
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b], ctx, [_dtypes.half, _dtypes.float32, ])
  (x, cs_prev, h_prev, w, wci, wcf, wco, b) = _inputs_T
  seq_len_max = _ops.convert_to_tensor(seq_len_max, _dtypes.int64)
  _inputs_flat = [seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b]
  _attrs = ("forget_bias", forget_bias, "cell_clip", cell_clip,
  "use_peephole", use_peephole, "T", _attr_T)
  _result = _execute.execute(b"BlockLSTM", 7, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BlockLSTM", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMOutput._make(_result)
  return _result

_BlockLSTMGradOutput = collections.namedtuple(
    "BlockLSTMGrad",
    ["x_grad", "cs_prev_grad", "h_prev_grad", "w_grad", "wci_grad", "wcf_grad", "wco_grad", "b_grad"])


TV_BlockLSTMGrad_T = TypeVar("TV_BlockLSTMGrad_T", _atypes.Float32, _atypes.Half)

def block_lstm_grad(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTMGrad_T], cs_prev: Annotated[Any, TV_BlockLSTMGrad_T], h_prev: Annotated[Any, TV_BlockLSTMGrad_T], w: Annotated[Any, TV_BlockLSTMGrad_T], wci: Annotated[Any, TV_BlockLSTMGrad_T], wcf: Annotated[Any, TV_BlockLSTMGrad_T], wco: Annotated[Any, TV_BlockLSTMGrad_T], b: Annotated[Any, TV_BlockLSTMGrad_T], i: Annotated[Any, TV_BlockLSTMGrad_T], cs: Annotated[Any, TV_BlockLSTMGrad_T], f: Annotated[Any, TV_BlockLSTMGrad_T], o: Annotated[Any, TV_BlockLSTMGrad_T], ci: Annotated[Any, TV_BlockLSTMGrad_T], co: Annotated[Any, TV_BlockLSTMGrad_T], h: Annotated[Any, TV_BlockLSTMGrad_T], cs_grad: Annotated[Any, TV_BlockLSTMGrad_T], h_grad: Annotated[Any, TV_BlockLSTMGrad_T], use_peephole: bool, name=None):
  r"""Computes the LSTM cell backward propagation for the entire time sequence.

  This implementation is to be used in conjunction of LSTMBlock.

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`.
      The input gate over the whole time sequence.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh over the whole time sequence.
    f: A `Tensor`. Must have the same type as `x`.
      The forget gate over the whole time sequence.
    o: A `Tensor`. Must have the same type as `x`.
      The output gate over the whole time sequence.
    ci: A `Tensor`. Must have the same type as `x`.
      The cell input over the whole time sequence.
    co: A `Tensor`. Must have the same type as `x`.
      The cell after the tanh over the whole time sequence.
    h: A `Tensor`. Must have the same type as `x`.
      The output h vector over the whole time sequence.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad, b_grad).

    x_grad: A `Tensor`. Has the same type as `x`.
    cs_prev_grad: A `Tensor`. Has the same type as `x`.
    h_prev_grad: A `Tensor`. Has the same type as `x`.
    w_grad: A `Tensor`. Has the same type as `x`.
    wci_grad: A `Tensor`. Has the same type as `x`.
    wcf_grad: A `Tensor`. Has the same type as `x`.
    wco_grad: A `Tensor`. Has the same type as `x`.
    b_grad: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BlockLSTMGrad", name, seq_len_max, x, cs_prev, h_prev, w, wci,
        wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad, "use_peephole",
        use_peephole)
      _result = _BlockLSTMGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return block_lstm_grad_eager_fallback(
          seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o,
          ci, co, h, cs_grad, h_grad, use_peephole=use_peephole, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BlockLSTMGrad", seq_len_max=seq_len_max, x=x, cs_prev=cs_prev,
                         h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b,
                         i=i, cs=cs, f=f, o=o, ci=ci, co=co, h=h,
                         cs_grad=cs_grad, h_grad=h_grad,
                         use_peephole=use_peephole, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_peephole", _op._get_attr_bool("use_peephole"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BlockLSTMGrad", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMGradOutput._make(_result)
  return _result

BlockLSTMGrad = tf_export("raw_ops.BlockLSTMGrad")(_ops.to_raw_op(block_lstm_grad))


def block_lstm_grad_eager_fallback(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTMGrad_T], cs_prev: Annotated[Any, TV_BlockLSTMGrad_T], h_prev: Annotated[Any, TV_BlockLSTMGrad_T], w: Annotated[Any, TV_BlockLSTMGrad_T], wci: Annotated[Any, TV_BlockLSTMGrad_T], wcf: Annotated[Any, TV_BlockLSTMGrad_T], wco: Annotated[Any, TV_BlockLSTMGrad_T], b: Annotated[Any, TV_BlockLSTMGrad_T], i: Annotated[Any, TV_BlockLSTMGrad_T], cs: Annotated[Any, TV_BlockLSTMGrad_T], f: Annotated[Any, TV_BlockLSTMGrad_T], o: Annotated[Any, TV_BlockLSTMGrad_T], ci: Annotated[Any, TV_BlockLSTMGrad_T], co: Annotated[Any, TV_BlockLSTMGrad_T], h: Annotated[Any, TV_BlockLSTMGrad_T], cs_grad: Annotated[Any, TV_BlockLSTMGrad_T], h_grad: Annotated[Any, TV_BlockLSTMGrad_T], use_peephole: bool, name, ctx):
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad], ctx, [_dtypes.half, _dtypes.float32, ])
  (x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad) = _inputs_T
  seq_len_max = _ops.convert_to_tensor(seq_len_max, _dtypes.int64)
  _inputs_flat = [seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad]
  _attrs = ("use_peephole", use_peephole, "T", _attr_T)
  _result = _execute.execute(b"BlockLSTMGrad", 8, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BlockLSTMGrad", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMGradOutput._make(_result)
  return _result

_BlockLSTMGradV2Output = collections.namedtuple(
    "BlockLSTMGradV2",
    ["x_grad", "cs_prev_grad", "h_prev_grad", "w_grad", "wci_grad", "wcf_grad", "wco_grad", "b_grad"])


TV_BlockLSTMGradV2_T = TypeVar("TV_BlockLSTMGradV2_T", _atypes.Float32, _atypes.Half)

def block_lstm_grad_v2(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTMGradV2_T], cs_prev: Annotated[Any, TV_BlockLSTMGradV2_T], h_prev: Annotated[Any, TV_BlockLSTMGradV2_T], w: Annotated[Any, TV_BlockLSTMGradV2_T], wci: Annotated[Any, TV_BlockLSTMGradV2_T], wcf: Annotated[Any, TV_BlockLSTMGradV2_T], wco: Annotated[Any, TV_BlockLSTMGradV2_T], b: Annotated[Any, TV_BlockLSTMGradV2_T], i: Annotated[Any, TV_BlockLSTMGradV2_T], cs: Annotated[Any, TV_BlockLSTMGradV2_T], f: Annotated[Any, TV_BlockLSTMGradV2_T], o: Annotated[Any, TV_BlockLSTMGradV2_T], ci: Annotated[Any, TV_BlockLSTMGradV2_T], co: Annotated[Any, TV_BlockLSTMGradV2_T], h: Annotated[Any, TV_BlockLSTMGradV2_T], cs_grad: Annotated[Any, TV_BlockLSTMGradV2_T], h_grad: Annotated[Any, TV_BlockLSTMGradV2_T], use_peephole: bool, name=None):
  r"""Computes the LSTM cell backward propagation for the entire time sequence.

  This implementation is to be used in conjunction of BlockLSTMV2.

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`.
      The input gate over the whole time sequence.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh over the whole time sequence.
    f: A `Tensor`. Must have the same type as `x`.
      The forget gate over the whole time sequence.
    o: A `Tensor`. Must have the same type as `x`.
      The output gate over the whole time sequence.
    ci: A `Tensor`. Must have the same type as `x`.
      The cell input over the whole time sequence.
    co: A `Tensor`. Must have the same type as `x`.
      The cell after the tanh over the whole time sequence.
    h: A `Tensor`. Must have the same type as `x`.
      The output h vector over the whole time sequence.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad, b_grad).

    x_grad: A `Tensor`. Has the same type as `x`.
    cs_prev_grad: A `Tensor`. Has the same type as `x`.
    h_prev_grad: A `Tensor`. Has the same type as `x`.
    w_grad: A `Tensor`. Has the same type as `x`.
    wci_grad: A `Tensor`. Has the same type as `x`.
    wcf_grad: A `Tensor`. Has the same type as `x`.
    wco_grad: A `Tensor`. Has the same type as `x`.
    b_grad: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BlockLSTMGradV2", name, seq_len_max, x, cs_prev, h_prev, w,
        wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad,
        "use_peephole", use_peephole)
      _result = _BlockLSTMGradV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return block_lstm_grad_v2_eager_fallback(
          seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o,
          ci, co, h, cs_grad, h_grad, use_peephole=use_peephole, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BlockLSTMGradV2", seq_len_max=seq_len_max, x=x, cs_prev=cs_prev,
                           h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b,
                           i=i, cs=cs, f=f, o=o, ci=ci, co=co, h=h,
                           cs_grad=cs_grad, h_grad=h_grad,
                           use_peephole=use_peephole, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_peephole", _op._get_attr_bool("use_peephole"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BlockLSTMGradV2", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMGradV2Output._make(_result)
  return _result

BlockLSTMGradV2 = tf_export("raw_ops.BlockLSTMGradV2")(_ops.to_raw_op(block_lstm_grad_v2))


def block_lstm_grad_v2_eager_fallback(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTMGradV2_T], cs_prev: Annotated[Any, TV_BlockLSTMGradV2_T], h_prev: Annotated[Any, TV_BlockLSTMGradV2_T], w: Annotated[Any, TV_BlockLSTMGradV2_T], wci: Annotated[Any, TV_BlockLSTMGradV2_T], wcf: Annotated[Any, TV_BlockLSTMGradV2_T], wco: Annotated[Any, TV_BlockLSTMGradV2_T], b: Annotated[Any, TV_BlockLSTMGradV2_T], i: Annotated[Any, TV_BlockLSTMGradV2_T], cs: Annotated[Any, TV_BlockLSTMGradV2_T], f: Annotated[Any, TV_BlockLSTMGradV2_T], o: Annotated[Any, TV_BlockLSTMGradV2_T], ci: Annotated[Any, TV_BlockLSTMGradV2_T], co: Annotated[Any, TV_BlockLSTMGradV2_T], h: Annotated[Any, TV_BlockLSTMGradV2_T], cs_grad: Annotated[Any, TV_BlockLSTMGradV2_T], h_grad: Annotated[Any, TV_BlockLSTMGradV2_T], use_peephole: bool, name, ctx):
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad], ctx, [_dtypes.half, _dtypes.float32, ])
  (x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad) = _inputs_T
  seq_len_max = _ops.convert_to_tensor(seq_len_max, _dtypes.int64)
  _inputs_flat = [seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad]
  _attrs = ("use_peephole", use_peephole, "T", _attr_T)
  _result = _execute.execute(b"BlockLSTMGradV2", 8, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BlockLSTMGradV2", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMGradV2Output._make(_result)
  return _result

_BlockLSTMV2Output = collections.namedtuple(
    "BlockLSTMV2",
    ["i", "cs", "f", "o", "ci", "co", "h"])


TV_BlockLSTMV2_T = TypeVar("TV_BlockLSTMV2_T", _atypes.Float32, _atypes.Half)

def block_lstmv2(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTMV2_T], cs_prev: Annotated[Any, TV_BlockLSTMV2_T], h_prev: Annotated[Any, TV_BlockLSTMV2_T], w: Annotated[Any, TV_BlockLSTMV2_T], wci: Annotated[Any, TV_BlockLSTMV2_T], wcf: Annotated[Any, TV_BlockLSTMV2_T], wco: Annotated[Any, TV_BlockLSTMV2_T], b: Annotated[Any, TV_BlockLSTMV2_T], cell_clip:float=0, use_peephole:bool=False, name=None):
  r"""Computes the LSTM cell forward propagation for all the time steps.

  This is equivalent to applying LSTMBlockCell in a loop, like so:

  ```python
  for x1 in unpack(x):
    i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
      x1, cs_prev, h_prev, w, wci, wcf, wco, b)
    cs_prev = cs1
    h_prev = h1
    i.append(i1)
    cs.append(cs1)
    f.append(f1)
    o.append(o1)
    ci.append(ci1)
    co.append(co1)
    h.append(h1)
  return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)

  Note that unlike LSTMBlockCell (and BlockLSTM) which uses ICFO gate layout,
  this op uses IFCO. So in order for the following snippet to be equivalent
  all gate-related outputs should be reordered.
  ```

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    cell_clip: An optional `float`. Defaults to `0`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).

    i: A `Tensor`. Has the same type as `x`.
    cs: A `Tensor`. Has the same type as `x`.
    f: A `Tensor`. Has the same type as `x`.
    o: A `Tensor`. Has the same type as `x`.
    ci: A `Tensor`. Has the same type as `x`.
    co: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BlockLSTMV2", name, seq_len_max, x, cs_prev, h_prev, w, wci,
        wcf, wco, b, "cell_clip", cell_clip, "use_peephole", use_peephole)
      _result = _BlockLSTMV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return block_lstmv2_eager_fallback(
          seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b,
          cell_clip=cell_clip, use_peephole=use_peephole, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if cell_clip is None:
    cell_clip = 0
  cell_clip = _execute.make_float(cell_clip, "cell_clip")
  if use_peephole is None:
    use_peephole = False
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BlockLSTMV2", seq_len_max=seq_len_max, x=x, cs_prev=cs_prev,
                       h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b,
                       cell_clip=cell_clip, use_peephole=use_peephole,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("cell_clip", _op.get_attr("cell_clip"), "use_peephole",
              _op._get_attr_bool("use_peephole"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BlockLSTMV2", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMV2Output._make(_result)
  return _result

BlockLSTMV2 = tf_export("raw_ops.BlockLSTMV2")(_ops.to_raw_op(block_lstmv2))


def block_lstmv2_eager_fallback(seq_len_max: Annotated[Any, _atypes.Int64], x: Annotated[Any, TV_BlockLSTMV2_T], cs_prev: Annotated[Any, TV_BlockLSTMV2_T], h_prev: Annotated[Any, TV_BlockLSTMV2_T], w: Annotated[Any, TV_BlockLSTMV2_T], wci: Annotated[Any, TV_BlockLSTMV2_T], wcf: Annotated[Any, TV_BlockLSTMV2_T], wco: Annotated[Any, TV_BlockLSTMV2_T], b: Annotated[Any, TV_BlockLSTMV2_T], cell_clip: float, use_peephole: bool, name, ctx):
  if cell_clip is None:
    cell_clip = 0
  cell_clip = _execute.make_float(cell_clip, "cell_clip")
  if use_peephole is None:
    use_peephole = False
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b], ctx, [_dtypes.half, _dtypes.float32, ])
  (x, cs_prev, h_prev, w, wci, wcf, wco, b) = _inputs_T
  seq_len_max = _ops.convert_to_tensor(seq_len_max, _dtypes.int64)
  _inputs_flat = [seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b]
  _attrs = ("cell_clip", cell_clip, "use_peephole", use_peephole, "T",
  _attr_T)
  _result = _execute.execute(b"BlockLSTMV2", 7, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BlockLSTMV2", _inputs_flat, _attrs, _result)
  _result = _BlockLSTMV2Output._make(_result)
  return _result

_GRUBlockCellOutput = collections.namedtuple(
    "GRUBlockCell",
    ["r", "u", "c", "h"])


TV_GRUBlockCell_T = TypeVar("TV_GRUBlockCell_T", bound=_atypes.Float32)

def gru_block_cell(x: Annotated[Any, TV_GRUBlockCell_T], h_prev: Annotated[Any, TV_GRUBlockCell_T], w_ru: Annotated[Any, TV_GRUBlockCell_T], w_c: Annotated[Any, TV_GRUBlockCell_T], b_ru: Annotated[Any, TV_GRUBlockCell_T], b_c: Annotated[Any, TV_GRUBlockCell_T], name=None):
  r"""Computes the GRU cell forward propagation for 1 time step.

  Args
      x: Input to the GRU cell.
      h_prev: State input from the previous GRU cell.
      w_ru: Weight matrix for the reset and update gate.
      w_c: Weight matrix for the cell connection gate.
      b_ru: Bias vector for the reset and update gate.
      b_c: Bias vector for the cell connection gate.

  Returns
      r: Output of the reset gate.
      u: Output of the update gate.
      c: Output of the cell connection gate.
      h: Current state of the GRU cell.

  Note on notation of the variables:

  Concatenation of a and b is represented by a_b
  Element-wise dot product of a and b is represented by ab
  Element-wise dot product is represented by \circ
  Matrix multiplication is represented by *

  Biases are initialized with :
  `b_ru` - constant_initializer(1.0)
  `b_c` - constant_initializer(0.0)

  This kernel op implements the following mathematical equations:

  ```
  x_h_prev = [x, h_prev]

  [r_bar u_bar] = x_h_prev * w_ru + b_ru

  r = sigmoid(r_bar)
  u = sigmoid(u_bar)

  h_prevr = h_prev \circ r

  x_h_prevr = [x h_prevr]

  c_bar = x_h_prevr * w_c + b_c
  c = tanh(c_bar)

  h = (1-u) \circ c + u \circ h_prev
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w_ru: A `Tensor`. Must have the same type as `x`.
    w_c: A `Tensor`. Must have the same type as `x`.
    b_ru: A `Tensor`. Must have the same type as `x`.
    b_c: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r, u, c, h).

    r: A `Tensor`. Has the same type as `x`.
    u: A `Tensor`. Has the same type as `x`.
    c: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GRUBlockCell", name, x, h_prev, w_ru, w_c, b_ru, b_c)
      _result = _GRUBlockCellOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return gru_block_cell_eager_fallback(
          x, h_prev, w_ru, w_c, b_ru, b_c, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GRUBlockCell", x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c, b_ru=b_ru,
                        b_c=b_c, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GRUBlockCell", _inputs_flat, _attrs, _result)
  _result = _GRUBlockCellOutput._make(_result)
  return _result

GRUBlockCell = tf_export("raw_ops.GRUBlockCell")(_ops.to_raw_op(gru_block_cell))


def gru_block_cell_eager_fallback(x: Annotated[Any, TV_GRUBlockCell_T], h_prev: Annotated[Any, TV_GRUBlockCell_T], w_ru: Annotated[Any, TV_GRUBlockCell_T], w_c: Annotated[Any, TV_GRUBlockCell_T], b_ru: Annotated[Any, TV_GRUBlockCell_T], b_c: Annotated[Any, TV_GRUBlockCell_T], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, h_prev, w_ru, w_c, b_ru, b_c], ctx, [_dtypes.float32, ])
  (x, h_prev, w_ru, w_c, b_ru, b_c) = _inputs_T
  _inputs_flat = [x, h_prev, w_ru, w_c, b_ru, b_c]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"GRUBlockCell", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GRUBlockCell", _inputs_flat, _attrs, _result)
  _result = _GRUBlockCellOutput._make(_result)
  return _result

_GRUBlockCellGradOutput = collections.namedtuple(
    "GRUBlockCellGrad",
    ["d_x", "d_h_prev", "d_c_bar", "d_r_bar_u_bar"])


TV_GRUBlockCellGrad_T = TypeVar("TV_GRUBlockCellGrad_T", bound=_atypes.Float32)

def gru_block_cell_grad(x: Annotated[Any, TV_GRUBlockCellGrad_T], h_prev: Annotated[Any, TV_GRUBlockCellGrad_T], w_ru: Annotated[Any, TV_GRUBlockCellGrad_T], w_c: Annotated[Any, TV_GRUBlockCellGrad_T], b_ru: Annotated[Any, TV_GRUBlockCellGrad_T], b_c: Annotated[Any, TV_GRUBlockCellGrad_T], r: Annotated[Any, TV_GRUBlockCellGrad_T], u: Annotated[Any, TV_GRUBlockCellGrad_T], c: Annotated[Any, TV_GRUBlockCellGrad_T], d_h: Annotated[Any, TV_GRUBlockCellGrad_T], name=None):
  r"""Computes the GRU cell back-propagation for 1 time step.

  Args
      x: Input to the GRU cell.
      h_prev: State input from the previous GRU cell.
      w_ru: Weight matrix for the reset and update gate.
      w_c: Weight matrix for the cell connection gate.
      b_ru: Bias vector for the reset and update gate.
      b_c: Bias vector for the cell connection gate.
      r: Output of the reset gate.
      u: Output of the update gate.
      c: Output of the cell connection gate.
      d_h: Gradients of the h_new wrt to objective function.

  Returns
      d_x: Gradients of the x wrt to objective function.
      d_h_prev: Gradients of the h wrt to objective function.
      d_c_bar Gradients of the c_bar wrt to objective function.
      d_r_bar_u_bar Gradients of the r_bar & u_bar wrt to objective function.

  This kernel op implements the following mathematical equations:

  Note on notation of the variables:

  Concatenation of a and b is represented by a_b
  Element-wise dot product of a and b is represented by ab
  Element-wise dot product is represented by \circ
  Matrix multiplication is represented by *

  Additional notes for clarity:

  `w_ru` can be segmented into 4 different matrices.
  ```
  w_ru = [w_r_x w_u_x
          w_r_h_prev w_u_h_prev]
  ```
  Similarly, `w_c` can be segmented into 2 different matrices.
  ```
  w_c = [w_c_x w_c_h_prevr]
  ```
  Same goes for biases.
  ```
  b_ru = [b_ru_x b_ru_h]
  b_c = [b_c_x b_c_h]
  ```
  Another note on notation:
  ```
  d_x = d_x_component_1 + d_x_component_2

  where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
  and d_x_component_2 = d_c_bar * w_c_x^T

  d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + d_h \circ u
  where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
  ```

  Mathematics behind the Gradients below:
  ```
  d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
  d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)

  d_r_bar_u_bar = [d_r_bar d_u_bar]

  [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T

  [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T

  d_x = d_x_component_1 + d_x_component_2

  d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
  ```
  Below calculation is performed in the python wrapper for the Gradients
  (not in the gradient kernel.)
  ```
  d_w_ru = x_h_prevr^T * d_c_bar

  d_w_c = x_h_prev^T * d_r_bar_u_bar

  d_b_ru = sum of d_r_bar_u_bar along axis = 0

  d_b_c = sum of d_c_bar along axis = 0
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w_ru: A `Tensor`. Must have the same type as `x`.
    w_c: A `Tensor`. Must have the same type as `x`.
    b_ru: A `Tensor`. Must have the same type as `x`.
    b_c: A `Tensor`. Must have the same type as `x`.
    r: A `Tensor`. Must have the same type as `x`.
    u: A `Tensor`. Must have the same type as `x`.
    c: A `Tensor`. Must have the same type as `x`.
    d_h: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d_x, d_h_prev, d_c_bar, d_r_bar_u_bar).

    d_x: A `Tensor`. Has the same type as `x`.
    d_h_prev: A `Tensor`. Has the same type as `x`.
    d_c_bar: A `Tensor`. Has the same type as `x`.
    d_r_bar_u_bar: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GRUBlockCellGrad", name, x, h_prev, w_ru, w_c, b_ru, b_c, r, u,
        c, d_h)
      _result = _GRUBlockCellGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return gru_block_cell_grad_eager_fallback(
          x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GRUBlockCellGrad", x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c, b_ru=b_ru,
                            b_c=b_c, r=r, u=u, c=c, d_h=d_h, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GRUBlockCellGrad", _inputs_flat, _attrs, _result)
  _result = _GRUBlockCellGradOutput._make(_result)
  return _result

GRUBlockCellGrad = tf_export("raw_ops.GRUBlockCellGrad")(_ops.to_raw_op(gru_block_cell_grad))


def gru_block_cell_grad_eager_fallback(x: Annotated[Any, TV_GRUBlockCellGrad_T], h_prev: Annotated[Any, TV_GRUBlockCellGrad_T], w_ru: Annotated[Any, TV_GRUBlockCellGrad_T], w_c: Annotated[Any, TV_GRUBlockCellGrad_T], b_ru: Annotated[Any, TV_GRUBlockCellGrad_T], b_c: Annotated[Any, TV_GRUBlockCellGrad_T], r: Annotated[Any, TV_GRUBlockCellGrad_T], u: Annotated[Any, TV_GRUBlockCellGrad_T], c: Annotated[Any, TV_GRUBlockCellGrad_T], d_h: Annotated[Any, TV_GRUBlockCellGrad_T], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h], ctx, [_dtypes.float32, ])
  (x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h) = _inputs_T
  _inputs_flat = [x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"GRUBlockCellGrad", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GRUBlockCellGrad", _inputs_flat, _attrs, _result)
  _result = _GRUBlockCellGradOutput._make(_result)
  return _result

_LSTMBlockCellOutput = collections.namedtuple(
    "LSTMBlockCell",
    ["i", "cs", "f", "o", "ci", "co", "h"])


TV_LSTMBlockCell_T = TypeVar("TV_LSTMBlockCell_T", _atypes.Float32, _atypes.Half)

def lstm_block_cell(x: Annotated[Any, TV_LSTMBlockCell_T], cs_prev: Annotated[Any, TV_LSTMBlockCell_T], h_prev: Annotated[Any, TV_LSTMBlockCell_T], w: Annotated[Any, TV_LSTMBlockCell_T], wci: Annotated[Any, TV_LSTMBlockCell_T], wcf: Annotated[Any, TV_LSTMBlockCell_T], wco: Annotated[Any, TV_LSTMBlockCell_T], b: Annotated[Any, TV_LSTMBlockCell_T], forget_bias:float=1, cell_clip:float=3, use_peephole:bool=False, name=None):
  r"""Computes the LSTM cell forward propagation for 1 time step.

  This implementation uses 1 weight matrix and 1 bias vector, and there's an
  optional peephole connection.

  This kernel op implements the following mathematical equations:

  ```python
  xh = [x, h_prev]
  [i, f, ci, o] = xh * w + b
  f = f + forget_bias

  if not use_peephole:
    wci = wcf = wco = 0

  i = sigmoid(cs_prev * wci + i)
  f = sigmoid(cs_prev * wcf + f)
  ci = tanh(ci)

  cs = ci .* i + cs_prev .* f
  cs = clip(cs, cell_clip)

  o = sigmoid(cs * wco + o)
  co = tanh(cs)
  h = co .* o
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the cell state at previous time step.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Output of the previous cell at previous time step.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).

    i: A `Tensor`. Has the same type as `x`.
    cs: A `Tensor`. Has the same type as `x`.
    f: A `Tensor`. Has the same type as `x`.
    o: A `Tensor`. Has the same type as `x`.
    ci: A `Tensor`. Has the same type as `x`.
    co: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LSTMBlockCell", name, x, cs_prev, h_prev, w, wci, wcf, wco, b,
        "forget_bias", forget_bias, "cell_clip", cell_clip, "use_peephole",
        use_peephole)
      _result = _LSTMBlockCellOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lstm_block_cell_eager_fallback(
          x, cs_prev, h_prev, w, wci, wcf, wco, b, forget_bias=forget_bias,
          cell_clip=cell_clip, use_peephole=use_peephole, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if forget_bias is None:
    forget_bias = 1
  forget_bias = _execute.make_float(forget_bias, "forget_bias")
  if cell_clip is None:
    cell_clip = 3
  cell_clip = _execute.make_float(cell_clip, "cell_clip")
  if use_peephole is None:
    use_peephole = False
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LSTMBlockCell", x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci,
                         wcf=wcf, wco=wco, b=b, forget_bias=forget_bias,
                         cell_clip=cell_clip, use_peephole=use_peephole,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("forget_bias", _op.get_attr("forget_bias"), "cell_clip",
              _op.get_attr("cell_clip"), "use_peephole",
              _op._get_attr_bool("use_peephole"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LSTMBlockCell", _inputs_flat, _attrs, _result)
  _result = _LSTMBlockCellOutput._make(_result)
  return _result

LSTMBlockCell = tf_export("raw_ops.LSTMBlockCell")(_ops.to_raw_op(lstm_block_cell))


def lstm_block_cell_eager_fallback(x: Annotated[Any, TV_LSTMBlockCell_T], cs_prev: Annotated[Any, TV_LSTMBlockCell_T], h_prev: Annotated[Any, TV_LSTMBlockCell_T], w: Annotated[Any, TV_LSTMBlockCell_T], wci: Annotated[Any, TV_LSTMBlockCell_T], wcf: Annotated[Any, TV_LSTMBlockCell_T], wco: Annotated[Any, TV_LSTMBlockCell_T], b: Annotated[Any, TV_LSTMBlockCell_T], forget_bias: float, cell_clip: float, use_peephole: bool, name, ctx):
  if forget_bias is None:
    forget_bias = 1
  forget_bias = _execute.make_float(forget_bias, "forget_bias")
  if cell_clip is None:
    cell_clip = 3
  cell_clip = _execute.make_float(cell_clip, "cell_clip")
  if use_peephole is None:
    use_peephole = False
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b], ctx, [_dtypes.half, _dtypes.float32, ])
  (x, cs_prev, h_prev, w, wci, wcf, wco, b) = _inputs_T
  _inputs_flat = [x, cs_prev, h_prev, w, wci, wcf, wco, b]
  _attrs = ("forget_bias", forget_bias, "cell_clip", cell_clip,
  "use_peephole", use_peephole, "T", _attr_T)
  _result = _execute.execute(b"LSTMBlockCell", 7, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LSTMBlockCell", _inputs_flat, _attrs, _result)
  _result = _LSTMBlockCellOutput._make(_result)
  return _result

_LSTMBlockCellGradOutput = collections.namedtuple(
    "LSTMBlockCellGrad",
    ["cs_prev_grad", "dicfo", "wci_grad", "wcf_grad", "wco_grad"])


TV_LSTMBlockCellGrad_T = TypeVar("TV_LSTMBlockCellGrad_T", _atypes.Float32, _atypes.Half)

def lstm_block_cell_grad(x: Annotated[Any, TV_LSTMBlockCellGrad_T], cs_prev: Annotated[Any, TV_LSTMBlockCellGrad_T], h_prev: Annotated[Any, TV_LSTMBlockCellGrad_T], w: Annotated[Any, TV_LSTMBlockCellGrad_T], wci: Annotated[Any, TV_LSTMBlockCellGrad_T], wcf: Annotated[Any, TV_LSTMBlockCellGrad_T], wco: Annotated[Any, TV_LSTMBlockCellGrad_T], b: Annotated[Any, TV_LSTMBlockCellGrad_T], i: Annotated[Any, TV_LSTMBlockCellGrad_T], cs: Annotated[Any, TV_LSTMBlockCellGrad_T], f: Annotated[Any, TV_LSTMBlockCellGrad_T], o: Annotated[Any, TV_LSTMBlockCellGrad_T], ci: Annotated[Any, TV_LSTMBlockCellGrad_T], co: Annotated[Any, TV_LSTMBlockCellGrad_T], cs_grad: Annotated[Any, TV_LSTMBlockCellGrad_T], h_grad: Annotated[Any, TV_LSTMBlockCellGrad_T], use_peephole: bool, name=None):
  r"""Computes the LSTM cell backward propagation for 1 timestep.

  This implementation is to be used in conjunction of LSTMBlockCell.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      The previous cell state.
    h_prev: A `Tensor`. Must have the same type as `x`. The previous h state.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`. The input gate.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh.
    f: A `Tensor`. Must have the same type as `x`. The forget gate.
    o: A `Tensor`. Must have the same type as `x`. The output gate.
    ci: A `Tensor`. Must have the same type as `x`. The cell input.
    co: A `Tensor`. Must have the same type as `x`. The cell after the tanh.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether the cell uses peephole connections.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad).

    cs_prev_grad: A `Tensor`. Has the same type as `x`.
    dicfo: A `Tensor`. Has the same type as `x`.
    wci_grad: A `Tensor`. Has the same type as `x`.
    wcf_grad: A `Tensor`. Has the same type as `x`.
    wco_grad: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LSTMBlockCellGrad", name, x, cs_prev, h_prev, w, wci, wcf, wco,
        b, i, cs, f, o, ci, co, cs_grad, h_grad, "use_peephole", use_peephole)
      _result = _LSTMBlockCellGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lstm_block_cell_grad_eager_fallback(
          x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co,
          cs_grad, h_grad, use_peephole=use_peephole, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LSTMBlockCellGrad", x=x, cs_prev=cs_prev, h_prev=h_prev, w=w,
                             wci=wci, wcf=wcf, wco=wco, b=b, i=i, cs=cs, f=f,
                             o=o, ci=ci, co=co, cs_grad=cs_grad,
                             h_grad=h_grad, use_peephole=use_peephole,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_peephole", _op._get_attr_bool("use_peephole"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LSTMBlockCellGrad", _inputs_flat, _attrs, _result)
  _result = _LSTMBlockCellGradOutput._make(_result)
  return _result

LSTMBlockCellGrad = tf_export("raw_ops.LSTMBlockCellGrad")(_ops.to_raw_op(lstm_block_cell_grad))


def lstm_block_cell_grad_eager_fallback(x: Annotated[Any, TV_LSTMBlockCellGrad_T], cs_prev: Annotated[Any, TV_LSTMBlockCellGrad_T], h_prev: Annotated[Any, TV_LSTMBlockCellGrad_T], w: Annotated[Any, TV_LSTMBlockCellGrad_T], wci: Annotated[Any, TV_LSTMBlockCellGrad_T], wcf: Annotated[Any, TV_LSTMBlockCellGrad_T], wco: Annotated[Any, TV_LSTMBlockCellGrad_T], b: Annotated[Any, TV_LSTMBlockCellGrad_T], i: Annotated[Any, TV_LSTMBlockCellGrad_T], cs: Annotated[Any, TV_LSTMBlockCellGrad_T], f: Annotated[Any, TV_LSTMBlockCellGrad_T], o: Annotated[Any, TV_LSTMBlockCellGrad_T], ci: Annotated[Any, TV_LSTMBlockCellGrad_T], co: Annotated[Any, TV_LSTMBlockCellGrad_T], cs_grad: Annotated[Any, TV_LSTMBlockCellGrad_T], h_grad: Annotated[Any, TV_LSTMBlockCellGrad_T], use_peephole: bool, name, ctx):
  use_peephole = _execute.make_bool(use_peephole, "use_peephole")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad], ctx, [_dtypes.half, _dtypes.float32, ])
  (x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad) = _inputs_T
  _inputs_flat = [x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad]
  _attrs = ("use_peephole", use_peephole, "T", _attr_T)
  _result = _execute.execute(b"LSTMBlockCellGrad", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LSTMBlockCellGrad", _inputs_flat, _attrs, _result)
  _result = _LSTMBlockCellGradOutput._make(_result)
  return _result


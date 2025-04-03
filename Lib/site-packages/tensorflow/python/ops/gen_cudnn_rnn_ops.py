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
_CudnnRNNOutput = collections.namedtuple(
    "CudnnRNN",
    ["output", "output_h", "output_c", "reserve_space"])


TV_CudnnRNN_T = TypeVar("TV_CudnnRNN_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn(input: Annotated[Any, TV_CudnnRNN_T], input_h: Annotated[Any, TV_CudnnRNN_T], input_c: Annotated[Any, TV_CudnnRNN_T], params: Annotated[Any, TV_CudnnRNN_T], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, is_training:bool=True, name=None):
  r"""A RNN backed by cuDNN.

  Computes the RNN from the input and initial states, with respect to the params
  buffer.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
    the actual computation before the first layer. 'skip_input' is only allowed
    when input_size == num_units; 'auto_select' implies 'skip_input' when
    input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
  input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  output: A 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  is_training: Indicates whether this operation is used for inference or
    training.
  reserve_space: An opaque tensor that can be used in backprop calculation. It
    is only produced if is_training is false.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    is_training: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_h, output_c, reserve_space).

    output: A `Tensor`. Has the same type as `input`.
    output_h: A `Tensor`. Has the same type as `input`.
    output_c: A `Tensor`. Has the same type as `input`.
    reserve_space: A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNN", name, input, input_h, input_c, params, "rnn_mode",
        rnn_mode, "input_mode", input_mode, "direction", direction, "dropout",
        dropout, "seed", seed, "seed2", seed2, "is_training", is_training)
      _result = _CudnnRNNOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_eager_fallback(
          input, input_h, input_c, params, rnn_mode=rnn_mode,
          input_mode=input_mode, direction=direction, dropout=dropout,
          seed=seed, seed2=seed2, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNN", input=input, input_h=input_h, input_c=input_c,
                    params=params, rnn_mode=rnn_mode, input_mode=input_mode,
                    direction=direction, dropout=dropout, seed=seed,
                    seed2=seed2, is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNN", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNOutput._make(_result)
  return _result

CudnnRNN = tf_export("raw_ops.CudnnRNN")(_ops.to_raw_op(cudnn_rnn))


def cudnn_rnn_eager_fallback(input: Annotated[Any, TV_CudnnRNN_T], input_h: Annotated[Any, TV_CudnnRNN_T], input_c: Annotated[Any, TV_CudnnRNN_T], params: Annotated[Any, TV_CudnnRNN_T], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, is_training: bool, name, ctx):
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_h, input_c, params) = _inputs_T
  _inputs_flat = [input, input_h, input_c, params]
  _attrs = ("T", _attr_T, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2,
  "is_training", is_training)
  _result = _execute.execute(b"CudnnRNN", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNN", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNOutput._make(_result)
  return _result

_CudnnRNNBackpropOutput = collections.namedtuple(
    "CudnnRNNBackprop",
    ["input_backprop", "input_h_backprop", "input_c_backprop", "params_backprop"])


TV_CudnnRNNBackprop_T = TypeVar("TV_CudnnRNNBackprop_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_backprop(input: Annotated[Any, TV_CudnnRNNBackprop_T], input_h: Annotated[Any, TV_CudnnRNNBackprop_T], input_c: Annotated[Any, TV_CudnnRNNBackprop_T], params: Annotated[Any, TV_CudnnRNNBackprop_T], output: Annotated[Any, TV_CudnnRNNBackprop_T], output_h: Annotated[Any, TV_CudnnRNNBackprop_T], output_c: Annotated[Any, TV_CudnnRNNBackprop_T], output_backprop: Annotated[Any, TV_CudnnRNNBackprop_T], output_h_backprop: Annotated[Any, TV_CudnnRNNBackprop_T], output_c_backprop: Annotated[Any, TV_CudnnRNNBackprop_T], reserve_space: Annotated[Any, TV_CudnnRNNBackprop_T], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, name=None):
  r"""Backprop step of CudnnRNN.

  Compute the backprop of both data and weights in a RNN.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      the actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
  input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  output: A 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  output_backprop: A 3-D tensor with the same shape as output in the forward pass.
  output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
      pass.
  output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
      pass.
  reserve_space: The same reserve_space produced in for forward operation.
  input_backprop: The backprop to input in the forward pass. Has the same shape
      as input.
  input_h_backprop: The backprop to input_h in the forward pass. Has the same
      shape as input_h.
  input_c_backprop: The backprop to input_c in the forward pass. Has the same
      shape as input_c.
  params_backprop: The backprop to the params buffer in the forward pass. Has the
      same shape as params.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    output: A `Tensor`. Must have the same type as `input`.
    output_h: A `Tensor`. Must have the same type as `input`.
    output_c: A `Tensor`. Must have the same type as `input`.
    output_backprop: A `Tensor`. Must have the same type as `input`.
    output_h_backprop: A `Tensor`. Must have the same type as `input`.
    output_c_backprop: A `Tensor`. Must have the same type as `input`.
    reserve_space: A `Tensor`. Must have the same type as `input`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_h_backprop, input_c_backprop, params_backprop).

    input_backprop: A `Tensor`. Has the same type as `input`.
    input_h_backprop: A `Tensor`. Has the same type as `input`.
    input_c_backprop: A `Tensor`. Has the same type as `input`.
    params_backprop: A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNBackprop", name, input, input_h, input_c, params,
        output, output_h, output_c, output_backprop, output_h_backprop,
        output_c_backprop, reserve_space, "rnn_mode", rnn_mode, "input_mode",
        input_mode, "direction", direction, "dropout", dropout, "seed", seed,
        "seed2", seed2)
      _result = _CudnnRNNBackpropOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_backprop_eager_fallback(
          input, input_h, input_c, params, output, output_h, output_c,
          output_backprop, output_h_backprop, output_c_backprop,
          reserve_space, rnn_mode=rnn_mode, input_mode=input_mode,
          direction=direction, dropout=dropout, seed=seed, seed2=seed2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNBackprop", input=input, input_h=input_h, input_c=input_c,
                            params=params, output=output, output_h=output_h,
                            output_c=output_c,
                            output_backprop=output_backprop,
                            output_h_backprop=output_h_backprop,
                            output_c_backprop=output_c_backprop,
                            reserve_space=reserve_space, rnn_mode=rnn_mode,
                            input_mode=input_mode, direction=direction,
                            dropout=dropout, seed=seed, seed2=seed2,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNBackprop", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNBackpropOutput._make(_result)
  return _result

CudnnRNNBackprop = tf_export("raw_ops.CudnnRNNBackprop")(_ops.to_raw_op(cudnn_rnn_backprop))


def cudnn_rnn_backprop_eager_fallback(input: Annotated[Any, TV_CudnnRNNBackprop_T], input_h: Annotated[Any, TV_CudnnRNNBackprop_T], input_c: Annotated[Any, TV_CudnnRNNBackprop_T], params: Annotated[Any, TV_CudnnRNNBackprop_T], output: Annotated[Any, TV_CudnnRNNBackprop_T], output_h: Annotated[Any, TV_CudnnRNNBackprop_T], output_c: Annotated[Any, TV_CudnnRNNBackprop_T], output_backprop: Annotated[Any, TV_CudnnRNNBackprop_T], output_h_backprop: Annotated[Any, TV_CudnnRNNBackprop_T], output_c_backprop: Annotated[Any, TV_CudnnRNNBackprop_T], reserve_space: Annotated[Any, TV_CudnnRNNBackprop_T], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, name, ctx):
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space) = _inputs_T
  _inputs_flat = [input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space]
  _attrs = ("T", _attr_T, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"CudnnRNNBackprop", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNBackprop", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNBackpropOutput._make(_result)
  return _result

_CudnnRNNBackpropV2Output = collections.namedtuple(
    "CudnnRNNBackpropV2",
    ["input_backprop", "input_h_backprop", "input_c_backprop", "params_backprop"])


TV_CudnnRNNBackpropV2_T = TypeVar("TV_CudnnRNNBackpropV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_backprop_v2(input: Annotated[Any, TV_CudnnRNNBackpropV2_T], input_h: Annotated[Any, TV_CudnnRNNBackpropV2_T], input_c: Annotated[Any, TV_CudnnRNNBackpropV2_T], params: Annotated[Any, TV_CudnnRNNBackpropV2_T], output: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_h: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_c: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_backprop: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_h_backprop: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_c_backprop: Annotated[Any, TV_CudnnRNNBackpropV2_T], reserve_space: Annotated[Any, TV_CudnnRNNBackpropV2_T], host_reserved: Annotated[Any, _atypes.Int8], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, name=None):
  r"""Backprop step of CudnnRNN.

  Compute the backprop of both data and weights in a RNN. Takes an extra
      "host_reserved" inupt than CudnnRNNBackprop, which is used to determine RNN
      cudnnRNNAlgo_t and cudnnMathType_t.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicates whether there is a linear projection between the input and
      the actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
  input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  output: A 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  output_backprop: A 3-D tensor with the same shape as output in the forward pass.
  output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
      pass.
  output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
      pass.
  reserve_space: The same reserve_space produced in the forward operation.
  host_reserved: The same host_reserved produced in the forward operation.
  input_backprop: The backprop to input in the forward pass. Has the same shape
      as input.
  input_h_backprop: The backprop to input_h in the forward pass. Has the same
      shape as input_h.
  input_c_backprop: The backprop to input_c in the forward pass. Has the same
      shape as input_c.
  params_backprop: The backprop to the params buffer in the forward pass. Has the
      same shape as params.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    output: A `Tensor`. Must have the same type as `input`.
    output_h: A `Tensor`. Must have the same type as `input`.
    output_c: A `Tensor`. Must have the same type as `input`.
    output_backprop: A `Tensor`. Must have the same type as `input`.
    output_h_backprop: A `Tensor`. Must have the same type as `input`.
    output_c_backprop: A `Tensor`. Must have the same type as `input`.
    reserve_space: A `Tensor`. Must have the same type as `input`.
    host_reserved: A `Tensor` of type `int8`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_h_backprop, input_c_backprop, params_backprop).

    input_backprop: A `Tensor`. Has the same type as `input`.
    input_h_backprop: A `Tensor`. Has the same type as `input`.
    input_c_backprop: A `Tensor`. Has the same type as `input`.
    params_backprop: A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNBackpropV2", name, input, input_h, input_c, params,
        output, output_h, output_c, output_backprop, output_h_backprop,
        output_c_backprop, reserve_space, host_reserved, "rnn_mode", rnn_mode,
        "input_mode", input_mode, "direction", direction, "dropout", dropout,
        "seed", seed, "seed2", seed2)
      _result = _CudnnRNNBackpropV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_backprop_v2_eager_fallback(
          input, input_h, input_c, params, output, output_h, output_c,
          output_backprop, output_h_backprop, output_c_backprop,
          reserve_space, host_reserved, rnn_mode=rnn_mode,
          input_mode=input_mode, direction=direction, dropout=dropout,
          seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNBackpropV2", input=input, input_h=input_h, input_c=input_c,
                              params=params, output=output, output_h=output_h,
                              output_c=output_c,
                              output_backprop=output_backprop,
                              output_h_backprop=output_h_backprop,
                              output_c_backprop=output_c_backprop,
                              reserve_space=reserve_space,
                              host_reserved=host_reserved, rnn_mode=rnn_mode,
                              input_mode=input_mode, direction=direction,
                              dropout=dropout, seed=seed, seed2=seed2,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNBackpropV2", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNBackpropV2Output._make(_result)
  return _result

CudnnRNNBackpropV2 = tf_export("raw_ops.CudnnRNNBackpropV2")(_ops.to_raw_op(cudnn_rnn_backprop_v2))


def cudnn_rnn_backprop_v2_eager_fallback(input: Annotated[Any, TV_CudnnRNNBackpropV2_T], input_h: Annotated[Any, TV_CudnnRNNBackpropV2_T], input_c: Annotated[Any, TV_CudnnRNNBackpropV2_T], params: Annotated[Any, TV_CudnnRNNBackpropV2_T], output: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_h: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_c: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_backprop: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_h_backprop: Annotated[Any, TV_CudnnRNNBackpropV2_T], output_c_backprop: Annotated[Any, TV_CudnnRNNBackpropV2_T], reserve_space: Annotated[Any, TV_CudnnRNNBackpropV2_T], host_reserved: Annotated[Any, _atypes.Int8], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, name, ctx):
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space) = _inputs_T
  host_reserved = _ops.convert_to_tensor(host_reserved, _dtypes.int8)
  _inputs_flat = [input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space, host_reserved]
  _attrs = ("T", _attr_T, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"CudnnRNNBackpropV2", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNBackpropV2", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNBackpropV2Output._make(_result)
  return _result

_CudnnRNNBackpropV3Output = collections.namedtuple(
    "CudnnRNNBackpropV3",
    ["input_backprop", "input_h_backprop", "input_c_backprop", "params_backprop"])


TV_CudnnRNNBackpropV3_T = TypeVar("TV_CudnnRNNBackpropV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_backprop_v3(input: Annotated[Any, TV_CudnnRNNBackpropV3_T], input_h: Annotated[Any, TV_CudnnRNNBackpropV3_T], input_c: Annotated[Any, TV_CudnnRNNBackpropV3_T], params: Annotated[Any, TV_CudnnRNNBackpropV3_T], sequence_lengths: Annotated[Any, _atypes.Int32], output: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_h: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_c: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_backprop: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_h_backprop: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_c_backprop: Annotated[Any, TV_CudnnRNNBackpropV3_T], reserve_space: Annotated[Any, TV_CudnnRNNBackpropV3_T], host_reserved: Annotated[Any, _atypes.Int8], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, num_proj:int=0, time_major:bool=True, name=None):
  r"""Backprop step of CudnnRNNV3.

  Compute the backprop of both data and weights in a RNN. Takes an extra
      "sequence_lengths" input than CudnnRNNBackprop.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicates whether there is a linear projection between the input and
      the actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: If time_major is true, this is a 3-D tensor with the shape of
      [seq_length, batch_size, input_size]. If time_major is false, the shape is
      [batch_size, seq_length, input_size].
  input_h: If time_major is true, this is a 3-D tensor with the shape of
      [num_layer * dir, batch_size, num_units]. If time_major is false, the shape
      is [batch_size, num_layer * dir, num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  sequence_lengths: a vector of lengths of each input sequence.
  output: If time_major is true, this is a 3-D tensor with the shape of
      [seq_length, batch_size, dir * num_units]. If time_major is false, the
      shape is [batch_size, seq_length, dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  output_backprop: A 3-D tensor with the same shape as output in the forward pass.
  output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
      pass.
  output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
      pass.
  time_major: Indicates whether the input/output format is time major or batch
      major.
  reserve_space: The same reserve_space produced in the forward operation.
  input_backprop: The backprop to input in the forward pass. Has the same shape
      as input.
  input_h_backprop: The backprop to input_h in the forward pass. Has the same
      shape as input_h.
  input_c_backprop: The backprop to input_c in the forward pass. Has the same
      shape as input_c.
  params_backprop: The backprop to the params buffer in the forward pass. Has the
      same shape as params.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    sequence_lengths: A `Tensor` of type `int32`.
    output: A `Tensor`. Must have the same type as `input`.
    output_h: A `Tensor`. Must have the same type as `input`.
    output_c: A `Tensor`. Must have the same type as `input`.
    output_backprop: A `Tensor`. Must have the same type as `input`.
    output_h_backprop: A `Tensor`. Must have the same type as `input`.
    output_c_backprop: A `Tensor`. Must have the same type as `input`.
    reserve_space: A `Tensor`. Must have the same type as `input`.
    host_reserved: A `Tensor` of type `int8`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    time_major: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_h_backprop, input_c_backprop, params_backprop).

    input_backprop: A `Tensor`. Has the same type as `input`.
    input_h_backprop: A `Tensor`. Has the same type as `input`.
    input_c_backprop: A `Tensor`. Has the same type as `input`.
    params_backprop: A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNBackpropV3", name, input, input_h, input_c, params,
        sequence_lengths, output, output_h, output_c, output_backprop,
        output_h_backprop, output_c_backprop, reserve_space, host_reserved,
        "rnn_mode", rnn_mode, "input_mode", input_mode, "direction",
        direction, "dropout", dropout, "seed", seed, "seed2", seed2,
        "num_proj", num_proj, "time_major", time_major)
      _result = _CudnnRNNBackpropV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_backprop_v3_eager_fallback(
          input, input_h, input_c, params, sequence_lengths, output, output_h,
          output_c, output_backprop, output_h_backprop, output_c_backprop,
          reserve_space, host_reserved, rnn_mode=rnn_mode,
          input_mode=input_mode, direction=direction, dropout=dropout,
          seed=seed, seed2=seed2, num_proj=num_proj, time_major=time_major,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  if time_major is None:
    time_major = True
  time_major = _execute.make_bool(time_major, "time_major")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNBackpropV3", input=input, input_h=input_h, input_c=input_c,
                              params=params,
                              sequence_lengths=sequence_lengths,
                              output=output, output_h=output_h,
                              output_c=output_c,
                              output_backprop=output_backprop,
                              output_h_backprop=output_h_backprop,
                              output_c_backprop=output_c_backprop,
                              reserve_space=reserve_space,
                              host_reserved=host_reserved, rnn_mode=rnn_mode,
                              input_mode=input_mode, direction=direction,
                              dropout=dropout, seed=seed, seed2=seed2,
                              num_proj=num_proj, time_major=time_major,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "num_proj",
              _op._get_attr_int("num_proj"), "time_major",
              _op._get_attr_bool("time_major"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNBackpropV3", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNBackpropV3Output._make(_result)
  return _result

CudnnRNNBackpropV3 = tf_export("raw_ops.CudnnRNNBackpropV3")(_ops.to_raw_op(cudnn_rnn_backprop_v3))


def cudnn_rnn_backprop_v3_eager_fallback(input: Annotated[Any, TV_CudnnRNNBackpropV3_T], input_h: Annotated[Any, TV_CudnnRNNBackpropV3_T], input_c: Annotated[Any, TV_CudnnRNNBackpropV3_T], params: Annotated[Any, TV_CudnnRNNBackpropV3_T], sequence_lengths: Annotated[Any, _atypes.Int32], output: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_h: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_c: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_backprop: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_h_backprop: Annotated[Any, TV_CudnnRNNBackpropV3_T], output_c_backprop: Annotated[Any, TV_CudnnRNNBackpropV3_T], reserve_space: Annotated[Any, TV_CudnnRNNBackpropV3_T], host_reserved: Annotated[Any, _atypes.Int8], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, num_proj: int, time_major: bool, name, ctx):
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  if time_major is None:
    time_major = True
  time_major = _execute.make_bool(time_major, "time_major")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space) = _inputs_T
  sequence_lengths = _ops.convert_to_tensor(sequence_lengths, _dtypes.int32)
  host_reserved = _ops.convert_to_tensor(host_reserved, _dtypes.int8)
  _inputs_flat = [input, input_h, input_c, params, sequence_lengths, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space, host_reserved]
  _attrs = ("T", _attr_T, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2,
  "num_proj", num_proj, "time_major", time_major)
  _result = _execute.execute(b"CudnnRNNBackpropV3", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNBackpropV3", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNBackpropV3Output._make(_result)
  return _result


TV_CudnnRNNCanonicalToParams_T = TypeVar("TV_CudnnRNNCanonicalToParams_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_canonical_to_params(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], weights: Annotated[List[Any], TV_CudnnRNNCanonicalToParams_T], biases: Annotated[List[Any], TV_CudnnRNNCanonicalToParams_T], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_CudnnRNNCanonicalToParams_T]:
  r"""Converts CudnnRNN params from canonical form to usable form.

  Writes a set of weights into the opaque params buffer so they can be used in
  upcoming training or inferences.

  Note that the params buffer may not be compatible across different GPUs. So any
  save and restoration should be converted to and from the canonical weights and
  biases.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  weights: the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  biases: the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  num_params: number of parameter sets for all layers.
      Each layer may contain multiple parameter sets, with each set consisting of
      a weight matrix and a bias vector.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    weights: A list of at least 1 `Tensor` objects with the same type in: `bfloat16`, `half`, `float32`, `float64`.
    biases: A list with the same length as `weights` of `Tensor` objects with the same type as `weights`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNCanonicalToParams", name, num_layers, num_units,
        input_size, weights, biases, "rnn_mode", rnn_mode, "input_mode",
        input_mode, "direction", direction, "dropout", dropout, "seed", seed,
        "seed2", seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_canonical_to_params_eager_fallback(
          num_layers, num_units, input_size, weights, biases,
          rnn_mode=rnn_mode, input_mode=input_mode, direction=direction,
          dropout=dropout, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'weights' argument to "
        "'cudnn_rnn_canonical_to_params' Op, not %r." % weights)
  _attr_num_params = len(weights)
  if not isinstance(biases, (list, tuple)):
    raise TypeError(
        "Expected list for 'biases' argument to "
        "'cudnn_rnn_canonical_to_params' Op, not %r." % biases)
  if len(biases) != _attr_num_params:
    raise ValueError(
        "List argument 'biases' to 'cudnn_rnn_canonical_to_params' Op with length %d "
        "must match length %d of argument 'weights'." %
        (len(biases), _attr_num_params))
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNCanonicalToParams", num_layers=num_layers,
                                     num_units=num_units,
                                     input_size=input_size, weights=weights,
                                     biases=biases, rnn_mode=rnn_mode,
                                     input_mode=input_mode,
                                     direction=direction, dropout=dropout,
                                     seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_params",
              _op._get_attr_int("num_params"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNCanonicalToParams", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CudnnRNNCanonicalToParams = tf_export("raw_ops.CudnnRNNCanonicalToParams")(_ops.to_raw_op(cudnn_rnn_canonical_to_params))


def cudnn_rnn_canonical_to_params_eager_fallback(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], weights: Annotated[List[Any], TV_CudnnRNNCanonicalToParams_T], biases: Annotated[List[Any], TV_CudnnRNNCanonicalToParams_T], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_CudnnRNNCanonicalToParams_T]:
  if not isinstance(weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'weights' argument to "
        "'cudnn_rnn_canonical_to_params' Op, not %r." % weights)
  _attr_num_params = len(weights)
  if not isinstance(biases, (list, tuple)):
    raise TypeError(
        "Expected list for 'biases' argument to "
        "'cudnn_rnn_canonical_to_params' Op, not %r." % biases)
  if len(biases) != _attr_num_params:
    raise ValueError(
        "List argument 'biases' to 'cudnn_rnn_canonical_to_params' Op with length %d "
        "must match length %d of argument 'weights'." %
        (len(biases), _attr_num_params))
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(weights) + list(biases), ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_T = [_inputs_T[:_attr_num_params]] + _inputs_T[_attr_num_params:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (weights, biases) = _inputs_T
  num_layers = _ops.convert_to_tensor(num_layers, _dtypes.int32)
  num_units = _ops.convert_to_tensor(num_units, _dtypes.int32)
  input_size = _ops.convert_to_tensor(input_size, _dtypes.int32)
  _inputs_flat = [num_layers, num_units, input_size] + list(weights) + list(biases)
  _attrs = ("T", _attr_T, "num_params", _attr_num_params, "rnn_mode",
  rnn_mode, "input_mode", input_mode, "direction", direction, "dropout",
  dropout, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"CudnnRNNCanonicalToParams", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNCanonicalToParams", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CudnnRNNCanonicalToParamsV2_T = TypeVar("TV_CudnnRNNCanonicalToParamsV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_canonical_to_params_v2(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], weights: Annotated[List[Any], TV_CudnnRNNCanonicalToParamsV2_T], biases: Annotated[List[Any], TV_CudnnRNNCanonicalToParamsV2_T], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, num_proj:int=0, name=None) -> Annotated[Any, TV_CudnnRNNCanonicalToParamsV2_T]:
  r"""Converts CudnnRNN params from canonical form to usable form. It supports the projection in LSTM.

  Writes a set of weights into the opaque params buffer so they can be used in
  upcoming training or inferences.

  Note that the params buffer may not be compatible across different GPUs. So any
  save and restoration should be converted to and from the canonical weights and
  biases.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  weights: the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  biases: the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  num_params_weights: number of weight parameter matrix for all layers.
  num_params_biases: number of bias parameter vector for all layers.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.
  num_proj: The output dimensionality for the projection matrices. If None or 0,
      no projection is performed.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    weights: A list of at least 1 `Tensor` objects with the same type in: `bfloat16`, `half`, `float32`, `float64`.
    biases: A list of at least 1 `Tensor` objects with the same type as `weights`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNCanonicalToParamsV2", name, num_layers, num_units,
        input_size, weights, biases, "rnn_mode", rnn_mode, "input_mode",
        input_mode, "direction", direction, "dropout", dropout, "seed", seed,
        "seed2", seed2, "num_proj", num_proj)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_canonical_to_params_v2_eager_fallback(
          num_layers, num_units, input_size, weights, biases,
          rnn_mode=rnn_mode, input_mode=input_mode, direction=direction,
          dropout=dropout, seed=seed, seed2=seed2, num_proj=num_proj,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'weights' argument to "
        "'cudnn_rnn_canonical_to_params_v2' Op, not %r." % weights)
  _attr_num_params_weights = len(weights)
  if not isinstance(biases, (list, tuple)):
    raise TypeError(
        "Expected list for 'biases' argument to "
        "'cudnn_rnn_canonical_to_params_v2' Op, not %r." % biases)
  _attr_num_params_biases = len(biases)
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNCanonicalToParamsV2", num_layers=num_layers,
                                       num_units=num_units,
                                       input_size=input_size, weights=weights,
                                       biases=biases, rnn_mode=rnn_mode,
                                       input_mode=input_mode,
                                       direction=direction, dropout=dropout,
                                       seed=seed, seed2=seed2,
                                       num_proj=num_proj, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_params_weights",
              _op._get_attr_int("num_params_weights"), "num_params_biases",
              _op._get_attr_int("num_params_biases"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "num_proj",
              _op._get_attr_int("num_proj"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNCanonicalToParamsV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CudnnRNNCanonicalToParamsV2 = tf_export("raw_ops.CudnnRNNCanonicalToParamsV2")(_ops.to_raw_op(cudnn_rnn_canonical_to_params_v2))


def cudnn_rnn_canonical_to_params_v2_eager_fallback(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], weights: Annotated[List[Any], TV_CudnnRNNCanonicalToParamsV2_T], biases: Annotated[List[Any], TV_CudnnRNNCanonicalToParamsV2_T], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, num_proj: int, name, ctx) -> Annotated[Any, TV_CudnnRNNCanonicalToParamsV2_T]:
  if not isinstance(weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'weights' argument to "
        "'cudnn_rnn_canonical_to_params_v2' Op, not %r." % weights)
  _attr_num_params_weights = len(weights)
  if not isinstance(biases, (list, tuple)):
    raise TypeError(
        "Expected list for 'biases' argument to "
        "'cudnn_rnn_canonical_to_params_v2' Op, not %r." % biases)
  _attr_num_params_biases = len(biases)
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(weights) + list(biases), ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_T = [_inputs_T[:_attr_num_params_weights]] + _inputs_T[_attr_num_params_weights:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (weights, biases) = _inputs_T
  num_layers = _ops.convert_to_tensor(num_layers, _dtypes.int32)
  num_units = _ops.convert_to_tensor(num_units, _dtypes.int32)
  input_size = _ops.convert_to_tensor(input_size, _dtypes.int32)
  _inputs_flat = [num_layers, num_units, input_size] + list(weights) + list(biases)
  _attrs = ("T", _attr_T, "num_params_weights", _attr_num_params_weights,
  "num_params_biases", _attr_num_params_biases, "rnn_mode", rnn_mode,
  "input_mode", input_mode, "direction", direction, "dropout", dropout,
  "seed", seed, "seed2", seed2, "num_proj", num_proj)
  _result = _execute.execute(b"CudnnRNNCanonicalToParamsV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNCanonicalToParamsV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CudnnRNNParamsSize_T = TypeVar("TV_CudnnRNNParamsSize_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_CudnnRNNParamsSize_S = TypeVar("TV_CudnnRNNParamsSize_S", _atypes.Int32, _atypes.Int64)

def cudnn_rnn_params_size(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], T: TV_CudnnRNNParamsSize_T, S: TV_CudnnRNNParamsSize_S, rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, num_proj:int=0, name=None) -> Annotated[Any, TV_CudnnRNNParamsSize_S]:
  r"""Computes size of weights that can be used by a Cudnn RNN model.

  Return the params size that can be used by the Cudnn RNN model. Subsequent
  weight allocation and initialization should use this size.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
    The actual computation before the first layer. 'skip_input' is only allowed
    when input_size == num_units; 'auto_select' implies 'skip_input' when
    input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
    dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.
  params_size: The size of the params buffer that should be allocated and
    initialized for this RNN model. Note that this params buffer may not be
    compatible across GPUs. Please use CudnnRNNParamsWeights and
    CudnnRNNParamsBiases to save and restore them in a way that is compatible
    across different runs.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    T: A `tf.DType` from: `tf.bfloat16, tf.half, tf.float32, tf.float64`.
    S: A `tf.DType` from: `tf.int32, tf.int64`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `S`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNParamsSize", name, num_layers, num_units, input_size,
        "T", T, "S", S, "rnn_mode", rnn_mode, "input_mode", input_mode,
        "direction", direction, "dropout", dropout, "seed", seed, "seed2",
        seed2, "num_proj", num_proj)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_params_size_eager_fallback(
          num_layers, num_units, input_size, T=T, S=S, rnn_mode=rnn_mode,
          input_mode=input_mode, direction=direction, dropout=dropout,
          seed=seed, seed2=seed2, num_proj=num_proj, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  S = _execute.make_type(S, "S")
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNParamsSize", num_layers=num_layers, num_units=num_units,
                              input_size=input_size, T=T, S=S,
                              rnn_mode=rnn_mode, input_mode=input_mode,
                              direction=direction, dropout=dropout, seed=seed,
                              seed2=seed2, num_proj=num_proj, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "S", _op._get_attr_type("S"),
              "rnn_mode", _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "num_proj",
              _op._get_attr_int("num_proj"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNParamsSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CudnnRNNParamsSize = tf_export("raw_ops.CudnnRNNParamsSize")(_ops.to_raw_op(cudnn_rnn_params_size))


def cudnn_rnn_params_size_eager_fallback(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], T: TV_CudnnRNNParamsSize_T, S: TV_CudnnRNNParamsSize_S, rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, num_proj: int, name, ctx) -> Annotated[Any, TV_CudnnRNNParamsSize_S]:
  T = _execute.make_type(T, "T")
  S = _execute.make_type(S, "S")
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  num_layers = _ops.convert_to_tensor(num_layers, _dtypes.int32)
  num_units = _ops.convert_to_tensor(num_units, _dtypes.int32)
  input_size = _ops.convert_to_tensor(input_size, _dtypes.int32)
  _inputs_flat = [num_layers, num_units, input_size]
  _attrs = ("T", T, "S", S, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2,
  "num_proj", num_proj)
  _result = _execute.execute(b"CudnnRNNParamsSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNParamsSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_CudnnRNNParamsToCanonicalOutput = collections.namedtuple(
    "CudnnRNNParamsToCanonical",
    ["weights", "biases"])


TV_CudnnRNNParamsToCanonical_T = TypeVar("TV_CudnnRNNParamsToCanonical_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_params_to_canonical(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], params: Annotated[Any, TV_CudnnRNNParamsToCanonical_T], num_params: int, rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, name=None):
  r"""Retrieves CudnnRNN params in canonical form.

  Retrieves a set of weights from the opaque params buffer that can be saved and
  restored in a way compatible with future runs.

  Note that the params buffer may not be compatible across different GPUs. So any
  save and restoration should be converted to and from the canonical weights and
  biases.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  num_params: number of parameter sets for all layers.
      Each layer may contain multiple parameter sets, with each set consisting of
      a weight matrix and a bias vector.
  weights: the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  biases: the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    params: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    num_params: An `int` that is `>= 1`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (weights, biases).

    weights: A list of `num_params` `Tensor` objects with the same type as `params`.
    biases: A list of `num_params` `Tensor` objects with the same type as `params`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNParamsToCanonical", name, num_layers, num_units,
        input_size, params, "num_params", num_params, "rnn_mode", rnn_mode,
        "input_mode", input_mode, "direction", direction, "dropout", dropout,
        "seed", seed, "seed2", seed2)
      _result = _CudnnRNNParamsToCanonicalOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_params_to_canonical_eager_fallback(
          num_layers, num_units, input_size, params, num_params=num_params,
          rnn_mode=rnn_mode, input_mode=input_mode, direction=direction,
          dropout=dropout, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_params = _execute.make_int(num_params, "num_params")
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNParamsToCanonical", num_layers=num_layers,
                                     num_units=num_units,
                                     input_size=input_size, params=params,
                                     num_params=num_params, rnn_mode=rnn_mode,
                                     input_mode=input_mode,
                                     direction=direction, dropout=dropout,
                                     seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_params",
              _op._get_attr_int("num_params"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNParamsToCanonical", _inputs_flat, _attrs, _result)
  _result = [_result[:num_params]] + _result[num_params:]
  _result = _result[:1] + [_result[1:]]
  _result = _CudnnRNNParamsToCanonicalOutput._make(_result)
  return _result

CudnnRNNParamsToCanonical = tf_export("raw_ops.CudnnRNNParamsToCanonical")(_ops.to_raw_op(cudnn_rnn_params_to_canonical))


def cudnn_rnn_params_to_canonical_eager_fallback(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], params: Annotated[Any, TV_CudnnRNNParamsToCanonical_T], num_params: int, rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, name, ctx):
  num_params = _execute.make_int(num_params, "num_params")
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (params,) = _execute.args_to_matching_eager([params], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  num_layers = _ops.convert_to_tensor(num_layers, _dtypes.int32)
  num_units = _ops.convert_to_tensor(num_units, _dtypes.int32)
  input_size = _ops.convert_to_tensor(input_size, _dtypes.int32)
  _inputs_flat = [num_layers, num_units, input_size, params]
  _attrs = ("T", _attr_T, "num_params", num_params, "rnn_mode", rnn_mode,
  "input_mode", input_mode, "direction", direction, "dropout", dropout,
  "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"CudnnRNNParamsToCanonical", num_params +
                             num_params, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNParamsToCanonical", _inputs_flat, _attrs, _result)
  _result = [_result[:num_params]] + _result[num_params:]
  _result = _result[:1] + [_result[1:]]
  _result = _CudnnRNNParamsToCanonicalOutput._make(_result)
  return _result

_CudnnRNNParamsToCanonicalV2Output = collections.namedtuple(
    "CudnnRNNParamsToCanonicalV2",
    ["weights", "biases"])


TV_CudnnRNNParamsToCanonicalV2_T = TypeVar("TV_CudnnRNNParamsToCanonicalV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnn_params_to_canonical_v2(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], params: Annotated[Any, TV_CudnnRNNParamsToCanonicalV2_T], num_params_weights: int, num_params_biases: int, rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, num_proj:int=0, name=None):
  r"""Retrieves CudnnRNN params in canonical form. It supports the projection in LSTM.

  Retrieves a set of weights from the opaque params buffer that can be saved and
  restored in a way compatible with future runs.

  Note that the params buffer may not be compatible across different GPUs. So any
  save and restoration should be converted to and from the canonical weights and
  biases.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  num_params_weights: number of weight parameter matrix for all layers.
  num_params_biases: number of bias parameter vector for all layers.
  weights: the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  biases: the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.
  num_proj: The output dimensionality for the projection matrices. If None or 0,
      no projection is performed.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    params: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    num_params_weights: An `int` that is `>= 1`.
    num_params_biases: An `int` that is `>= 1`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (weights, biases).

    weights: A list of `num_params_weights` `Tensor` objects with the same type as `params`.
    biases: A list of `num_params_biases` `Tensor` objects with the same type as `params`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNParamsToCanonicalV2", name, num_layers, num_units,
        input_size, params, "num_params_weights", num_params_weights,
        "num_params_biases", num_params_biases, "rnn_mode", rnn_mode,
        "input_mode", input_mode, "direction", direction, "dropout", dropout,
        "seed", seed, "seed2", seed2, "num_proj", num_proj)
      _result = _CudnnRNNParamsToCanonicalV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnn_params_to_canonical_v2_eager_fallback(
          num_layers, num_units, input_size, params,
          num_params_weights=num_params_weights,
          num_params_biases=num_params_biases, rnn_mode=rnn_mode,
          input_mode=input_mode, direction=direction, dropout=dropout,
          seed=seed, seed2=seed2, num_proj=num_proj, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_params_weights = _execute.make_int(num_params_weights, "num_params_weights")
  num_params_biases = _execute.make_int(num_params_biases, "num_params_biases")
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNParamsToCanonicalV2", num_layers=num_layers,
                                       num_units=num_units,
                                       input_size=input_size, params=params,
                                       num_params_weights=num_params_weights,
                                       num_params_biases=num_params_biases,
                                       rnn_mode=rnn_mode,
                                       input_mode=input_mode,
                                       direction=direction, dropout=dropout,
                                       seed=seed, seed2=seed2,
                                       num_proj=num_proj, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_params_weights",
              _op._get_attr_int("num_params_weights"), "num_params_biases",
              _op._get_attr_int("num_params_biases"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "num_proj",
              _op._get_attr_int("num_proj"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNParamsToCanonicalV2", _inputs_flat, _attrs, _result)
  _result = [_result[:num_params_weights]] + _result[num_params_weights:]
  _result = _result[:1] + [_result[1:]]
  _result = _CudnnRNNParamsToCanonicalV2Output._make(_result)
  return _result

CudnnRNNParamsToCanonicalV2 = tf_export("raw_ops.CudnnRNNParamsToCanonicalV2")(_ops.to_raw_op(cudnn_rnn_params_to_canonical_v2))


def cudnn_rnn_params_to_canonical_v2_eager_fallback(num_layers: Annotated[Any, _atypes.Int32], num_units: Annotated[Any, _atypes.Int32], input_size: Annotated[Any, _atypes.Int32], params: Annotated[Any, TV_CudnnRNNParamsToCanonicalV2_T], num_params_weights: int, num_params_biases: int, rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, num_proj: int, name, ctx):
  num_params_weights = _execute.make_int(num_params_weights, "num_params_weights")
  num_params_biases = _execute.make_int(num_params_biases, "num_params_biases")
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  _attr_T, (params,) = _execute.args_to_matching_eager([params], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  num_layers = _ops.convert_to_tensor(num_layers, _dtypes.int32)
  num_units = _ops.convert_to_tensor(num_units, _dtypes.int32)
  input_size = _ops.convert_to_tensor(input_size, _dtypes.int32)
  _inputs_flat = [num_layers, num_units, input_size, params]
  _attrs = ("T", _attr_T, "num_params_weights", num_params_weights,
  "num_params_biases", num_params_biases, "rnn_mode", rnn_mode, "input_mode",
  input_mode, "direction", direction, "dropout", dropout, "seed", seed,
  "seed2", seed2, "num_proj", num_proj)
  _result = _execute.execute(b"CudnnRNNParamsToCanonicalV2",
                             num_params_weights + num_params_biases,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNParamsToCanonicalV2", _inputs_flat, _attrs, _result)
  _result = [_result[:num_params_weights]] + _result[num_params_weights:]
  _result = _result[:1] + [_result[1:]]
  _result = _CudnnRNNParamsToCanonicalV2Output._make(_result)
  return _result

_CudnnRNNV2Output = collections.namedtuple(
    "CudnnRNNV2",
    ["output", "output_h", "output_c", "reserve_space", "host_reserved"])


TV_CudnnRNNV2_T = TypeVar("TV_CudnnRNNV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnnv2(input: Annotated[Any, TV_CudnnRNNV2_T], input_h: Annotated[Any, TV_CudnnRNNV2_T], input_c: Annotated[Any, TV_CudnnRNNV2_T], params: Annotated[Any, TV_CudnnRNNV2_T], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, is_training:bool=True, name=None):
  r"""A RNN backed by cuDNN.

  Computes the RNN from the input and initial states, with respect to the params
  buffer. Produces one extra output "host_reserved" than CudnnRNN.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicates whether there is a linear projection between the input and
    the actual computation before the first layer. 'skip_input' is only allowed
    when input_size == num_units; 'auto_select' implies 'skip_input' when
    input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
  input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  output: A 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  is_training: Indicates whether this operation is used for inference or
    training.
  reserve_space: An opaque tensor that can be used in backprop calculation. It
    is only produced if is_training is true.
  host_reserved: An opaque tensor that can be used in backprop calculation. It is
    only produced if is_training is true. It is output on host memory rather than
    device memory.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    is_training: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_h, output_c, reserve_space, host_reserved).

    output: A `Tensor`. Has the same type as `input`.
    output_h: A `Tensor`. Has the same type as `input`.
    output_c: A `Tensor`. Has the same type as `input`.
    reserve_space: A `Tensor`. Has the same type as `input`.
    host_reserved: A `Tensor` of type `int8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNV2", name, input, input_h, input_c, params, "rnn_mode",
        rnn_mode, "input_mode", input_mode, "direction", direction, "dropout",
        dropout, "seed", seed, "seed2", seed2, "is_training", is_training)
      _result = _CudnnRNNV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnnv2_eager_fallback(
          input, input_h, input_c, params, rnn_mode=rnn_mode,
          input_mode=input_mode, direction=direction, dropout=dropout,
          seed=seed, seed2=seed2, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNV2", input=input, input_h=input_h, input_c=input_c,
                      params=params, rnn_mode=rnn_mode, input_mode=input_mode,
                      direction=direction, dropout=dropout, seed=seed,
                      seed2=seed2, is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNV2", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNV2Output._make(_result)
  return _result

CudnnRNNV2 = tf_export("raw_ops.CudnnRNNV2")(_ops.to_raw_op(cudnn_rnnv2))


def cudnn_rnnv2_eager_fallback(input: Annotated[Any, TV_CudnnRNNV2_T], input_h: Annotated[Any, TV_CudnnRNNV2_T], input_c: Annotated[Any, TV_CudnnRNNV2_T], params: Annotated[Any, TV_CudnnRNNV2_T], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, is_training: bool, name, ctx):
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_h, input_c, params) = _inputs_T
  _inputs_flat = [input, input_h, input_c, params]
  _attrs = ("T", _attr_T, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2,
  "is_training", is_training)
  _result = _execute.execute(b"CudnnRNNV2", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNV2", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNV2Output._make(_result)
  return _result

_CudnnRNNV3Output = collections.namedtuple(
    "CudnnRNNV3",
    ["output", "output_h", "output_c", "reserve_space", "host_reserved"])


TV_CudnnRNNV3_T = TypeVar("TV_CudnnRNNV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def cudnn_rnnv3(input: Annotated[Any, TV_CudnnRNNV3_T], input_h: Annotated[Any, TV_CudnnRNNV3_T], input_c: Annotated[Any, TV_CudnnRNNV3_T], params: Annotated[Any, TV_CudnnRNNV3_T], sequence_lengths: Annotated[Any, _atypes.Int32], rnn_mode:str="lstm", input_mode:str="linear_input", direction:str="unidirectional", dropout:float=0, seed:int=0, seed2:int=0, num_proj:int=0, is_training:bool=True, time_major:bool=True, name=None):
  r"""A RNN backed by cuDNN.

  Computes the RNN from the input and initial states, with respect to the params
  buffer. Accepts one extra input "sequence_lengths" than CudnnRNN.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicates whether there is a linear projection between the input and
    the actual computation before the first layer. 'skip_input' is only allowed
    when input_size == num_units; 'auto_select' implies 'skip_input' when
    input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: If time_major is true, this is a 3-D tensor with the shape of
      [seq_length, batch_size, input_size]. If time_major is false, the shape is
      [batch_size, seq_length, input_size].
  input_h: If time_major is true, this is a 3-D tensor with the shape of
      [num_layer * dir, batch_size, num_units]. If time_major is false, the shape
      is [batch_size, num_layer * dir, num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  sequence_lengths: a vector of lengths of each input sequence.
  output: If time_major is true, this is a 3-D tensor with the shape of
      [seq_length, batch_size, dir * num_units]. If time_major is false, the
      shape is [batch_size, seq_length, dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  is_training: Indicates whether this operation is used for inference or
    training.
  time_major: Indicates whether the input/output format is time major or batch
      major.
  reserve_space: An opaque tensor that can be used in backprop calculation. It
    is only produced if is_training is true.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    sequence_lengths: A `Tensor` of type `int32`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    is_training: An optional `bool`. Defaults to `True`.
    time_major: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_h, output_c, reserve_space, host_reserved).

    output: A `Tensor`. Has the same type as `input`.
    output_h: A `Tensor`. Has the same type as `input`.
    output_c: A `Tensor`. Has the same type as `input`.
    reserve_space: A `Tensor`. Has the same type as `input`.
    host_reserved: A `Tensor` of type `int8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CudnnRNNV3", name, input, input_h, input_c, params,
        sequence_lengths, "rnn_mode", rnn_mode, "input_mode", input_mode,
        "direction", direction, "dropout", dropout, "seed", seed, "seed2",
        seed2, "num_proj", num_proj, "is_training", is_training, "time_major",
        time_major)
      _result = _CudnnRNNV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cudnn_rnnv3_eager_fallback(
          input, input_h, input_c, params, sequence_lengths,
          rnn_mode=rnn_mode, input_mode=input_mode, direction=direction,
          dropout=dropout, seed=seed, seed2=seed2, num_proj=num_proj,
          is_training=is_training, time_major=time_major, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  if time_major is None:
    time_major = True
  time_major = _execute.make_bool(time_major, "time_major")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CudnnRNNV3", input=input, input_h=input_h, input_c=input_c,
                      params=params, sequence_lengths=sequence_lengths,
                      rnn_mode=rnn_mode, input_mode=input_mode,
                      direction=direction, dropout=dropout, seed=seed,
                      seed2=seed2, num_proj=num_proj, is_training=is_training,
                      time_major=time_major, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "rnn_mode",
              _op.get_attr("rnn_mode"), "input_mode",
              _op.get_attr("input_mode"), "direction",
              _op.get_attr("direction"), "dropout", _op.get_attr("dropout"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "num_proj",
              _op._get_attr_int("num_proj"), "is_training",
              _op._get_attr_bool("is_training"), "time_major",
              _op._get_attr_bool("time_major"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CudnnRNNV3", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNV3Output._make(_result)
  return _result

CudnnRNNV3 = tf_export("raw_ops.CudnnRNNV3")(_ops.to_raw_op(cudnn_rnnv3))


def cudnn_rnnv3_eager_fallback(input: Annotated[Any, TV_CudnnRNNV3_T], input_h: Annotated[Any, TV_CudnnRNNV3_T], input_c: Annotated[Any, TV_CudnnRNNV3_T], params: Annotated[Any, TV_CudnnRNNV3_T], sequence_lengths: Annotated[Any, _atypes.Int32], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, num_proj: int, is_training: bool, time_major: bool, name, ctx):
  if rnn_mode is None:
    rnn_mode = "lstm"
  rnn_mode = _execute.make_str(rnn_mode, "rnn_mode")
  if input_mode is None:
    input_mode = "linear_input"
  input_mode = _execute.make_str(input_mode, "input_mode")
  if direction is None:
    direction = "unidirectional"
  direction = _execute.make_str(direction, "direction")
  if dropout is None:
    dropout = 0
  dropout = _execute.make_float(dropout, "dropout")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if num_proj is None:
    num_proj = 0
  num_proj = _execute.make_int(num_proj, "num_proj")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  if time_major is None:
    time_major = True
  time_major = _execute.make_bool(time_major, "time_major")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, input_h, input_c, params) = _inputs_T
  sequence_lengths = _ops.convert_to_tensor(sequence_lengths, _dtypes.int32)
  _inputs_flat = [input, input_h, input_c, params, sequence_lengths]
  _attrs = ("T", _attr_T, "rnn_mode", rnn_mode, "input_mode", input_mode,
  "direction", direction, "dropout", dropout, "seed", seed, "seed2", seed2,
  "num_proj", num_proj, "is_training", is_training, "time_major", time_major)
  _result = _execute.execute(b"CudnnRNNV3", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CudnnRNNV3", _inputs_flat, _attrs, _result)
  _result = _CudnnRNNV3Output._make(_result)
  return _result


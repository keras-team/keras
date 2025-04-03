# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import importlib

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.stablehlo as hlo

import numpy as np

from jaxlib import xla_client
from .gpu_common_utils import GpuLibNotLinkedError

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuda_rnn = importlib.import_module(f"{cuda_module_name}._rnn", package="jaxlib")
  except ImportError:
    _cuda_rnn = None
  else:
    break

if _cuda_rnn:
  for _name, _value in _cuda_rnn.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform='CUDA')
  compute_rnn_workspace_reserve_space_sizes = _cuda_rnn.compute_rnn_workspace_reserve_space_sizes


for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hip_rnn = importlib.import_module(f"{rocm_module_name}._rnn", package="jaxlib")
  except ImportError:
    _hip_rnn = None
  else:
    break

if _hip_rnn:
  for _name, _value in _hip_rnn.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform='ROCM')
  compute_rnn_workspace_reserve_space_sizes = _hip_rnn.compute_rnn_workspace_reserve_space_sizes


def _rnn_fwd_lowering(_rnn, platform, ctx, input, h_0, c_0, weights, seq_lengths, *,
                       input_size: int, hidden_size: int, num_layers: int,
                       dropout: bool, bidirectional: bool,
                       cudnn_allow_tf32: bool):
  """CuDnn RNN."""
  out_dtype = ctx.avals_out[0].dtype
  if out_dtype == np.float32:
    out_type = ir.F32Type.get()
  elif out_dtype == np.float64:
    out_type = ir.F64Type.get()
  elif out_dtype == np.complex64:
    out_type = ir.ComplexType.get(ir.F32Type.get())
  elif out_dtype == np.complex128:
    out_type = ir.ComplexType.get(ir.F64Type.get())
  else:
    raise ValueError(f'Unknown output type {out_dtype}')

  output_type = ir.RankedTensorType.get(ctx.avals_out[0].shape, out_type)
  batch_size = ctx.avals_in[0].shape[0]
  max_seq_length = ctx.avals_in[0].shape[1]
  # workspace_shape = ctx.avals_out[3].shape
  workspace_size, _ = compute_rnn_workspace_reserve_space_sizes(
      input_size, hidden_size, num_layers, batch_size, max_seq_length,
      dropout, bidirectional, cudnn_allow_tf32)
  workspace_shape = (workspace_size,)
  workspace_type = ir.RankedTensorType.get(workspace_shape, ir.F32Type.get())
  reserve_space_shape = ctx.avals_out[3].shape
  reserve_space_type = ir.RankedTensorType.get(reserve_space_shape,
                                               ir.F32Type.get())
  if not _rnn:
    raise GpuLibNotLinkedError()

  opaque = _rnn.build_rnn_descriptor(input_size, hidden_size, num_layers,
                                     batch_size, max_seq_length, dropout,
                                     bidirectional, cudnn_allow_tf32,
                                     workspace_shape[0],
                                     reserve_space_shape[0])

  i32_type = ir.IntegerType.get_signless(32)
  out = hlo.CustomCallOp(
      [output_type, h_0.type, c_0.type, workspace_type, reserve_space_type],
      [input, h_0, c_0, weights, seq_lengths],
      call_target_name=ir.StringAttr.get(f"{platform}dnn_rnn"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
  )
  return out.results[:-2] + out.results[-1:]  # drop workspace output

cudnn_rnn_lowering = partial(_rnn_fwd_lowering, _cuda_rnn, "cu")
miopen_rnn_lowering = partial(_rnn_fwd_lowering, _hip_rnn, "hip")


def _hlo_zeros_f32(shape):
  return hlo.constant(
      ir.DenseElementsAttr.get(
          np.zeros(shape, dtype=np.float32), type=ir.F32Type.get()))


def _rnn_bwd_lowering(_rnn, platform, ctx, dy, dhn, dcn, x, h0, c0, w, y,
                           reserve_space, seq_lengths, *, input_size: int,
                           hidden_size: int, num_layers: int, dropout: bool,
                           bidirectional: bool, cudnn_allow_tf32: bool):
  """CuDnn RNN Backward pass."""
  batch_size = ctx.avals_in[3].shape[0]
  max_seq_length = ctx.avals_in[3].shape[1]
  workspace_size, _ = compute_rnn_workspace_reserve_space_sizes(
      input_size, hidden_size, num_layers, batch_size, max_seq_length,
      dropout, bidirectional, cudnn_allow_tf32)
  workspace_shape = (workspace_size,)
  workspace_type = ir.RankedTensorType.get(workspace_shape, ir.F32Type.get())
  reserve_space_shape = ctx.avals_in[8].shape

  if _rnn is None:
    raise RuntimeError("cuda couldn't be imported")
  opaque = _rnn.build_rnn_descriptor(input_size, hidden_size, num_layers,
                                     batch_size, max_seq_length, dropout,
                                     bidirectional, cudnn_allow_tf32,
                                     workspace_shape[0],
                                     reserve_space_shape[0])

  i32_type = ir.IntegerType.get_signless(32)
  zeroed_dw = _hlo_zeros_f32(ctx.avals_out[3].shape)
  out = hlo.CustomCallOp(
      [x.type, h0.type, c0.type, w.type, workspace_type], [
          dy, dhn, dcn, x, h0, c0, w, y, reserve_space, zeroed_dw,
          seq_lengths
      ],
      call_target_name=ir.StringAttr.get(f"{platform}dnn_rnn_bwd"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      output_operand_aliases=ir.ArrayAttr.get([
          hlo.OutputOperandAlias.get(
              output_tuple_indices=[3],
              operand_index=9,
              operand_tuple_indices=[])
      ]))
  return out.results[:-1]  # drop workspace output

cudnn_rnn_bwd_lowering = partial(_rnn_bwd_lowering, _cuda_rnn, "cu")
miopen_rnn_bwd_lowering = partial(_rnn_bwd_lowering, _hip_rnn, "hip")

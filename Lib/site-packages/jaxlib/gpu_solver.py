# Copyright 2019 The JAX Authors.
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

import numpy as np

from jaxlib import xla_client

from .hlo_helpers import custom_call

try:
  from .cuda import _blas as _cublas  # pytype: disable=import-error
except ImportError:
  for cuda_module_name in ["jax_cuda12_plugin"]:
    try:
      _cublas = importlib.import_module(f"{cuda_module_name}._blas")
    except ImportError:
      _cublas = None
    else:
      break

if _cublas:
  for _name, _value in _cublas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cusolver = importlib.import_module(
        f"{cuda_module_name}._solver", package="jaxlib"
    )
  except ImportError:
    _cusolver = None
  else:
    break

if _cusolver:
  for _name, _value in _cusolver.registrations().items():
    # TODO(danfm): Clean up after all legacy custom calls are ported.
    api_version = 1 if _name.endswith("_ffi") else 0
    xla_client.register_custom_call_target(_name, _value, platform="CUDA",
                                           api_version=api_version)

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuhybrid = importlib.import_module(
        f"{cuda_module_name}._hybrid", package="jaxlib"
    )
  except ImportError:
    _cuhybrid = None
  else:
    break

if _cuhybrid:
  for _name, _value in _cuhybrid.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA",
                                           api_version=1)

try:
  from .rocm import _blas as _hipblas  # pytype: disable=import-error
except ImportError:
  for rocm_module_name in ["jax_rocm60_plugin"]:
    try:
      _hipblas = importlib.import_module(f"{rocm_module_name}._blas")
    except:
      _hipblas = None
    else:
      break

if _hipblas:
  for _name, _value in _hipblas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hipsolver = importlib.import_module(
        f"{rocm_module_name}._solver", package="jaxlib"
    )
  except ImportError:
    _hipsolver = None
  else:
    break

if _hipsolver:
  for _name, _value in _hipsolver.registrations().items():
    # TODO(danfm): Clean up after all legacy custom calls are ported.
    api_version = 1 if _name.endswith("_ffi") else 0
    xla_client.register_custom_call_target(_name, _value, platform="ROCM",
                                           api_version=api_version)

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hiphybrid = importlib.import_module(
        f"{rocm_module_name}._hybrid", package="jaxlib"
    )
  except ImportError:
    _hiphybrid = None
  else:
    break

if _hiphybrid:
  for _name, _value in _hiphybrid.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM",
                                           api_version=1)

def initialize_hybrid_kernels():
  if _cuhybrid:
    _cuhybrid.initialize()
  if _hiphybrid:
    _hiphybrid.initialize()

def has_magma():
  if _cuhybrid:
    return _cuhybrid.has_magma()
  if _hiphybrid:
    return _hiphybrid.has_magma()
  return False

def _csrlsvqr_hlo(platform, gpu_solver, dtype, data,
                  indices, indptr, b, tol, reorder):
  """Sparse solver via QR decomposition. CUDA only."""
  b_type = ir.RankedTensorType(b.type)
  data_type = ir.RankedTensorType(data.type)

  n = b_type.shape[0]
  nnz = data_type.shape[0]
  opaque = gpu_solver.build_csrlsvqr_descriptor(
      np.dtype(dtype), n, nnz, reorder, tol
  )

  out = custom_call(
      f"{platform}solver_csrlsvqr",  # call_target_name
      result_types=[b.type],
      operands=[data, indptr, indices, b],
      backend_config=opaque,  # backend_config
      operand_layouts=[(0,), (0,), (0,), (0,)],  # operand_layouts
      result_layouts=[(0,)]  # result_layouts
  ).results
  return out

cuda_csrlsvqr = partial(_csrlsvqr_hlo, "cu", _cusolver)

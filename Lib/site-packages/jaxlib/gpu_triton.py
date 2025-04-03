# Copyright 2023 The JAX Authors.
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
import importlib

from jaxlib import xla_client

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuda_triton = importlib.import_module(
        f"{cuda_module_name}._triton", package="jaxlib"
    )
  except ImportError:
    _cuda_triton = None
  else:
    break

if _cuda_triton:
  xla_client.register_custom_call_target(
      "triton_kernel_call", _cuda_triton.get_custom_call(),
      platform='CUDA')
  TritonKernelCall = _cuda_triton.TritonKernelCall
  TritonAutotunedKernelCall = _cuda_triton.TritonAutotunedKernelCall
  TritonKernel = _cuda_triton.TritonKernel
  create_array_parameter = _cuda_triton.create_array_parameter
  create_scalar_parameter = _cuda_triton.create_scalar_parameter
  get_compute_capability = _cuda_triton.get_compute_capability
  get_arch_details = _cuda_triton.get_arch_details
  get_custom_call = _cuda_triton.get_custom_call
  get_serialized_metadata = _cuda_triton.get_serialized_metadata

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hip_triton = importlib.import_module(
        f"{rocm_module_name}._triton", package="jaxlib"
    )
  except ImportError:
    _hip_triton = None
  else:
    break

if _hip_triton:
  xla_client.register_custom_call_target(
      "triton_kernel_call", _hip_triton.get_custom_call(),
      platform='ROCM')
  TritonKernelCall = _hip_triton.TritonKernelCall
  TritonAutotunedKernelCall = _hip_triton.TritonAutotunedKernelCall
  TritonKernel = _hip_triton.TritonKernel
  create_array_parameter = _hip_triton.create_array_parameter
  create_scalar_parameter = _hip_triton.create_scalar_parameter
  get_compute_capability = _hip_triton.get_compute_capability
  get_arch_details = _hip_triton.get_arch_details
  get_custom_call = _hip_triton.get_custom_call
  get_serialized_metadata = _hip_triton.get_serialized_metadata

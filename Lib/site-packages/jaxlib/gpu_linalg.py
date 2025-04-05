# Copyright 2021 The JAX Authors.
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
    _cuda_linalg = importlib.import_module(
        f"{cuda_module_name}._linalg", package="jaxlib"
    )
  except ImportError:
    _cuda_linalg = None
  else:
    break

if _cuda_linalg:
  for _name, _value in _cuda_linalg.registrations().items():
    api_version = (1
                   if _name.endswith("lu_pivots_to_permutation")
                   or _name.endswith("_ffi") else 0)
    xla_client.register_custom_call_target(
        _name, _value, platform="CUDA", api_version=api_version
    )

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hip_linalg = importlib.import_module(
        f"{rocm_module_name}._linalg", package="jaxlib"
    )
  except ImportError:
    _hip_linalg = None
  else:
    break

if _hip_linalg:
  for _name, _value in _hip_linalg.registrations().items():
    api_version = (1
                   if _name.endswith("lu_pivots_to_permutation")
                   or _name.endswith("_ffi") else 0)
    xla_client.register_custom_call_target(
        _name, _value, platform="ROCM", api_version=api_version
    )

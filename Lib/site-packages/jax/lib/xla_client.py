# Copyright 2024 The JAX Authors.
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

from jax._src.lax.fft import FftType as _FftType
from jax._src.lib import xla_client as _xc

get_topology_for_devices = _xc.get_topology_for_devices
heap_profile = _xc.heap_profile
mlir_api_version = _xc.mlir_api_version
Client = _xc.Client
CompileOptions = _xc.CompileOptions
DeviceAssignment = _xc.DeviceAssignment
Frame = _xc.Frame
HloSharding = _xc.HloSharding
OpSharding = _xc.OpSharding
Traceback = _xc.Traceback

_deprecations = {
    # Finalized 2024-12-11; remove after 2025-3-11
    "_xla": (
        "jax.lib.xla_client._xla was removed in JAX v0.4.38; use jax.lib.xla_extension.",
        None,
    ),
    "bfloat16": (
        "jax.lib.xla_client.bfloat16 was removed in JAX v0.4.38; use ml_dtypes.bfloat16.",
        None,
    ),
    # Finalized 2024-12-23; remove after 2024-03-23
    "Device": (
        "jax.lib.xla_client.Device is deprecated; use jax.Device instead.",
        None,
    ),
    "XlaRuntimeError": (
        (
            "jax.lib.xla_client.XlaRuntimeError is deprecated; use"
            " jax.errors.JaxRuntimeError."
        ),
        None,
    ),
    # Added Oct 10 2024
    "FftType": (
        "jax.lib.xla_client.FftType is deprecated; use jax.lax.FftType.",
        _FftType,
    ),
    "PaddingType": (
        (
            "jax.lib.xla_client.PaddingType is deprecated; this type is unused"
            " by JAX so there is no replacement."
        ),
        _xc.PaddingType,
    ),
    # Added Oct 11 2024
    "dtype_to_etype": (
        "dtype_to_etype is deprecated; use StableHLO instead.",
        _xc.dtype_to_etype,
    ),
    "ops": (
        "ops is deprecated; use StableHLO instead.",
        _xc.ops,
    ),
    "register_custom_call_target": (
        "register_custom_call_target is deprecated; use the JAX FFI instead "
        "(https://jax.readthedocs.io/en/latest/ffi.html)",
        _xc.register_custom_call_target,
    ),
    "shape_from_pyval": (
        "shape_from_pyval is deprecated; use StableHLO instead.",
        _xc.shape_from_pyval,
    ),
    "PrimitiveType": (
        "PrimitiveType is deprecated; use StableHLO instead.",
        _xc.PrimitiveType,
    ),
    "Shape": (
        "Shape is deprecated; use StableHLO instead.",
        _xc.Shape,
    ),
    "XlaBuilder": (
        "XlaBuilder is deprecated; use StableHLO instead.",
        _xc.XlaBuilder,
    ),
    "XlaComputation": (
        "XlaComputation is deprecated; use StableHLO instead.",
        _xc.XlaComputation,
    ),
    # Added Nov 20 2024
    "ArrayImpl": (
        "jax.lib.xla_client.ArrayImpl is deprecated; use jax.Array instead.",
        _xc.ArrayImpl,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  dtype_to_etype = _xc.dtype_to_etype
  ops = _xc.ops
  register_custom_call_target = _xc.register_custom_call_target
  shape_from_pyval = _xc.shape_from_pyval
  ArrayImpl = _xc.ArrayImpl
  Device = _xc.Device
  FftType = _FftType
  PaddingType = _xc.PaddingType
  PrimitiveType = _xc.PrimitiveType
  Shape = _xc.Shape
  XlaBuilder = _xc.XlaBuilder
  XlaComputation = _xc.XlaComputation
  XlaRuntimeError = _xc.XlaRuntimeError
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _FftType
del _xc

# Copyright 2018 The JAX Authors.
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

# This module is largely a wrapper around `jaxlib` that performs version
# checking on import.

from __future__ import annotations

import gc
import os
import pathlib
import re

try:
  import jaxlib as jaxlib
except ModuleNotFoundError as err:
  raise ModuleNotFoundError(
    'jax requires jaxlib to be installed. See '
    'https://github.com/jax-ml/jax#installation for installation instructions.'
    ) from err

import jax.version
from jax.version import _minimum_jaxlib_version as _minimum_jaxlib_version_str
try:
  import jaxlib.version
except Exception as err:
  # jaxlib is too old to have version number.
  msg = f'This version of jax requires jaxlib version >= {_minimum_jaxlib_version_str}.'
  raise ImportError(msg) from err


# Checks the jaxlib version before importing anything else from jaxlib.
# Returns the jaxlib version string.
def check_jaxlib_version(jax_version: str, jaxlib_version: str,
                         minimum_jaxlib_version: str) -> tuple[int, ...]:
  # Regex to match a dotted version prefix 0.1.23.456.789 of a PEP440 version.
  # PEP440 allows a number of non-numeric suffixes, which we allow also.
  # We currently do not allow an epoch.
  version_regex = re.compile(r"[0-9]+(?:\.[0-9]+)*")
  def _parse_version(v: str) -> tuple[int, ...]:
    m = version_regex.match(v)
    if m is None:
      raise ValueError(f"Unable to parse jaxlib version '{v}'")
    return tuple(int(x) for x in m.group(0).split('.'))

  _jax_version = _parse_version(jax_version)
  _minimum_jaxlib_version = _parse_version(minimum_jaxlib_version)
  _jaxlib_version = _parse_version(jaxlib_version)

  if _jaxlib_version < _minimum_jaxlib_version:
    msg = (f'jaxlib is version {jaxlib_version}, but this version '
           f'of jax requires version >= {minimum_jaxlib_version}.')
    raise RuntimeError(msg)

  if _jaxlib_version > _jax_version:
    raise RuntimeError(
        f'jaxlib version {jaxlib_version} is newer than and '
        f'incompatible with jax version {jax_version}. Please '
        'update your jax and/or jaxlib packages.')
  return _jaxlib_version


version_str = jaxlib.version.__version__
version = check_jaxlib_version(
  jax_version=jax.version.__version__,
  jaxlib_version=jaxlib.version.__version__,
  minimum_jaxlib_version=jax.version._minimum_jaxlib_version)

# Before importing any C compiled modules from jaxlib, first import the CPU
# feature guard module to verify that jaxlib was compiled in a way that only
# uses instructions that are present on this machine.
import jaxlib.cpu_feature_guard as cpu_feature_guard
cpu_feature_guard.check_cpu_features()

import jaxlib.utils as utils  # noqa: F401
import jaxlib.xla_client as xla_client
import jaxlib.lapack as lapack  # noqa: F401

xla_extension = xla_client._xla
pytree = xla_client._xla.pytree
jax_jit = xla_client._xla.jax_jit
pmap_lib = xla_client._xla.pmap_lib

# XLA garbage collection: see https://github.com/jax-ml/jax/issues/14882
def _xla_gc_callback(*args):
  xla_client._xla.collect_garbage()
gc.callbacks.append(_xla_gc_callback)

try:
  import jaxlib.cuda._versions as cuda_versions  # pytype: disable=import-error  # noqa: F401
except ImportError:
  try:
    import jax_cuda12_plugin._versions as cuda_versions  # pytype: disable=import-error  # noqa: F401
  except ImportError:
    cuda_versions = None

import jaxlib.gpu_solver as gpu_solver  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_sparse as gpu_sparse  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_prng as gpu_prng  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_linalg as gpu_linalg  # pytype: disable=import-error  # noqa: F401
import jaxlib.hlo_helpers as hlo_helpers  # pytype: disable=import-error  # noqa: F401

# Jaxlib code is split between the Jax and the Tensorflow repositories.
# Only for the internal usage of the JAX developers, we expose a version
# number that can be used to perform changes without breaking the main
# branch on the Jax github.
xla_extension_version: int = getattr(xla_client, '_version', 0)

import jaxlib.gpu_rnn as gpu_rnn  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_triton as gpu_triton # pytype: disable=import-error  # noqa: F401

try:
  import jaxlib.mosaic.python.mosaic_gpu as mosaic_gpu_dialect  # pytype: disable=import-error
except ImportError:
  # TODO(bchetioui): Remove this when minimum jaxlib version >= 0.4.36.
  # Jaxlib doesn't contain Mosaic GPU dialect bindings.
  mosaic_gpu_dialect = None  # type: ignore

import jaxlib.mosaic.python.tpu as tpu  # pytype: disable=import-error  # noqa: F401

# Version number for MLIR:Python APIs, provided by jaxlib.
mlir_api_version = xla_client.mlir_api_version

# TODO(rocm): check if we need the same for rocm.

def _cuda_path() -> str | None:
  def _try_cuda_root_environment_variable() -> str | None:
    """Use `CUDA_ROOT` environment variable if set."""
    return os.environ.get('CUDA_ROOT', None)

  def _try_cuda_nvcc_import() -> str | None:
    """Try to import `cuda_nvcc` and get its path directly.

    If the pip package `nvidia-cuda-nvcc-cu11` is installed, it should have
    both of the things XLA looks for in the cuda path, namely `bin/ptxas` and
    `nvvm/libdevice/libdevice.10.bc`.
    """
    try:
      from nvidia import cuda_nvcc  # pytype: disable=import-error
    except ImportError:
      return None

    if hasattr(cuda_nvcc, '__file__') and cuda_nvcc.__file__ is not None:
      # `cuda_nvcc` is a regular package.
      cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent
    elif hasattr(cuda_nvcc, '__path__') and cuda_nvcc.__path__ is not None:
      # `cuda_nvcc` is a namespace package, which might have multiple paths.
      cuda_nvcc_path = None
      for path in cuda_nvcc.__path__:
        if (pathlib.Path(path) / 'bin' / 'ptxas').exists():
          cuda_nvcc_path = pathlib.Path(path)
          break
    else:
      return None

    return str(cuda_nvcc_path)

  if (path := _try_cuda_root_environment_variable()) is not None:
    return path
  elif (path := _try_cuda_nvcc_import()) is not None:
    return path

  return None

cuda_path = _cuda_path()

guard_lib = xla_client._xla.guard_lib
Device = xla_client._xla.Device

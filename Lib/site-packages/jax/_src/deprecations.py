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

from dataclasses import dataclass
import functools
from types import ModuleType
import warnings

# Module __getattr__ factory that warns if deprecated names are used.
#
# Example usage:
# from jax._src.interpreters.pxla import (
#   Mesh as _deprecated_Mesh,
# )
#
# _deprecations = {
#   # Added Feb 8, 2023:
#   "Mesh": (
#     "jax.interpreters.pxla.Mesh is deprecated. Use jax.sharding.Mesh.",
#     _deprecated_Mesh,
#   ),
# }
#
# from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
# __getattr__ = _deprecation_getattr(__name__, _deprecations)
# del _deprecation_getattr

# Note that type checkers such as Pytype will not know about the deprecated
# names. If it is desirable that a deprecated name is known to the type checker,
# add:
# import typing
# if typing.TYPE_CHECKING:
#   from jax._src.interpreters.pxla import (
#     Mesh as Mesh,
#   )
# del typing
def deprecation_getattr(module, deprecations):
  @functools.cache
  def getattr(name):
    if name in deprecations:
      message, fn = deprecations[name]
      if fn is None:  # Is the deprecation accelerated?
        raise AttributeError(message)
      warnings.warn(message, DeprecationWarning, stacklevel=2)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return getattr


def accelerate_getattr_deprecation(module: ModuleType, name: str) -> None:
  """Accelerate the deprecation of a module-level attribute.

  Raises an AttributeError instead of a DeprecationWarning upon attribute access.
  Used in Google-internal code to implement faster deprecation.
  """
  message, _ = module._deprecations[name]
  module._deprecations[name] = (message, None)

def is_accelerated_attribute(module: ModuleType, name: str) -> bool:
  """Returns true if given name is accelerated.

  Raises an error if name is not a deprecated attribute in module.
  """
  return module._deprecations[name][1] is None

# The following mechanism is a separate one, for registering and
# accelerating deprecations that are not imports (for example, deprecations
# of a function argument).
# Maps a globally unique string ID to a DeprecationState, which tracks whether
# the deprecation is accelerated.
# The intent is that non-accelerated deprecations will warn, and accelerated
# deprecations will error.

@dataclass
class DeprecationState:
  accelerated: bool = False

_registered_deprecations: dict[str, DeprecationState] = {}


def register(deprecation_id: str) -> None:
  _registered_deprecations[deprecation_id] = DeprecationState()


def unregister(deprecation_id: str) -> None:
  if deprecation_id not in _registered_deprecations:
    raise ValueError(f"{deprecation_id=!r} not registered.")
  _registered_deprecations.pop(deprecation_id)


def accelerate(deprecation_id: str) -> None:
  if deprecation_id not in _registered_deprecations:
    raise ValueError(f"{deprecation_id=!r} not registered.")
  _registered_deprecations[deprecation_id].accelerated = True


def is_accelerated(deprecation_id: str) -> bool:
  if deprecation_id not in _registered_deprecations:
    raise ValueError(f"{deprecation_id=!r} not registered.")
  return _registered_deprecations[deprecation_id].accelerated


def warn(deprecation_id: str, message: str, stacklevel: int) -> None:
  """Warns about a deprecation, or errors if the deprecation is accelerated."""
  if is_accelerated(deprecation_id):
    raise ValueError(message)
  else:
    warnings.warn(message, category=DeprecationWarning,
                  stacklevel=stacklevel + 1)


# Register a number of deprecations: we do this here to ensure they're
# always registered by the time `accelerate` and `is_acelerated` are called.
register('jax-aval-named-shape')
register('jax-dlpack-import-legacy')
register('jax-nn-one-hot-float-input')
register("jax-numpy-astype-complex-to-real")
register("jax-numpy-array-none")
register('jax-numpy-clip-args')
register('jax-numpy-linalg-matrix_rank-tol')
register('jax-numpy-linalg-pinv-rcond')
register('jax-numpy-quantile-interpolation')
register('jax-numpy-reduction-non-boolean-where')
register('jax-numpy-trimzeros-not-1d-array')
register('pallas-gpu-triton')
register('jax-scipy-special-sph-harm')

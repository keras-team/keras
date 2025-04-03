# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deprecation helpers."""

import warnings

# Module __getattr__ factory that warns if deprecated names are used.
#
# Example usage:
# from optax.contrib.dpsgd import  dpsgd as _deprecated_dpsgd
#
# _deprecations = {
#   # Added Apr 2024:
#   "dpsgd": (
#     "optax.dpsgd is deprecated. Use optax.contrib.dpsgd instead.",
#     _deprecated_dpsgd,
#   ),
# }
#
# from optax._src.deprecations import deprecation_getattr as _deprecation_getattr  # pylint: disable=line-too-long
# __getattr__ = _deprecation_getattr(__name__, _deprecations)
# del _deprecation_getattr


# Note that type checkers such as Pytype will not know about the deprecated
# names. If it is desirable that a deprecated name is known to the type checker,
# add:
# import typing
# if typing.TYPE_CHECKING:
#   from optax.contrib import dpsgd
# del typing


def deprecation_getattr(module, deprecations):
  def _getattr(name):
    if name in deprecations:
      message, fn = deprecations[name]
      if fn is None:  # Is the deprecation accelerated?
        raise AttributeError(message)
      warnings.warn(message, DeprecationWarning, stacklevel=2)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return _getattr



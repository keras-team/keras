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

"""A LazyLoader class."""

from collections.abc import Callable, Sequence
import importlib
import sys
from typing import Any


def attach(package_name: str, submodules: Sequence[str]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
  """Lazily loads submodules of a package.

  Returns:
    A tuple of ``__getattr__``, ``__dir__`` function and ``__all__`` --
    a list of available global names, which can be used to replace the
    corresponding definitions in the package.

  Raises:
    RuntimeError: If the ``__name__`` of the caller cannot be determined.
  """
  owner_name = sys._getframe(1).f_globals.get("__name__")
  if owner_name is None:
    raise RuntimeError("Cannot determine the ``__name__`` of the caller.")

  __all__ = list(submodules)

  def __getattr__(name: str) -> Any:
    if name in submodules:
      value = importlib.import_module(f"{package_name}.{name}")
      # Update module-level globals to avoid calling ``__getattr__`` again
      # for this ``name``.
      setattr(sys.modules[owner_name], name, value)
      return value
    raise AttributeError(f"module '{package_name}' has no attribute '{name}")

  def __dir__() -> list[str]:
    return __all__

  return __getattr__, __dir__, __all__

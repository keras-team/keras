# Copyright 2024 The etils Authors.
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

"""Lazy imports."""

from collections.abc import Iterator
import contextlib
import functools
import sys
from typing import Any

from etils.epy import lazy_imports_utils
from etils.epy import reraise_utils


@contextlib.contextmanager
def lazy_api_imports(
    globals_: dict[str, Any],
    *,
    error_msg: str | None = None,
) -> Iterator[None]:
  """Lazy-import an API (`__init__.py`).

  Usage:

  ```python
  with epy.lazy_api_imports(globals()):
    from my_project import Obj1
    from my_project import OtherObj
    from my_project import my_function
  ```

  Contrary to `epy.lazy_imports()` which works on modules (and resolve the
  imports) during first access. This function is intended to be used on
  `__init__.py` files, such as all imported symbols are lazy and only resolved
  when the symbol is accessed.

  Args:
    globals_: The module `globals()`. Will be updated to add a `__getattr__`
    error_msg: A additional message to append to the `ImportError` if the import
      fails. Can use `{symbol_name}` dynamic placeholder.

  Yields:
    None
  """
  try:
    before = set(globals_)
    with lazy_imports_utils.lazy_imports():
      yield
  finally:
    after = set(globals_)

  all_imported_symbols = after - before
  imported_symbols = {
      k: v
      for k in all_imported_symbols
      if isinstance(v := globals_[k], lazy_imports_utils.LazyModule)
  }
  if len(all_imported_symbols) != len(imported_symbols):
    raise ValueError(
        'Unexpected imported symbols: '
        f'{set(all_imported_symbols) - set(imported_symbols)}.'
    )
  for name in imported_symbols:
    del globals_[name]  # Remove so `module.__getattr__` is triggered

  # Note this will only works if the `__getattr__` is defined before the
  # `lazy_api_imports`, which is quite unlikely.
  assert '__getattr__' not in globals_
  assert '__dir__' not in globals_
  globals_['__getattr__'] = functools.partial(
      _getattr,
      module_name=globals_['__name__'],
      imported_symbols=imported_symbols,
      error_msg=error_msg,
  )
  globals_['__dir__'] = functools.partial(
      _dir,
      globals_=globals_,
      imported_symbols=imported_symbols,
  )


def _getattr(
    name: str,
    *,
    module_name: str,
    imported_symbols: dict[str, lazy_imports_utils.LazyModule | Any],
    error_msg: str | None,
) -> Any:
  """Module `__getattr__` that lazy-imports symbols."""
  if name not in imported_symbols:
    raise AttributeError(
        f'module {module_name!r} has no attribute {name!r}',
        name=name,
        obj=sys.modules.get(module_name),
    )
  symbol_or_lazy_module = imported_symbols[name]

  # symbol already loaded
  if not isinstance(symbol_or_lazy_module, lazy_imports_utils.LazyModule):
    return symbol_or_lazy_module

  # Otherwise, load the symbol
  lazy_module = symbol_or_lazy_module
  with lazy_module._maybe_adhoc():  # pylint: disable=protected-access
    try:
      symbol = _import_symbol(
          import_qualname=lazy_module.module_name,
          parent_module_name=module_name,
      )
    except ImportError as e:
      if error_msg:
        reraise_utils.reraise(
            e,
            prefix=error_msg.format(symbol_name=lazy_module.module_name),
        )
      else:
        raise

  imported_symbols[name] = symbol
  return symbol


def _dir(
    *,
    globals_: dict[str, Any],
    imported_symbols: dict[str, lazy_imports_utils.LazyModule | Any],
) -> list[str]:
  """Module `__dir__` that lazy-imports symbols."""
  return list(globals_) + list(imported_symbols)


def _import_symbol(import_qualname: str, parent_module_name: str) -> Any:
  """Import the lazy-symbol."""
  module_name, obj_name = import_qualname.rsplit('.', 1)
  if module_name == parent_module_name:
    # To avoid infinite recursion, import sub-modules as
    # `import parent_module.submodule` rather than
    # `from parent_module import submodule`
    module = __import__(f'{module_name}.{obj_name}')
    parts = module_name.split('.')[1:] + [obj_name]
    for name in parts:
      module = getattr(module, name)
    return module
  else:
    # Import symbols as `from module import obj` to supports functions,
    # classes, etc.
    module = __import__(module_name, fromlist=[obj_name])
    return getattr(module, obj_name)

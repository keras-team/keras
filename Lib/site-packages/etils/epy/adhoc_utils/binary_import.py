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

"""Wrapper around binary_import that supports colab."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import functools
import sys
from typing import Any, Optional

from etils.epy import contextlib as epy_contextlib
from etils.epy import lazy_imports_utils
from etils.epy import py_utils
from etils.epy.adhoc_utils import curr_args
from etils.epy.adhoc_utils import module_utils
from etils.epy.adhoc_utils import utils as adhoc_utils

import __main__  # pylint: disable=g-bad-import-order


@functools.cache
def _is_ipython_terminal() -> bool:
  """Returns True if running in a IPython terminal/XManager CLI environment."""
  # XManager CLI trigger binary imports

  # `epy` is imported before the `runpy.run_module(`, so main is still the
  # XManager binary
  # On Colab, `__main__.__file__` do not exists.
  main_file = getattr(__main__, '__file__', None)
  if main_file and main_file.endswith('xmanager2/client/cli/xm_cli.py'):
    return True

  # In case `epy` is imported after the XManager CLI, detecting we're in
  # `xmanager launch` is non-trivial because the script is launched with
  # `runpy.run_module(`, hiding some XManager internals and overwriting
  # `__main__`.
  if any(flag.startswith('--xm_launch_script=') for flag in sys.argv):
    return True

  if IPython := sys.modules.get('IPython'):  # pylint: disable=invalid-name
    ipython = IPython.get_ipython()
    if ipython and type(ipython).__name__ == 'TerminalInteractiveShell':
      return True
  return False


@contextlib.contextmanager
def binary_adhoc(
    source: Optional[g3_utils.Source] = None,
    *,
    restrict: None | py_utils.StrOrStrList = None,
    reload: None | py_utils.StrOrStrList = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Iterator[None]:
  yield

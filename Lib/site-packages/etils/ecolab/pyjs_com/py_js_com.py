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

"""Communication between Python and Javascript.

Could be released as a separate module once it also support non-colab notebooks.
"""

import abc
import functools
import sys
import traceback
from typing import Any, Callable, TypeVar

from etils import epath
from etils import epy
import IPython
import IPython.display


_FnT = TypeVar('_FnT')

_Json = Any
_Fn = Callable[..., Any]  # (*args: Json, **kwargs: Json) -> Json

# Coms doc is defined in
# https://jupyter-notebook.readthedocs.io/en/stable/comms.html


def _is_notebook_colab() -> bool:
  """Returns True if notebook is colab."""
  return 'google.colab' in sys.modules


class _NotebookBackend(abc.ABC):
  """Backend interface."""

  @abc.abstractmethod
  def wrap_output(self, out: _Json):
    """Eventually wrap the Json output."""
    raise NotImplementedError

  @abc.abstractmethod
  def register_fn(self, fn: _Fn) -> None:
    """Register the function to be called in Javascript."""
    raise NotImplementedError


class _Colab(_NotebookBackend):
  """Backend for Colab."""

  def wrap_output(self, out):
    return IPython.display.JSON(out)

  def register_fn(self, fn: _Fn) -> None:
    from google.colab import output  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    # TODO(epot): Fragile if multiple functions have the same name. How to
    # specify namespace ?
    output.register_callback(fn.__name__, fn)


class _Jupyter(_NotebookBackend):
  """Backend for Jupyter notebooks."""

  def wrap_output(self, out):
    return out

  def register_fn(self, fn: _Fn) -> None:
    def target_func(comm, open_msg):
      del open_msg

      @comm.on_msg
      def _recv(msg):
        data = msg['content']['data']
        out = fn(*data['args'], **data['kwargs'])
        comm.send(out)

    ipython = IPython.get_ipython()
    ipython.kernel.comm_manager.register_target(fn.__name__, target_func)


def register_js_fn(fn: _FnT) -> _FnT:
  r"""Decorator to make a function callable from Javascript.

  Usage:

  In Python:

  ```python
  @register_js_fn
  def my_fn(*args, **kwargs):
    return {'x': 123}
  ```

  The function can then be called from Javascript:

  ```python
  # Currently has to be executed in the same cell to install the library
  IPython.display.display(IPython.display.HTML(ecolab.pyjs_import()))

  IPython.display.HTML(\"\"\"
  <script>
    async function main() {
      out = await call_python('my_fn', [1, 2], {z: 3});
      console.log(out['sum']);  // my_fn(1, 2, z=3)  == {'sum': 6}
    }
    main();
  </script>
  \"\"\")
  ```

  Note that Javascript require the `pyjs_com.js_import()` statement to be
  present in the HTML from the cell.

  Args:
    fn: The Python function, can return any json-like value or dict

  Returns:
    The Python function, unmodified
  """

  # No-op when running on tests
  if not epy.is_notebook():
    return fn

  if _is_notebook_colab():
    backend = _Colab()
  else:
    backend = _Jupyter()

  @functools.wraps(fn)
  def decorated(*args, **kwargs):
    try:
      out = fn(*args, **kwargs)
      # Wrap non-dict values inside JSON
      if not isinstance(out, dict):
        out = {'__etils_pyjs__': out}
      # Eventually wrap the output
      return backend.wrap_output(out)
    except Exception as e:
      traceback.print_exception(e)
      raise

  backend.register_fn(decorated)
  return fn


# TODO(epot): Automatically add js-import if `register_js_fn` was called. This
# could be done similarly to `auto_inspect()`.


# TODO(epot): Host on gstatic and dynamically generate the URL from the
# local file hash.
# Auto-detect adhoc import or add a flag to load locally modified `.js`
@functools.lru_cache()
def js_import() -> str:
  """`<script></script>` to import to add in the HTML."""
  path = epath.resource_path('etils.ecolab') / 'pyjs_com/py_js_com.js'
  return f'<script>{path.read_text()}</script>'

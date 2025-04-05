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

"""Utils to auto-apply `ecolab.inspect`."""

import functools

from etils.ecolab import ip_utils
from etils.ecolab import pyjs_com
from etils.ecolab.inspects import core
from etils.ecolab.inspects import nodes
from etils.ecolab.inspects import resource_utils
import IPython
import IPython.display


@functools.lru_cache(None)
def auto_inspect() -> None:
  """Add a button on each cell outputs to switch output to `ecolab.inspect`."""
  ip_utils.register_once(
      'post_run_cell',
      _post_run_cell_add_inspect,
      '__is_auto_inspect__',
  )


def _post_run_cell_add_inspect(*args) -> None:
  """Callback after cell execution to add the `inspect` button."""
  del args  # Future version of IPython will have a `result` arg

  # TODO(epot): Detect if IPython has output
  # Currently, this add an output because `_` is set to the previous
  # value if no output is set.
  ip = IPython.get_ipython()

  # TODO(epot): Store the `last_result` in a weakref to avoid memory leaks.
  last_result = ip.ev('_')
  root = nodes.Node.from_obj(last_result)

  # TODO(epot): Should not load all `css`/`js` everytime ?
  # Especially the main inspect one which is used only after activation.
  html_content = IPython.display.HTML(f"""
      {pyjs_com.js_import()}
      {resource_utils.resource_import('auto_activate.css')}
      {resource_utils.resource_import('auto_activate.js')}
      {resource_utils.resource_import('theme.css')}
      {resource_utils.resource_import('main.js')}

      <script>
        add_auto_activate("{root.id}");
      </script>
  """)
  IPython.display.display(html_content)


@pyjs_com.register_js_fn
def get_inspect_html(id_: str) -> str:
  """Returns the inspect content."""
  node = nodes.Node.from_id(id_)
  return core.main_inspect_html(node)

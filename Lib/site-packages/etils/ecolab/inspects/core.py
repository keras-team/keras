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

"""Inspect API entry point."""

from __future__ import annotations

from etils import epy
from etils.ecolab import pyjs_com
from etils.ecolab.inspects import html_helper as H
from etils.ecolab.inspects import nodes
from etils.ecolab.inspects import resource_utils
import IPython.display


@pyjs_com.register_js_fn
def get_html_content(id_: str) -> str:
  """Returns the inner content of the block id.

  Is called the first time a block is expanded.

  Args:
    id_: Id of the block to load

  Returns:
    The html to add.
  """
  try:
    node = nodes.Node.from_id(id_)
    return node.inner_html
  except Exception as e:  # pylint: disable=broad-except
    epy.reraise(
        e,
        prefix=(
            '`ecolab.inspect` internal error. Please report an issue'
            '.\n'
        ),
    )


def inspect(obj: object) -> None:
  """Inspect all attributes of a Python object interactivelly.

  Args:
    obj: Any object to inspect (module, class, dict,...).
  """
  root = nodes.Node.from_obj(obj)

  html_content = IPython.display.HTML(f"""
      {resource_utils.resource_import('theme.css')}
      {pyjs_com.js_import()}
      {resource_utils.resource_import('main.js')}

      {main_inspect_html(root)}
      <script>
        load_content("{root.id}");
      </script>
      """)
  IPython.display.display(html_content)


def main_inspect_html(root: nodes.Node) -> str:
  """Main HTML content."""
  return H.ul(class_='tree-root')(root.header_html)

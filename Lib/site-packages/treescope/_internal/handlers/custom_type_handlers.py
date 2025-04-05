# Copyright 2024 The Treescope Authors.
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

"""Handler for custom types via __treescope_repr__ or the global registry."""

from __future__ import annotations

from typing import Any

from treescope import renderers
from treescope import rendering_parts
from treescope import type_registries
from treescope._internal import object_inspection


def handle_via_treescope_repr_method(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
    *,
    method_name: str = "__treescope_repr__",
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a type by calling its ``__treescope_repr__`` method, if it exists.

  The ``__treescope_repr__`` method can be used to add treescope support to
  custom classes. The method is expected to return a rendering in treescope's
  internal intermediate representation.

  The exact structure of the intermediate representation is an
  implementation detail and may change in future releases. Users should instead
  build a rendering using the high-level helpers in `treescope.repr_lib`, or
  the lower-level definitions in `treescope.rendering_parts`.

  Args:
    node: The node to render.
    path: An optional path to this node from the root.
    subtree_renderer: The renderer for subtrees of this node.
    method_name: Optional override for the name of the method to call. This is
      primarily intended for compatibility with treescope's original
      implementation as part of Penzai, which used __penzai_repr__.

  Returns:
    A rendering of this node, if it implements the __treescope_repr__ extension
    method.
  """
  treescope_repr_method = object_inspection.safely_get_real_method(
      node, method_name
  )
  if treescope_repr_method:
    return treescope_repr_method(path, subtree_renderer)
  else:
    return NotImplemented


def handle_via_global_registry(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a type by looking it up in the global handler registry.

  If it is not feasible to define ``__treescope_repr__`` for a type, it can
  instead be registered in the global handler registry. This is a dictionary
  mapping types to functions that render a node of that type.

  The exact structure of the intermediate representation is an
  implementation detail and may change in future releases. Users should instead
  build a rendering using the high-level helpers in `treescope.repr_lib`, or
  the lower-level definitions in `treescope.rendering_parts`.

  Args:
    node: The node to render.
    path: An optional path to this node from the root.
    subtree_renderer: The renderer for subtrees of this node.

  Returns:
    A rendering of this node, if it was found in the global registry.
  """
  maybe_handler = type_registries.lookup_treescope_handler_for_type(type(node))
  if maybe_handler:
    return maybe_handler(node, path, subtree_renderer)
  else:
    return NotImplemented

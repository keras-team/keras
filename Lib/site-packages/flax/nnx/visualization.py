# Copyright 2024 The Flax Authors.
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

import typing as tp

import treescope  # type: ignore[import-untyped]
from treescope import rendering_parts, renderers

try:
  from IPython import get_ipython

  in_ipython = get_ipython() is not None
except ImportError:
  in_ipython = False


def display(*args):
  """Display the given objects using the Treescope pretty-printer.

  If treescope is not installed or the code is not running in IPython,
  ``display`` will print the objects instead.
  """
  if not in_ipython:
    for x in args:
      print(x)
    return

  for x in args:
    treescope.display(x, ignore_exceptions=True, autovisualize=True)


def render_object_constructor(
  object_type: type[tp.Any],
  attributes: tp.Mapping[str, tp.Any],
  path: str | None,
  subtree_renderer: renderers.TreescopeSubtreeRenderer,
  roundtrippable: bool = False,
  color: str | None = None,
  first_line_annotation: rendering_parts.RenderableTreePart | None = None,
) -> rendering_parts.Rendering:
  """Renders an object in "constructor format", similar to a dataclass.

  This produces a rendering like `Foo(bar=1, baz=2)`, where Foo identifies the
  type of the object, and bar and baz are the names of the attributes of the
  object. It is a *requirement* that these are the actual attributes of the
  object, which can be accessed via `obj.bar` or similar; otherwise, the
  path renderings will break.

  This can be used from within a `__treescope_repr__` implementation via ::

    def __treescope_repr__(self, path, subtree_renderer):
      return repr_lib.render_object_constructor(
          object_type=type(self),
          attributes=<dict of attributes here>,
          path=path,
          subtree_renderer=subtree_renderer,
      )

  Args:
    object_type: The type of the object.
    attributes: The attributes of the object, which will be rendered as keyword
      arguments to the constructor.
    path: The path to the object. When `render_object_constructor` is called
      from `__treescope_repr__`, this should come from the `path` argument to
      `__treescope_repr__`.
    subtree_renderer: The renderer to use to render subtrees. When
      `render_object_constructor` is called from `__treescope_repr__`, this
      should come from the `subtree_renderer` argument to `__treescope_repr__`.
    roundtrippable: Whether evaluating the rendering as Python code will produce
      an object that is equal to the original object. This implies that the
      keyword arguments are actually the keyword arguments to the constructor,
      and not some other attributes of the object.
    color: The background color to use for the object rendering. If None, does
      not use a background color. A utility for assigning a random color based
      on a string key is given in `treescope.formatting_util`.
    first_line_annotation: An annotation for the first line of the node when it
      is expanded.

  Returns:
    A rendering of the object, suitable for returning from `__treescope_repr__`.
  """
  if roundtrippable:
    constructor = rendering_parts.siblings(
      rendering_parts.maybe_qualified_type_name(object_type), '('
    )
    closing_suffix = rendering_parts.text(')')
  else:
    constructor = rendering_parts.siblings(
      rendering_parts.roundtrip_condition(roundtrip=rendering_parts.text('<')),
      rendering_parts.maybe_qualified_type_name(object_type),
      '(',
    )
    closing_suffix = rendering_parts.siblings(
      ')',
      rendering_parts.roundtrip_condition(roundtrip=rendering_parts.text('>')),
    )

  children = []
  for i, (name, value) in enumerate(attributes.items()):
    child_path = None if path is None else f'{path}.{name}'

    if i < len(attributes) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = rendering_parts.siblings(
        ',',
        rendering_parts.fold_condition(collapsed=rendering_parts.text(' ')),
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = rendering_parts.fold_condition(
        expanded=rendering_parts.text(',')
      )

    child_line = rendering_parts.build_full_line_with_annotations(
      rendering_parts.siblings_with_annotations(
        f'{name}=',
        subtree_renderer(value, path=child_path),
      ),
      comma_after,
    )
    children.append(child_line)

  return rendering_parts.build_foldable_tree_node_from_children(
    prefix=constructor,
    children=children,
    suffix=closing_suffix,
    path=path,
    background_color=color,
    first_line_annotation=first_line_annotation,
  )
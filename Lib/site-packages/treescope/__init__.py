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

"""Treescope: An interactive HTML pretty-printer and array visualizer.

You can configure treescope as the default IPython pretty-printer using ::

  treescope.basic_interactive_setup()

or, for more control: ::

  treescope.register_as_default()
  treescope.register_autovisualize_magic()
  treescope.active_autovisualizer.set_globally(
      treescope.ArrayAutovisualizer()
  )

You can also pretty-print individual values using `treescope.show` or
`treescope.display`.
"""
from __future__ import annotations

# pylint: disable=g-importing-member,g-multiple-import,unused-import

import typing

from . import _internal
from . import canonical_aliases
from . import context
from . import dataclass_util
from . import figures
from . import formatting_util
from . import handlers
from . import lowering
from . import ndarray_adapters
from . import renderers
from . import rendering_parts
from . import repr_lib
from . import type_registries

from ._internal.api.abbreviation import (
    abbreviation_threshold,
    roundtrip_abbreviation_threshold,
)
from ._internal.api.array_autovisualizer import (
    ArrayAutovisualizer,
)
from ._internal.api.arrayviz import (
    default_diverging_colormap,
    default_sequential_colormap,
    integer_digitbox,
    render_array,
    render_array_sharding,
)
from ._internal.api.autovisualize import (
    IPythonVisualization,
    VisualizationFromTreescopePart,
    ChildAutovisualizer,
    Autovisualizer,
    active_autovisualizer,
)
from ._internal.api.default_renderer import (
    active_renderer,
    active_expansion_strategy,
    render_to_html,
    render_to_text,
    using_expansion_strategy,
)
from ._internal.api.ipython_integration import (
    default_magic_autovisualizer,
    basic_interactive_setup,
    display,
    register_as_default,
    register_autovisualize_magic,
    register_context_manager_magic,
    show,
)

if typing.TYPE_CHECKING:
  Callable = typing.Callable

# Type annotations and docstrings for imported constants. Used by sphinx to
# generate documentation.

default_diverging_colormap: context.ContextualValue[
    list[tuple[int, int, int]]
] = default_diverging_colormap
"""Default diverging colormap.

Used by `render_array` when ``around_zero`` is True. Intended for user
customization in an interactive setting.
"""

default_sequential_colormap: context.ContextualValue[
    list[tuple[int, int, int]]
] = default_sequential_colormap
"""Default sequential colormap.

Used by `render_array` when ``around_zero`` is False. Intended for user
customization in an interactive setting.
"""

active_autovisualizer: context.ContextualValue[Autovisualizer | None] = (
    active_autovisualizer
)
"""The active autovisualizer to use when rendering a tree to HTML.

This can be overridden interactively to enable rich visualizations in
treescope. Users are free to set this to an arbitrary renderer of their
choice; a common choice is arrayviz's `ArrayAutovisualizer()`.
"""

active_renderer: context.ContextualValue[renderers.TreescopeRenderer] = (
    active_renderer
)
"""The default renderer to use when rendering a tree to HTML.

This determines the set of handlers and postprocessors to use when
rendering an object.

This can be overridden locally to reconfigure how nodes are rendered
with treescope. Users are free to set this to an arbitrary renderer of
their choice, and programs should not assume the renderer has a
particular form. Library functions can retrieve the current value of
this to render objects in a user-configurable way, and can optionally
make further adjustments using `TreescopeRenderer.extend_with`.
"""

active_expansion_strategy: context.ContextualValue[
    Callable[[rendering_parts.RenderableTreePart], None]
] = active_expansion_strategy
"""The default expansion strategy to use when rendering a tree to HTML.

Expansion strategies are used to figure out how deeply to unfold an object
by default. They should operate by setting the expand states of the
foldable nodes inside the object.
"""

default_magic_autovisualizer: context.ContextualValue[Autovisualizer] = (
    default_magic_autovisualizer
)
"""The default autovisualizer to use for the ``%%autovisualize`` magic.

This can be overridden interactively to customize the autovisualizer
used by ``%%autovisualize``.
"""

abbreviation_threshold: context.ContextualValue[int | None] = (
    abbreviation_threshold
)
"""Depth threshold for abbreviating large outputs in normal mode.

This value sets the depth at which values should be abbreviated (replaced by
... markers) when their parents are collapsed. Threshold 1 means that children
of a collapsed node should be abbreviated. Threshold 2 means that grandchildren
of a collapsed node should be abbreviated. Threshold None means that no
abbreviation should be performed.
"""

roundtrip_abbreviation_threshold: context.ContextualValue[int | None] = (
    roundtrip_abbreviation_threshold
)
"""Depth threshold for abbreviating large outputs in roundtrip mode.

This value sets the depth at which values should be abbreviated (replaced by
... markers) when their parents are collapsed. Threshold 1 means that children
of a collapsed node should be abbreviated. Threshold 2 means that grandchildren
of a collapsed node should be abbreviated. Threshold None means that no
abbreviation should be performed.

Note that if abbreviation is enabled in roundtrip mode, outputs may not be
fully roundtrippable due to the abbreviated children.
"""

# Package version.
__version__ = '0.1.9'


# Set up canonical aliases for the treescope API itself.
def _setup_canonical_aliases_for_api():
  import types  # pylint: disable=import-outside-toplevel

  for key, value in globals().items():
    if isinstance(value, (type, types.FunctionType)):
      canonical_aliases.add_alias(
          value, canonical_aliases.ModuleAttributePath(__name__, (key,))
      )


_setup_canonical_aliases_for_api()
del _setup_canonical_aliases_for_api, typing

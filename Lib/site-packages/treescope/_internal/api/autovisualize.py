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

"""User-configurable automatic visualization of subtrees.

This module provides a quick but flexible interface for customizing the
output of Treescope by replacing subtrees by rendered figures. The intended
use case is to make it easy to generate visualizations for arrays, parameters,
or other information, even if they are deeply nested inside larger data
structures.

This module also defines a particular automatic visualizer that uses Arrayviz
to visualize all arrays in a tree, which can be useful for exploratory analysis
of models and data.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Protocol

from treescope import context
from treescope import rendering_parts
from treescope._internal import object_inspection


@dataclasses.dataclass
class IPythonVisualization:
  """Used by autovisualizers to replace a subtree with a display object.

  Attributes:
    display_object: An object to render, e.g. a figure.
    replace: Whether to replace the subtree completely with this display
      object. If False, the display object will be rendered below the object.
  """

  display_object: object_inspection.HasReprHtml
  replace: bool = False


@dataclasses.dataclass
class VisualizationFromTreescopePart:
  """Used by advanced autovisualizers to directly insert Treescope content.

  Attributes:
    rendering: A custom treescope rendering which will completely replace the
      subtree.
  """

  rendering: rendering_parts.RenderableAndLineAnnotations


@dataclasses.dataclass
class ChildAutovisualizer:
  """Used by autovisualizers to switch to a different autovisualizer.

  This can be used to enable conditional logic, e.g. "when rendering an object
  of type Foo, visualize objects of type Bar". Note that the returned child
  autovisualizer is also allowed to close over things from the current call,
  e.g. you can return a lambda function that checks if it is rendering a
  particular child of the original node.

  Attributes:
    autovisualizer: The new autovisualizer to use while rendering this value's
      children.
  """

  autovisualizer: Autovisualizer


class Autovisualizer(Protocol):
  """Protocol for autovisualizers.

  An autovisualizer is a callable object that can be used to customize
  Treescope for rendering purposes. The active autovisualizer is called once
  for each subtree of a rendered PyTree, and can choose to:

  * render it normally (but possibly customize its children),

  * replace the rendering of that PyTree with a custom rich display object,

  * or replace the current autovisualizer with a new one that will be active
    while rendering the children.
  """

  @abc.abstractmethod
  def __call__(
      self, value: Any, path: str | None
  ) -> (
      IPythonVisualization
      | VisualizationFromTreescopePart
      | ChildAutovisualizer
      | None
  ):
    """Runs the autovisualizer on a node at a given path.

    Args:
      value: A value being rendered in treescope.
      path: Optionally, a path to this node, represented as a string that can be
        used to reach this node from the root (e.g. ".foo.bar['baz']").

    Returns:
      A visualization for this subtree, a child autovisualizer to use while
      rendering the node's children, or None to continue rendering the node
      normally using the current autovisualizer.
    """
    raise NotImplementedError()


active_autovisualizer: context.ContextualValue[Autovisualizer | None] = (
    context.ContextualValue(
        module=__name__,
        qualname="active_autovisualizer",
        initial_value=None,
    )
)

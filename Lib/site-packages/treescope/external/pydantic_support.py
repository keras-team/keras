# Copyright 2025 The Treescope Authors.
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

"""Lazy setup logic for adding Pydantic support to treescope."""
from __future__ import annotations

from treescope import renderers
from treescope import rendering_parts
from treescope import repr_lib
from treescope import type_registries

# pylint: disable=import-outside-toplevel
try:
  import pydantic
except ImportError:
  omegaconf = None
# pylint: enable=import-outside-toplevel


def render_pydantic_model(
    node: pydantic.BaseModel,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> rendering_parts.Rendering | type(NotImplemented):
  """Renders a pydantic model."""
  if pydantic.__version__.startswith("1."):
    fields = type(node).__fields__
  else:
    fields = type(node).model_fields
  return repr_lib.render_object_constructor(
      type(node),
      attributes={k: getattr(node, k) for k in fields.keys()},
      path=path,
      subtree_renderer=subtree_renderer,
      roundtrippable=True,
  )


def set_up_pydantic() -> None:
  """Registers handlers for pydantic types."""
  if pydantic is None:
    raise RuntimeError(
        "Cannot set up pydantic support in treescope: pydantic cannot be"
        " imported."
    )
  type_registries.TREESCOPE_HANDLER_REGISTRY[pydantic.BaseModel] = (
      render_pydantic_model
  )

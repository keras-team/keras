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

"""Containers for conveniently rendering multiple objects together.

These containers make it possible to group together multiple children so that
they render as a single figure-like display object in IPython, and allow
embedding arbitrary treescope-renderable objects or treescope-compatible styles
as figures.
"""

from __future__ import annotations

import math
from typing import Any

from treescope import rendering_parts
from treescope._internal import arrayviz_impl
from treescope._internal import figures_impl
from treescope._internal import object_inspection
from treescope._internal.api import arrayviz
from treescope._internal.api import default_renderer
from treescope._internal.parts import basic_parts
from treescope._internal.parts import embedded_iframe


def inline(
    *subfigures: Any, wrap: bool = False
) -> figures_impl.TreescopeFigure:
  """Returns a figure that arranges a set of displayable objects along a line.

  Args:
    *subfigures: Subfigures to display inline.
    wrap: Whether to wrap (insert newlines) between words at the end of a line.

  Returns:
    A figure which can be rendered in IPython or used to build more complex
    figures.
  """
  siblings = rendering_parts.siblings(
      *(treescope_part_from_display_object(subfig) for subfig in subfigures)
  )
  if wrap:
    return figures_impl.TreescopeFigure(figures_impl.AllowWordWrap(siblings))
  else:
    return figures_impl.TreescopeFigure(figures_impl.PreventWordWrap(siblings))


def indented(subfigure: Any) -> figures_impl.TreescopeFigure:
  """Returns a figure object that displays a value with an indent.

  Args:
    subfigure: A value to render indented.
  """
  return figures_impl.TreescopeFigure(
      rendering_parts.indented_children([
          rendering_parts.vertical_space("0.25em"),
          treescope_part_from_display_object(subfigure),
          rendering_parts.vertical_space("0.25em"),
      ])
  )


def styled(subfigure: Any, style: str) -> figures_impl.TreescopeFigure:
  """Returns a CSS-styled version of the first figure.

  Args:
    subfigure: A value to render.
    style: A CSS style string.
  """
  return figures_impl.TreescopeFigure(
      rendering_parts.custom_style(
          treescope_part_from_display_object(subfigure), style
      )
  )


def with_font_size(
    subfigure: Any, size: str | float
) -> figures_impl.TreescopeFigure:
  """Returns a scaled version of the first figure.

  Args:
    subfigure: A value to render.
    size: A multiplier for the font size (as a float) or a string giving a
      specific CSS font size (e.g. "14pt" or "2em").
  """
  if isinstance(size, str):
    style = f"font-size: {size}"
  else:
    style = f"font-size: {size}em"
  return figures_impl.TreescopeFigure(
      rendering_parts.custom_style(
          treescope_part_from_display_object(subfigure), style
      )
  )


def with_color(subfigure: Any, color: str) -> figures_impl.TreescopeFigure:
  """Returns a colored version of the first figure.

  Args:
    subfigure: A value to render.
    color: Any CSS color string.
  """
  return figures_impl.TreescopeFigure(
      rendering_parts.custom_style(
          treescope_part_from_display_object(subfigure), f"color: {color}"
      )
  )


def bolded(subfigure: Any) -> figures_impl.TreescopeFigure:
  """Returns a bolded version of the first figure.

  Args:
    subfigure: A value to render.
  """
  return figures_impl.TreescopeFigure(
      rendering_parts.custom_style(
          treescope_part_from_display_object(subfigure), "font-weight: bold"
      )
  )


def text_on_color(
    text: str,
    value: float,
    vmax: float = 1.0,
    vmin: float | None = None,
    colormap: list[tuple[int, int, int]] | None = None,
) -> figures_impl.TreescopeFigure:
  """Renders some text on colored background, with the default colormap.

  Args:
    text: Text to display.
    value: Value to color the background with.
    vmax: Maximum value for the colormap.
    vmin: Minimum value for the colormap. Defaults to ``-vmax``.
    colormap: Explicit colormap to use. If not provided, uses the default
      diverging colormap if ``vmin`` is not provided, or the default sequential
      colormap if ``vmin`` is provided.

  Returns:
    A rendering of this word, formatted similarly to how this value would be
    formatted in an array visualization.
  """
  if colormap is None:
    if vmin is None:
      colormap = arrayviz.default_diverging_colormap.get()
    else:
      colormap = arrayviz.default_sequential_colormap.get()

  if vmin is None:
    vmin = -vmax

  if value > vmax:
    is_out_of_bounds = True
    color = colormap[-1]
  elif value < vmin:
    is_out_of_bounds = True
    color = colormap[0]
  else:
    is_out_of_bounds = False
    relative = (value - vmin) / (vmax - vmin)
    continuous_index = relative * (len(colormap) - 1)
    base_index = int(math.floor(continuous_index))
    offset = continuous_index - base_index
    if base_index == len(colormap) - 1:
      base_index -= 1
      offset = 1
    assert base_index < len(colormap) - 1
    color = [
        v0 * (1 - offset) + v1 * offset
        for v0, v1 in zip(colormap[base_index], colormap[base_index + 1])
    ]

  r, g, b = color
  # Logic matches `contrasting_style_string` in arrayviz.js
  # https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
  intensity = 0.2126 * r + 0.7152 * g + 0.0722 * b
  if intensity > 128:
    text_color = "black"
  else:
    text_color = "white"
  return figures_impl.TreescopeFigure(
      arrayviz_impl.ValueColoredTextbox(
          child=rendering_parts.text(text),
          text_color=text_color,
          background_color=f"rgb({r} {g} {b})",
          out_of_bounds=is_out_of_bounds,
          value=value,
      )
  )


def figure_from_treescope_rendering_part(
    part: rendering_parts.RenderableTreePart,
) -> figures_impl.TreescopeFigure:
  """Returns a figure object that displays a Treescope rendering part.

  Args:
    part: A Treescope rendering part to display, usually constructed via
      `repr_lib` or `rendering_parts`.

  Returns:
    A figure object that can be rendered in IPython.
  """
  return figures_impl.TreescopeFigure(part)


def treescope_part_from_display_object(
    value: Any,
) -> rendering_parts.RenderableTreePart:
  """Converts an arbitrary object to a renderable treescope part if possible.

  Behavior depends on the type of `value`:

  * If ``value`` is an instance of `TreescopeFigure`, unwraps the
    underlying treescope part.
  * If ``value`` is a string, returns a rendering of that string.
  * If ``value`` has a ``_repr_html_`` method (but isn't an instance of
    `TreescopeFigure`), returns an embedded iframe with the given HTML
    contents.
  * Otherwise, renders the value using the default treescope renderer, but
    strips off any top-level comments / copy button annotations.

  The typical use is to provide helper constructors for containers to allow
  rendering lots of different objects in the "obvious" way.

  Args:
    value: Value to wrap.

  Returns:
    A renderable treescope part showing the value.
  """
  if isinstance(value, figures_impl.TreescopeFigure):
    return value.treescope_part
  elif isinstance(value, str):
    return basic_parts.Text(value)
  else:
    maybe_html = object_inspection.to_html(value)
    if maybe_html:
      return figures_impl.InlineBlock(
          embedded_iframe.embedded_iframe(
              maybe_html,
              fallback_in_text_mode=basic_parts.Text(object.__repr__(value)),
          )
      )
    else:
      return default_renderer.build_foldable_representation(value).renderable

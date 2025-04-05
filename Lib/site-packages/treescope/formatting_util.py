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

"""Utilities for formatting and rendering."""

import hashlib
import warnings


def oklch_color(
    lightness: float, chroma: float, hue: float, alpha: float = 1.0
) -> str:
  """Constructs an OKLCH CSS color.

  See https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/oklch and
  https://oklch.com/.

  Args:
    lightness: Lightness argument (0 to 1)
    chroma: Chroma argument (0 to 0.5 in practice; large values are clipped)
    hue: Hue argument (0 to 360)
    alpha: Opacity (0 to 1)

  Returns:
    A CSS color string.
  """
  return f"oklch({lightness} {chroma} {hue} / {alpha})"


def color_from_string(
    key_string: str,
    lightness: float = 0.785,
    chroma: float = 0.103,
    alpha: float = 1.0,
) -> str:
  """Derives a color whose hue is keyed by a string.

  The main use for this is to automatically generate a color for the CSS OKLCH
  color space, which can be used to set a type-dependent color for an object in
  treescope.

  The default values of ``lightness`` and ``chroma`` are set up to give a
  consistent range of colors that are all within gamut.

  Args:
    key_string: A string that should be hashed to determine a hue.
    lightness: Lightness argument (0 to 1)
    chroma: Chroma argument (0 to 0.5 in practice; large values are clipped)
    alpha: Opacity (0 to 1)

  Returns:
    A pseudo-uniformly-distributed number between 0 and 360, suitable for use
    as the third argument to the CSS ``oklch(...)`` function.
  """
  # Derive a 32-bit fingerprint from the type.
  fingerprint = int.from_bytes(
      hashlib.sha256(key_string.encode("utf-8")).digest(), byteorder="little"
  )
  # fingerprint = farmhash.fingerprint32(key_string)
  # Convert it into a uniformly-distributed float in [0, 1].
  uniform = (fingerprint % (2**16)) / (2**16)
  # Convert to a hue.
  return oklch_color(lightness, chroma, 360 * uniform, alpha)


def parse_simple_color_and_pattern_spec(
    requested_color: str | tuple[str, str],
    typename_for_warning: str | None = None,
) -> tuple[str | None, str | None]:
  """Parses a background color and pattern from a user-provided color request.

  Args:
    requested_color: A color request, which is either a single CSS color or a
      tuple of outline and background colors.
    typename_for_warning: If provided, and the color is invalid, a warning will
      be issued with this typename as context.

  Returns:
    A tuple (background_color, background_pattern) that can be passed to
    the Treescope low-level representation construction functions (such as
    `build_foldable_tree_node_from_children`) that will configure it with the
    given background color and outline.
  """
  if isinstance(requested_color, str):
    background_color = requested_color
    background_pattern = None
  elif isinstance(requested_color, tuple) and len(requested_color) == 2:
    background_color = requested_color[0]
    background_pattern = (
        f"linear-gradient({requested_color[1]},{requested_color[1]})"
    )
  else:
    if typename_for_warning:
      warnings.warn(
          f"{typename_for_warning} requested an invalid color:"
          f" {requested_color} (not a string or a tuple)"
      )
    background_color = None
    background_pattern = None
  return background_color, background_pattern

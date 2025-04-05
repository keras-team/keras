# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layout utilities."""

from collections.abc import Sequence
import itertools
import re

from jax._src.lib.mlir import ir

from .fragmented_array import WGSplatFragLayout, WGStridedFragLayout


_splat_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGSplatFragLayout<\[(?P<shape>.*)\]>$"
)


def to_splat_fragmented_layout_attr(layout: WGSplatFragLayout) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGSplatFragLayout attribute from a WGSplatFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGSplatFragLayout<{list(layout.shape)}>"
  )


def from_splat_fragmented_layout_attr(attr: ir.Attribute) -> WGSplatFragLayout:
  """Constructs a WGSplatFragLayout from a #mosaic_gpu.WGSplatFragLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.WGSplatFragLayout
      attribute.
  """
  match = _splat_fragmented_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGSplatFragLayout attribute, got {attr}"
    )

  return WGSplatFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(","))
  )


def is_splat_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(re.search(_splat_fragmented_layout_attr_pattern, str(attr)))


_strided_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGStridedFragLayout<\[(?P<shape>.*)\],"
    r" (?P<vector_size>\d+)>$"
)

def to_strided_fragmented_layout_attr(
    layout: WGStridedFragLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGStridedFragLayout attribute from a WGStridedFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGStridedFragLayout<{list(layout.shape)},"
      f" {layout.vec_size}>"
  )


def from_strided_fragmented_layout_attr(
    attr: ir.Attribute,
) -> WGStridedFragLayout:
  """Constructs a WGStridedFragLayout from a #mosaic_gpu.WGStridedFragLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.WGStridedFragLayout
      attribute.
  """
  match = _strided_fragmented_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGStridedFragLayout attribute, got {attr}"
    )

  return WGStridedFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(",")),
      vec_size=int(match.group("vector_size")),
  )


def is_strided_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(re.search(_strided_fragmented_layout_attr_pattern, str(attr)))


def in_layouts(op: ir.OpView) -> Sequence[ir.Attribute]:
  """Returns the in_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_layouts attribute.
  """
  if "in_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an in_layouts attribute.")
  return op.attributes["in_layouts"]  # type: ignore


def out_layouts(op: ir.OpView) -> Sequence[ir.Attribute]:
  """Returns the out_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_layouts attribute.
  """
  if "out_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an out_layouts attribute.")
  return op.attributes["out_layouts"]  # type: ignore


def should_have_layout(op: ir.OpView) -> bool:
  """Returns 'true' if the operation should be assigned a layout."""

  is_array = lambda v: ir.VectorType.isinstance(v.type)
  return any(map(is_array, itertools.chain(op.operands, op.results)))  # type: ignore


def has_in_layouts_set(op: ir.OpView) -> bool:
  return "in_layouts" in op.attributes


def has_out_layouts_set(op: ir.OpView) -> bool:
  return "out_layouts" in op.attributes


def has_any_layout_set(op: ir.OpView) -> bool:
  return has_in_layouts_set(op) or has_out_layouts_set(op)

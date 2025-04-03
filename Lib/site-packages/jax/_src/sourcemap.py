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

"""
An implementation of sourcemaps following `TC39 <https://tc39.es/source-map>`_.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from typing import Union

# A Segment encodes how parts in the generated source relate to the original source.
# Each segment is made up of 1, 4 or 5 variable-length fields. For their semantics see
# https://tc39.es/source-map/#mappings-structure
Segment = Union[
    tuple[int], tuple[int, int, int, int], tuple[int, int, int, int, int]
]

# Mappings are sequences of segments for each line in the generated source.
Mappings = Sequence[Sequence[Segment]]


@dataclass(frozen=True)
class SourceMap:
  version: int
  # file: str
  # source_root: str
  sources: Sequence[str]
  sources_content: Sequence[str]
  names: Sequence[str]
  mappings: Mappings

  @classmethod
  def from_json(cls, json_data: str) -> SourceMap:
    """Deserialize a source map from JSON."""
    data = json.loads(json_data)
    return cls(
        version=data["version"],
        sources=data["sources"],
        sources_content=data["sourcesContent"],
        names=data["names"],
        mappings=deserialize_mappings(data["mappings"]),
    )

  def to_json(self) -> str:
    """Serialize a source map to JSON."""
    data = {
        "version": self.version,
        "sources": self.sources,
        "sourcesContent": self.sources_content,
        "names": self.names,
        "mappings": serialize_mappings(self.mappings),
    }
    return json.dumps(data)


VLQ_SIGN_MASK = 0x01
VLQ_MORE_MASK = 0x20
VLQ_VALUE_MASK = 0x1F
VLQ_VALUE_BITWIDTH = 5
VLQ_ALPHABET = (
    list(range(ord("A"), ord("Z") + 1))
    + list(range(ord("a"), ord("z") + 1))
    + list(range(ord("0"), ord("9") + 1))
    + [ord("+"), ord("/")]
)


def make_vlq_decode_table():
  lookup = {c: d for d, c in enumerate(VLQ_ALPHABET)}
  return [lookup.get(i, None) for i in range(256)]


VLQ_DECODE_TABLE = make_vlq_decode_table()


def decode_vlq(enc: Iterable[int]) -> int:
  """Decode a Base-64-VLQ into an integer."""
  enc_iter = iter(enc)
  d = VLQ_DECODE_TABLE[next(enc_iter)]
  sign = bool(d & VLQ_SIGN_MASK)
  value = (d & VLQ_VALUE_MASK) >> 1
  # Compensate for first quantum containing sign as LSB:
  shift = -1

  while d & VLQ_MORE_MASK:
    shift += VLQ_VALUE_BITWIDTH
    d = VLQ_DECODE_TABLE[next(enc_iter)]
    value |= (d & VLQ_VALUE_MASK) << shift

  return -value if sign else value


def encode_vlq(value: int) -> bytes:
  """Encode an integer into a Base-64-VLQ."""
  # Move sign to LSB
  value = ((-value) << 1 | 1) if value < 0 else value << 1
  buf = []

  while True:
    d = value & VLQ_VALUE_MASK
    value >>= VLQ_VALUE_BITWIDTH
    more = value > 0
    if more:
      d |= VLQ_MORE_MASK
    buf.append(VLQ_ALPHABET[d])
    if not more:
      break
  return bytes(buf)


def decode_segment(enc: Iterable[int]) -> Segment:
  """Decode a sequence of VLQs into a segment."""
  enc_iter = iter(enc)
  col = decode_vlq(enc_iter)
  try:
    source = decode_vlq(enc_iter)
  except StopIteration:
    # Stopping here is fine (1-segment).
    return (col,)
  source_line = decode_vlq(enc_iter)
  source_col = decode_vlq(enc_iter)
  try:
    name = decode_vlq(enc_iter)
  except StopIteration:
    # Stopping here is fine too (4-segment).
    return col, source, source_line, source_col
  # (5-segment)
  return col, source, source_line, source_col, name


def encode_segment(seg: Segment) -> bytes:
  """Encode a segment into a sequence of VLQs."""
  return b"".join(encode_vlq(value) for value in seg)


def deserialize_mappings(mappings_str: str) -> Mappings:
  """Decode a string of TC39 mapping data."""
  mappings_bytes = bytes(mappings_str, encoding="ascii")
  return [
      list(map(decode_segment, mapping.split(b","))) if mapping else []
      for mapping in mappings_bytes.split(b";")
  ]


def serialize_mappings(mappings: Mappings) -> str:
  """Encode mappings into a string of TC39 mapping data."""
  enc = b";".join(
      b",".join(encode_segment(seg) for seg in segs) for segs in mappings
  )
  return enc.decode("ascii")


class MappingsGenerator:
  """MappingsGenerator is a builder API for mappings.

  TC39 mapping data is inconvenient to emit directly: in an effort to compress
  data
  it encodes most indices using values _relative_ to the previous element.
  MappingsGenerator simplifies things by taking absolute indices everywhere.
  """

  def __init__(self):
    self._last_col = None
    self._last_source = 0
    self._last_source_line = 0
    self._last_source_col = 0
    self._last_name = 0
    self._mappings = []
    self._cur_group = None

  def new_group(self):
    """Start a new group (line)."""
    self._last_col = 0
    self._cur_group = []
    self._mappings.append(self._cur_group)

  def new_segment(self, *seg):
    """Start a new source mapping segment in the current group.

    Args:
      *seg: A segment as in TC39, but all indices are absolute. See
        https://tc39.es/source-map/#mappings-structure for details.

    Raises:
      RuntimeError: If no current group exists.
    """
    assert len(seg) >= 1
    group = self._cur_group
    if group is None:
      raise RuntimeError("No current group. Forgot to call new_group()?")

    col = seg[0] - self._last_col
    self._last_col = seg[0]

    if len(seg) == 1:
      group.append((col,))
      return

    source = seg[1] - self._last_source
    self._last_source = seg[1]
    source_line = seg[2] - self._last_source_line
    self._last_source_line = seg[2]
    source_col = seg[3] - self._last_source_col
    self._last_source_col = seg[3]

    if len(seg) == 4:
      group.append((col, source, source_line, source_col))
      return

    name = seg[4] - self._last_name
    self._last_name = seg[4]

    if len(seg) == 5:
      group.append((col, source, source_line, source_col, name))
      return

    assert False, "invalid segment"

  def mappings(self) -> Mappings:
    """Return the mapping as a list of segments per line."""
    return self._mappings

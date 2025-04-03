# Copyright 2021 The JAX Authors.
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
#
# Wadler-Lindig pretty printer.
#
# References:
# Wadler, P., 1998. A prettier printer. Journal of Functional Programming,
# pp.223-244.
#
# Lindig, C. 2000. Strictly Pretty.
# https://lindig.github.io/papers/strictly-pretty-2000.pdf
#
# Hafiz, A. 2021. Strictly Annotated: A Pretty-Printer With Support for
# Annotations. https://ayazhafiz.com/articles/21/strictly-annotated
#

from __future__ import annotations

from collections.abc import Sequence
import enum
from functools import partial
import sys
from typing import Any, NamedTuple

from jax._src import config
from jax._src import util

try:
  import colorama  # pytype: disable=import-error
except ImportError:
  colorama = None


_PPRINT_USE_COLOR = config.bool_state(
    'jax_pprint_use_color',
    True,
    help='Enable jaxpr pretty-printing with colorful syntax highlighting.'
)

def _can_use_color() -> bool:
  try:
    # Check if we're in IPython or Colab
    ipython = get_ipython()  # type: ignore[name-defined]
    shell = ipython.__class__.__name__
    if shell == "ZMQInteractiveShell":
      # Jupyter Notebook
      return True
    elif "colab" in str(ipython.__class__):
      # Google Colab (external or internal)
      return True
  except NameError:
    pass
  # Otherwise check if we're in a terminal
  return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

CAN_USE_COLOR = _can_use_color()

class Doc(util.StrictABC):
  __slots__ = ()

  def format(
    self, width: int = 80, *, use_color: bool | None = None,
    annotation_prefix: str = " # ",
    source_map: list[list[tuple[int, int, Any]]] | None = None
  ) -> str:
    """
    Formats a pretty-printer document as a string.

    Args:
    source_map: for each line in the output, contains a list of
      (start column, end column, source) tuples. Each tuple associates a
      region of output text with a source.
    """
    if use_color is None:
      use_color = CAN_USE_COLOR and _PPRINT_USE_COLOR.value
    return _format(self, width, use_color=use_color,
                   annotation_prefix=annotation_prefix, source_map=source_map)

  def __str__(self):
    return self.format()

  def __add__(self, other: Doc) -> Doc:
    return concat([self, other])

class _NilDoc(Doc):
  def __repr__(self): return "nil"

_nil = _NilDoc()

class _TextDoc(Doc):
  __slots__ = ("text", "annotation")
  text: str
  annotation: str | None

  def __init__(self, text: str, annotation: str | None = None):
    assert isinstance(text, str), text
    assert annotation is None or isinstance(annotation, str), annotation
    self.text = text
    self.annotation = annotation

  def __repr__(self):
    if self.annotation is not None:
      return f"text(\"{self.text}\", annotation=\"{self.annotation}\")"
    else:
      return f"text(\"{self.text}\")"

class _ConcatDoc(Doc):
  __slots__ = ("children",)
  children: list[Doc]

  def __init__(self, children: Sequence[Doc]):
    self.children = list(children)
    assert all(isinstance(doc, Doc) for doc in self.children), self.children

  def __repr__(self): return f"concat({self.children})"

class _BreakDoc(Doc):
  __slots__ = ("text",)
  text: str

  def __init__(self, text: str):
    assert isinstance(text, str), text
    self.text = text

  def __repr__(self): return f"break({self.text})"

class _GroupDoc(Doc):
  __slots__ = ("child",)
  child: Doc

  def __init__(self, child: Doc):
    assert isinstance(child, Doc), child
    self.child = child

  def __repr__(self): return f"group({self.child})"

class _NestDoc(Doc):
  __slots__ = ("n", "child",)
  n: int
  child: Doc

  def __init__(self, n: int, child: Doc):
    assert isinstance(child, Doc), child
    self.n = n
    self.child = child

  def __repr__(self): return f"nest({self.n, self.child})"


_NO_SOURCE = object()

class _SourceMapDoc(Doc):
  __slots__ = ("child", "source")
  child: Doc
  source: Any

  def __init__(self, child: Doc, source: Any):
    assert isinstance(child, Doc), child
    self.child = child
    self.source = source

  def __repr__(self): return f"source({self.child}, {self.source})"


Color = enum.Enum("Color", ["BLACK", "RED", "GREEN", "YELLOW", "BLUE",
                            "MAGENTA", "CYAN", "WHITE", "RESET"])
Intensity = enum.Enum("Intensity", ["DIM", "NORMAL", "BRIGHT"])

class _ColorDoc(Doc):
  __slots__ = ("foreground", "background", "intensity", "child")
  foreground: Color | None
  background: Color | None
  intensity: Intensity | None
  child: Doc

  def __init__(self, child: Doc, *, foreground: Color | None = None,
               background: Color | None = None,
               intensity: Intensity | None = None):
    assert isinstance(child, Doc), child
    self.child = child
    self.foreground = foreground
    self.background = background
    self.intensity = intensity


_BreakMode = enum.Enum("_BreakMode", ["FLAT", "BREAK"])


# In Lindig's paper fits() and format() are defined recursively. This is a
# non-recursive formulation using an explicit stack, necessary because Python
# doesn't have a tail recursion optimization.

def _fits(doc: Doc, width: int, agenda: list[tuple[int, _BreakMode, Doc]]
         ) -> bool:
  while width >= 0 and len(agenda) > 0:
    i, m, doc = agenda.pop()
    if isinstance(doc, _NilDoc):
      pass
    elif isinstance(doc, _TextDoc):
      width -= len(doc.text)
    elif isinstance(doc, _ConcatDoc):
      agenda.extend((i, m, d) for d in reversed(doc.children))
    elif isinstance(doc, _BreakDoc):
      if m == _BreakMode.BREAK:
        return True
      width -= len(doc.text)
    elif isinstance(doc, _NestDoc):
      agenda.append((i + doc.n, m, doc.child))
    elif isinstance(doc, _GroupDoc):
      agenda.append((i, _BreakMode.FLAT, doc.child))
    elif isinstance(doc, _ColorDoc) or isinstance(doc, _SourceMapDoc):
      agenda.append((i, m, doc.child))
    else:
      raise ValueError("Invalid document ", doc)

  return width >= 0


# Annotation layout: A flat group is sparse if there are no breaks between
# annotations.
def _sparse(doc: Doc) -> bool:
  agenda = [doc]
  num_annotations = 0
  seen_break = False
  while len(agenda) > 0:
    doc = agenda.pop()
    if isinstance(doc, _NilDoc):
      pass
    elif isinstance(doc, _TextDoc):
      if doc.annotation is not None:
        if num_annotations >= 1 and seen_break:
          return False
        num_annotations += 1
    elif isinstance(doc, _ConcatDoc):
      agenda.extend(reversed(doc.children))
    elif isinstance(doc, _BreakDoc):
      seen_break = True
    elif isinstance(doc, _NestDoc):
      agenda.append(doc.child)
    elif isinstance(doc, _GroupDoc):
      agenda.append(doc.child)
    elif isinstance(doc, _ColorDoc) or isinstance(doc, _SourceMapDoc):
      agenda.append(doc.child)
    else:
      raise ValueError("Invalid document ", doc)

  return True

class _ColorState(NamedTuple):
  foreground: Color
  background: Color
  intensity: Intensity

class _State(NamedTuple):
  indent: int
  mode: _BreakMode
  doc: Doc
  color: _ColorState
  source_map: Any

class _Line(NamedTuple):
  text: str
  width: int
  annotations: str | None | list[str]


def _update_color(use_color: bool, state: _ColorState, update: _ColorState
                 ) -> tuple[_ColorState, str]:
  if not use_color or colorama is None:
    return update, ""
  color_str = ""
  if state.foreground != update.foreground:
    color_str += getattr(colorama.Fore, str(update.foreground.name))
  if state.background != update.background:
    color_str += getattr(colorama.Back, str(update.background.name))
  if state.intensity != update.intensity:
    color_str += colorama.Style.NORMAL  # pytype: disable=unsupported-operands
    color_str += getattr(colorama.Style, str(update.intensity.name))
  return update, color_str


def _align_annotations(lines):
  # TODO: Hafiz also implements a local alignment mode, where groups of lines
  # with annotations are aligned together.
  maxlen = max(l.width for l in lines)
  out = []
  for l in lines:
    if len(l.annotations) == 0:
      out.append(l._replace(annotations=None))
    elif len(l.annotations) == 1:
      out.append(l._replace(text=l.text + " " * (maxlen - l.width),
                            annotations=l.annotations[0]))
    else:
      out.append(l._replace(text=l.text + " " * (maxlen - l.width),
                            annotations=l.annotations[0]))
      for a in l.annotations[1:]:
        out.append(_Line(text=" " * maxlen, width=l.width, annotations=a))
  return out



def _format(
  doc: Doc, width: int, *, use_color: bool, annotation_prefix: str,
  source_map: list[list[tuple[int, int, Any]]] | None
) -> str:
  lines = []
  default_colors = _ColorState(Color.RESET, Color.RESET, Intensity.NORMAL)
  annotation_colors = _ColorState(Color.RESET, Color.RESET, Intensity.DIM)
  color_state = default_colors
  source_start = 0       # The column at which the current source region starts.
  source = _NO_SOURCE    # The currently active source region.
  line_source_map = []  # Source maps for the current line of text.
  agenda = [_State(0, _BreakMode.BREAK, doc, default_colors, source)]
  k = 0
  line_text = ""
  line_annotations = []
  while len(agenda) > 0:
    i, m, doc, color, agenda_source = agenda.pop()
    if source_map is not None and agenda_source != source:
      pos = len(line_text)
      if source_start != pos and source is not _NO_SOURCE:
        line_source_map.append((source_start, pos, source))
      source = agenda_source
      source_start = pos
    if isinstance(doc, _NilDoc):
      pass
    elif isinstance(doc, _TextDoc):
      color_state, color_str = _update_color(use_color, color_state, color)
      line_text += color_str
      line_text += doc.text
      if doc.annotation is not None:
        line_annotations.append(doc.annotation)
      k += len(doc.text)
    elif isinstance(doc, _ConcatDoc):
      agenda.extend(_State(i, m, d, color, source)
                    for d in reversed(doc.children))
    elif isinstance(doc, _BreakDoc):
      if m == _BreakMode.BREAK:
        if len(line_annotations) > 0:
          color_state, color_str = _update_color(use_color, color_state,
                                                 annotation_colors)
          line_text += color_str
        lines.append(_Line(line_text, k, line_annotations))
        if source_map is not None:
          pos = len(line_text)
          if source_start != pos and source is not _NO_SOURCE:
            line_source_map.append((source_start, pos, source))
          source_map.append(line_source_map)
          line_source_map = []
          source_start = i
        line_text = " " * i
        line_annotations = []
        k = i
      else:
        color_state, color_str = _update_color(use_color, color_state, color)
        line_text += color_str
        line_text += doc.text
        k += len(doc.text)
    elif isinstance(doc, _NestDoc):
      agenda.append(_State(i + doc.n, m, doc.child, color, source))
    elif isinstance(doc, _GroupDoc):
      # In Lindig's paper, _fits is passed the remainder of the document.
      # I'm pretty sure that's a bug and we care only if the current group fits!
      if (_sparse(doc)
          and _fits(doc, width - k, [(i, _BreakMode.FLAT, doc.child)])):
        agenda.append(_State(i, _BreakMode.FLAT, doc.child, color, source))
      else:
        agenda.append(_State(i, _BreakMode.BREAK, doc.child, color, source))
    elif isinstance(doc, _ColorDoc):
      color = _ColorState(doc.foreground or color.foreground,
                          doc.background or color.background,
                          doc.intensity or color.intensity)
      agenda.append(_State(i, m, doc.child, color, source))
    elif isinstance(doc, _SourceMapDoc):
      agenda.append(_State(i, m, doc.child, color, doc.source))
    else:
      raise ValueError("Invalid document ", doc)

  if len(line_annotations) > 0:
    color_state, color_str = _update_color(use_color, color_state,
                                           annotation_colors)
    line_text += color_str
  if source_map is not None:
    pos = len(line_text)
    if source_start != pos and source is not _NO_SOURCE:
      line_source_map.append((source_start, pos, source))
    source_map.append(line_source_map)
  lines.append(_Line(line_text, k, line_annotations))
  lines = _align_annotations(lines)
  out = "\n".join(
    l.text if l.annotations is None
    else f"{l.text}{annotation_prefix}{l.annotations}" for l in lines)
  color_state, color_str = _update_color(use_color, color_state,
                                         default_colors)
  return out + color_str




# Public API.

def nil() -> Doc:
  """An empty document."""
  return _nil

def text(s: str, annotation: str | None = None) -> Doc:
  """Literal text."""
  return _TextDoc(s, annotation)

def concat(docs: Sequence[Doc]) -> Doc:
  """Concatenation of documents."""
  docs = list(docs)
  if len(docs) == 1:
    return docs[0]
  return _ConcatDoc(docs)

def brk(text: str = " ") -> Doc:
  """A break.

  Prints either as a newline or as `text`, depending on the enclosing group.
  """
  return _BreakDoc(text)

def group(doc: Doc) -> Doc:
  """Layout alternative groups.

  Prints the group with its breaks as their text (typically spaces) if the
  entire group would fit on the line when printed that way. Otherwise, breaks
  inside the group as printed as newlines.
  """
  return _GroupDoc(doc)

def nest(n: int, doc: Doc) -> Doc:
  """Increases the indentation level by `n`."""
  return _NestDoc(n, doc)


def color(doc: Doc, *, foreground: Color | None = None,
          background: Color | None = None,
          intensity: Intensity | None = None):
  """ANSI colors.

  Overrides the foreground/background/intensity of the text for the child doc.
  Requires use_colors=True to be set when printing and the `colorama` package
  to be installed; otherwise does nothing.
  """
  return _ColorDoc(doc, foreground=foreground, background=background,
                   intensity=intensity)


def source_map(doc: Doc, source: Any):
  """Source mapping.

  A source map associates a region of the pretty-printer's text output with a
  source location that produced it. For the purposes of the pretty printer a
  ``source`` may be any object: we require only that we can compare sources for
  equality. A text region to source object mapping can be populated as a side
  output of the ``format`` method.
  """
  return _SourceMapDoc(doc, source)

type_annotation = partial(color, intensity=Intensity.NORMAL,
                          foreground=Color.MAGENTA)
keyword = partial(color, intensity=Intensity.BRIGHT, foreground=Color.BLUE)


def join(sep: Doc, docs: Sequence[Doc]) -> Doc:
  """Concatenates `docs`, separated by `sep`."""
  docs = list(docs)
  if len(docs) == 0:
    return nil()
  xs = [docs[0]]
  for doc in docs[1:]:
    xs.append(sep)
    xs.append(doc)
  return concat(xs)

# Copyright 2024 The etils Authors.
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

"""HTML builder for Python objects."""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import html
import inspect
import types
from typing import Any, Callable, ClassVar, Generic, Type, TypeVar, Union
import uuid

from etils import enp
from etils.ecolab.inspects import attrs
from etils.ecolab.inspects import html_helper as H
from google.protobuf import message

# Import both C++ and Python API
from google.protobuf.internal import containers

_T = TypeVar('_T')


# All nodes are loaded globally and never cleared. Indeed, it's not possible
# to know when the Javascript is deleted (cell is cleared).
# It could create RAM issues by keeping object alive for too long. Could
# use weakref if this become an issue in practice.
_ALL_NODES: dict[str, Node] = {}


@dataclasses.dataclass
class Node:
  """Node base class.

  Each node correspond to a `<li>` element in the nested tree. When the node
  is expanded, `inner_html` is add the child nodes.

  Attributes:
    id: HTML id used to identify the node.
  """

  id: str = dataclasses.field(init=False)

  def __post_init__(self):
    # TODO(epot): Could likely have shorter/smarter uuids
    self.id = str(uuid.uuid1())
    _ALL_NODES[self.id] = self

  @classmethod
  def from_id(cls, id_: str) -> Node:
    """Returns the cached node from the HTML id."""
    return _ALL_NODES[id_]

  @classmethod
  def from_obj(cls, obj: object, *, name: str = '') -> ObjectNode:
    """Factory of a node from any object."""
    for sub_cls in [
        BuiltinNode,
        FnNode,
        MappingNode,
        SequenceNode,
        SetNode,
        ClsNode,
        ArrayNode,
        ProtoNode,
        ExceptionNode,
        ObjectNode,
    ]:
      if isinstance(obj, sub_cls.MATCH_TYPES):
        break
    else:
      raise TypeError(f'Unexpected object {obj!r}.')

    return sub_cls(obj=obj, name=name)  # pytype: disable=wrong-arg-types

  @property
  def header_html(self) -> str:
    """`<li>` one-line content."""
    raise NotImplementedError

  @property
  def inner_html(self) -> str:
    """Inner content when the item is expanded."""
    raise NotImplementedError

  def _li(self, *, clickable: bool = True) -> Callable[..., str]:
    """`<li>` section, called inside `header_html`."""
    if clickable:
      class_ = 'register-onclick-expand'
    else:
      class_ = 'caret-invisible'

    def apply(*content):
      # TODO(epot): Non-clickable nodes likely do not need ids
      return H.li(id=self.id)(H.span(class_=['caret', class_])(*content))

    return apply


@dataclasses.dataclass
class SubsectionNode(Node):
  """Expandable subsection of an object (to group object attributes).

  Example: `[[Methods]]`,...

  ```
  > obj:
    > ...
    > [[Methods]]
      > ...
    > [[Special attributes]]
      > ...
  ```
  """

  name: str
  children: list[Node]

  @property
  def header_html(self) -> str:
    return self._li()(H.span(class_=['preview'])(f'[[{self.name}]]'))

  @property
  def inner_html(self) -> str:
    children = [c.header_html for c in self.children]
    return H.ul(class_=['collapsible'])(*children)


@dataclasses.dataclass
class HtmlNode(Node):
  """Simple non clickable node that only displaying HTML."""

  html: str

  @property
  def header_html(self) -> str:
    return self._li(clickable=False)(self.html)


@dataclasses.dataclass
class KeyValNode(Node):
  """The (k, v) items for `list`, `dict`."""

  key: object
  value: object

  # TODO(epot):
  # * Cleaner implementation.
  # * Make both key and value expandable (when not built-in) ?

  @property
  def header_html(self) -> str:
    # * Built-ins are not be expandable.
    is_builtin = isinstance(self.value, BuiltinNode.MATCH_TYPES)
    return self._li(clickable=not is_builtin)(
        _obj_html_repr(self.key), ': ', _obj_html_repr(self.value)
    )

  @property
  def inner_html(self) -> str:
    return Node.from_obj(self.value).inner_html


@dataclasses.dataclass
class ObjectNode(Node, Generic[_T]):
  """Any Python objects."""

  obj: _T
  name: str

  MATCH_TYPES: ClassVar[type[Any] | tuple[type[Any], ...]] = object

  @property
  def header_html(self) -> str:
    if self.is_root:
      prefix = ''
    else:
      prefix = f'{self.name}: '
    return self._li(clickable=not self.is_leaf)(
        H.span(class_=['key-main'])(f'{prefix}{self.header_repr}')
    )

  @property
  def inner_html(self) -> str:
    children = [c.header_html for c in self.children]
    return H.ul(class_=['collapsible'])(*children)

  @property
  def children(self) -> list[Node]:
    """Extract all attributes."""
    children = [
        Node.from_obj(v, name=k) for k, v in attrs.get_attrs(self.obj).items()
    ]

    magic_attrs = []
    fn_attrs = []
    private_attrs = []
    val_attrs = []

    for c in children:
      if c.name.startswith('__') and c.name.endswith('__'):
        magic_attrs.append(c)
      elif c.name.startswith('_'):
        private_attrs.append(c)
      elif isinstance(c, FnNode):
        fn_attrs.append(c)
      else:
        val_attrs.append(c)

    children = val_attrs
    if fn_attrs:
      children.append(SubsectionNode(children=fn_attrs, name='Methods'))
    if private_attrs:
      children.append(SubsectionNode(children=private_attrs, name='Private'))
    if magic_attrs:
      children.append(
          SubsectionNode(children=magic_attrs, name='Special attributes')
      )

    return children

  @property
  def header_repr(self) -> str:
    return _obj_html_repr(self.obj)

  @property
  def is_root(self) -> bool:
    """Returns `True` if the node is top-level."""
    return not bool(self.name)

  @property
  def is_leaf(self) -> bool:
    """Returns `True` if the node cannot be recursed into."""
    return False


@dataclasses.dataclass
class BuiltinNode(ObjectNode[Union[int, float, bool, str, bytes, None]]):  # pytype: disable=bad-concrete-type
  """`int`, `float`, `bytes`, `str`,..."""

  MATCH_TYPES = (type(None), int, float, bool, str, bytes, type(...))

  # TODO(epot): For subclasses, print the actual type somewhere ? Same for list,
  # dict,... ?

  @property
  def is_leaf(self) -> bool:
    # Can recurse into built-ins only if they are roots
    return not self.is_root


@dataclasses.dataclass
class MappingNode(ObjectNode[collections.abc.Mapping]):  # pytype: disable=bad-concrete-type
  """`dict` like."""

  MATCH_TYPES = (
      dict,
      collections.abc.Mapping,
      # Proto map fields (both C++ and Python API)
      containers.MessageMap,
      containers.ScalarMap,
  )

  @property
  def children(self) -> list[Node]:
    return [
        KeyValNode(key=k, value=v) for k, v in self.obj.items()
    ] + super().children


@dataclasses.dataclass
class SetNode(ObjectNode[collections.abc.Set]):  # pytype: disable=bad-concrete-type
  """`set` like."""

  MATCH_TYPES = (set, frozenset, collections.abc.Set)

  @property
  def children(self) -> list[Node]:
    return [KeyValNode(key=id(v), value=v) for v in self.obj] + super().children


@dataclasses.dataclass
class SequenceNode(ObjectNode[Union[list, tuple]]):
  """`list` like."""

  MATCH_TYPES = (
      list,
      tuple,
      # Proto repeated fields (both C++ and Python API)
      containers.RepeatedScalarFieldContainer,
      containers.RepeatedCompositeFieldContainer,
  )

  @property
  def children(self) -> list[Node]:
    return [
        KeyValNode(key=i, value=v) for i, v in enumerate(self.obj)
    ] + super().children


@dataclasses.dataclass
class FnNode(ObjectNode[Callable[..., Any]]):
  """Function."""

  MATCH_TYPES = (
      types.FunctionType,
      types.BuiltinFunctionType,
      types.BuiltinMethodType,
      types.MethodType,
      types.MethodDescriptorType,
      types.ClassMethodDescriptorType,
      types.MemberDescriptorType,
      types.WrapperDescriptorType,
      types.MethodWrapperType,
      functools.partial,
  )

  @property
  def header_repr(self) -> str:
    return _fn_html_repr(self.obj)

  @property
  def children(self) -> list[Node]:
    return _fn_children(self.obj) + super().children

  # TODO(epot): Expand the function should display docstring, signature,
  # annotations, default values


@dataclasses.dataclass
class ArrayNode(ObjectNode[enp.typing.Array]):
  """Array."""

  MATCH_TYPES = enp.lazy.LazyArray

  # TODO(epot): Also print array values in the one-line description ?

  @property
  def children(self) -> list[Node]:
    # When expanded, print the array values
    return [
        HtmlNode(_obj_html_repr(self.obj, array_values=True))
    ] + super().children


@dataclasses.dataclass
class ClsNode(ObjectNode[Type[Any]]):
  """Type."""

  MATCH_TYPES = type

  # TODO(epot): Add link to source code

  @property
  def children(self) -> list[Node]:
    if isinstance(self.obj, type) and self.obj.mro is type.mro:
      mro = self.obj.mro(self.obj)  # `type.mro()` do not work
    else:
      mro = self.obj.mro()
    # Add `[[mro]]` subsection
    # TODO(epot): Also add `[[mro]]` section to all objects (but hidden inside)
    # private.
    return super().children + [
        SubsectionNode(
            children=[Node.from_obj(cls) for cls in mro],
            name='mro',
        )
    ]


_PROTO_FIELD_NAMES = frozenset({
    'DESCRIPTOR',
    'Extensions',
})


class ProtoNode(ObjectNode[message.Message]):
  """Proto."""

  MATCH_TYPES = message.Message

  @property
  def children(self) -> list[Node]:
    # Filter the proto-specific attributes to hide them in a separate section
    val_attrs = []
    proto_attrs = []
    other_attrs = []
    for c in super().children:
      if isinstance(c, SubsectionNode):
        other_attrs.append(c)
      elif not isinstance(c, ObjectNode):
        raise ValueError(f'Unexpected child {c!r}')
      elif c.name in _PROTO_FIELD_NAMES:
        proto_attrs.append(c)
      else:
        val_attrs.append(c)

    return (
        val_attrs
        + other_attrs
        + [SubsectionNode(children=proto_attrs, name='Proto')]
    )


@dataclasses.dataclass
class ExceptionNode(ObjectNode[attrs.ExceptionWrapper]):
  """Exception."""

  MATCH_TYPES = attrs.ExceptionWrapper

  # TODO(epot): Could expand with the traceback

  @property
  def is_leaf(self) -> bool:
    return True


def _obj_html_repr(obj: object, *, array_values: bool = False) -> str:
  """Returns the object representation."""
  if isinstance(obj, type(None)):
    type_ = 'null'
  elif isinstance(obj, type(...)):
    type_ = 'null'
  elif isinstance(obj, (int, float)):
    type_ = 'number'
  elif isinstance(obj, bool):
    type_ = 'boolean'
  elif isinstance(obj, str):
    type_ = 'string'
    obj = _truncate_long_str(obj, expand_new_lines=True)
  elif isinstance(obj, bytes):
    type_ = 'string'
    obj = _truncate_long_str(repr(obj))
  elif not array_values and isinstance(obj, enp.lazy.LazyArray):
    type_ = 'number'
    obj = enp.ArraySpec.from_array(obj)
  elif isinstance(obj, attrs.ExceptionWrapper):
    type_ = 'error'
    obj = obj.e
  else:
    type_ = 'preview'
    try:
      obj = repr(obj)
    except Exception as e:  # pylint: disable=broad-except
      return _obj_html_repr(attrs.ExceptionWrapper(e))
    obj = _truncate_long_str(obj)

  if not isinstance(obj, str):
    obj = repr(obj)
    obj = html.escape(obj)
  return H.span(class_=[type_])(obj)


def _truncate_long_str(value: str, *, expand_new_lines: bool = False) -> str:
  """Truncate long strings."""
  value = html.escape(value)
  if expand_new_lines:
    # `repr` replace `\n` by `\\n` on str, so only apply it on the short version
    short_value = repr(value)
  else:
    short_value = value
  # TODO(epot): Could have a better expand section which truncate long string
  # (e.g. > 100 lines)
  # TODO(epot): Better CSS (with button)
  if len(short_value) > 80:
    if expand_new_lines:
      # On the long value, add the same braces `'` or `"`
      long_value = short_value[0] + value + short_value[-1]
    else:
      long_value = value
    return H.span(class_=['content-switch'])(
        # Short version
        H.span(class_=['content-version-short'])(
            short_value[:80]
            + H.span(
                class_=['content-switch-expand', 'register-onclick-switch']
            )('...')
        ),
        # Long version
        H.span(class_=['content-version-long', 'register-onclick-switch'])(
            long_value
        ),
    )
  else:
    return short_value


def _fn_children(fn: Callable[..., Any]) -> list[Node]:
  """Build the fn children."""
  children = []
  # Add docstring
  try:
    doc = fn.__doc__
  except Exception:  # pylint: disable=broad-except
    pass
  else:
    if doc:
      children.append(HtmlNode(_obj_html_repr(doc)))

  # Add signature
  try:
    sig = inspect.Signature.from_callable(fn)
  except Exception:  # pylint: disable=broad-except
    # Many builtins do not expose any signature information
    pass
  else:
    for param in sig.parameters.values():
      name = param.name
      if param.kind == inspect.Parameter.VAR_POSITIONAL:
        name = f'*{name}'
      elif param.kind == inspect.Parameter.VAR_KEYWORD:
        name = f'**{name}'
      name = H.span(class_=['preview'])(name)

      if param.annotation is inspect.Parameter.empty:
        annotations = ''
      else:
        annotations = f': {_obj_html_repr(param.annotation)}'
      if param.default is inspect.Parameter.empty:
        default = ''
      else:
        default = f' = {_obj_html_repr(param.default)}'
      node = HtmlNode(name + annotations + default)
      children.append(node)
  return children


def _fn_signature_repr(fn: Callable[..., Any]) -> str:
  """Constructs the signature representation of a function."""
  try:
    sig = inspect.Signature.from_callable(fn)
  except Exception:  # pylint: disable=broad-except
    # Many builtins do not expose any signature information
    return '...'

  parts = []
  is_first = True
  is_kwarg_only = False
  for param in sig.parameters.values():
    if not is_first:
      parts.append(', ')
    is_first = False
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
      parts.append('*')
      is_kwarg_only = True
    elif param.kind == inspect.Parameter.VAR_KEYWORD:
      parts.append('**')
      is_kwarg_only = True
    elif param.kind == inspect.Parameter.KEYWORD_ONLY and not is_kwarg_only:
      is_kwarg_only = True
      parts.append('*, ')
    parts.append(param.name)
  return ''.join(parts)


def _fn_html_repr(fn: Callable[..., Any]) -> str:
  """Constructs the signature representation of a function."""
  # TODO(epot): Special partial case (should likely be another custom node)
  if isinstance(fn, functools.partial):
    return _obj_html_repr(fn)

  try:
    # TODO(epot): When using built-in, should display `object.__repr__` rather
    # than `A.__repr__`
    fn_name = fn.__qualname__
  except Exception:  # pylint: disable=broad-except
    # Not sure when this could happen
    return _obj_html_repr(fn)

  # Do not add annotations/default in the one-line repr (would be too
  # boilerplate)

  sig_str = f'{fn_name}({_fn_signature_repr(fn)})'
  sig_str = _truncate_long_str(sig_str)
  return H.span(class_='fn')('ƒ') + ' ' + H.span(class_=['preview'])(sig_str)

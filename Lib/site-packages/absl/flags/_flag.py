# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains Flag class - information about single command-line flag.

Do NOT import this module directly. Import the flags package and use the
aliases defined at the package level instead.
"""

from collections import abc
import copy
import enum
import functools
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union
from xml.dom import minidom

from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _helpers

_T = TypeVar('_T')
_ET = TypeVar('_ET', bound=enum.Enum)


@functools.total_ordering
class Flag(Generic[_T]):
  """Information about a command-line flag.

  Attributes:
    name: the name for this flag
    default: the default value for this flag
    default_unparsed: the unparsed default value for this flag.
    default_as_str: default value as repr'd string, e.g., "'true'"
      (or None)
    value: the most recent parsed value of this flag set by :meth:`parse`
    help: a help string or None if no help is available
    short_name: the single letter alias for this flag (or None)
    boolean: if 'true', this flag does not accept arguments
    present: true if this flag was parsed from command line flags
    parser: an :class:`~absl.flags.ArgumentParser` object
    serializer: an ArgumentSerializer object
    allow_override: the flag may be redefined without raising an error,
      and newly defined flag overrides the old one.
    allow_override_cpp: use the flag from C++ if available the flag
      definition is replaced by the C++ flag after init
    allow_hide_cpp: use the Python flag despite having a C++ flag with
      the same name (ignore the C++ flag)
    using_default_value: the flag value has not been set by user
    allow_overwrite: the flag may be parsed more than once without
      raising an error, the last set value will be used
    allow_using_method_names: whether this flag can be defined even if
      it has a name that conflicts with a FlagValues method.
    validators: list of the flag validators.

  The only public method of a ``Flag`` object is :meth:`parse`, but it is
  typically only called by a :class:`~absl.flags.FlagValues` object.  The
  :meth:`parse` method is a thin wrapper around the
  :meth:`ArgumentParser.parse()<absl.flags.ArgumentParser.parse>` method.  The
  parsed value is saved in ``.value``, and the ``.present`` attribute is
  updated.  If this flag was already present, an Error is raised.

  :meth:`parse` is also called during ``__init__`` to parse the default value
  and initialize the ``.value`` attribute.  This enables other python modules to
  safely use flags even if the ``__main__`` module neglects to parse the
  command line arguments.  The ``.present`` attribute is cleared after
  ``__init__`` parsing.  If the default value is set to ``None``, then the
  ``__init__`` parsing step is skipped and the ``.value`` attribute is
  initialized to None.

  Note: The default value is also presented to the user in the help
  string, so it is important that it be a legal value for this flag.
  """

  # NOTE: pytype doesn't find defaults without this.
  default: Optional[_T]
  default_as_str: Optional[str]
  default_unparsed: Union[Optional[_T], str]

  parser: _argument_parser.ArgumentParser[_T]

  def __init__(
      self,
      parser: _argument_parser.ArgumentParser[_T],
      serializer: Optional[_argument_parser.ArgumentSerializer[_T]],
      name: str,
      default: Union[Optional[_T], str],
      help_string: Optional[str],
      short_name: Optional[str] = None,
      boolean: bool = False,
      allow_override: bool = False,
      allow_override_cpp: bool = False,
      allow_hide_cpp: bool = False,
      allow_overwrite: bool = True,
      allow_using_method_names: bool = False,
  ) -> None:
    self.name = name

    if not help_string:
      help_string = '(no help available)'

    self.help = help_string
    self.short_name = short_name
    self.boolean = boolean
    self.present = 0
    self.parser = parser  # type: ignore[annotation-type-mismatch]
    self.serializer = serializer
    self.allow_override = allow_override
    self.allow_override_cpp = allow_override_cpp
    self.allow_hide_cpp = allow_hide_cpp
    self.allow_overwrite = allow_overwrite
    self.allow_using_method_names = allow_using_method_names

    self.using_default_value = True
    self._value: Optional[_T] = None
    self.validators: List[Any] = []
    if self.allow_hide_cpp and self.allow_override_cpp:
      raise _exceptions.Error(
          "Can't have both allow_hide_cpp (means use Python flag) and "
          'allow_override_cpp (means use C++ flag after InitGoogle)')

    self._set_default(default)

  @property
  def value(self) -> Optional[_T]:
    return self._value

  @value.setter
  def value(self, value: Optional[_T]):
    self._value = value

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return self is other

  def __lt__(self, other):
    if isinstance(other, Flag):
      return id(self) < id(other)
    return NotImplemented

  def __bool__(self):
    raise TypeError('A Flag instance would always be True. '
                    'Did you mean to test the `.value` attribute?')

  def __getstate__(self):
    raise TypeError("can't pickle Flag objects")

  def __copy__(self):
    raise TypeError('%s does not support shallow copies. '
                    'Use copy.deepcopy instead.' % type(self).__name__)

  def __deepcopy__(self, memo: Dict[int, Any]) -> 'Flag[_T]':
    result = object.__new__(type(self))
    result.__dict__ = copy.deepcopy(self.__dict__, memo)
    return result

  def _get_parsed_value_as_string(self, value: Optional[_T]) -> Optional[str]:
    """Returns parsed flag value as string."""
    if value is None:
      return None
    if self.serializer:
      return repr(self.serializer.serialize(value))
    if self.boolean:
      if value:
        return repr('true')
      else:
        return repr('false')
    return repr(str(value))

  def parse(self, argument: Union[str, _T]) -> None:
    """Parses string and sets flag value.

    Args:
      argument: str or the correct flag value type, argument to be parsed.
    """
    if self.present and not self.allow_overwrite:
      raise _exceptions.IllegalFlagValueError(
          'flag --%s=%s: already defined as %s' % (
              self.name, argument, self.value))
    self.value = self._parse(argument)
    self.present += 1

  def _parse(self, argument: Union[str, _T]) -> Optional[_T]:
    """Internal parse function.

    It returns the parsed value, and does not modify class states.

    Args:
      argument: str or the correct flag value type, argument to be parsed.

    Returns:
      The parsed value.
    """
    try:
      return self.parser.parse(argument)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as e:
      # Recast as IllegalFlagValueError.
      raise _exceptions.IllegalFlagValueError(
          'flag --%s=%s: %s' % (self.name, argument, e))

  def unparse(self) -> None:
    self.value = self.default
    self.using_default_value = True
    self.present = 0

  def serialize(self) -> str:
    """Serializes the flag."""
    return self._serialize(self.value)

  def _serialize(self, value: Optional[_T]) -> str:
    """Internal serialize function."""
    if value is None:
      return ''
    if self.boolean:
      if value:
        return '--%s' % self.name
      else:
        return '--no%s' % self.name
    else:
      if not self.serializer:
        raise _exceptions.Error(
            'Serializer not present for flag %s' % self.name)
      return '--%s=%s' % (self.name, self.serializer.serialize(value))

  def _set_default(self, value: Union[Optional[_T], str]) -> None:
    """Changes the default value (and current value too) for this Flag."""
    self.default_unparsed = value
    if value is None:
      self.default = None
    else:
      self.default = self._parse_from_default(value)
    self.default_as_str = self._get_parsed_value_as_string(self.default)
    if self.using_default_value:
      self.value = self.default

  # This is split out so that aliases can skip regular parsing of the default
  # value.
  def _parse_from_default(self, value: Union[str, _T]) -> Optional[_T]:
    return self._parse(value)

  def flag_type(self) -> str:
    """Returns a str that describes the type of the flag.

    NOTE: we use strings, and not the types.*Type constants because
    our flags can have more exotic types, e.g., 'comma separated list
    of strings', 'whitespace separated list of strings', etc.
    """
    return self.parser.flag_type()

  def _create_xml_dom_element(
      self, doc: minidom.Document, module_name: str, is_key: bool = False
  ) -> minidom.Element:
    """Returns an XML element that contains this flag's information.

    This is information that is relevant to all flags (e.g., name,
    meaning, etc.).  If you defined a flag that has some other pieces of
    info, then please override _ExtraXMLInfo.

    Please do NOT override this method.

    Args:
      doc: minidom.Document, the DOM document it should create nodes from.
      module_name: str,, the name of the module that defines this flag.
      is_key: boolean, True iff this flag is key for main module.

    Returns:
      A minidom.Element instance.
    """
    element = doc.createElement('flag')
    if is_key:
      element.appendChild(_helpers.create_xml_dom_element(doc, 'key', 'yes'))
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'file', module_name))
    # Adds flag features that are relevant for all flags.
    element.appendChild(_helpers.create_xml_dom_element(doc, 'name', self.name))
    if self.short_name:
      element.appendChild(_helpers.create_xml_dom_element(
          doc, 'short_name', self.short_name))
    if self.help:
      element.appendChild(_helpers.create_xml_dom_element(
          doc, 'meaning', self.help))
    # The default flag value can either be represented as a string like on the
    # command line, or as a Python object.  We serialize this value in the
    # latter case in order to remain consistent.
    if self.serializer and not isinstance(self.default, str):
      if self.default is not None:
        default_serialized = self.serializer.serialize(self.default)
      else:
        default_serialized = ''
    else:
      default_serialized = self.default  # type: ignore[assignment]
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'default', default_serialized))
    value_serialized = self._serialize_value_for_xml(self.value)
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'current', value_serialized))
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'type', self.flag_type()))
    # Adds extra flag features this flag may have.
    for e in self._extra_xml_dom_elements(doc):
      element.appendChild(e)
    return element

  def _serialize_value_for_xml(self, value: Optional[_T]) -> Any:
    """Returns the serialized value, for use in an XML help text."""
    return value

  def _extra_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    """Returns extra info about this flag in XML.

    "Extra" means "not already included by _create_xml_dom_element above."

    Args:
      doc: minidom.Document, the DOM document it should create nodes from.

    Returns:
      A list of minidom.Element.
    """
    # Usually, the parser knows the extra details about the flag, so
    # we just forward the call to it.
    return self.parser._custom_xml_dom_elements(doc)  # pylint: disable=protected-access


class BooleanFlag(Flag[bool]):
  """Basic boolean flag.

  Boolean flags do not take any arguments, and their value is either
  ``True`` (1) or ``False`` (0).  The false value is specified on the command
  line by prepending the word ``'no'`` to either the long or the short flag
  name.

  For example, if a Boolean flag was created whose long name was
  ``'update'`` and whose short name was ``'x'``, then this flag could be
  explicitly unset through either ``--noupdate`` or ``--nox``.
  """

  def __init__(
      self,
      name: str,
      default: Union[Optional[bool], str],
      help: Optional[str],  # pylint: disable=redefined-builtin
      short_name: Optional[str] = None,
      **args
  ) -> None:
    p = _argument_parser.BooleanParser()
    super().__init__(p, None, name, default, help, short_name, True, **args)


class EnumFlag(Flag[str]):
  """Basic enum flag; its value can be any string from list of enum_values."""

  parser: _argument_parser.EnumParser

  def __init__(
      self,
      name: str,
      default: Optional[str],
      help: Optional[str],  # pylint: disable=redefined-builtin
      enum_values: Iterable[str],
      short_name: Optional[str] = None,
      case_sensitive: bool = True,
      **args
  ):
    p = _argument_parser.EnumParser(enum_values, case_sensitive)
    g: _argument_parser.ArgumentSerializer[str]
    g = _argument_parser.ArgumentSerializer()
    super().__init__(p, g, name, default, help, short_name, **args)
    self.parser = p
    self.help = '<%s>: %s' % ('|'.join(p.enum_values), self.help)

  def _extra_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = []
    for enum_value in self.parser.enum_values:
      elements.append(_helpers.create_xml_dom_element(
          doc, 'enum_value', enum_value))
    return elements


class EnumClassFlag(Flag[_ET]):
  """Basic enum flag; its value is an enum class's member."""

  parser: _argument_parser.EnumClassParser

  def __init__(
      self,
      name: str,
      default: Union[Optional[_ET], str],
      help: Optional[str],  # pylint: disable=redefined-builtin
      enum_class: Type[_ET],
      short_name: Optional[str] = None,
      case_sensitive: bool = False,
      **args
  ):
    p = _argument_parser.EnumClassParser(
        enum_class, case_sensitive=case_sensitive
    )
    g: _argument_parser.EnumClassSerializer[_ET]
    g = _argument_parser.EnumClassSerializer(lowercase=not case_sensitive)
    super().__init__(p, g, name, default, help, short_name, **args)
    self.parser = p
    self.help = '<%s>: %s' % ('|'.join(p.member_names), self.help)

  def _extra_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = []
    for enum_value in self.parser.enum_class.__members__.keys():
      elements.append(_helpers.create_xml_dom_element(
          doc, 'enum_value', enum_value))
    return elements


class MultiFlag(Generic[_T], Flag[List[_T]]):
  """A flag that can appear multiple time on the command-line.

  The value of such a flag is a list that contains the individual values
  from all the appearances of that flag on the command-line.

  See the __doc__ for Flag for most behavior of this class.  Only
  differences in behavior are described here:

    * The default value may be either a single value or an iterable of values.
      A single value is transformed into a single-item list of that value.

    * The value of the flag is always a list, even if the option was
      only supplied once, and even if the default value is a single
      value
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.help += ';\n    repeat this option to specify a list of values'

  def parse(self, arguments: Union[str, _T, Iterable[_T]]):  # pylint: disable=arguments-renamed
    """Parses one or more arguments with the installed parser.

    Args:
      arguments: a single argument or a list of arguments (typically a
        list of default values); a single argument is converted
        internally into a list containing one item.
    """
    new_values = self._parse(arguments)
    if self.present:
      assert self.value is not None
      self.value.extend(new_values)
    else:
      self.value = new_values
    self.present += len(new_values)

  def _parse(self, arguments: Union[str, _T, Iterable[_T]]) -> List[_T]:  # pylint: disable=arguments-renamed
    arguments_list: List[Union[str, _T]]

    if isinstance(arguments, str):
      arguments_list = [arguments]

    elif isinstance(arguments, abc.Iterable):
      arguments_list = list(arguments)

    else:
      # Default value may be a list of values.  Most other arguments
      # will not be, so convert them into a single-item list to make
      # processing simpler below.
      arguments_list = [arguments]

    return [super(MultiFlag, self)._parse(item) for item in arguments_list]  # type: ignore

  def _serialize(self, value: Optional[List[_T]]) -> str:
    """See base class."""
    if not self.serializer:
      raise _exceptions.Error(
          'Serializer not present for flag %s' % self.name)
    if value is None:
      return ''

    serialized_items = [
        super(MultiFlag, self)._serialize(value_item)  # type: ignore[arg-type]
        for value_item in value
    ]

    return '\n'.join(serialized_items)

  def flag_type(self):
    """See base class."""
    return 'multi ' + self.parser.flag_type()

  def _extra_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = []
    if hasattr(self.parser, 'enum_values'):
      for enum_value in self.parser.enum_values:  # pytype: disable=attribute-error
        elements.append(_helpers.create_xml_dom_element(
            doc, 'enum_value', enum_value))
    return elements


class MultiEnumClassFlag(MultiFlag[_ET]):  # pytype: disable=not-indexable
  """A multi_enum_class flag.

  See the __doc__ for MultiFlag for most behaviors of this class.  In addition,
  this class knows how to handle enum.Enum instances as values for this flag
  type.
  """

  parser: _argument_parser.EnumClassParser[_ET]  # type: ignore[assignment]

  def __init__(
      self,
      name: str,
      default: Union[None, Iterable[_ET], _ET, Iterable[str], str],
      help_string: str,
      enum_class: Type[_ET],
      case_sensitive: bool = False,
      **args
  ):
    p = _argument_parser.EnumClassParser(
        enum_class, case_sensitive=case_sensitive)
    g: _argument_parser.EnumClassListSerializer
    g = _argument_parser.EnumClassListSerializer(
        list_sep=',', lowercase=not case_sensitive)
    super().__init__(p, g, name, default, help_string, **args)
    # NOTE: parser should be typed EnumClassParser[_ET] but the constructor
    # restricts the available interface to ArgumentParser[str].
    self.parser = p
    # NOTE: serializer should be non-Optional but this isn't inferred.
    self.serializer = g
    self.help = (
        '<%s>: %s;\n    repeat this option to specify a list of values' %
        ('|'.join(p.member_names), help_string or '(no help available)'))

  def _extra_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = []
    for enum_value in self.parser.enum_class.__members__.keys():  # pytype: disable=attribute-error
      elements.append(_helpers.create_xml_dom_element(
          doc, 'enum_value', enum_value))
    return elements

  def _serialize_value_for_xml(self, value):
    """See base class."""
    if value is not None:
      if not self.serializer:
        raise _exceptions.Error(
            'Serializer not present for flag %s' % self.name
        )
      value_serialized = self.serializer.serialize(value)
    else:
      value_serialized = ''
    return value_serialized

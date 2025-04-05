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

"""Contains base classes used to parse and convert arguments.

Do NOT import this module directly. Import the flags package and use the
aliases defined at the package level instead.
"""

import collections
import csv
import enum
import io
import string
from typing import Any, Dict, Generic, Iterable, List, Optional, Sequence, Type, TypeVar, Union
from xml.dom import minidom

from absl.flags import _helpers

_T = TypeVar('_T')
_ET = TypeVar('_ET', bound=enum.Enum)
_N = TypeVar('_N', int, float)


class _ArgumentParserCache(type):
  """Metaclass used to cache and share argument parsers among flags."""

  _instances: Dict[Any, Any] = {}

  def __call__(cls, *args, **kwargs):
    """Returns an instance of the argument parser cls.

    This method overrides behavior of the __new__ methods in
    all subclasses of ArgumentParser (inclusive). If an instance
    for cls with the same set of arguments exists, this instance is
    returned, otherwise a new instance is created.

    If any keyword arguments are defined, or the values in args
    are not hashable, this method always returns a new instance of
    cls.

    Args:
      *args: Positional initializer arguments.
      **kwargs: Initializer keyword arguments.

    Returns:
      An instance of cls, shared or new.
    """
    if kwargs:
      return type.__call__(cls, *args, **kwargs)
    else:
      instances = cls._instances
      key = (cls,) + tuple(args)
      try:
        return instances[key]
      except KeyError:
        # No cache entry for key exists, create a new one.
        return instances.setdefault(key, type.__call__(cls, *args))
      except TypeError:
        # An object in args cannot be hashed, always return
        # a new instance.
        return type.__call__(cls, *args)


class ArgumentParser(Generic[_T], metaclass=_ArgumentParserCache):
  """Base class used to parse and convert arguments.

  The :meth:`parse` method checks to make sure that the string argument is a
  legal value and convert it to a native type.  If the value cannot be
  converted, it should throw a ``ValueError`` exception with a human
  readable explanation of why the value is illegal.

  Subclasses should also define a syntactic_help string which may be
  presented to the user to describe the form of the legal values.

  Argument parser classes must be stateless, since instances are cached
  and shared between flags. Initializer arguments are allowed, but all
  member variables must be derived from initializer arguments only.
  """

  syntactic_help: str = ''

  def parse(self, argument: str) -> Optional[_T]:
    """Parses the string argument and returns the native value.

    By default it returns its argument unmodified.

    Args:
      argument: string argument passed in the commandline.

    Raises:
      ValueError: Raised when it fails to parse the argument.
      TypeError: Raised when the argument has the wrong type.

    Returns:
      The parsed value in native type.
    """
    if not isinstance(argument, str):
      raise TypeError('flag value must be a string, found "{}"'.format(
          type(argument)))
    return argument  # type: ignore[return-value]

  def flag_type(self) -> str:
    """Returns a string representing the type of the flag."""
    return 'string'

  def _custom_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    """Returns a list of minidom.Element to add additional flag information.

    Args:
      doc: minidom.Document, the DOM document it should create nodes from.
    """
    del doc  # Unused.
    return []


class ArgumentSerializer(Generic[_T]):
  """Base class for generating string representations of a flag value."""

  def serialize(self, value: _T) -> str:
    """Returns a serialized string of the value."""
    return str(value)


class NumericParser(ArgumentParser[_N]):
  """Parser of numeric values.

  Parsed value may be bounded to a given upper and lower bound.
  """

  lower_bound: Optional[_N]
  upper_bound: Optional[_N]

  def is_outside_bounds(self, val: _N) -> bool:
    """Returns whether the value is outside the bounds or not."""
    return ((self.lower_bound is not None and val < self.lower_bound) or
            (self.upper_bound is not None and val > self.upper_bound))

  def parse(self, argument: Union[str, _N]) -> _N:
    """See base class."""
    val = self.convert(argument)
    if self.is_outside_bounds(val):
      raise ValueError('%s is not %s' % (val, self.syntactic_help))
    return val

  def _custom_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = []
    if self.lower_bound is not None:
      elements.append(_helpers.create_xml_dom_element(
          doc, 'lower_bound', self.lower_bound))
    if self.upper_bound is not None:
      elements.append(_helpers.create_xml_dom_element(
          doc, 'upper_bound', self.upper_bound))
    return elements

  def convert(self, argument: Union[str, _N]) -> _N:
    """Returns the correct numeric value of argument.

    Subclass must implement this method, and raise TypeError if argument is not
    string or has the right numeric type.

    Args:
      argument: string argument passed in the commandline, or the numeric type.

    Raises:
      TypeError: Raised when argument is not a string or the right numeric type.
      ValueError: Raised when failed to convert argument to the numeric value.
    """
    raise NotImplementedError


class FloatParser(NumericParser[float]):
  """Parser of floating point values.

  Parsed value may be bounded to a given upper and lower bound.
  """
  number_article = 'a'
  number_name = 'number'
  syntactic_help = ' '.join((number_article, number_name))

  def __init__(
      self,
      lower_bound: Optional[float] = None,
      upper_bound: Optional[float] = None,
  ) -> None:
    super().__init__()
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    sh = self.syntactic_help
    if lower_bound is not None and upper_bound is not None:
      sh = ('%s in the range [%s, %s]' % (sh, lower_bound, upper_bound))
    elif lower_bound == 0:
      sh = 'a non-negative %s' % self.number_name
    elif upper_bound == 0:
      sh = 'a non-positive %s' % self.number_name
    elif upper_bound is not None:
      sh = '%s <= %s' % (self.number_name, upper_bound)
    elif lower_bound is not None:
      sh = '%s >= %s' % (self.number_name, lower_bound)
    self.syntactic_help = sh

  def convert(self, argument: Union[int, float, str]) -> float:
    """Returns the float value of argument."""
    if (
        (isinstance(argument, int) and not isinstance(argument, bool))
        or isinstance(argument, float)
        or isinstance(argument, str)
    ):
      return float(argument)
    else:
      raise TypeError(
          'Expect argument to be a string, int, or float, found {}'.format(
              type(argument)))

  def flag_type(self) -> str:
    """See base class."""
    return 'float'


class IntegerParser(NumericParser[int]):
  """Parser of an integer value.

  Parsed value may be bounded to a given upper and lower bound.
  """
  number_article = 'an'
  number_name = 'integer'
  syntactic_help = ' '.join((number_article, number_name))

  def __init__(
      self, lower_bound: Optional[int] = None, upper_bound: Optional[int] = None
  ) -> None:
    super().__init__()
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    sh = self.syntactic_help
    if lower_bound is not None and upper_bound is not None:
      sh = ('%s in the range [%s, %s]' % (sh, lower_bound, upper_bound))
    elif lower_bound == 1:
      sh = 'a positive %s' % self.number_name
    elif upper_bound == -1:
      sh = 'a negative %s' % self.number_name
    elif lower_bound == 0:
      sh = 'a non-negative %s' % self.number_name
    elif upper_bound == 0:
      sh = 'a non-positive %s' % self.number_name
    elif upper_bound is not None:
      sh = '%s <= %s' % (self.number_name, upper_bound)
    elif lower_bound is not None:
      sh = '%s >= %s' % (self.number_name, lower_bound)
    self.syntactic_help = sh

  def convert(self, argument: Union[int, str]) -> int:
    """Returns the int value of argument."""
    if isinstance(argument, int) and not isinstance(argument, bool):
      return argument
    elif isinstance(argument, str):
      base = 10
      if len(argument) > 2 and argument[0] == '0':
        if argument[1] == 'o':
          base = 8
        elif argument[1] == 'x':
          base = 16
      return int(argument, base)
    else:
      raise TypeError('Expect argument to be a string or int, found {}'.format(
          type(argument)))

  def flag_type(self) -> str:
    """See base class."""
    return 'int'


class BooleanParser(ArgumentParser[bool]):
  """Parser of boolean values."""

  def parse(self, argument: Union[str, int]) -> bool:
    """See base class."""
    if isinstance(argument, str):
      if argument.lower() in ('true', 't', '1'):
        return True
      elif argument.lower() in ('false', 'f', '0'):
        return False
      else:
        raise ValueError('Non-boolean argument to boolean flag', argument)
    elif isinstance(argument, int):
      # Only allow bool or integer 0, 1.
      # Note that float 1.0 == True, 0.0 == False.
      bool_value = bool(argument)
      if argument == bool_value:
        return bool_value
      else:
        raise ValueError('Non-boolean argument to boolean flag', argument)

    raise TypeError('Non-boolean argument to boolean flag', argument)

  def flag_type(self) -> str:
    """See base class."""
    return 'bool'


class EnumParser(ArgumentParser[str]):
  """Parser of a string enum value (a string value from a given set)."""

  def __init__(
      self, enum_values: Iterable[str], case_sensitive: bool = True
  ) -> None:
    """Initializes EnumParser.

    Args:
      enum_values: [str], a non-empty list of string values in the enum.
      case_sensitive: bool, whether or not the enum is to be case-sensitive.

    Raises:
      ValueError: When enum_values is empty.
    """
    if not enum_values:
      raise ValueError(f'enum_values cannot be empty, found "{enum_values}"')
    if isinstance(enum_values, str):
      raise ValueError(f'enum_values cannot be a str, found "{enum_values}"')
    super().__init__()
    self.enum_values = list(enum_values)
    self.case_sensitive = case_sensitive

  def parse(self, argument: str) -> str:
    """Determines validity of argument and returns the correct element of enum.

    Args:
      argument: str, the supplied flag value.

    Returns:
      The first matching element from enum_values.

    Raises:
      ValueError: Raised when argument didn't match anything in enum.
    """
    if self.case_sensitive:
      if argument not in self.enum_values:
        raise ValueError('value should be one of <%s>' %
                         '|'.join(self.enum_values))
      else:
        return argument
    else:
      if argument.upper() not in [value.upper() for value in self.enum_values]:
        raise ValueError('value should be one of <%s>' %
                         '|'.join(self.enum_values))
      else:
        return [value for value in self.enum_values
                if value.upper() == argument.upper()][0]

  def flag_type(self) -> str:
    """See base class."""
    return 'string enum'


class EnumClassParser(ArgumentParser[_ET]):
  """Parser of an Enum class member."""

  def __init__(
      self, enum_class: Type[_ET], case_sensitive: bool = True
  ) -> None:
    """Initializes EnumParser.

    Args:
      enum_class: class, the Enum class with all possible flag values.
      case_sensitive: bool, whether or not the enum is to be case-sensitive. If
        False, all member names must be unique when case is ignored.

    Raises:
      TypeError: When enum_class is not a subclass of Enum.
      ValueError: When enum_class is empty.
    """
    if not issubclass(enum_class, enum.Enum):
      raise TypeError(f'{enum_class} is not a subclass of Enum.')
    if not enum_class.__members__:
      raise ValueError('enum_class cannot be empty, but "{}" is empty.'
                       .format(enum_class))
    if not case_sensitive:
      members = collections.Counter(
          name.lower() for name in enum_class.__members__)
      duplicate_keys = {
          member for member, count in members.items() if count > 1
      }
      if duplicate_keys:
        raise ValueError(
            'Duplicate enum values for {} using case_sensitive=False'.format(
                duplicate_keys))

    super().__init__()
    self.enum_class = enum_class
    self._case_sensitive = case_sensitive
    if case_sensitive:
      self._member_names = tuple(enum_class.__members__)
    else:
      self._member_names = tuple(
          name.lower() for name in enum_class.__members__)

  @property
  def member_names(self) -> Sequence[str]:
    """The accepted enum names, in lowercase if not case sensitive."""
    return self._member_names

  def parse(self, argument: Union[_ET, str]) -> _ET:
    """Determines validity of argument and returns the correct element of enum.

    Args:
      argument: str or Enum class member, the supplied flag value.

    Returns:
      The first matching Enum class member in Enum class.

    Raises:
      ValueError: Raised when argument didn't match anything in enum.
    """
    if isinstance(argument, self.enum_class):
      return argument  # pytype: disable=bad-return-type
    elif not isinstance(argument, str):
      raise ValueError(
          '{} is not an enum member or a name of a member in {}'.format(
              argument, self.enum_class))
    key = EnumParser(
        self._member_names, case_sensitive=self._case_sensitive).parse(argument)
    if self._case_sensitive:
      return self.enum_class[key]
    else:
      # If EnumParser.parse() return a value, we're guaranteed to find it
      # as a member of the class
      return next(value for name, value in self.enum_class.__members__.items()
                  if name.lower() == key.lower())

  def flag_type(self) -> str:
    """See base class."""
    return 'enum class'


class ListSerializer(Generic[_T], ArgumentSerializer[List[_T]]):

  def __init__(self, list_sep: str) -> None:
    self.list_sep = list_sep

  def serialize(self, value: List[_T]) -> str:
    """See base class."""
    return self.list_sep.join([str(x) for x in value])


class EnumClassListSerializer(ListSerializer[_ET]):
  """A serializer for :class:`MultiEnumClass` flags.

  This serializer simply joins the output of `EnumClassSerializer` using a
  provided separator.
  """

  _element_serializer: 'EnumClassSerializer'

  def __init__(self, list_sep: str, **kwargs) -> None:
    """Initializes EnumClassListSerializer.

    Args:
      list_sep: String to be used as a separator when serializing
      **kwargs: Keyword arguments to the `EnumClassSerializer` used to serialize
        individual values.
    """
    super().__init__(list_sep)
    self._element_serializer = EnumClassSerializer(**kwargs)

  def serialize(self, value: Union[_ET, List[_ET]]) -> str:
    """See base class."""
    if isinstance(value, list):
      return self.list_sep.join(
          self._element_serializer.serialize(x) for x in value)
    else:
      return self._element_serializer.serialize(value)


class CsvListSerializer(ListSerializer[str]):

  def serialize(self, value: List[str]) -> str:
    """Serializes a list as a CSV string or unicode."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=self.list_sep)
    writer.writerow([str(x) for x in value])
    serialized_value = output.getvalue().strip()

    # We need the returned value to be pure ascii or Unicodes so that
    # when the xml help is generated they are usefully encodable.
    return str(serialized_value)


class EnumClassSerializer(ArgumentSerializer[_ET]):
  """Class for generating string representations of an enum class flag value."""

  def __init__(self, lowercase: bool) -> None:
    """Initializes EnumClassSerializer.

    Args:
      lowercase: If True, enum member names are lowercased during serialization.
    """
    self._lowercase = lowercase

  def serialize(self, value: _ET) -> str:
    """Returns a serialized string of the Enum class value."""
    as_string = str(value.name)
    return as_string.lower() if self._lowercase else as_string


class BaseListParser(ArgumentParser):
  """Base class for a parser of lists of strings.

  To extend, inherit from this class; from the subclass ``__init__``, call::

      super().__init__(token, name)

  where token is a character used to tokenize, and name is a description
  of the separator.
  """

  def __init__(
      self, token: Optional[str] = None, name: Optional[str] = None
  ) -> None:
    assert name
    super().__init__()
    self._token = token
    self._name = name
    self.syntactic_help = 'a %s separated list' % self._name

  def parse(self, argument: str) -> List[str]:
    """See base class."""
    if isinstance(argument, list):
      return argument
    elif not argument:
      return []
    else:
      return [s.strip() for s in argument.split(self._token)]

  def flag_type(self) -> str:
    """See base class."""
    return '%s separated list of strings' % self._name


class ListParser(BaseListParser):
  """Parser for a comma-separated list of strings."""

  def __init__(self) -> None:
    super().__init__(',', 'comma')

  def parse(self, argument: Union[str, List[str]]) -> List[str]:
    """Parses argument as comma-separated list of strings."""
    if isinstance(argument, list):
      return argument
    elif not argument:
      return []
    else:
      try:
        return [s.strip() for s in list(csv.reader([argument], strict=True))[0]]
      except csv.Error as e:
        # Provide a helpful report for case like
        #   --listflag="$(printf 'hello,\nworld')"
        # IOW, list flag values containing naked newlines.  This error
        # was previously "reported" by allowing csv.Error to
        # propagate.
        raise ValueError('Unable to parse the value %r as a %s: %s'
                         % (argument, self.flag_type(), e))

  def _custom_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = super()._custom_xml_dom_elements(doc)
    elements.append(_helpers.create_xml_dom_element(
        doc, 'list_separator', repr(',')))
    return elements


class WhitespaceSeparatedListParser(BaseListParser):
  """Parser for a whitespace-separated list of strings."""

  def __init__(self, comma_compat: bool = False) -> None:
    """Initializer.

    Args:
      comma_compat: bool, whether to support comma as an additional separator.
          If False then only whitespace is supported.  This is intended only for
          backwards compatibility with flags that used to be comma-separated.
    """
    self._comma_compat = comma_compat
    name = 'whitespace or comma' if self._comma_compat else 'whitespace'
    super().__init__(None, name)

  def parse(self, argument: Union[str, List[str]]) -> List[str]:
    """Parses argument as whitespace-separated list of strings.

    It also parses argument as comma-separated list of strings if requested.

    Args:
      argument: string argument passed in the commandline.

    Returns:
      [str], the parsed flag value.
    """
    if isinstance(argument, list):
      return argument
    elif not argument:
      return []
    else:
      if self._comma_compat:
        argument = argument.replace(',', ' ')
      return argument.split()

  def _custom_xml_dom_elements(
      self, doc: minidom.Document
  ) -> List[minidom.Element]:
    elements = super()._custom_xml_dom_elements(doc)
    separators = list(string.whitespace)
    if self._comma_compat:
      separators.append(',')
    separators.sort()
    for sep_char in separators:
      elements.append(_helpers.create_xml_dom_element(
          doc, 'list_separator', repr(sep_char)))
    return elements

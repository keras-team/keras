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
"""Defines the FlagValues class - registry of 'Flag' objects.

Do NOT import this module directly. Import the flags package and use the
aliases defined at the package level instead.
"""

import copy
import logging
import os
import sys
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Set, TextIO, Tuple, TypeVar, Union
from xml.dom import minidom

from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
from absl.flags._flag import Flag

# Add flagvalues module to disclaimed module ids.
_helpers.disclaim_module_ids.add(id(sys.modules[__name__]))

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)  # pytype: disable=not-supported-yet


class FlagValues:
  """Registry of :class:`~absl.flags.Flag` objects.

  A :class:`FlagValues` can then scan command line arguments, passing flag
  arguments through to the 'Flag' objects that it owns.  It also
  provides easy access to the flag values.  Typically only one
  :class:`FlagValues` object is needed by an application:
  :const:`FLAGS`.

  This class is heavily overloaded:

  :class:`Flag` objects are registered via ``__setitem__``::

       FLAGS['longname'] = x   # register a new flag

  The ``.value`` attribute of the registered :class:`~absl.flags.Flag` objects
  can be accessed as attributes of this :class:`FlagValues` object, through
  ``__getattr__``.  Both the long and short name of the original
  :class:`~absl.flags.Flag` objects can be used to access its value::

       FLAGS.longname  # parsed flag value
       FLAGS.x  # parsed flag value (short name)

  Command line arguments are scanned and passed to the registered
  :class:`~absl.flags.Flag` objects through the ``__call__`` method.  Unparsed
  arguments, including ``argv[0]`` (e.g. the program name) are returned::

       argv = FLAGS(sys.argv)  # scan command line arguments

  The original registered :class:`~absl.flags.Flag` objects can be retrieved
  through the use of the dictionary-like operator, ``__getitem__``::

       x = FLAGS['longname']   # access the registered Flag object

  The ``str()`` operator of a :class:`absl.flags.FlagValues` object provides
  help for all of the registered :class:`~absl.flags.Flag` objects.
  """

  _HAS_DYNAMIC_ATTRIBUTES = True

  # A note on collections.abc.Mapping:
  # FlagValues defines __getitem__, __iter__, and __len__. It makes perfect
  # sense to let it be a collections.abc.Mapping class. However, we are not
  # able to do so. The mixin methods, e.g. keys, values, are not uncommon flag
  # names. Those flag values would not be accessible via the FLAGS.xxx form.

  __dict__: Dict[str, Any]

  def __init__(self):
    # Since everything in this class is so heavily overloaded, the only
    # way of defining and using fields is to access __dict__ directly.

    # Dictionary: flag name (string) -> Flag object.
    self.__dict__['__flags'] = {}

    # Set: name of hidden flag (string).
    # Holds flags that should not be directly accessible from Python.
    self.__dict__['__hiddenflags'] = set()

    # Dictionary: module name (string) -> list of Flag objects that are defined
    # by that module.
    self.__dict__['__flags_by_module'] = {}
    # Dictionary: module id (int) -> list of Flag objects that are defined by
    # that module.
    self.__dict__['__flags_by_module_id'] = {}
    # Dictionary: module name (string) -> list of Flag objects that are
    # key for that module.
    self.__dict__['__key_flags_by_module'] = {}

    # Bool: True if flags were parsed.
    self.__dict__['__flags_parsed'] = False

    # Bool: True if unparse_flags() was called.
    self.__dict__['__unparse_flags_called'] = False

    # None or Method(name, value) to call from __setattr__ for an unknown flag.
    self.__dict__['__set_unknown'] = None

    # A set of banned flag names. This is to prevent users from accidentally
    # defining a flag that has the same name as a method on this class.
    # Users can still allow defining the flag by passing
    # allow_using_method_names=True in DEFINE_xxx functions.
    self.__dict__['__banned_flag_names'] = frozenset(dir(FlagValues))

    # Bool: Whether to use GNU style scanning.
    self.__dict__['__use_gnu_getopt'] = True

    # Bool: Whether use_gnu_getopt has been explicitly set by the user.
    self.__dict__['__use_gnu_getopt_explicitly_set'] = False

    # Function: Takes a flag name as parameter, returns a tuple
    # (is_retired, type_is_bool).
    self.__dict__['__is_retired_flag_func'] = None

  def set_gnu_getopt(self, gnu_getopt: bool = True) -> None:
    """Sets whether or not to use GNU style scanning.

    GNU style allows mixing of flag and non-flag arguments. See
    http://docs.python.org/library/getopt.html#getopt.gnu_getopt

    Args:
      gnu_getopt: bool, whether or not to use GNU style scanning.
    """
    self.__dict__['__use_gnu_getopt'] = gnu_getopt
    self.__dict__['__use_gnu_getopt_explicitly_set'] = True

  def is_gnu_getopt(self) -> bool:
    return self.__dict__['__use_gnu_getopt']

  def _flags(self) -> Dict[str, Flag]:
    return self.__dict__['__flags']

  def flags_by_module_dict(self) -> Dict[str, List[Flag]]:
    """Returns the dictionary of module_name -> list of defined flags.

    Returns:
      A dictionary.  Its keys are module names (strings).  Its values
      are lists of Flag objects.
    """
    return self.__dict__['__flags_by_module']

  def flags_by_module_id_dict(self) -> Dict[int, List[Flag]]:
    """Returns the dictionary of module_id -> list of defined flags.

    Returns:
      A dictionary.  Its keys are module IDs (ints).  Its values
      are lists of Flag objects.
    """
    return self.__dict__['__flags_by_module_id']

  def key_flags_by_module_dict(self) -> Dict[str, List[Flag]]:
    """Returns the dictionary of module_name -> list of key flags.

    Returns:
      A dictionary.  Its keys are module names (strings).  Its values
      are lists of Flag objects.
    """
    return self.__dict__['__key_flags_by_module']

  def register_flag_by_module(self, module_name: str, flag: Flag) -> None:
    """Records the module that defines a specific flag.

    We keep track of which flag is defined by which module so that we
    can later sort the flags by module.

    Args:
      module_name: str, the name of a Python module.
      flag: Flag, the Flag instance that is key to the module.
    """
    flags_by_module = self.flags_by_module_dict()
    flags_by_module.setdefault(module_name, []).append(flag)

  def register_flag_by_module_id(self, module_id: int, flag: Flag) -> None:
    """Records the module that defines a specific flag.

    Args:
      module_id: int, the ID of the Python module.
      flag: Flag, the Flag instance that is key to the module.
    """
    flags_by_module_id = self.flags_by_module_id_dict()
    flags_by_module_id.setdefault(module_id, []).append(flag)

  def register_key_flag_for_module(self, module_name: str, flag: Flag) -> None:
    """Specifies that a flag is a key flag for a module.

    Args:
      module_name: str, the name of a Python module.
      flag: Flag, the Flag instance that is key to the module.
    """
    key_flags_by_module = self.key_flags_by_module_dict()
    # The list of key flags for the module named module_name.
    key_flags = key_flags_by_module.setdefault(module_name, [])
    # Add flag, but avoid duplicates.
    if flag not in key_flags:
      key_flags.append(flag)

  def _flag_is_registered(self, flag_obj: Flag) -> bool:
    """Checks whether a Flag object is registered under long name or short name.

    Args:
      flag_obj: Flag, the Flag instance to check for.

    Returns:
      bool, True iff flag_obj is registered under long name or short name.
    """
    flag_dict = self._flags()
    # Check whether flag_obj is registered under its long name.
    name = flag_obj.name
    if name in flag_dict and flag_dict[name] == flag_obj:
      return True
    # Check whether flag_obj is registered under its short name.
    short_name = flag_obj.short_name
    if (
        short_name is not None
        and short_name in flag_dict
        and flag_dict[short_name] == flag_obj
    ):
      return True
    return False

  def _cleanup_unregistered_flag_from_module_dicts(
      self, flag_obj: Flag
  ) -> None:
    """Cleans up unregistered flags from all module -> [flags] dictionaries.

    If flag_obj is registered under either its long name or short name, it
    won't be removed from the dictionaries.

    Args:
      flag_obj: Flag, the Flag instance to clean up for.
    """
    if self._flag_is_registered(flag_obj):
      return
    # Materialize dict values to list to avoid concurrent modification.
    for flags_in_module in [
        *self.flags_by_module_dict().values(),
        *self.flags_by_module_id_dict().values(),
        *self.key_flags_by_module_dict().values(),
    ]:
      # While (as opposed to if) takes care of multiple occurrences of a
      # flag in the list for the same module.
      while flag_obj in flags_in_module:
        flags_in_module.remove(flag_obj)

  def get_flags_for_module(self, module: Union[str, Any]) -> List[Flag]:
    """Returns the list of flags defined by a module.

    Args:
      module: module|str, the module to get flags from.

    Returns:
      [Flag], a new list of Flag instances.  Caller may update this list as
      desired: none of those changes will affect the internals of this
      FlagValue instance.
    """
    if not isinstance(module, str):
      module = module.__name__
    if module == '__main__':
      module = sys.argv[0]

    return list(self.flags_by_module_dict().get(module, []))

  def get_key_flags_for_module(self, module: Union[str, Any]) -> List[Flag]:
    """Returns the list of key flags for a module.

    Args:
      module: module|str, the module to get key flags from.

    Returns:
      [Flag], a new list of Flag instances.  Caller may update this list as
      desired: none of those changes will affect the internals of this
      FlagValue instance.
    """
    if not isinstance(module, str):
      module = module.__name__
    if module == '__main__':
      module = sys.argv[0]

    # Any flag is a key flag for the module that defined it.  NOTE:
    # key_flags is a fresh list: we can update it without affecting the
    # internals of this FlagValues object.
    key_flags = self.get_flags_for_module(module)

    # Take into account flags explicitly declared as key for a module.
    for flag in self.key_flags_by_module_dict().get(module, []):
      if flag not in key_flags:
        key_flags.append(flag)
    return key_flags

  # TODO(yileiyang): Restrict default to Optional[str].
  def find_module_defining_flag(
      self, flagname: str, default: Optional[_T] = None
  ) -> Union[str, Optional[_T]]:
    """Return the name of the module defining this flag, or default.

    Args:
      flagname: str, name of the flag to lookup.
      default: Value to return if flagname is not defined. Defaults to None.

    Returns:
      The name of the module which registered the flag with this name.
      If no such module exists (i.e. no flag with this name exists),
      we return default.
    """
    registered_flag = self._flags().get(flagname)
    if registered_flag is None:
      return default
    for module, flags in self.flags_by_module_dict().items():
      for flag in flags:
        # It must compare the flag with the one in _flags. This is because a
        # flag might be overridden only for its long name (or short name),
        # and only its short name (or long name) is considered registered.
        if (flag.name == registered_flag.name and
            flag.short_name == registered_flag.short_name):
          return module
    return default

  # TODO(yileiyang): Restrict default to Optional[str].
  def find_module_id_defining_flag(
      self, flagname: str, default: Optional[_T] = None
  ) -> Union[int, Optional[_T]]:
    """Return the ID of the module defining this flag, or default.

    Args:
      flagname: str, name of the flag to lookup.
      default: Value to return if flagname is not defined. Defaults to None.

    Returns:
      The ID of the module which registered the flag with this name.
      If no such module exists (i.e. no flag with this name exists),
      we return default.
    """
    registered_flag = self._flags().get(flagname)
    if registered_flag is None:
      return default
    for module_id, flags in self.flags_by_module_id_dict().items():
      for flag in flags:
        # It must compare the flag with the one in _flags. This is because a
        # flag might be overridden only for its long name (or short name),
        # and only its short name (or long name) is considered registered.
        if (flag.name == registered_flag.name and
            flag.short_name == registered_flag.short_name):
          return module_id
    return default

  def _register_unknown_flag_setter(
      self, setter: Callable[[str, Any], None]
  ) -> None:
    """Allow set default values for undefined flags.

    Args:
      setter: Method(name, value) to call to __setattr__ an unknown flag. Must
        raise NameError or ValueError for invalid name/value.
    """
    self.__dict__['__set_unknown'] = setter

  def _set_unknown_flag(self, name: str, value: _T) -> _T:
    """Returns value if setting flag |name| to |value| returned True.

    Args:
      name: str, name of the flag to set.
      value: Value to set.

    Returns:
      Flag value on successful call.

    Raises:
      UnrecognizedFlagError
      IllegalFlagValueError
    """
    setter = self.__dict__['__set_unknown']
    if setter:
      try:
        setter(name, value)
        return value
      except (TypeError, ValueError):  # Flag value is not valid.
        raise _exceptions.IllegalFlagValueError(
            f'"{value}" is not valid for --{name}'
        )
      except NameError:  # Flag name is not valid.
        pass
    raise _exceptions.UnrecognizedFlagError(name, value)

  def append_flag_values(self, flag_values: 'FlagValues') -> None:
    """Appends flags registered in another FlagValues instance.

    Args:
      flag_values: FlagValues, the FlagValues instance from which to copy flags.
    """
    for flag_name, flag in flag_values._flags().items():  # pylint: disable=protected-access
      # Each flags with short_name appears here twice (once under its
      # normal name, and again with its short name).  To prevent
      # problems (DuplicateFlagError) with double flag registration, we
      # perform a check to make sure that the entry we're looking at is
      # for its normal name.
      if flag_name == flag.name:
        try:
          self[flag_name] = flag
        except _exceptions.DuplicateFlagError:
          raise _exceptions.DuplicateFlagError.from_flag(
              flag_name, self, other_flag_values=flag_values)

  def remove_flag_values(
      self, flag_values: 'Union[FlagValues, Iterable[str]]'
  ) -> None:
    """Remove flags that were previously appended from another FlagValues.

    Args:
      flag_values: FlagValues, the FlagValues instance containing flags to
        remove.
    """
    for flag_name in flag_values:
      self.__delattr__(flag_name)

  def __setitem__(self, name: str, flag: Flag) -> None:
    """Registers a new flag variable."""
    fl = self._flags()
    if not isinstance(flag, _flag.Flag):
      raise _exceptions.IllegalFlagValueError(
          f'Expect Flag instances, found type {type(flag)}. '
          "Maybe you didn't mean to use FlagValue.__setitem__?")
    if not isinstance(name, str):
      raise _exceptions.Error('Flag name must be a string')
    if not name:
      raise _exceptions.Error('Flag name cannot be empty')
    if ' ' in name:
      raise _exceptions.Error('Flag name cannot contain a space')
    self._check_method_name_conflicts(name, flag)
    if name in fl and not flag.allow_override and not fl[name].allow_override:
      module, module_name = _helpers.get_calling_module_object_and_name()
      if (self.find_module_defining_flag(name) == module_name and
          id(module) != self.find_module_id_defining_flag(name)):
        # If the flag has already been defined by a module with the same name,
        # but a different ID, we can stop here because it indicates that the
        # module is simply being imported a subsequent time.
        return
      raise _exceptions.DuplicateFlagError.from_flag(name, self)
    # If a new flag overrides an old one, we need to cleanup the old flag's
    # modules if it's not registered.
    flags_to_cleanup = set()
    short_name: Optional[str] = flag.short_name
    if short_name is not None:
      if (short_name in fl and not flag.allow_override and
          not fl[short_name].allow_override):
        raise _exceptions.DuplicateFlagError.from_flag(short_name, self)
      if short_name in fl and fl[short_name] != flag:
        flags_to_cleanup.add(fl[short_name])
      fl[short_name] = flag
    if (name not in fl  # new flag
        or fl[name].using_default_value or not flag.using_default_value):
      if name in fl and fl[name] != flag:
        flags_to_cleanup.add(fl[name])
      fl[name] = flag
    for f in flags_to_cleanup:
      self._cleanup_unregistered_flag_from_module_dicts(f)

  def __dir__(self) -> List[str]:
    """Returns list of names of all defined flags.

    Useful for TAB-completion in ipython.

    Returns:
      [str], a list of names of all defined flags.
    """
    return sorted(self._flags())

  def __getitem__(self, name: str) -> Flag:
    """Returns the Flag object for the flag --name."""
    return self._flags()[name]

  def _hide_flag(self, name):
    """Marks the flag --name as hidden."""
    self.__dict__['__hiddenflags'].add(name)

  def __getattr__(self, name: str) -> Any:
    """Retrieves the 'value' attribute of the flag --name."""
    flag_entry = self._flags().get(name)
    if flag_entry is None:
      raise AttributeError(name)
    if name in self.__dict__['__hiddenflags']:
      raise AttributeError(name)

    if self.__dict__['__flags_parsed'] or flag_entry.present:
      return flag_entry.value
    else:
      raise _exceptions.UnparsedFlagAccessError(
          'Trying to access flag --%s before flags were parsed.' % name)

  def __setattr__(self, name: str, value: _T) -> _T:
    """Sets the 'value' attribute of the flag --name."""
    self._set_attributes(**{name: value})
    return value

  def _set_attributes(self, **attributes: Any) -> None:
    """Sets multiple flag values together, triggers validators afterwards."""
    fl = self._flags()
    known_flag_vals = {}
    known_flag_used_defaults = {}
    try:
      for name, value in attributes.items():
        if name in self.__dict__['__hiddenflags']:
          raise AttributeError(name)
        flag_entry = fl.get(name)
        if flag_entry is not None:
          orig = flag_entry.value
          flag_entry.value = value
          known_flag_vals[name] = orig
        else:
          self._set_unknown_flag(name, value)
      for name in known_flag_vals:
        self._assert_validators(fl[name].validators)
        known_flag_used_defaults[name] = fl[name].using_default_value
        fl[name].using_default_value = False
    except:
      for name, orig in known_flag_vals.items():
        fl[name].value = orig
      for name, orig in known_flag_used_defaults.items():
        fl[name].using_default_value = orig
      # NOTE: We do not attempt to undo unknown flag side effects because we
      # cannot reliably undo the user-configured behavior.
      raise

  def validate_all_flags(self) -> None:
    """Verifies whether all flags pass validation.

    Raises:
      AttributeError: Raised if validators work with a non-existing flag.
      IllegalFlagValueError: Raised if validation fails for at least one
          validator.
    """
    all_validators = set()
    for flag in self._flags().values():
      all_validators.update(flag.validators)
    self._assert_validators(all_validators)

  def _assert_validators(
      self, validators: Iterable[_validators_classes.Validator]
  ) -> None:
    """Asserts if all validators in the list are satisfied.

    It asserts validators in the order they were created.

    Args:
      validators: Iterable(validators.Validator), validators to be verified.

    Raises:
      AttributeError: Raised if validators work with a non-existing flag.
      IllegalFlagValueError: Raised if validation fails for at least one
          validator.
    """
    messages = []
    bad_flags: Set[str] = set()
    for validator in sorted(
        validators, key=lambda validator: validator.insertion_index):
      try:
        if isinstance(validator, _validators_classes.SingleFlagValidator):
          if validator.flag_name in bad_flags:
            continue
        elif isinstance(validator, _validators_classes.MultiFlagsValidator):
          if bad_flags & set(validator.flag_names):
            continue
        validator.verify(self)
      except _exceptions.ValidationError as e:
        if isinstance(validator, _validators_classes.SingleFlagValidator):
          bad_flags.add(validator.flag_name)
        elif isinstance(validator, _validators_classes.MultiFlagsValidator):
          bad_flags.update(set(validator.flag_names))
        message = validator.print_flags_with_values(self)
        messages.append('%s: %s' % (message, str(e)))
    if messages:
      raise _exceptions.IllegalFlagValueError('\n'.join(messages))

  def __delattr__(self, flag_name: str) -> None:
    """Deletes a previously-defined flag from a flag object.

    This method makes sure we can delete a flag by using

      del FLAGS.<flag_name>

    E.g.,

      flags.DEFINE_integer('foo', 1, 'Integer flag.')
      del flags.FLAGS.foo

    If a flag is also registered by its the other name (long name or short
    name), the other name won't be deleted.

    Args:
      flag_name: str, the name of the flag to be deleted.

    Raises:
      AttributeError: Raised when there is no registered flag named flag_name.
    """
    fl = self._flags()
    flag_entry = fl.get(flag_name)
    if flag_entry is None:
      raise AttributeError(flag_name)
    del fl[flag_name]

    self._cleanup_unregistered_flag_from_module_dicts(flag_entry)

  def set_default(self, name: str, value: Any) -> None:
    """Changes the default value of the named flag object.

    The flag's current value is also updated if the flag is currently using
    the default value, i.e. not specified in the command line, and not set
    by FLAGS.name = value.

    Args:
      name: str, the name of the flag to modify.
      value: The new default value.

    Raises:
      UnrecognizedFlagError: Raised when there is no registered flag named name.
      IllegalFlagValueError: Raised when value is not valid.
    """
    fl = self._flags()
    flag_entry = fl.get(name)
    if flag_entry is None:
      self._set_unknown_flag(name, value)
      return
    flag_entry._set_default(value)  # pylint: disable=protected-access
    self._assert_validators(flag_entry.validators)

  def __contains__(self, name: str) -> bool:
    """Returns True if name is a value (flag) in the dict."""
    return name in self._flags()

  def __len__(self) -> int:
    return len(self.__dict__['__flags'])

  def __iter__(self) -> Iterator[str]:
    return iter(self._flags())

  def __call__(
      self, argv: Sequence[str], known_only: bool = False
  ) -> List[str]:
    """Parses flags from argv; stores parsed flags into this FlagValues object.

    All unparsed arguments are returned.

    Args:
       argv: a tuple/list of strings.
       known_only: bool, if True, parse and remove known flags; return the rest
         untouched. Unknown flags specified by --undefok are not returned.

    Returns:
       The list of arguments not parsed as options, including argv[0].

    Raises:
       Error: Raised on any parsing error.
       TypeError: Raised on passing wrong type of arguments.
       ValueError: Raised on flag value parsing error.
    """
    if isinstance(argv, (str, bytes)):
      raise TypeError(
          'argv should be a tuple/list of strings, not bytes or string.')
    if not argv:
      raise ValueError(
          'argv cannot be an empty list, and must contain the program name as '
          'the first element.')

    # This pre parses the argv list for --flagfile=<> options.
    program_name = argv[0]
    args = self.read_flags_from_files(argv[1:], force_gnu=False)

    # Parse the arguments.
    unknown_flags, unparsed_args = self._parse_args(args, known_only)

    # Handle unknown flags by raising UnrecognizedFlagError.
    # Note some users depend on us raising this particular error.
    for name, value in unknown_flags:
      suggestions = _helpers.get_flag_suggestions(name, list(self))
      raise _exceptions.UnrecognizedFlagError(
          name, value, suggestions=suggestions)

    self.mark_as_parsed()
    self.validate_all_flags()
    return [program_name] + unparsed_args

  def __getstate__(self) -> Any:
    raise TypeError("can't pickle FlagValues")

  def __copy__(self) -> Any:
    raise TypeError('FlagValues does not support shallow copies. '
                    'Use absl.testing.flagsaver or copy.deepcopy instead.')

  def __deepcopy__(self, memo) -> Any:
    result = object.__new__(type(self))
    result.__dict__.update(copy.deepcopy(self.__dict__, memo))
    return result

  def _set_is_retired_flag_func(self, is_retired_flag_func):
    """Sets a function for checking retired flags.

    Do not use it. This is a private absl API used to check retired flags
    registered by the absl C++ flags library.

    Args:
      is_retired_flag_func: Callable(str) -> (bool, bool), a function takes flag
        name as parameter, returns a tuple (is_retired, type_is_bool).
    """
    self.__dict__['__is_retired_flag_func'] = is_retired_flag_func

  def _parse_args(
      self, args: List[str], known_only: bool
  ) -> Tuple[List[Tuple[str, Any]], List[str]]:
    """Helper function to do the main argument parsing.

    This function goes through args and does the bulk of the flag parsing.
    It will find the corresponding flag in our flag dictionary, and call its
    .parse() method on the flag value.

    Args:
      args: [str], a list of strings with the arguments to parse.
      known_only: bool, if True, parse and remove known flags; return the rest
        untouched. Unknown flags specified by --undefok are not returned.

    Returns:
      A tuple with the following:
          unknown_flags: List of (flag name, arg) for flags we don't know about.
          unparsed_args: List of arguments we did not parse.

    Raises:
       Error: Raised on any parsing error.
       ValueError: Raised on flag value parsing error.
    """
    unparsed_names_and_args: List[Tuple[Optional[str], str]] = []
    undefok: Set[str] = set()
    retired_flag_func = self.__dict__['__is_retired_flag_func']

    flag_dict = self._flags()
    args_it = iter(args)
    del args
    for arg in args_it:
      value = None

      def get_value() -> str:
        try:
          return next(args_it) if value is None else value  # pylint: disable=cell-var-from-loop
        except StopIteration:
          raise _exceptions.Error('Missing value for flag ' + arg) from None  # pylint: disable=cell-var-from-loop

      if not arg.startswith('-'):
        # A non-argument: default is break, GNU is skip.
        unparsed_names_and_args.append((None, arg))
        if self.is_gnu_getopt():
          continue
        else:
          break

      if arg == '--':
        if known_only:
          unparsed_names_and_args.append((None, arg))
        break

      # At this point, arg must start with '-'.
      if arg.startswith('--'):
        arg_without_dashes = arg[2:]
      else:
        arg_without_dashes = arg[1:]

      if '=' in arg_without_dashes:
        name, value = arg_without_dashes.split('=', 1)
      else:
        name, value = arg_without_dashes, None

      if not name:
        # The argument is all dashes (including one dash).
        unparsed_names_and_args.append((None, arg))
        if self.is_gnu_getopt():
          continue
        else:
          break

      # --undefok is a special case.
      if name == 'undefok':
        value = get_value()
        undefok.update(v.strip() for v in value.split(','))
        undefok.update('no' + v.strip() for v in value.split(','))
        continue

      flag = flag_dict.get(name)
      if flag is not None:
        if flag.boolean and value is None:
          value = 'true'
        else:
          value = get_value()
      elif name.startswith('no') and len(name) > 2:
        # Boolean flags can take the form of --noflag, with no value.
        noflag = flag_dict.get(name[2:])
        if noflag is not None and noflag.boolean:
          if value is not None:
            raise ValueError(arg + ' does not take an argument')
          flag = noflag
          value = 'false'

      if retired_flag_func and flag is None:
        is_retired, is_bool = retired_flag_func(name)

        # If we didn't recognize that flag, but it starts with
        # "no" then maybe it was a boolean flag specified in the
        # --nofoo form.
        if not is_retired and name.startswith('no'):
          is_retired, is_bool = retired_flag_func(name[2:])
          is_retired = is_retired and is_bool

        if is_retired:
          if not is_bool and value is None:
            # This happens when a non-bool retired flag is specified
            # in format of "--flag value".
            get_value()
          logging.error(
              'Flag "%s" is retired and should no longer be specified. See '
              'https://abseil.io/tips/90.',
              name,
          )
          continue

      if flag is not None:
        # LINT.IfChange
        flag.parse(value)
        flag.using_default_value = False
        # LINT.ThenChange(../testing/flagsaver.py:flag_override_parsing)
      else:
        unparsed_names_and_args.append((name, arg))

    unknown_flags = []
    unparsed_args = []
    for arg_name, arg in unparsed_names_and_args:
      if arg_name is None:
        # Positional arguments.
        unparsed_args.append(arg)
      elif arg_name in undefok:
        # Remove undefok flags.
        continue
      else:
        # This is an unknown flag.
        if known_only:
          unparsed_args.append(arg)
        else:
          unknown_flags.append((arg_name, arg))

    unparsed_args.extend(list(args_it))
    return unknown_flags, unparsed_args

  def is_parsed(self) -> bool:
    """Returns whether flags were parsed."""
    return self.__dict__['__flags_parsed']

  def mark_as_parsed(self) -> None:
    """Explicitly marks flags as parsed.

    Use this when the caller knows that this FlagValues has been parsed as if
    a ``__call__()`` invocation has happened.  This is only a public method for
    use by things like appcommands which do additional command like parsing.
    """
    self.__dict__['__flags_parsed'] = True

  def unparse_flags(self) -> None:
    """Unparses all flags to the point before any FLAGS(argv) was called."""
    for f in self._flags().values():
      f.unparse()
    # We log this message before marking flags as unparsed to avoid a
    # problem when the logging library causes flags access.
    logging.info('unparse_flags() called; flags access will now raise errors.')
    self.__dict__['__flags_parsed'] = False
    self.__dict__['__unparse_flags_called'] = True

  def flag_values_dict(self) -> Dict[str, Any]:
    """Returns a dictionary that maps flag names to flag values."""
    return {name: flag.value for name, flag in list(self._flags().items())}

  def __str__(self):
    """Returns a help string for all known flags."""
    return self.get_help()

  def get_help(
      self, prefix: str = '', include_special_flags: bool = True
  ) -> str:
    """Returns a help string for all known flags.

    Args:
      prefix: str, per-line output prefix.
      include_special_flags: bool, whether to include description of
        SPECIAL_FLAGS, i.e. --flagfile and --undefok.

    Returns:
      str, formatted help message.
    """
    flags_by_module = self.flags_by_module_dict()
    if flags_by_module:
      modules = sorted(flags_by_module)
      # Print the help for the main module first, if possible.
      main_module = sys.argv[0]
      if main_module in modules:
        modules.remove(main_module)
        modules = [main_module] + modules
      return self._get_help_for_modules(modules, prefix, include_special_flags)
    else:
      output_lines: List[str] = []
      # Just print one long list of flags.
      values = list(self._flags().values())
      if include_special_flags:
        values.extend(_helpers.SPECIAL_FLAGS._flags().values())  # pylint: disable=protected-access
      self._render_flag_list(values, output_lines, prefix)
      return '\n'.join(output_lines)

  def _get_help_for_modules(self, modules, prefix, include_special_flags):
    """Returns the help string for a list of modules.

    Private to absl.flags package.

    Args:
      modules: List[str], a list of modules to get the help string for.
      prefix: str, a string that is prepended to each generated help line.
      include_special_flags: bool, whether to include description of
        SPECIAL_FLAGS, i.e. --flagfile and --undefok.
    """
    output_lines = []
    for module in modules:
      self._render_our_module_flags(module, output_lines, prefix)
    if include_special_flags:
      self._render_module_flags(
          'absl.flags',
          _helpers.SPECIAL_FLAGS._flags().values(),  # pylint: disable=protected-access  # pytype: disable=attribute-error
          output_lines,
          prefix,
      )
    return '\n'.join(output_lines)

  def _render_module_flags(self, module, flags, output_lines, prefix=''):
    """Returns a help string for a given module."""
    if not isinstance(module, str):
      module = module.__name__
    output_lines.append('\n%s%s:' % (prefix, module))
    self._render_flag_list(flags, output_lines, prefix + '  ')

  def _render_our_module_flags(self, module, output_lines, prefix=''):
    """Returns a help string for a given module."""
    flags = self.get_flags_for_module(module)
    if flags:
      self._render_module_flags(module, flags, output_lines, prefix)

  def _render_our_module_key_flags(self, module, output_lines, prefix=''):
    """Returns a help string for the key flags of a given module.

    Args:
      module: module|str, the module to render key flags for.
      output_lines: [str], a list of strings.  The generated help message lines
        will be appended to this list.
      prefix: str, a string that is prepended to each generated help line.
    """
    key_flags = self.get_key_flags_for_module(module)
    if key_flags:
      self._render_module_flags(module, key_flags, output_lines, prefix)

  def module_help(self, module: Any) -> str:
    """Describes the key flags of a module.

    Args:
      module: module|str, the module to describe the key flags for.

    Returns:
      str, describing the key flags of a module.
    """
    helplist: List[str] = []
    self._render_our_module_key_flags(module, helplist)
    return '\n'.join(helplist)

  def main_module_help(self) -> str:
    """Describes the key flags of the main module.

    Returns:
      str, describing the key flags of the main module.
    """
    return self.module_help(sys.argv[0])

  def _render_flag_list(self, flaglist, output_lines, prefix='  '):
    fl = self._flags()
    special_fl = _helpers.SPECIAL_FLAGS._flags()  # pylint: disable=protected-access  # pytype: disable=attribute-error
    flaglist = [(flag.name, flag) for flag in flaglist]
    flaglist.sort()
    flagset = {}
    for (name, flag) in flaglist:
      # It's possible this flag got deleted or overridden since being
      # registered in the per-module flaglist.  Check now against the
      # canonical source of current flag information, the _flags.
      if fl.get(name, None) != flag and special_fl.get(name, None) != flag:
        # a different flag is using this name now
        continue
      # only print help once
      if flag in flagset:
        continue
      flagset[flag] = 1
      flaghelp = ''
      if flag.short_name:
        flaghelp += '-%s,' % flag.short_name
      if flag.boolean:
        flaghelp += '--[no]%s:' % flag.name
      else:
        flaghelp += '--%s:' % flag.name
      flaghelp += ' '
      if flag.help:
        flaghelp += flag.help
      flaghelp = _helpers.text_wrap(
          flaghelp, indent=prefix + '  ', firstline_indent=prefix)
      if flag.default_as_str:
        flaghelp += '\n'
        flaghelp += _helpers.text_wrap(
            '(default: %s)' % flag.default_as_str, indent=prefix + '  ')
      if flag.parser.syntactic_help:
        flaghelp += '\n'
        flaghelp += _helpers.text_wrap(
            '(%s)' % flag.parser.syntactic_help, indent=prefix + '  ')
      output_lines.append(flaghelp)

  def get_flag_value(self, name: str, default: Any) -> Any:  # pylint: disable=invalid-name
    """Returns the value of a flag (if not None) or a default value.

    Args:
      name: str, the name of a flag.
      default: Default value to use if the flag value is None.

    Returns:
      Requested flag value or default.
    """

    value = self.__getattr__(name)
    if value is not None:  # Can't do if not value, b/c value might be '0' or ""
      return value
    else:
      return default

  def _is_flag_file_directive(self, flag_string):
    """Checks whether flag_string contain a --flagfile=<foo> directive."""
    if isinstance(flag_string, str):
      if flag_string.startswith('--flagfile='):
        return 1
      elif flag_string == '--flagfile':
        return 1
      elif flag_string.startswith('-flagfile='):
        return 1
      elif flag_string == '-flagfile':
        return 1
      else:
        return 0
    return 0

  def _extract_filename(self, flagfile_str):
    """Returns filename from a flagfile_str of form -[-]flagfile=filename.

    The cases of --flagfile foo and -flagfile foo shouldn't be hitting
    this function, as they are dealt with in the level above this
    function.

    Args:
      flagfile_str: str, the flagfile string.

    Returns:
      str, the filename from a flagfile_str of form -[-]flagfile=filename.

    Raises:
      Error: Raised when illegal --flagfile is provided.
    """
    if flagfile_str.startswith('--flagfile='):
      return os.path.expanduser((flagfile_str[(len('--flagfile=')):]).strip())
    elif flagfile_str.startswith('-flagfile='):
      return os.path.expanduser((flagfile_str[(len('-flagfile=')):]).strip())
    else:
      raise _exceptions.Error('Hit illegal --flagfile type: %s' % flagfile_str)

  def _get_flag_file_lines(self, filename, parsed_file_stack=None):
    """Returns the useful (!=comments, etc) lines from a file with flags.

    Args:
      filename: str, the name of the flag file.
      parsed_file_stack: [str], a list of the names of the files that we have
        recursively encountered at the current depth. MUTATED BY THIS FUNCTION
        (but the original value is preserved upon successfully returning from
        function call).

    Returns:
      List of strings. See the note below.

    NOTE(springer): This function checks for a nested --flagfile=<foo>
    tag and handles the lower file recursively. It returns a list of
    all the lines that _could_ contain command flags. This is
    EVERYTHING except whitespace lines and comments (lines starting
    with '#' or '//').
    """
    # For consistency with the cpp version, ignore empty values.
    if not filename:
      return []
    if parsed_file_stack is None:
      parsed_file_stack = []
    # We do a little safety check for reparsing a file we've already encountered
    # at a previous depth.
    if filename in parsed_file_stack:
      sys.stderr.write('Warning: Hit circular flagfile dependency. Ignoring'
                       ' flagfile: %s\n' % (filename,))
      return []
    else:
      parsed_file_stack.append(filename)

    line_list = []  # All line from flagfile.
    flag_line_list = []  # Subset of lines w/o comments, blanks, flagfile= tags.
    try:
      file_obj = open(filename)
    except OSError as e_msg:
      raise _exceptions.CantOpenFlagFileError(
          'ERROR:: Unable to open flagfile: %s' % e_msg)

    with file_obj:
      line_list = file_obj.readlines()

    # This is where we check each line in the file we just read.
    for line in line_list:
      if line.isspace():
        pass
      # Checks for comment (a line that starts with '#').
      elif line.startswith('#') or line.startswith('//'):
        pass
      # Checks for a nested "--flagfile=<bar>" flag in the current file.
      # If we find one, recursively parse down into that file.
      elif self._is_flag_file_directive(line):
        sub_filename = self._extract_filename(line)
        included_flags = self._get_flag_file_lines(
            sub_filename, parsed_file_stack=parsed_file_stack)
        flag_line_list.extend(included_flags)
      else:
        # Any line that's not a comment or a nested flagfile should get
        # copied into 2nd position.  This leaves earlier arguments
        # further back in the list, thus giving them higher priority.
        flag_line_list.append(line.strip())

    parsed_file_stack.pop()
    return flag_line_list

  def read_flags_from_files(
      self, argv: Sequence[str], force_gnu: bool = True
  ) -> List[str]:
    """Processes command line args, but also allow args to be read from file.

    Args:
      argv: [str], a list of strings, usually sys.argv[1:], which may contain
        one or more flagfile directives of the form --flagfile="./filename".
        Note that the name of the program (sys.argv[0]) should be omitted.
      force_gnu: bool, if False, --flagfile parsing obeys the
        FLAGS.is_gnu_getopt() value. If True, ignore the value and always follow
        gnu_getopt semantics.

    Returns:
      A new list which has the original list combined with what we read
      from any flagfile(s).

    Raises:
      IllegalFlagValueError: Raised when --flagfile is provided with no
          argument.

    This function is called by FLAGS(argv).
    It scans the input list for a flag that looks like:
    --flagfile=<somefile>. Then it opens <somefile>, reads all valid key
    and value pairs and inserts them into the input list in exactly the
    place where the --flagfile arg is found.

    Note that your application's flags are still defined the usual way
    using absl.flags DEFINE_flag() type functions.

    Notes (assuming we're getting a commandline of some sort as our input):

    * For duplicate flags, the last one we hit should "win".
    * Since flags that appear later win, a flagfile's settings can be "weak"
        if the --flagfile comes at the beginning of the argument sequence,
        and it can be "strong" if the --flagfile comes at the end.
    * A further "--flagfile=<otherfile.cfg>" CAN be nested in a flagfile.
        It will be expanded in exactly the spot where it is found.
    * In a flagfile, a line beginning with # or // is a comment.
    * Entirely blank lines _should_ be ignored.
    """
    rest_of_args = argv
    new_argv = []
    while rest_of_args:
      current_arg = rest_of_args[0]
      rest_of_args = rest_of_args[1:]
      if self._is_flag_file_directive(current_arg):
        # This handles the case of -(-)flagfile foo.  In this case the
        # next arg really is part of this one.
        if current_arg == '--flagfile' or current_arg == '-flagfile':
          if not rest_of_args:
            raise _exceptions.IllegalFlagValueError(
                '--flagfile with no argument')
          flag_filename = os.path.expanduser(rest_of_args[0])
          rest_of_args = rest_of_args[1:]
        else:
          # This handles the case of (-)-flagfile=foo.
          flag_filename = self._extract_filename(current_arg)
        new_argv.extend(self._get_flag_file_lines(flag_filename))
      else:
        new_argv.append(current_arg)
        # Stop parsing after '--', like getopt and gnu_getopt.
        if current_arg == '--':
          break
        # Stop parsing after a non-flag, like getopt.
        if not current_arg.startswith('-'):
          if not force_gnu and not self.__dict__['__use_gnu_getopt']:
            break
        else:
          if ('=' not in current_arg and rest_of_args and
              not rest_of_args[0].startswith('-')):
            # If this is an occurrence of a legitimate --x y, skip the value
            # so that it won't be mistaken for a standalone arg.
            fl = self._flags()
            name = current_arg.lstrip('-')
            if name in fl and not fl[name].boolean:
              current_arg = rest_of_args[0]
              rest_of_args = rest_of_args[1:]
              new_argv.append(current_arg)

    if rest_of_args:
      new_argv.extend(rest_of_args)

    return new_argv

  def flags_into_string(self) -> str:
    """Returns a string with the flags assignments from this FlagValues object.

    This function ignores flags whose value is None.  Each flag
    assignment is separated by a newline.

    NOTE: MUST mirror the behavior of the C++ CommandlineFlagsIntoString
    from https://github.com/gflags/gflags.

    Returns:
      str, the string with the flags assignments from this FlagValues object.
      The flags are ordered by (module_name, flag_name).
    """
    module_flags = sorted(self.flags_by_module_dict().items())
    s = ''
    for unused_module_name, flags in module_flags:
      flags = sorted(flags, key=lambda f: f.name)
      for flag in flags:
        if flag.value is not None:
          s += flag.serialize() + '\n'
    return s

  def append_flags_into_file(self, filename: str) -> None:
    """Appends all flags assignments from this FlagInfo object to a file.

    Output will be in the format of a flagfile.

    NOTE: MUST mirror the behavior of the C++ AppendFlagsIntoFile
    from https://github.com/gflags/gflags.

    Args:
      filename: str, name of the file.
    """
    with open(filename, 'a') as out_file:
      out_file.write(self.flags_into_string())

  def write_help_in_xml_format(self, outfile: Optional[TextIO] = None) -> None:
    """Outputs flag documentation in XML format.

    NOTE: We use element names that are consistent with those used by
    the C++ command-line flag library, from
    https://github.com/gflags/gflags.
    We also use a few new elements (e.g., <key>), but we do not
    interfere / overlap with existing XML elements used by the C++
    library.  Please maintain this consistency.

    Args:
      outfile: File object we write to.  Default None means sys.stdout.
    """
    doc = minidom.Document()
    all_flag = doc.createElement('AllFlags')
    doc.appendChild(all_flag)

    all_flag.appendChild(
        _helpers.create_xml_dom_element(doc, 'program',
                                        os.path.basename(sys.argv[0])))

    usage_doc = sys.modules['__main__'].__doc__
    if not usage_doc:
      usage_doc = '\nUSAGE: %s [flags]\n' % sys.argv[0]
    else:
      usage_doc = usage_doc.replace('%s', sys.argv[0])
    all_flag.appendChild(
        _helpers.create_xml_dom_element(doc, 'usage', usage_doc))

    # Get list of key flags for the main module.
    key_flags = self.get_key_flags_for_module(sys.argv[0])

    flags_by_module = self.flags_by_module_dict()
    # Sort flags by declaring module name and next by flag name.
    for module_name in sorted(flags_by_module.keys()):
      flag_list = [(f.name, f) for f in flags_by_module[module_name]]
      flag_list.sort()
      for unused_flag_name, flag in flag_list:
        is_key = flag in key_flags
        all_flag.appendChild(
            flag._create_xml_dom_element(  # pylint: disable=protected-access
                doc,
                module_name,
                is_key=is_key))

    outfile = outfile or sys.stdout
    outfile.write(
        doc.toprettyxml(indent='  ', encoding='utf-8').decode('utf-8'))
    outfile.flush()

  def _check_method_name_conflicts(self, name: str, flag: Flag):
    if flag.allow_using_method_names:
      return
    short_name = flag.short_name
    flag_names = {name} if short_name is None else {name, short_name}
    for flag_name in flag_names:
      if flag_name in self.__dict__['__banned_flag_names']:
        raise _exceptions.FlagNameConflictsWithMethodError(
            'Cannot define a flag named "{name}". It conflicts with a method '
            'on class "{class_name}". To allow defining it, use '
            'allow_using_method_names and access the flag value with '
            "FLAGS['{name}'].value. FLAGS.{name} returns the method, "
            'not the flag value.'.format(
                name=flag_name, class_name=type(self).__name__))


FLAGS = FlagValues()


class FlagHolder(Generic[_T_co]):
  """Holds a defined flag.

  This facilitates a cleaner api around global state. Instead of::

      flags.DEFINE_integer('foo', ...)
      flags.DEFINE_integer('bar', ...)

      def method():
        # prints parsed value of 'bar' flag
        print(flags.FLAGS.foo)
        # runtime error due to typo or possibly bad coding style.
        print(flags.FLAGS.baz)

  it encourages code like::

      _FOO_FLAG = flags.DEFINE_integer('foo', ...)
      _BAR_FLAG = flags.DEFINE_integer('bar', ...)

      def method():
        print(_FOO_FLAG.value)
        print(_BAR_FLAG.value)

  since the name of the flag appears only once in the source code.
  """

  value: _T_co

  def __init__(
      self,
      flag_values: FlagValues,
      flag: Flag[_T_co],
      ensure_non_none_value: bool = False,
  ):
    """Constructs a FlagHolder instance providing typesafe access to flag.

    Args:
      flag_values: The container the flag is registered to.
      flag: The flag object for this flag.
      ensure_non_none_value: Is the value of the flag allowed to be None.
    """
    self._flagvalues = flag_values
    # We take the entire flag object, but only keep the name. Why?
    # - We want FlagHolder[T] to be generic container
    # - flag_values contains all flags, so has no reference to T.
    # - typecheckers don't like to see a generic class where none of the ctor
    #   arguments refer to the generic type.
    self._name = flag.name
    # We intentionally do NOT check if the default value is None.
    # This allows future use of this for "required flags with None default"
    self._ensure_non_none_value = ensure_non_none_value

  def __eq__(self, other):
    raise TypeError(
        "unsupported operand type(s) for ==: '{0}' and '{1}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__, type(other).__name__))

  def __bool__(self):
    raise TypeError(
        "bool() not supported for instances of type '{0}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__))

  __nonzero__ = __bool__

  @property
  def name(self) -> str:
    return self._name

  @property  # type: ignore[no-redef]
  def value(self) -> _T_co:
    """Returns the value of the flag.

    If ``_ensure_non_none_value`` is ``True``, then return value is not
    ``None``.

    Raises:
      UnparsedFlagAccessError: if flag parsing has not finished.
      IllegalFlagValueError: if value is None unexpectedly.
    """
    val = getattr(self._flagvalues, self._name)
    if self._ensure_non_none_value and val is None:
      raise _exceptions.IllegalFlagValueError(
          'Unexpected None value for flag %s' % self._name)
    return val

  @property
  def default(self) -> _T_co:
    """Returns the default value of the flag."""
    return self._flagvalues[self._name].default  # type: ignore[return-value]

  @property
  def present(self) -> bool:
    """Returns True if the flag was parsed from command-line flags."""
    return bool(self._flagvalues[self._name].present)

  def serialize(self) -> str:
    """Returns a serialized representation of the flag."""
    return self._flagvalues[self._name].serialize()


def resolve_flag_ref(
    flag_ref: Union[str, FlagHolder], flag_values: FlagValues
) -> Tuple[str, FlagValues]:
  """Helper to validate and resolve a flag reference argument."""
  if isinstance(flag_ref, FlagHolder):
    new_flag_values = flag_ref._flagvalues  # pylint: disable=protected-access
    if flag_values != FLAGS and flag_values != new_flag_values:
      raise ValueError(
          'flag_values must not be customized when operating on a FlagHolder')
    return flag_ref.name, new_flag_values
  return flag_ref, flag_values


def resolve_flag_refs(
    flag_refs: Sequence[Union[str, FlagHolder]], flag_values: FlagValues
) -> Tuple[List[str], FlagValues]:
  """Helper to validate and resolve flag reference list arguments."""
  fv = None
  names = []
  for ref in flag_refs:
    if isinstance(ref, FlagHolder):
      newfv = ref._flagvalues  # pylint: disable=protected-access
      name = ref.name
    else:
      newfv = flag_values
      name = ref
    if fv and fv != newfv:
      raise ValueError(
          'multiple FlagValues instances used in invocation. '
          'FlagHolders must be registered to the same FlagValues instance as '
          'do flag names, if provided.')
    fv = newfv
    names.append(name)
  if fv is None:
    raise ValueError('flag_refs argument must not be empty')
  return names, fv

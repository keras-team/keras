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

"""Decorator and context manager for saving and restoring flag values.

There are many ways to save and restore.  Always use the most convenient method
for a given use case.

Here are examples of each method.  They all call ``do_stuff()`` while
``FLAGS.someflag`` is temporarily set to ``'foo'``::

    from absl.testing import flagsaver

    # Use a decorator which can optionally override flags via arguments.
    @flagsaver.flagsaver(someflag='foo')
    def some_func():
      do_stuff()

    # Use a decorator which can optionally override flags with flagholders.
    @flagsaver.flagsaver((module.FOO_FLAG, 'foo'), (other_mod.BAR_FLAG, 23))
    def some_func():
      do_stuff()

    # Use a decorator which does not override flags itself.
    @flagsaver.flagsaver
    def some_func():
      FLAGS.someflag = 'foo'
      do_stuff()

    # Use a context manager which can optionally override flags via arguments.
    with flagsaver.flagsaver(someflag='foo'):
      do_stuff()

    # Save and restore the flag values yourself.
    saved_flag_values = flagsaver.save_flag_values()
    try:
      FLAGS.someflag = 'foo'
      do_stuff()
    finally:
      flagsaver.restore_flag_values(saved_flag_values)

    # Use the parsing version to emulate users providing the flags.
    # Note that all flags must be provided as strings (unparsed).
    @flagsaver.as_parsed(some_int_flag='123')
    def some_func():
      # Because the flag was parsed it is considered "present".
      assert FLAGS.some_int_flag.present
      do_stuff()

    # flagsaver.as_parsed() can also be used as a context manager just like
    # flagsaver.flagsaver()
    with flagsaver.as_parsed(some_int_flag='123'):
      do_stuff()

    # The flagsaver.as_parsed() interface also supports FlagHolder objects.
    @flagsaver.as_parsed((module.FOO_FLAG, 'foo'), (other_mod.BAR_FLAG, '23'))
    def some_func():
      do_stuff()

    # Using as_parsed with a multi_X flag requires a sequence of strings.
    @flagsaver.as_parsed(some_multi_int_flag=['123', '456'])
    def some_func():
      assert FLAGS.some_multi_int_flag.present
      do_stuff()

    # If a flag name includes non-identifier characters it can be specified like
    # so:
    @flagsaver.as_parsed(**{'i-like-dashes': 'true'})
    def some_func():
      do_stuff()

We save and restore a shallow copy of each Flag object's ``__dict__`` attribute.
This preserves all attributes of the flag, such as whether or not it was
overridden from its default value.

WARNING: Currently a flag that is saved and then deleted cannot be restored.  An
exception will be raised.  However if you *add* a flag after saving flag values,
and then restore flag values, the added flag will be deleted with no errors.
"""

import collections
import functools
import inspect
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Type, TypeVar, Union, overload

from absl import flags

FLAGS = flags.FLAGS


# The type of pre/post wrapped functions.
_CallableT = TypeVar('_CallableT', bound=Callable)


@overload
def flagsaver(func: _CallableT) -> _CallableT:
  ...


@overload
def flagsaver(
    *args: Tuple[flags.FlagHolder, Any], **kwargs: Any
) -> '_FlagOverrider':
  ...


def flagsaver(*args, **kwargs):
  """The main flagsaver interface. See module doc for usage."""
  return _construct_overrider(_FlagOverrider, *args, **kwargs)  # type: ignore[bad-return-type]


@overload
def as_parsed(*args: Tuple[flags.FlagHolder, Union[str, Sequence[str]]],
              **kwargs: Union[str, Sequence[str]]) -> '_ParsingFlagOverrider':
  ...


@overload
def as_parsed(func: _CallableT) -> _CallableT:
  ...


def as_parsed(*args, **kwargs):
  """Overrides flags by parsing strings, saves flag state similar to flagsaver.

  This function can be used as either a decorator or context manager similar to
  flagsaver.flagsaver(). However, where flagsaver.flagsaver() directly sets the
  flags to new values, this function will parse the provided arguments as if
  they were provided on the command line. Among other things, this will cause
  `FLAGS['flag_name'].present == True`.

  A note on unparsed input: For many flag types, the unparsed version will be
  a single string. However for multi_x (multi_string, multi_integer, multi_enum)
  the unparsed version will be a Sequence of strings.

  Args:
    *args: Tuples of FlagHolders and their unparsed value.
    **kwargs: The keyword args are flag names, and the values are unparsed
      values.

  Returns:
    _ParsingFlagOverrider that serves as a context manager or decorator. Will
    save previous flag state and parse new flags, then on cleanup it will
    restore the previous flag state.
  """
  return _construct_overrider(_ParsingFlagOverrider, *args, **kwargs)


# NOTE: the order of these overload declarations matters. The type checker will
# pick the first match which could be incorrect.
@overload
def _construct_overrider(
    flag_overrider_cls: Type['_ParsingFlagOverrider'],
    *args: Tuple[flags.FlagHolder, Union[str, Sequence[str]]],
    **kwargs: Union[str, Sequence[str]]) -> '_ParsingFlagOverrider':
  ...


@overload
def _construct_overrider(
    flag_overrider_cls: Type['_FlagOverrider'], func: _CallableT
) -> _CallableT:
  ...


@overload
def _construct_overrider(
    flag_overrider_cls: Type['_FlagOverrider'],
    *args: Tuple[flags.FlagHolder, Any],
    **kwargs: Any,
) -> '_FlagOverrider':
  ...


def _construct_overrider(flag_overrider_cls, *args, **kwargs):
  """Handles the args/kwargs returning an instance of flag_overrider_cls.

  If flag_overrider_cls is _FlagOverrider then values should be native python
  types matching the python types. Otherwise if flag_overrider_cls is
  _ParsingFlagOverrider the values should be strings or sequences of strings.

  Args:
    flag_overrider_cls: The class that will do the overriding.
    *args: Tuples of FlagHolder and the new flag value.
    **kwargs: Keword args mapping flag name to new flag value.

  Returns:
    A _FlagOverrider to be used as a decorator or context manager.
  """
  if not args:
    return flag_overrider_cls(**kwargs)
  # args can be [func] if used as `@flagsaver` instead of `@flagsaver(...)`
  if len(args) == 1 and callable(args[0]):
    if kwargs:
      raise ValueError(
          "It's invalid to specify both positional and keyword parameters.")
    func = args[0]
    if inspect.isclass(func):
      raise TypeError('@flagsaver.flagsaver cannot be applied to a class.')
    return _wrap(flag_overrider_cls, func, {})
  # args can be a list of (FlagHolder, value) pairs.
  # In which case they augment any specified kwargs.
  for arg in args:
    if not isinstance(arg, tuple) or len(arg) != 2:
      raise ValueError('Expected (FlagHolder, value) pair, found %r' % (arg,))
    holder, value = arg
    if not isinstance(holder, flags.FlagHolder):
      raise ValueError('Expected (FlagHolder, value) pair, found %r' % (arg,))
    if holder.name in kwargs:
      raise ValueError('Cannot set --%s multiple times' % holder.name)
    kwargs[holder.name] = value
  return flag_overrider_cls(**kwargs)


def save_flag_values(
    flag_values: flags.FlagValues = FLAGS,
) -> Dict[str, Dict[str, Any]]:
  """Returns copy of flag values as a dict.

  Args:
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      saved. This should almost never need to be overridden.

  Returns:
    Dictionary mapping keys to values. Keys are flag names, values are
    corresponding ``__dict__`` members. E.g. ``{'key': value_dict, ...}``.
  """
  return {name: _copy_flag_dict(flag_values[name]) for name in flag_values}


def restore_flag_values(
    saved_flag_values: Mapping[str, Dict[str, Any]],
    flag_values: flags.FlagValues = FLAGS,
) -> None:
  """Restores flag values based on the dictionary of flag values.

  Args:
    saved_flag_values: {'flag_name': value_dict, ...}
    flag_values: FlagValues, the FlagValues instance from which the flag will be
      restored. This should almost never need to be overridden.
  """
  new_flag_names = list(flag_values)
  for name in new_flag_names:
    saved = saved_flag_values.get(name)
    if saved is None:
      # If __dict__ was not saved delete "new" flag.
      delattr(flag_values, name)
    else:
      if flag_values[name].value != saved['_value']:
        flag_values[name].value = saved['_value']  # Ensure C++ value is set.
      flag_values[name].__dict__ = saved


@overload
def _wrap(flag_overrider_cls: Type['_FlagOverrider'], func: _CallableT,
          overrides: Mapping[str, Any]) -> _CallableT:
  ...


@overload
def _wrap(flag_overrider_cls: Type['_ParsingFlagOverrider'], func: _CallableT,
          overrides: Mapping[str, Union[str, Sequence[str]]]) -> _CallableT:
  ...


def _wrap(flag_overrider_cls, func, overrides):
  """Creates a wrapper function that saves/restores flag values.

  Args:
    flag_overrider_cls: The class that will be used as a context manager.
    func: This will be called between saving flags and restoring flags.
    overrides: Flag names mapped to their values. These flags will be set after
      saving the original flag state. The type of the values depends on if
      _FlagOverrider or _ParsingFlagOverrider was specified.

  Returns:
    A wrapped version of func.
  """

  @functools.wraps(func)
  def _flagsaver_wrapper(*args, **kwargs):
    """Wrapper function that saves and restores flags."""
    with flag_overrider_cls(**overrides):
      return func(*args, **kwargs)

  return _flagsaver_wrapper


class _FlagOverrider:
  """Overrides flags for the duration of the decorated function call.

  It also restores all original values of flags after decorated method
  completes.
  """

  def __init__(self, **overrides: Any):
    self._overrides = overrides
    self._saved_flag_values = None

  def __call__(self, func: _CallableT) -> _CallableT:
    if inspect.isclass(func):
      raise TypeError('flagsaver cannot be applied to a class.')
    return _wrap(self.__class__, func, self._overrides)

  def __enter__(self):
    self._saved_flag_values = save_flag_values(FLAGS)
    try:
      FLAGS._set_attributes(**self._overrides)
    except:
      # It may fail because of flag validators.
      restore_flag_values(self._saved_flag_values, FLAGS)
      raise

  def __exit__(self, exc_type, exc_value, traceback):
    restore_flag_values(self._saved_flag_values, FLAGS)


class _ParsingFlagOverrider(_FlagOverrider):
  """Context manager for overriding flags.

  Simulates command line parsing.

  This is simlar to _FlagOverrider except that all **overrides should be
  strings or sequences of strings, and when context is entered this class calls
  .parse(value)

  This results in the flags having .present set properly.
  """

  def __init__(self, **overrides: Union[str, Sequence[str]]):
    for flag_name, new_value in overrides.items():
      if isinstance(new_value, str):
        continue
      if (isinstance(new_value, collections.abc.Sequence) and
          all(isinstance(single_value, str) for single_value in new_value)):
        continue
      raise TypeError(
          f'flagsaver.as_parsed() cannot parse {flag_name}. Expected a single '
          f'string or sequence of strings but {type(new_value)} was provided.')
    super().__init__(**overrides)

  def __enter__(self):
    self._saved_flag_values = save_flag_values(FLAGS)
    try:
      for flag_name, unparsed_value in self._overrides.items():
        # LINT.IfChange(flag_override_parsing)
        FLAGS[flag_name].parse(unparsed_value)
        FLAGS[flag_name].using_default_value = False
        # LINT.ThenChange()

      # Perform the validation on all modified flags. This is something that
      # FLAGS._set_attributes() does for you in _FlagOverrider.
      for flag_name in self._overrides:
        FLAGS._assert_validators(FLAGS[flag_name].validators)

    except KeyError as e:
      # If a flag doesn't exist, an UnrecognizedFlagError is more specific.
      restore_flag_values(self._saved_flag_values, FLAGS)
      raise flags.UnrecognizedFlagError('Unknown command line flag.') from e

    except:
      # It may fail because of flag validators or general parsing issues.
      restore_flag_values(self._saved_flag_values, FLAGS)
      raise


def _copy_flag_dict(flag: flags.Flag) -> Dict[str, Any]:
  """Returns a copy of the flag object's ``__dict__``.

  It's mostly a shallow copy of the ``__dict__``, except it also does a shallow
  copy of the validator list.

  Args:
    flag: flags.Flag, the flag to copy.

  Returns:
    A copy of the flag object's ``__dict__``.
  """
  copy = flag.__dict__.copy()
  copy['_value'] = flag.value  # Ensure correct restore for C++ flags.
  copy['validators'] = list(flag.validators)
  return copy

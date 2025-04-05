# Copyright 2024 The Flax Authors.
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

"""Global configuration flags for Flax."""

import os
from contextlib import contextmanager
from typing import Any, Generic, NoReturn, TypeVar, overload

_T = TypeVar('_T')


class Config:
  flax_use_flaxlib: bool
  # See https://google.github.io/pytype/faq.html.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self._values = {}

  def _add_option(self, name, default):
    if name in self._values:
      raise RuntimeError(f'Config option {name} already defined')
    self._values[name] = default

  def _read(self, name):
    try:
      return self._values[name]
    except KeyError:
      raise LookupError(f'Unrecognized config option: {name}')

  @overload
  def update(self, name: str, value: Any, /) -> None:
    ...

  @overload
  def update(self, holder: 'FlagHolder[_T]', value: _T, /) -> None:
    ...

  def update(self, name_or_holder, value, /):
    """Modify the value of a given flag.

    Args:
      name_or_holder: the name of the flag to modify or the corresponding
        flag holder object.
      value: new value to set.
    """
    name = name_or_holder
    if isinstance(name_or_holder, FlagHolder):
      name = name_or_holder.name
    if name not in self._values:
      raise LookupError(f'Unrecognized config option: {name}')
    self._values[name] = value

  def __repr__(self):
    values_repr = ', '.join(f'\n  {k}={v!r}' for k, v in self._values.items())
    return f'Config({values_repr}\n)'


config = Config()

# Config parsing utils


class FlagHolder(Generic[_T]):
  def __init__(self, name, help):
    self.name = name
    self.__name__ = name[4:] if name.startswith('flax_') else name
    self.__doc__ = f'Flag holder for `{name}`.\n\n{help}'

  def __bool__(self) -> NoReturn:
    raise TypeError(
      "bool() not supported for instances of type '{0}' "
      "(did you mean to use '{0}.value' instead?)".format(type(self).__name__)
    )

  @property
  def value(self) -> _T:
    return config._read(self.name)


def bool_flag(name: str, *, default: bool, help: str) -> FlagHolder[bool]:
  """Set up a boolean flag.

  Example::

    enable_foo = bool_flag(
        name='flax_enable_foo',
        default=False,
        help='Enable foo.',
    )

  Now the ``FLAX_ENABLE_FOO`` shell environment variable can be used to
  control the process-level value of the flag, in addition to using e.g.
  ``config.update("flax_enable_foo", True)`` directly.

  Args:
    name: converted to lowercase to define the name of the flag. It is
      converted to uppercase to define the corresponding shell environment
      variable.
    default: a default value for the flag.
    help: used to populate the docstring of the returned flag holder object.

  Returns:
    A flag holder object for accessing the value of the flag.
  """
  name = name.lower()
  config._add_option(name, static_bool_env(name.upper(), default))
  fh = FlagHolder[bool](name, help)
  setattr(Config, name, property(lambda _: fh.value, doc=help))
  return fh


def static_bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  This is deprecated. Please use bool_flag() unless your flag
  will be used in a static method and does not require runtime updates.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.
  Args:
    varname: the name of the variable
    default: the default boolean value
  Returns:
    boolean return value derived from defaults and environment.
  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(
      f'invalid truth value {val!r} for environment {varname!r}'
    )


@contextmanager
def temp_flip_flag(var_name: str, var_value: bool):
  """Context manager to temporarily flip feature flags for test functions.

  Args:
    var_name: the config variable name (without the 'flax_' prefix)
    var_value: the boolean value to set var_name to temporarily
  """
  old_value = getattr(config, f'flax_{var_name}')
  try:
    config.update(f'flax_{var_name}', var_value)
    yield
  finally:
    config.update(f'flax_{var_name}', old_value)


# Flax Global Configuration Variables:

flax_filter_frames = bool_flag(
  name='flax_filter_frames',
  default=True,
  help='Whether to hide flax-internal stack frames from tracebacks.',
)

flax_profile = bool_flag(
  name='flax_profile',
  default=True,
  help='Whether to run Module methods under jax.named_scope for profiles.',
)

flax_use_orbax_checkpointing = bool_flag(
  name='flax_use_orbax_checkpointing',
  default=True,
  help='Whether to use Orbax to save checkpoints.',
)

flax_preserve_adopted_names = bool_flag(
  name='flax_preserve_adopted_names',
  default=False,
  help="When adopting outside modules, don't clobber existing names.",
)

# TODO(marcuschiam): remove this feature flag once regular dict migration is complete
flax_return_frozendict = bool_flag(
  name='flax_return_frozendict',
  default=False,
  help='Whether to return FrozenDicts when calling init or apply.',
)

flax_fix_rng = bool_flag(
  name='flax_fix_rng_separator',
  default=False,
  help=(
    'Whether to add separator characters when folding in static data into'
    ' PRNG keys.'
  ),
)

flax_use_flaxlib = bool_flag(
  name='flax_use_flaxlib',
  default=False,
  help='Whether to use flaxlib for C++ acceleration.',
)
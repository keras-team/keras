# Copyright 2018 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import contextlib
import functools
import itertools
import logging
import os
import sys
from typing import Any, Generic, NoReturn, Optional, Protocol, TypeVar, cast

from jax._src.lib import guard_lib
from jax._src.lib import jax_jit
from jax._src.lib import xla_client
from jax._src import logging_config

config_ext = xla_client._xla.config

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


def bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    varname: the name of the variable
    default: the default boolean value
  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(f"invalid truth value {val!r} for environment {varname!r}")

def int_env(varname: str, default: int) -> int:
  """Read an environment variable and interpret it as an integer."""
  return int(os.getenv(varname, str(default)))


class ValueHolder(Protocol[_T]):
  """A holder for a configuration value.

  There are two kinds of value holders: ``Flag``, which is assigned exactly
  once and never modified after; and ``State``, which can be changed locally
  within a thread via a context manager.
  """

  value: _T

  def _set(self, value: _T) -> None: ...


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self._value_holders: dict[str, ValueHolder] = {}
    self.meta = {}
    self.use_absl = False
    self._contextmanager_flags = set()

  def update(self, name, val):
    if name not in self._value_holders:
      raise AttributeError(f"Unrecognized config option: {name}")
    self._value_holders[name]._set(val)

  def read(self, name):
    if name in self._contextmanager_flags:
      raise AttributeError(
          "For flags with a corresponding contextmanager, read their value "
          f"via e.g. `config.{name}` rather than `config.FLAGS.{name}`.")
    return self._read(name)

  def _read(self, name):
    try:
      return self._value_holders[name].value
    except KeyError:
      raise AttributeError(f"Unrecognized config option: {name}")

  @property
  def values(self):
    return {name: holder.value for name, holder in self._value_holders.items()}

  def add_option(self, name, holder, opt_type, meta_args, meta_kwargs):
    if name in self._value_holders:
      raise Exception(f"Config option {name} already defined")
    self._value_holders[name] = holder
    self.meta[name] = (opt_type, meta_args, meta_kwargs)

  def config_with_absl(self):
    """Registers absl flags for the JAX configs.

    E.g., for each JAX config defined using bool_state(), this method
    registers an absl boolean flag, with the same name.

    This is the recommended method to call if you use `app.run(main)` and you
    need JAX flags.

    Examples:

    ```python
    from absl import app
    import jax
    ...

    if __name__ == '__main__':
      jax.config.config_with_absl()
      app.run(main)
    ```

    """
    import absl.flags as absl_FLAGS  # noqa: F401  # pytype: disable=import-error
    from absl import app, flags as absl_flags  # pytype: disable=import-error

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = { bool: absl_flags.DEFINE_bool,
                  int:  absl_flags.DEFINE_integer,
                  float: absl_flags.DEFINE_float,
                  str:  absl_flags.DEFINE_string,
                  'enum': absl_flags.DEFINE_enum }

    for name, (flag_type, meta_args, meta_kwargs) in self.meta.items():
      holder = self._value_holders[name]
      absl_defs[flag_type](name, holder.value, *meta_args, **meta_kwargs)
    app.call_after_init(lambda: self.complete_absl_config(absl_flags))

  def complete_absl_config(self, absl_flags):
    # NOTE: avoid calling from outside this module. Instead, use
    # `config_with_absl()`, and (in rare cases) `parse_flags_with_absl()`.
    for name, holder in self._value_holders.items():
      try:
        flag = absl_flags.FLAGS[name]
      except KeyError:
        # This can happen if a new flag was added after config_with_absl() was
        # called, but before complete_absl_config was run. We could in principle
        # add code to DEFINE_... to register any newly added flags with ABSL
        # if config_with_absl() has already been called, but arguably the user
        # should have called config_with_absl() later.
        continue
      if flag.present:
        holder._set(flag.value)

  def parse_flags_with_absl(self):
    """Parses command-line args that start with --jax.

    This method should be used only by advanced users. Most users should use
    :meth:`config_with_absl` instead.

    This method has serious limitations: e.g., although it parses only the
    --jax* command-line args, it runs the validators of all registered absl
    flags, even non-JAX ones that have not been set yet; as such, for the
    non-JAX flags, the validators run on the default flag values, not on the
    values indicated by the command-line args.
    """
    global already_configured_with_absl
    if not already_configured_with_absl:
      # Extract just the --jax... flags (before the first --) from argv. In some
      # environments (e.g. ipython/colab) argv might be a mess of things
      # parseable by absl and other junk.
      jax_argv = itertools.takewhile(lambda a: a != '--', sys.argv)
      jax_argv = ['', *(a for a in jax_argv if a.startswith('--jax'))]

      import absl.flags  # pytype: disable=import-error
      self.config_with_absl()
      absl.flags.FLAGS(jax_argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True


def trace_context():
  """Returns a tuple of configuration values that affect tracing.

  These values are included in the cache key for linear_util.cache.

  Values included in this set should also most likely be included in
  the C++ JIT state, which is handled separately.
  """
  return (axis_env_state.value, mesh_context_manager.value,
          xla_metadata_context_manager.value,
          abstract_mesh_context_manager.value,
          device_context.value,
          compute_on_context_manager.value, enable_x64.value,
          numpy_rank_promotion.value, default_matmul_precision.value,
          dynamic_shapes.value,
          eager_constant_folding.value,
          numpy_dtype_promotion.value,
          default_device.value, random_seed_offset.value,
          threefry_partitionable.value,
          threefry_gpu_kernel_lowering.value,
          sharding_in_types.value,
          use_direct_linearize.value,
          softmax_custom_jvp.value,
          disable_jit.value,
          debug_key_reuse.value,
          jax_xla_profile_version.value,
          # Technically this affects jaxpr->stablehlo lowering, not tracing.
          hlo_source_file_canonicalization_regex.value,
          pgle_profiling_runs.value,
          enable_pgle.value,
          use_shardy_partitioner.value)

config = Config()

_read = config._read
update = config.update
parse_flags_with_absl = config.parse_flags_with_absl


class NoDefault: pass
no_default = NoDefault()

config_states = {}

class State(config_ext.Config[_T]):

  __slots__ = (
      '_name', '_update_thread_local_hook', '_update_global_hook',
      '_validator', '_default_context_manager_value', '__doc__', '__name__',
  )

  def __init__(
      self,
      name: str,
      default: _T,
      help,
      update_global_hook: Callable[[_T], None] | None = None,
      update_thread_local_hook: Callable[[_T | None], None] | None = None,
      validator: Callable[[Any], None] | None = None,
      extra_description: str = '',
      default_context_manager_value: Any = no_default,
      include_in_jit_key: bool = False,
  ):
    super().__init__(default, include_in_jit_key)
    self._name = name
    self.__name__ = name[4:] if name.startswith('jax_') else name
    self.__doc__ = (f"Context manager for `{name}` config option"
                    f"{extra_description}.\n\n{help}")
    self._update_global_hook = update_global_hook
    self._update_thread_local_hook = update_thread_local_hook
    self._validator = validator
    self._default_context_manager_value = default_context_manager_value
    if self._validator:
      self._validator(default)
    if self._update_global_hook:
      self._update_global_hook(default)
    config_states[name] = self

  def __bool__(self) -> NoReturn:
    raise TypeError(
        "bool() not supported for instances of type '{0}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__))

  def _set(self, value: _T) -> None:
    if self._validator:
      self._validator(value)
    self.set_global(value)
    if self._update_global_hook:
      self._update_global_hook(value)

  @contextlib.contextmanager
  def __call__(self, new_val: Any = no_default):
    if new_val is no_default:
      if self._default_context_manager_value is not no_default:
        new_val = self._default_context_manager_value  # default_context_manager_value provided to constructor
      else:
        # no default_value provided to constructor and no value provided as an
        # argument, so we raise an error
        raise TypeError(f"Context manager for {self.__name__} config option "
                        "requires an argument representing the new value for "
                        "the config option.")
    if self._validator:
      self._validator(new_val)
    prev_val = self.swap_local(new_val)
    if self._update_thread_local_hook:
      self._update_thread_local_hook(new_val)
    try:
      yield
    finally:
      self.set_local(prev_val)
      if self._update_thread_local_hook:
        if prev_val is config_ext.unset:
          self._update_thread_local_hook(None)
        else:
          self._update_thread_local_hook(cast(Optional[Any], prev_val))

  def _add_hooks(self, update_global_hook, update_thread_local_hook):
    """Private method that adds hooks to an existing context-manager.

    Used to avoid cyclic import dependencies."""
    self._update_thread_local_hook = update_thread_local_hook
    self._update_global_hook = update_global_hook
    update_global_hook(self.get_global())


UPGRADE_BOOL_HELP = (
    " This will be enabled by default in future versions of JAX, at which "
    "point all uses of the flag will be considered deprecated (following "
    "the `API compatibility policy "
    "<https://jax.readthedocs.io/en/latest/api_compatibility.html>`_).")

UPGRADE_BOOL_EXTRA_DESC = " (transient)"


def bool_state(
    name: str,
    default: bool,
    help: str,
    *,
    update_global_hook: Callable[[bool], None] | None = None,
    update_thread_local_hook: Callable[[bool | None], None] | None = None,
    upgrade: bool = False,
    extra_description: str = '',
    include_in_jit_key: bool = False,
) -> State[bool]:
  """Set up thread-local state and return a contextmanager for managing it.

  This function is a convenience wrapper. It defines a flag, environment
  variable, and corresponding thread-local state, which can be managed via the
  contextmanager it returns.

  The thread-local state value can be read via the ``config.<option_name>``
  attribute, where ``config`` is the singleton ``Config`` instance.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    default: boolean, a default value for the option.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.
    update_global_hook: a optional callback that is called with the updated
      value of the global state when it is altered or set initially.
    update_thread_local_hook: a optional callback that is called with the
      updated value of the thread-local state when it is altered or set
      initially.
    upgrade: optional indicator that this flag controls a canonical feature
      upgrade, so that it is `True` for the incoming functionality, `False`
      for the outgoing functionality to be deprecated.
    extra_description: string, optional: extra information to add to the
      summary description.

  Returns:
    A contextmanager to control the thread-local state value.

  Examples:

    ENABLE_FOO = config.bool_state(
        name='jax_enable_foo',
        default=False,
        help='Enable foo.')

    # Now the JAX_ENABLE_FOO shell environment variable and --jax_enable_foo
    # command-line flag can be used to control the process-level value of
    # the configuration option, in addition to using e.g.
    # ``config.update("jax_enable_foo", True)`` directly. We can also use a
    # context manager:

    with enable_foo(True):
      ...

  The value of the thread-local state or flag can be accessed via
  ``config.jax_enable_foo``. Reading it via ``config.FLAGS.jax_enable_foo`` is
  an error.
  """
  if not isinstance(default, bool):
    raise TypeError(f"Default value must be of type bool, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  default = bool_env(name.upper(), default)
  name = name.lower()
  if upgrade:
    help += ' ' + UPGRADE_BOOL_HELP
    extra_description += UPGRADE_BOOL_EXTRA_DESC
  config._contextmanager_flags.add(name)

  s = State[bool](
      name, default, help, update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      extra_description=extra_description, default_context_manager_value=True,
      include_in_jit_key=include_in_jit_key)
  config.add_option(name, s, bool, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s


def enum_state(
    name: str,
    enum_values: Sequence[str],
    default: str,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_jit_key: bool = False,
) -> State[str]:
  """Set up thread-local state and return a contextmanager for managing it.

  See docstring for ``bool_state``.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    enum_values: list of strings representing the possible values for the
      option.
    default: string, default value.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  if not isinstance(default, str):
    raise TypeError(f"Default value must be of type str, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default = os.getenv(name.upper(), default)
  if default not in enum_values:
    raise ValueError(f"Invalid value \"{default}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def validator(new_val):
    if type(new_val) is not str or new_val not in enum_values:
      raise ValueError(f"new enum value must be in {enum_values}, "
                       f"got {new_val} of type {type(new_val)}.")

  s = State[str](
      name,
      default,
      help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      validator=validator,
      include_in_jit_key=include_in_jit_key,
  )
  config.add_option(
      name, s, 'enum',
      meta_args=[],
      meta_kwargs={"enum_values": enum_values, "help": help}
  )
  setattr(Config, name, property(lambda _: s.value))
  return s


def optional_enum_state(
    name: str,
    enum_values: Sequence[str],
    default: str | None,
    help: str,
    *,
    update_global_hook: Callable[[str | None], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_jit_key: bool = False,
) -> State[str | None]:
  """Set up thread-local state and return a contextmanager for managing it.

  See docstring for ``bool_state``.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    enum_values: list of strings representing the possible values for the
      option.
    default: optional string, default value.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  if default is not None and not isinstance(default, str):
    raise TypeError(f"Default value must be of type str or None, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default = os.getenv(name.upper(), default)
  if default is not None and default not in enum_values:
    raise ValueError(f"Invalid value \"{default}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def validate(new_val):
    if (new_val is not None and
      (type(new_val) is not str or new_val not in enum_values)):
      raise ValueError(f"new enum value must be None or in {enum_values}, "
                       f"got {new_val} of type {type(new_val)}.")

  s = State['str | None'](
      name, default, help, update_global_hook, update_thread_local_hook,
      validate, include_in_jit_key=include_in_jit_key,
  )
  config.add_option(
      name, s, 'enum',
      meta_args=[],
      meta_kwargs={"enum_values": enum_values, "help": help}
  )
  setattr(Config, name, property(lambda _: s.value))
  return s


def int_state(
    name: str,
    default: int,
    help: str,
    *,
    update_global_hook: Callable[[int], None] | None = None,
    update_thread_local_hook: Callable[[int | None], None] | None = None,
    include_in_jit_key: bool = False,
) -> State[int]:
  """Set up thread-local state and return a contextmanager for managing it.

  See docstring for ``bool_state``.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    default: optional int, default value.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  if not isinstance(default, int):
    raise TypeError(f"Default value must be of type int, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default_env = os.getenv(name.upper())
  if default_env is not None:
    try:
      default = int(default_env)
    except ValueError:
      raise ValueError(f"Invalid value \"{default_env}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def validate(new_val):
    if new_val is not None and not isinstance(new_val, int):
      raise ValueError(f'new int config value must be None or of type int, '
                       f'got {new_val} of type {type(new_val)}')

  s = State[int](name, default, help, update_global_hook,
                 update_thread_local_hook, validate,
                 include_in_jit_key=include_in_jit_key)
  config.add_option(name, s, int, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s


def float_state(
    name: str,
    default: float,
    help: str,
    *,
    update_global_hook: Callable[[float], None] | None = None,
    update_thread_local_hook: Callable[[float | None], None] | None = None,
) -> State[float]:
  """Set up thread-local state and return a contextmanager for managing it.

  See docstring for ``bool_state``.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    default: default value.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  if not isinstance(default, float):
    raise TypeError(f"Default value must be of type float, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default_env = os.getenv(name.upper())
  if default_env is not None:
    try:
      default = float(default_env)
    except ValueError:
      raise ValueError(f"Invalid value \"{default_env}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def validate(new_val):
    if new_val is not None and not isinstance(new_val, (float, int)):
      raise ValueError(
        f'new float config value must be None or of type float, '
        f'got {new_val} of type {type(new_val)}')

  s = State[float](name, default, help, update_global_hook,
                   update_thread_local_hook, validate)
  config.add_option(name, s, float, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s


def string_state(
    name: str,
    default: str,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
) -> State[str]:
  """Set up thread-local state and return a contextmanager for managing it.

  See docstring for ``bool_state``.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    default: string, a default value for the option.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.
    update_global_hook: an optional callback that is called with the updated
      value of the global state when it is altered or set initially.
    update_thread_local_hook: an optional callback that is called with the
      updated value of the thread-local state when it is altered or set
      initially.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  if not isinstance(default, str):
    raise TypeError(f"Default value must be of type str, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")

  def validator(new_val):
    if not isinstance(new_val, str):
      raise TypeError('new string config value must be of type str,'
                       f' got {new_val} of type {type(new_val)}.')

  return string_or_object_state(
      name, default, help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      validator=validator)


def optional_string_state(
    name: str,
    default: str | None,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
) -> State[str | None]:
  """Set up thread-local state and return a contextmanager for managing it.

  See docstring for ``bool_state``.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    default: optional string, a default value for the option.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.
    update_global_hook: an optional callback that is called with the updated
      value of the global state when it is altered or set initially.
    update_thread_local_hook: an optional callback that is called with the
      updated value of the thread-local state when it is altered or set
      initially.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  if default is not None and not isinstance(default, str):
    raise TypeError(f"Default value must be of type str or None, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")

  def validator(new_val):
    if new_val is not None and not isinstance(new_val, str):
      raise ValueError('new string config value must be None or of type str,'
                       f' got {new_val} of type {type(new_val)}.')

  return string_or_object_state(
      name, default, help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      validator=validator)

def string_or_object_state(
    name: str,
    default: Any,
    help: str,
    *,
    update_global_hook: Callable[[Any], None] | None = None,
    update_thread_local_hook: Callable[[Any], None] | None = None,
    validator: Callable[[Any], None] | None = None,
) -> State[Any]:
  """Set up thread-local state and return a contextmanager for managing it.

  Similar to ``string_state``, except the context manager will accept
  any object, not just a string. Any value passed via command line flag or
  environment variable will be treated as a string.

  Args:
    name: string, converted to lowercase to define the name of the config
      option (and absl flag). It is converted to uppercase to define the
      corresponding shell environment variable.
    default: string, a default value for the option.
    help: string, used to populate the flag help information as well as the
      docstring of the returned context manager.
    update_global_hook: an optional callback that is called with the updated
      value of the global state when it is altered or set initially.
    update_thread_local_hook: an optional callback that is called with the
      updated value of the thread-local state when it is altered or set
      initially.
    validator: an optional callback that is called with the new
      value on any update, and should raise an error if the new value is
      invalid.

  Returns:
    A contextmanager to control the thread-local state value.
  """
  name = name.lower()
  default = os.getenv(name.upper(), default)
  config._contextmanager_flags.add(name)

  s = State[Any](
      name, default, help, update_global_hook, update_thread_local_hook,
      validator)
  setattr(Config, name, property(lambda _: s.value))
  config.add_option(name, s, str, meta_args=[], meta_kwargs={"help": help})
  return s


class Flag(Generic[_T]):

  __slots__ = ("_name", "value", "_update_hook")

  _name: str
  value: _T
  _update_hook: Callable[[Any], None] | None

  def __init__(self, name: str, default: _T,
               update_hook: Callable[[Any], None] | None = None):
    self._name = name
    self._update_hook = update_hook
    self._set(default)

  def __bool__(self) -> NoReturn:
    raise TypeError(
        "bool() not supported for instances of type '{0}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__))

  def _set(self, value: _T) -> None:
    self.value = value
    if self._update_hook is not None:
      self._update_hook(value)


def bool_flag(name, default, *args, **kwargs) -> Flag[bool]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, bool, args, kwargs)
  return holder


def int_flag(name, default, *args, **kwargs) -> Flag[int]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, int, args, kwargs)
  return holder


def float_flag(name, default, *args, **kwargs) -> Flag[float]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, float, args, kwargs)
  return holder


def string_flag(name, default, *args, **kwargs) -> Flag[str]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, str, args, kwargs)
  return holder


def enum_flag(name, default, *args, **kwargs) -> Flag[str]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, 'enum', args, kwargs)
  return holder


already_configured_with_absl = False


trace_state = config_ext.Config(None, include_in_jit_key=True)
axis_env_state = config_ext.Config((), include_in_jit_key=True)
mesh_context_manager = config_ext.Config((), include_in_jit_key=True)
abstract_mesh_context_manager = config_ext.Config((), include_in_jit_key=True)
device_context = config_ext.Config((), include_in_jit_key=True)
compute_on_context_manager = config_ext.Config((), include_in_jit_key=True)
xla_metadata_context_manager = config_ext.Config((), include_in_jit_key=True)


# TODO(b/214340779): remove flag when XLA:CPU is improved.
jax2tf_associative_scan_reductions = bool_state(
    name='jax2tf_associative_scan_reductions',
    default=False,
    help=(
        'JAX has two separate lowering rules for the cumulative reduction '
        'primitives (cumsum, cumprod, cummax, cummin). On CPUs and GPUs it uses '
        'a lax.associative_scan, while for TPUs it uses the HLO ReduceWindow. '
        'The latter has a slow implementation on CPUs and GPUs. '
        'By default, jax2tf uses the TPU lowering. Set this flag to True to '
        'use the associative scan lowering usage, and only if it makes a difference '
        'for your application. '
        'See the jax2tf README.md for more details.'
    )
)

jax2tf_default_native_serialization = bool_state(
    name='jax2tf_default_native_serialization',
    default=bool_env('JAX2TF_DEFAULT_NATIVE_SERIALIZATION', True),
    help=(
        'Sets the default value of the native_serialization parameter to '
        'jax2tf.convert. Prefer using the parameter instead of the flag, '
        'the flag may be removed in the future. '
        'Starting with JAX 0.4.31 non-native serialization is deprecated.'
    )
)

jax_serialization_version = int_state(
    name='jax_serialization_version',
    default=int_env('JAX_SERIALIZATION_VERSION', 0),  # We use 0 to detect default.
    help=(
        'DEPRECATED: use jax_export_calling_convention_version.'
    )
)

jax_export_calling_convention_version = int_state(
    name='jax_export_calling_convention_version',
    # Note: bump the default calling convention version at least one month after
    # we update XlaCallModule to support the new version, so that serialized
    # modules are forward compatible with deployed versions of XlaCallModule.
    # Version 9 of XlaCallModule is supported since October 27th, 2023.
    default=int_env('JAX_EXPORT_CALLING_CONVENTION_VERSION', 9),
    help=(
        'The calling convention version number to use for exporting. This must be '
        'within the range of versions supported by the tf.XlaCallModule '
        'used in your deployment environment. '
        'See https://jax.readthedocs.io/en/latest/export/shape_poly.html#calling-convention-versions.'
    )
)

export_ignore_forward_compatibility = bool_state(
    name='jax_export_ignore_forward_compatibility',
    default=bool_env('JAX_EXPORT_IGNORE_FORWARD_COMPATIBILIY', False),
    help=(
        'Whether to ignore the forward compatibility lowering rules. '
        'See https://jax.readthedocs.io/en/latest/export/export.html#compatibility-guarantees-for-custom-calls.'
    )
)

jax_platforms = optional_string_state(
    name='jax_platforms',
    default=None,
    help=(
        'Comma-separated list of platform names specifying which platforms jax '
        'should initialize. If any of the platforms in this list are not successfully '
        'initialized, an exception will be raised and the program will be aborted. '
        'The first platform in the list will be the default platform. '
        'For example, config.jax_platforms=cpu,tpu means that CPU and TPU backends '
        'will be initialized, and the CPU backend will be used unless otherwise '
        'specified. If TPU initialization fails, it will raise an exception. '
        'By default, jax will try to initialize all available '
        'platforms and will default to GPU or TPU if available, and fallback to CPU '
        'otherwise.'
        ))

jax_pjrt_client_create_options = optional_string_state(
    name='jax_pjrt_client_create_options',
    default=None,
    help=('A set of key-value pairs in the format of "k1:v1;k2:v2" strings '
          'provided to a device platform pjrt client as extra arguments.'))

enable_checks = bool_state(
    name='jax_enable_checks',
    default=False,
    help='Turn on invariant checking for JAX internals. Makes things slower.')

debug_key_reuse = bool_state(
    name='jax_debug_key_reuse',
    default=False,
    help=('Turn on experimental key reuse checking. With this configuration enabled,'
          ' typed PRNG keys (i.e. keys created with jax.random.key()) will have their'
          ' usage tracked, and incorrect reuse of a previously-used key will lead to'
          ' an error. Currently enabling this leads to a small Python overhead on'
          ' every call to a JIT-compiled function with keys as inputs or outputs.'))

check_tracer_leaks = bool_state(
    name='jax_check_tracer_leaks',
    default=False,
    help=('Turn on checking for leaked tracers as soon as a trace completes. '
          'Enabling leak checking may have performance impacts: some caching '
          'is disabled, and other overheads may be added. Additionally, be aware '
          'that some Python debuggers can cause false positives, so it is recommended '
          'to disable any debuggers while leak checking is enabled.'))
checking_leaks = functools.partial(check_tracer_leaks, True)

debug_nans = bool_state(
    name='jax_debug_nans',
    default=False,
    help=('Add nan checks to every operation. When a nan is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the nan.'))

debug_infs = bool_state(
    name='jax_debug_infs',
    default=False,
    help=('Add inf checks to every operation. When an inf is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the inf.'))

log_compiles = bool_state(
    name='jax_log_compiles',
    default=False,
    help=('Log a message each time `jit` or `pmap` compiles an XLA '
          'computation. Logging is performed with `logging`. When this '
          'option is set, the log level is WARNING; otherwise the level is '
          'DEBUG.'))

explain_cache_misses = bool_state(
    name='jax_explain_cache_misses',
    default=False,
    help=('Each time there is a miss on one of the main caches (e.g. the '
          'tracing cache), log an explanation.. Logging is performed with '
          '`logging`. When this option is set, the log level is WARNING; '
          'otherwise the level is DEBUG.'))

log_checkpoint_residuals = bool_state(
    name='jax_log_checkpoint_residuals',
    default=False,
    help=('Log a message every time jax.checkpoint (aka jax.remat) is '
          'partially evaluated (e.g. for autodiff), printing what residuals '
          'are saved.'))

pmap_shmap_merge = bool_state(
    name='jax_pmap_shmap_merge',
    default=False,
    upgrade=True,
    help='If True, pmap and shard_map API will be merged.')

# Remove after next JAX release on Jan 15, 2025.
if hasattr(jax_jit.global_state(), 'enable_memories'):
  def _update_jax_memories_global(val):
    jax_jit.global_state().enable_memories = val

  def _update_jax_memories_thread_local(val):
    jax_jit.thread_local_state().enable_memories = val

  enable_memories = bool_state(
      'jax_enable_memories',
      default=True,
      upgrade=True,
      update_global_hook=_update_jax_memories_global,
      update_thread_local_hook=_update_jax_memories_thread_local,
      help=("If True, will allow fetching memory kinds available on executable "
            "and annotate Shardings with it."))

spmd_mode = enum_state(
    name='jax_spmd_mode',
    enum_values=['allow_all', 'allow_jit'],
    default='allow_jit',
    help=("Decides whether Math on `jax.Array`'s that are not fully addressable "
          "(i.e. spans across multiple processes) is allowed. The options are: "
          "* allow_jit: Default, `pjit` and `jax.jit` computations are allowed "
          "    to execute on non-fully addressable `jax.Array`s\n"
          "* allow_all: `jnp`, normal math (like `a + b`, etc), `pjit`, "
          "    `jax.jit` and all other operations are allowed to "
          "    execute on non-fully addressable `jax.Array`s."))


distributed_debug = bool_state(
    name='jax_distributed_debug',
    default=False,
    help=('Enable logging useful for debugging multi-process distributed '
          'computations. Logging is performed with `logging` at WARNING '
          'level.'))

random_seed_offset = int_state(
    name='jax_random_seed_offset',
    default=0,
    help=('Offset to all random seeds (e.g. argument to jax.random.key()).'),
    include_in_jit_key=True,
)

legacy_prng_key = enum_state(
    name='jax_legacy_prng_key',
    enum_values=['allow', 'warn', 'error'],
    default='allow',
    help=('Specify the behavior when raw PRNG keys are passed to '
          'jax.random APIs.')
)

enable_custom_prng = bool_state(
    name='jax_enable_custom_prng',
    default=False,
    upgrade=True,
    help=('Enables an internal upgrade that allows one to define custom '
          'pseudo-random number generator implementations.'))

default_prng_impl = enum_state(
    name='jax_default_prng_impl',
    enum_values=['threefry2x32', 'rbg', 'unsafe_rbg'],
    default='threefry2x32',
    help=('Select the default PRNG implementation, used when one is not '
          'explicitly provided at seeding time.'))

threefry_partitionable = bool_state(
    name='jax_threefry_partitionable',
    default=True,
    upgrade=True,
    help=('Enables internal threefry PRNG implementation changes that '
          'render it automatically partitionable in some cases. Without this '
          'flag, using the standard jax.random pseudo-random number generation '
          'may result in extraneous communication and/or redundant distributed '
          'computation. With this flag, the communication overheads disappear '
          'in some cases.'),
    include_in_jit_key=True)

threefry_gpu_kernel_lowering = bool_state(
    name='jax_threefry_gpu_kernel_lowering',
    default=False,
    help=('On GPU, lower threefry PRNG operations to a kernel implementation. '
          'This makes compile times faster at a potential runtime memory '
          'cost.'),
    include_in_jit_key=True)

sharding_in_types = bool_state(
    name='jax_sharding_in_types',
    default=False,
    help=('When True, enables forward only sharding propagation in JAX and '
          'avals have sharding on them.'),
    include_in_jit_key=True)

use_direct_linearize = bool_state(
    name='jax_use_direct_linearize',
    default=False,
    help=('Use direct linearization instead JVP followed by partial eval'),
    include_in_jit_key=True)

data_dependent_tracing_fallback = bool_state(
    name='jax_data_dependent_tracing_fallback',
    default=False,
    help=('When True, falls back to trace dispatch based on data dependence '
          'instead of throwing an escaped tracer error.'))

softmax_custom_jvp = bool_state(
    name='jax_softmax_custom_jvp',
    default=False,
    upgrade=True,
    help=('Use a new custom_jvp rule for jax.nn.softmax. The new rule should '
          'improve memory usage and stability. Set True to use new '
          'behavior. See https://github.com/jax-ml/jax/pull/15677'),
    include_in_jit_key=True)


enable_custom_vjp_by_custom_transpose = bool_state(
    name='jax_enable_custom_vjp_by_custom_transpose',
    default=False,
    upgrade=True,
    help=('Enables an internal upgrade that implements `jax.custom_vjp` by '
          'reduction to `jax.custom_jvp` and `jax.custom_transpose`.'))

raise_persistent_cache_errors = bool_state(
    name='jax_raise_persistent_cache_errors',
    default=False,
    help=('If true, exceptions raised when reading or writing to the '
          'persistent compilation cache will be allowed through, halting '
          'program execution if not manually caught. If false, exceptions are '
          'caught and raised as warnings, allowing program execution to '
          'continue. Defaults to false so cache bugs or intermittent issues '
          'are non-fatal.'))

persistent_cache_min_compile_time_secs = float_state(
    name='jax_persistent_cache_min_compile_time_secs',
    default=1.,
    help=('The minimum compile time of a computation to be written to the '
          'persistent compilation cache. This threshold can be raised to '
          'decrease the number of entries written to the cache.'))

persistent_cache_min_entry_size_bytes = int_state(
    name='jax_persistent_cache_min_entry_size_bytes',
    default=0,
    help=('The minimum size (in bytes) of an entry that will be cached in the '
          'persistent compilation cache: '
          '* -1: disable the size restriction and prevent overrides. '
          '* Leave at default (0) to allow for overrides. The override will '
          '  typically ensure that the minimum size is optimal for the '
          '  filesystem being used for the cache. '
          '* > 0: the actual minimum size desired; no overrides.'))

# TODO: Change default to all
persistent_cache_enable_xla_caches = optional_string_state(
    name='jax_persistent_cache_enable_xla_caches',
    default='xla_gpu_per_fusion_autotune_cache_dir',
    help=('When the persistent cache is enabled, additional XLA caching will '
          'also be enabled automatically. This option can be used to configure'
          'which XLA caching methods will be enabled.'),
)

compilation_cache_include_metadata_in_key = bool_state(
    name='jax_compilation_cache_include_metadata_in_key',
    default=False,
    help=(
        'Include metadata, such as file names and line numbers, in the'
        ' compilation cache key. If false, the cache will still get hits even'
        ' if functions or files are moved, etc. However, it means that'
        ' executables loaded from the cache may have stale metadata, which'
        ' may show up in, e.g., profiles.'
    ),
)

hlo_source_file_canonicalization_regex = optional_string_state(
    name='jax_hlo_source_file_canonicalization_regex',
    default=None,
    help=('Used to canonicalize the source_path metadata of HLO instructions '
          'by removing the given regex. If set, re.sub() is called on each '
          'source_file with the given regex, and all matches are removed. '
          'This can be used to avoid spurious cache misses when using the '
          'persistent compilation cache, which includes HLO metadata in the '
          'cache key.'))

include_full_tracebacks_in_locations = bool_state(
    name='jax_include_full_tracebacks_in_locations',
    default=True,
    help=(
        'Include Python tracebacks in MLIR locations in IR emitted by JAX.'
    ),
)

traceback_in_locations_limit = int_state(
    name='jax_traceback_in_locations_limit',
    default=10,
    help=(
        'Limit the number of frames at the Python traceback frames included in '
        'MLIR locations. If set to the negative value, traceback will not be '
        'limited.'
    ),
)

share_binary_between_hosts = bool_state(
    name='jax_share_binary_between_hosts',
    default=False,
    help=(
        'If set to True, the compiled module will be shared between hosts '
        'directly.'
    ),
)

share_binary_between_hosts_timeout_ms = int_state(
    name='jax_share_binary_between_hosts_timeout_ms',
    default=20 * 60 * 1000,
    help='Timeout for the compiled module share.',
)

enable_pgle = bool_state(
    name='jax_enable_pgle',
    default=False,
    help=(
      'If set to True and the property jax_pgle_profiling_runs is set to '
      'greater than 0, the modules will be recompiled after running specified '
      'number times with collected data provided to the profile guided latency '
      'estimator.'
    ),
    include_in_jit_key=True,
)

pgle_profiling_runs = int_state(
    name='jax_pgle_profiling_runs',
    default=3,
    help=(
        'Amount of times module should be profiled before recompilation when '
        'PGLE is used.'
    ),
    include_in_jit_key=True,
)

pgle_aggregation_percentile = int_state(
    name='jax_pgle_aggregation_percentile',
    default=90,
    help='Percentile used to aggregate performance data between devices when '
         'PGLE is used.',
)

enable_compilation_cache = bool_state(
    name='jax_enable_compilation_cache',
    default=True,
    help=('If set to False, the compilation cache will be disabled regardless '
          'of whether set_cache_dir() was called. If set to True, the '
          'path could be set to a default value or via a call to '
          'set_cache_dir().'),
)

compilation_cache_dir = optional_string_state(
    name='jax_compilation_cache_dir',
    default=None,
    help=('Path for the cache. '
          'Precedence: '
          '1. A call to compilation_cache.set_cache_dir(). '
          '2. The value of this flag set in the command line or by default.'),
)

compilation_cache_max_size = int_state(
    name='jax_compilation_cache_max_size',
    default=-1,
    help=('The maximum size (in bytes) allowed for the persistent compilation '
          'cache. When set, the least recently accessed cache entry(s) '
          'will be deleted once the total cache directory size '
          'exceeds the specified limit. '
          'Caching will be disabled if this value is set to 0. A '
          'special value of -1 indicates no limit, allowing the cache '
          'size to grow indefinitely.'),
)

remove_custom_partitioning_ptr_from_cache_key = bool_state(
    name='jax_remove_custom_partitioning_ptr_from_cache_key',
    default=False,
    help=('If set to True, remove the custom partitioning pointer '
          'present in the precompiled stableHLO before hashing  '
          'during cache key computation. This is a potentially '
          'unsafe flag to set and only users who are sure of '
          'what they are trying to achieve should set it.'),
)

default_dtype_bits = enum_state(
    name='jax_default_dtype_bits',
    enum_values=['32', '64'],
    default='64',
    help=('Specify bit width of default dtypes, either 32-bit or 64-bit. '
          'This is a temporary flag that will be used during the process '
          'of deprecating the ``jax_enable_x64`` flag.'))

numpy_dtype_promotion = enum_state(
    name='jax_numpy_dtype_promotion',
    enum_values=['standard', 'strict'],
    default='standard',
    help=('Specify the rules used for implicit type promotion in operations '
          'between arrays. Options are "standard" or "strict"; in strict-mode, '
          'binary operations between arrays of differing strongly-specified '
          'dtypes will result in an error.'),
    include_in_jit_key=True)

disallow_mesh_context_manager = bool_state(
    name='jax_disallow_mesh_context_manager',
    default=False,
    help=(
        'If set to True, trying to use a mesh as a context manager will'
        ' result in a RuntimeError.'
    ),
)

def _update_x64_global(val):
  jax_jit.global_state().enable_x64 = val

def _update_x64_thread_local(val):
  jax_jit.thread_local_state().enable_x64 = val

enable_x64 = bool_state(
    name='jax_enable_x64',
    default=False,
    help='Enable 64-bit types to be used',
    update_global_hook=_update_x64_global,
    update_thread_local_hook=_update_x64_thread_local)

# TODO(phawkins): remove after fixing users of FLAGS.x64_enabled.
config._contextmanager_flags.remove('jax_enable_x64')

setattr(Config, "x64_enabled", property(lambda _: enable_x64.value))

def _update_default_device_global(val):
  jax_jit.global_state().default_device = val


def _update_default_device_thread_local(val):
  jax_jit.thread_local_state().default_device = val


def _validate_default_device(val):
  if (val is not None and
      not isinstance(val, xla_client.Device) and
      val not in ['cpu', 'gpu', 'tpu']):
    # TODO(skyewm): this is a workaround for non-PJRT Device types. Remove when
    # all JAX backends use a single C++ device interface.
    if 'Device' in str(type(val)):
      logger.info(
          'Allowing non-`xla_client.Device` default device: %s, type: %s',
          repr(val), type(val))
      return
    raise ValueError('jax.default_device must be passed either a Device object (e.g. '
                     f"`jax.devices('cpu')[0]`) or a platform name string like 'cpu' or 'gpu'"
                     f", got: {val!r}")


default_device = string_or_object_state(
    name='jax_default_device',
    default=None,
    help=(
        'Configure the default device for JAX operations. Set to a Device '
        'object (e.g. ``jax.devices("cpu")[0]``) to use that Device as the '
        'default device for JAX operations and jit\'d function calls (there is '
        'no effect on multi-device computations, e.g. pmapped function calls). '
        'Set to None to use the system default device. See '
        ':ref:`faq-data-placement` for more information on device placement.'),
    update_global_hook=_update_default_device_global,
    update_thread_local_hook=_update_default_device_thread_local,
    validator=_validate_default_device)

def _update_disable_jit_global(val):
  jax_jit.global_state().disable_jit = val

def _update_disable_jit_thread_local(val):
  jax_jit.thread_local_state().disable_jit = val

disable_jit = bool_state(
    name='jax_disable_jit',
    default=False,
    help=('Disable JIT compilation and just call original Python.'),
    update_global_hook=_update_disable_jit_global,
    update_thread_local_hook=_update_disable_jit_thread_local)


numpy_rank_promotion = enum_state(
    name='jax_numpy_rank_promotion',
    enum_values=['allow', 'warn', 'raise'],
    default='allow',
    help=('Control NumPy-style automatic rank promotion broadcasting '
          '("allow", "warn", or "raise").'),
    include_in_jit_key=True)

default_matmul_precision = optional_enum_state(
    name='jax_default_matmul_precision',
    enum_values=[
        # Legacy precision API values
        'default', 'high', 'highest', 'bfloat16', 'tensorfloat32', 'float32',
        # Dot algorithm presets
        'ANY_F8_ANY_F8_F32', 'ANY_F8_ANY_F8_F32_FAST_ACCUM', 'ANY_F8_ANY_F8_ANY',
        'ANY_F8_ANY_F8_ANY_FAST_ACCUM', 'F16_F16_F16', 'F16_F16_F32',
        'BF16_BF16_BF16', 'BF16_BF16_F32', 'BF16_BF16_F32_X3',
        'BF16_BF16_F32_X6', 'TF32_TF32_F32', 'TF32_TF32_F32_X3', 'F32_F32_F32',
        'F64_F64_F64',
    ],
    default=None,
    help=('Control the default matmul and conv precision for 32bit inputs.\n\n'

          'Some platforms, like TPU, offer configurable precision levels for '
          'matrix multiplication and convolution computations, trading off '
          'accuracy for speed. The precision can be controlled for each '
          'operation; for example, see the :func:`jax.lax.conv_general_dilated` '
          'and :func:`jax.lax.dot` docstrings. But it can be useful to control '
          'the default behavior obtained when an operation is not given a '
          'specific precision.\n\n'

          'This option can be used to control the default precision '
          'level for computations involved in matrix multiplication and '
          'convolution on 32bit inputs. The levels roughly describe the '
          "precision at which scalar products are computed. The 'bfloat16' "
          "option is the fastest and least precise; 'float32' is similar to "
          "full float32 precision; 'tensorfloat32' is intermediate.\n\n"

          'This parameter can also be used to specify an accumulation '
          '"algorithm" for functions that perform matrix multiplications, like '
          ':func:`jax.lax.dot`. To specify an algorithm, set this option to '
          'the name of a :class:`~jax.lax.DotAlgorithmPreset`.\n\n'),
    include_in_jit_key=True)


traceback_filtering = enum_state(
    name = 'jax_traceback_filtering',
    enum_values=["off", "tracebackhide", "remove_frames", "quiet_remove_frames",
                 "auto"],
    default="auto",
    help="Controls how JAX filters internal frames out of tracebacks.\n\n"
         "Valid values are:\n"
         " * \"off\": disables traceback filtering.\n"
         " * \"auto\": use \"tracebackhide\" if running under a sufficiently"
         " new IPython, or \"remove_frames\" otherwise.\n"
         " * \"tracebackhide\": adds \"__tracebackhide__\" annotations to"
         " hidden stack frames, which some traceback printers support.\n"
         " * \"remove_frames\": removes hidden frames from tracebacks, and adds"
         " the unfiltered traceback as a __cause__ of the exception.\n"
         " * \"quiet_remove_frames\": removes hidden frames from tracebacks, and adds"
         " a brief message (to the __cause__ of the exception) describing that this has"
         " happened.\n")

# This flag is for internal use.
# TODO(tianjianlu): Removes once we always enable cusparse lowering.
# TODO(b/262050896): Set to true after bug is fixed
bcoo_cusparse_lowering = bool_state(
    name='jax_bcoo_cusparse_lowering',
    default=False,
    help=('Enables lowering BCOO ops to cuSparse.'))

# TODO(mattjj): remove this flag when we ensure we only succeed at trace-staging
# if the intended backend can handle lowering the result
dynamic_shapes = bool_state(
    name='jax_dynamic_shapes',
    default=bool(os.getenv('JAX_DYNAMIC_SHAPES', '')),
    help=('Enables experimental features for staging out computations with '
          'dynamic shapes.'),
    include_in_jit_key=True)

# This is for stackless backward compat with e.g. equinox
eager_constant_folding = bool_state(
    name='eager_constant_folding',
    default=False,
    help=('Attempt constant folding during staging.'),
    include_in_jit_key=True)

# This flag is temporary during rollout of the remat barrier.
# TODO(parkers): Remove if there are no complaints.
remat_opt_barrier = bool_state(
    name='jax_remat_opt_barrier',
    default=True,
    help=('Enables using optimization-barrier op for lowering remat.'))

enable_remat_opt_pass = bool_state(
    name='jax_compiler_enable_remat_pass',
    default=True,
    help=('Config to enable / disable the rematerialization HLO pass. '
          'Useful to allow XLA to automatically trade off memory and '
          'compute when encountering OOM errors. However, you are '
          'likely to get better results manually with jax.checkpoint'))

# TODO(sharadmv,mattjj): set default to True, then remove
eager_pmap = bool_state(
    name='jax_eager_pmap',
    default=True,
    upgrade=True,
    help='Enable eager-mode pmap when jax_disable_jit is activated.')

no_tracing = bool_state(
    name='jax_no_tracing',
    default=False,
    help='Disallow tracing for JIT compilation.')

disable_vmap_shmap_error = bool_state(
    name='jax_disable_vmap_shmap_error',
    default=False,
    upgrade=False,
    help='Temporary workaround to disable an error check in vmap-of-shmap.')

# TODO(mattjj): remove once we land mutable array plumbing, or face great shame
custom_vjp_disable_shape_check = bool_state(
    name='jax_custom_vjp_disable_shape_check',
    default=False,
    upgrade=True,
    help='Disable the check from #19009 to enable some custom_vjp hacks.')

mutable_array_checks = bool_state(
    name='jax_mutable_array_checks',
    default=False,
    upgrade=True,
    help='Enable error checks for mutable arrays that rule out aliasing.')

xla_runtime_errors = bool_state(
    name='jax_experimental_unsafe_xla_runtime_errors',
    default=False,
    help=('Enable XLA runtime errors for jax.experimental.checkify.checks '
          'on CPU and GPU. These errors are async, might get lost and are not '
          'very readable. But, they crash the computation and enable you '
          'to write jittable checks without needing to checkify. Does not '
          'work under pmap/pjit.')
)

jax_xla_profile_version = int_state(
    name='jax_xla_profile_version',
    default=0,
    help=(
        'Optional profile version for XLA compilation. This is meaningful '
        'only when XLA is configured to support the remote compilation '
        'profile feature.'),
    include_in_jit_key=True,
)

@contextlib.contextmanager
def explicit_device_put_scope() -> Iterator[None]:
  """Indicates that the current context is an explicit device_put*() call."""
  state = guard_lib.thread_local_state()
  prev = state.explicit_device_put
  state.explicit_device_put = True
  try:
    yield
  finally:
    state.explicit_device_put = prev

@contextlib.contextmanager
def explicit_device_get_scope() -> Iterator[None]:
  """Indicates that the current context is an explicit device_get() call."""
  state = guard_lib.thread_local_state()
  prev = state.explicit_device_get
  state.explicit_device_get = True
  try:
    yield
  finally:
    state.explicit_device_get = prev

def _update_transfer_guard(state, key, val):
  """Applies the transfer guard level within guard_lib."""
  if val is None:
    setattr(state, key, None)
  elif val == 'allow':
    setattr(state, key, guard_lib.TransferGuardLevel.ALLOW)
  elif val == 'log':
    setattr(state, key, guard_lib.TransferGuardLevel.LOG)
  elif val == 'disallow':
    setattr(state, key, guard_lib.TransferGuardLevel.DISALLOW)
  elif val == 'log_explicit':
    setattr(state, key, guard_lib.TransferGuardLevel.LOG_EXPLICIT)
  elif val == 'disallow_explicit':
    setattr(state, key, guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT)
  else:
    assert False, f'Invalid transfer guard level {val}'

transfer_guard_host_to_device = optional_enum_state(
    name='jax_transfer_guard_host_to_device',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by guard_lib. Use None here to avoid accidentally
    # overriding --jax_transfer_guard.
    default=None,
    help=('Select the transfer guard level for host-to-device transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        guard_lib.global_state(), 'host_to_device', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        guard_lib.thread_local_state(), 'host_to_device', val))

transfer_guard_device_to_device = optional_enum_state(
    name='jax_transfer_guard_device_to_device',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by guard_lib. Use None here to avoid accidentally
    # overriding --jax_transfer_guard.
    default=None,
    help=('Select the transfer guard level for device-to-device transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        guard_lib.global_state(), 'device_to_device', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        guard_lib.thread_local_state(), 'device_to_device', val))

transfer_guard_device_to_host = optional_enum_state(
    name='jax_transfer_guard_device_to_host',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by guard_lib. Use None here to avoid
    # accidentally overriding --jax_transfer_guard.
    default=None,
    help=('Select the transfer guard level for device-to-host transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        guard_lib.global_state(), 'device_to_host', val
    ),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        guard_lib.thread_local_state(), 'device_to_host', val))

def _update_all_transfer_guard_global(val):
  for name in ('jax_transfer_guard_host_to_device',
               'jax_transfer_guard_device_to_device',
               'jax_transfer_guard_device_to_host'):
    config.update(name, val)

_transfer_guard = optional_enum_state(
    name='jax_transfer_guard',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by guard_lib. Use None here to avoid accidentally
    # overriding --jax_transfer_guard_*.
    default=None,
    help=('Select the transfer guard level for all transfers. This option is '
          'set-only; the transfer guard level for a specific direction should '
          'be read using the per-transfer direction option. '
          'Default is "allow".'),
    update_global_hook=_update_all_transfer_guard_global)

@contextlib.contextmanager
def transfer_guard(new_val: str) -> Iterator[None]:
  """A contextmanager to control the transfer guard level for all transfers.

  For more information, see
  https://jax.readthedocs.io/en/latest/transfer_guard.html

  Args:
    new_val: The new thread-local transfer guard level for all transfers.

  Yields:
    None.
  """
  with contextlib.ExitStack() as stack:
    stack.enter_context(transfer_guard_host_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_host(new_val))
    stack.enter_context(_transfer_guard(new_val))
    yield


def _update_garbage_collection_guard(state, key, val):
  """Applies the transfer guard level within guard_lib."""
  if val is None:
    setattr(state, key, None)
  elif val == 'allow':
    setattr(state, key, guard_lib.GarbageCollectionGuardLevel.ALLOW)
  elif val == 'log':
    setattr(state, key, guard_lib.GarbageCollectionGuardLevel.LOG)
  elif val == 'fatal':
    setattr(state, key, guard_lib.GarbageCollectionGuardLevel.FATAL)
  else:
    assert False, f'Invalid garbage collection guard level {val}'

array_garbage_collection_guard = optional_enum_state(
    name='jax_array_garbage_collection_guard',
    enum_values=['allow', 'log', 'fatal'],
    # The default is applied by guard_lib.
    default=None,
    help=(
        'Select garbage collection guard level for "jax.Array" objects.\nThis'
        ' option can be used to control what happens when a "jax.Array"'
        ' object is garbage collected. It is desirable for "jax.Array"'
        ' objects to be freed by Python reference couting rather than garbage'
        ' collection in order to avoid device memory being held by the arrays'
        ' until garbage collection occurs.\n\nValid values are:\n * "allow":'
        ' do not log garbage collection of "jax.Array" objects.\n * "log":'
        ' log an error when a "jax.Array" is garbage collected.\n * "fatal":'
        ' fatal error if a "jax.Array" is garbage collected.\nDefault is'
        ' "allow". Note that not all cycles may be detected.'
    ),
    update_global_hook=lambda val: _update_garbage_collection_guard(
        guard_lib.global_state(), 'garbage_collect_array', val
    ),
    update_thread_local_hook=lambda val: _update_garbage_collection_guard(
        guard_lib.thread_local_state(), 'garbage_collect_array', val
    ),
)

# Don't define a context manager since this isn't threadsafe.
string_state(
    name='jax_debug_log_modules',
    default='',
    help=('Comma-separated list of module names (e.g. "jax" or '
          '"jax._src.xla_bridge,jax._src.dispatch") to enable debug logging '
          'for.'),
    update_global_hook=logging_config.update_debug_log_modules)

# Don't define a context manager since this isn't threadsafe.
optional_enum_state(
    name='jax_logging_level',
    enum_values=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    default=logging.getLevelName(logging.getLogger("jax").level),
    help=('Set the corresponding logging level on all jax loggers. Only string'
          ' values from ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR",'
          ' "CRITICAL"] are accepted. If None, the logging level will not be'
          ' set. Includes C++ logging.'),
    update_global_hook=lambda logging_level: \
      logging_config.update_logging_level_global(logging_level=logging_level)
)

pmap_no_rank_reduction = bool_state(
    name='jax_pmap_no_rank_reduction',
    default=True,
    help='If True, pmap shards have a the same rank as their enclosing array.',
)

use_shardy_partitioner = bool_state(
    name='jax_use_shardy_partitioner',
    default=False,
    upgrade=True,
    help=(
        'Whether to lower to Shardy. Shardy is a new open sourced propagation '
        'framework for MLIR. Currently Shardy is experimental in JAX. See '
        'www.github.com/openxla/shardy'
    ),
    include_in_jit_key=True,
)

gpu_use_magma = enum_state(
    name='jax_use_magma',
    enum_values=['off', 'on', 'auto'],
    default='auto',
    help=(
        'Enable experimental support for MAGMA-backed lax.linalg.eig on GPU. '
        'See the documentation for lax.linalg.eig for more details about how '
        'to use this feature.'
    ),
)

exec_time_optimization_effort = float_state(
    name='jax_exec_time_optimization_effort',
    default=0.0,
    help='Effort for minimizing execution time (higher means more effort), valid range [-1.0, 1.0].'
)

memory_fitting_effort = float_state(
    name='jax_memory_fitting_effort',
    default=0.0,
    help='Effort for minimizing memory usage (higher means more effort), valid range [-1.0, 1.0].'
)

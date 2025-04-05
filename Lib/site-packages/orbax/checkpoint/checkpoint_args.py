# Copyright 2024 The Orbax Authors.
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

"""CheckpointArgs base class and registration."""

import dataclasses
import inspect
from typing import Tuple, Type, Union

from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import handler_type_registry

CheckpointHandler = checkpoint_handler.CheckpointHandler


@dataclasses.dataclass
class CheckpointArgs:
  """Base class for all checkpoint argument dataclasses.

  All :py:class:`CheckpointHandler` implementations should have corresponding
  :py:class:`CheckpointArgs` dataclasses, typically one for save and one for
  restore.

  Use one of the subclasses of :py:class:`CheckpointArgs` for your use case to
  specify how an object should be saved or restored.

  Typical usage::

    with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as ckptr:
      ckptr.save(
          path,
          args=ocp.args.StandardSave(train_state)
      )

  Example subclass::

    @ocp.args.register_with_handler(MyCheckpointHandler, for_save=True)
    @dataclasses.dataclass
    class MyCheckpointSave(ocp.args.CheckpointArgs):
      item: Any
      options: Any

    @ocp.args.register_with_handler(MyCheckpointHandler, for_restore=True)
    @dataclasses.dataclass
    class MyCheckpointRestore(ocp.args.CheckpointArgs):
      options: Any

  Example usage::

    ckptr.save(
        path,
        custom_state=MyCheckpointSave(item=..., options=...)
    )

    ckptr.save(
        path,
        custom_state=MyCheckpointRestore(options=...)
    )
  """

  pass

_SAVE_ARG_TO_HANDLER: dict[Type[CheckpointArgs], Type[CheckpointHandler]] = {}

_RESTORE_ARG_TO_HANDLER: dict[Type[CheckpointArgs], Type[CheckpointHandler]] = (
    {}
)


def register_with_handler(
    handler_cls: Type[CheckpointHandler],
    for_save: bool = False,
    for_restore: bool = False,
):
  """Registers a :py:class:`CheckpointArgs` subclass with a specific handler.

  This registration is necessary so that when the user passes uses this
  :py:class:`CheckpointArgs` class with :py:class:`CompositeCheckpointHandler`,
  we can automatically
  find the correct Handler to use to save this class.

  Note, `for_save` and `for_restore` may both be true, but cannot both be false.

  Args:
    handler_cls: `CheckpointHandler` to be associated with this `CheckpointArg`.
    for_save: indicates whether the `CheckpointArg` is registered as a save
      argument.
    for_restore: indicates whether the `CheckpointArg` is registered as a
      restore argument.

  Returns:
    Decorator.
  """
  if not for_save and not for_restore:
    raise ValueError('`for_save` and `for_restore` cannot both be False.')

  def decorator(cls: Type[CheckpointArgs]):
    if not issubclass(cls, CheckpointArgs):
      raise TypeError(
          f'{cls} must subclass `CheckpointArgs` in order to be registered.'
      )
    if for_save:
      _SAVE_ARG_TO_HANDLER[cls] = handler_cls
    if for_restore:
      _RESTORE_ARG_TO_HANDLER[cls] = handler_cls
    handler_type_registry.register_handler_type(handler_cls)
    return cls

  return decorator


def get_registered_handler_cls(
    arg: Union[Type[CheckpointArgs], CheckpointArgs]
) -> Type[CheckpointHandler]:
  """Returns the registered :py:class:`CheckpointHandler`."""
  if not inspect.isclass(arg):
    arg = type(arg)
  if not issubclass(arg, CheckpointArgs):
    raise TypeError(f'{arg} must be a subclass of `CheckpointArgs`.')
  if arg not in _SAVE_ARG_TO_HANDLER and arg not in _RESTORE_ARG_TO_HANDLER:
    raise ValueError(
        f'Unable to find registered `CheckpointHandler` for {arg}. Use'
        ' `register_with_handler`.'
    )
  if arg in _SAVE_ARG_TO_HANDLER:
    return _SAVE_ARG_TO_HANDLER[arg]
  else:
    return _RESTORE_ARG_TO_HANDLER[arg]


def get_registered_args_cls(
    handler: Union[Type[CheckpointHandler], CheckpointHandler]
) -> Tuple[Type[CheckpointArgs], Type[CheckpointArgs]]:
  """Returns the registered CheckpointArgs corresponding to the handler.

  Args:
    handler: `CheckpointHandler` instance or class.

  Returns:
    Tuple of (save, restore) `CheckpointArgs` classes.
  """
  save_args = None
  restore_args = None
  if not inspect.isclass(handler):
    handler = type(handler)
  for arg_cls, handler_cls in _SAVE_ARG_TO_HANDLER.items():
    if handler_cls == handler:
      save_args = arg_cls
      break
  if save_args is None:
    raise ValueError(
        f'Unable to find registered `CheckpointArgs` for save for {handler}.'
    )
  for arg_cls, handler_cls in _RESTORE_ARG_TO_HANDLER.items():
    if handler_cls == handler:
      restore_args = arg_cls
      break
  if restore_args is None:
    raise ValueError(
        f'Unable to find registered `CheckpointArgs` for restore for {handler}.'
    )
  return save_args, restore_args


def has_registered_args(
    handler: Union[Type[CheckpointHandler], CheckpointHandler]
) -> bool:
  try:
    get_registered_args_cls(handler)
  except ValueError:
    return False
  return True

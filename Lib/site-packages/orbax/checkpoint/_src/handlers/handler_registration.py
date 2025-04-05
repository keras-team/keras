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

"""Registry for checkpoint handlers (:py:class:`CheckpointHandler`).

A checkpoint handler registry is used to register a
:py:class:`CheckpointHandler` for specific a checkpointable item name and a
:py:class:`CheckpointArgs` type. When saving or restoring a checkpoint, the
registry can be used to find the appropriate handler for the item and its args.
The default implementation of the registry is
:py:class:`DefaultCheckpointHandlerRegistry`. See the docstring for the default
implementation for an example how to use the registry.
"""

from typing import MutableMapping, Optional, Protocol, Type, Union

from orbax.checkpoint import checkpoint_args
from orbax.checkpoint._src.handlers import checkpoint_handler

CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandler = checkpoint_handler.CheckpointHandler
HandlerRegistryMapping = MutableMapping[
    tuple[Optional[str], type[CheckpointArgs]], CheckpointHandler
]


class CheckpointHandlerRegistry(Protocol):
  """Protocol for `CheckpointHandlerRegistry`.

  A checkpoint handler registry is used to register
  :py:class:`CheckpointHandler`s for specific item names and
  :py:class:`CheckpointArgs` types. When saving or restoring a checkpoint, the
  registry can be used to find the appropriate handler for the item and its
  args. See :py:class:`DefaultCheckpointHandlerRegistry` for the default
  implementation.
  """

  def add(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
      handler: Optional[
          Union[CheckpointHandler, Type[CheckpointHandler]]
      ] = None,
      **kwargs,
  ) -> None:
    """Adds an entry to the registry."""
    ...

  def get(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> CheckpointHandler:
    """Gets an entry from the registry."""
    ...

  def has(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> bool:
    """Checks if an entry exists in the registry."""
    ...

  def get_all_entries(
      self,
  ) -> HandlerRegistryMapping:
    ...


class AlreadyExistsError(ValueError):
  """Raised when an entry already exists in the registry."""


class NoEntryError(KeyError):
  """Raised when no entry exists in the registry."""


def _get_args_type(
    args: Union[Type[CheckpointArgs], CheckpointArgs],
) -> type[CheckpointArgs]:
  if isinstance(args, type):
    return args
  else:
    return type(args)


class DefaultCheckpointHandlerRegistry(CheckpointHandlerRegistry):
  """Default implementation of :py:class:`CheckpointHandlerRegistry`.

  Usage::
    from orbax.checkpoint import handlers

    # Create a handler registry.
    registry = handlers.DefaultCheckpointHandlerRegistry()

    # Add custom save and restore args for a checkpointable item.
    registry.add(
        'item',
        FooCustomCheckpointSaveArgs,
        FooCustomCheckpointHandler,
    )
    registry.add(
        'item',
        FooCustomCheckpoinRestoreArgs,
        FooCustomCheckpointHandler,
    )

    # Retrieve the handler for the item and args.
    handler = registry.get('item', FooCustomCheckpointSaveArgs)
    assert isinstance(handler, FooCustomCheckpointHandler)


    # You can also register a handler for a general args type.
    registry.add(
        None,
        BarCustomCheckpointArgs,
        BarCustomCheckpointHandler,
    )

    # Retrieve the handler for the general args type (for an item which has
    # not been registered).
    handler = registry.get(None, BarCustomCheckpointSaveArgs)
    assert isinstance(handler, BarCustomCheckpointHandler)

    # Associate a `CheckpointArgs` (and `CheckpointHandler`) with a given item
    # name, provided that the `CheckpointHandler` is globally registered.
    registry.add('state', ocp.args.StandardSave)
    registry.add('state', ocp.args.StandardRestore)
  """

  def __init__(
      self, other_registry: Optional[CheckpointHandlerRegistry] = None
  ):
    self._registry: HandlerRegistryMapping = {}

    # Initialize the registry with entries from other registry.
    if other_registry:
      for (
          item,
          args_type,
      ), handler in other_registry.get_all_entries().items():
        self.add(item, args_type, handler)

  def add(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
      handler: Optional[
          Union[CheckpointHandler, Type[CheckpointHandler]]
      ] = None,
  ):
    """Adds an entry to the registry.

    Args:
      item: The item name. If None, the entry will be added as a general `arg`
        entry.
      args: The args type to be added to registry. If a concerete type is
        provided, the type will be added to the registry.
      handler: The handler. If a type is provided, an instance of the type will
        be added to the registry. If None, tries to find a globally registered
        handler for the given args type.

    Raises:
      AlreadyExistsError: If an entry for the given item and args type already
        exists in the registry.
      ValueError: If no globally registered handler can be found for the given
        args type and handler is provided as None or if the handler is globally
        registered but not default-constructible.
    """
    args_type = _get_args_type(args)

    if self.has(item, args_type):
      raise AlreadyExistsError(
          f'Entry for item={item} and args_type={args_type} already'
          ' exists in the registry.'
      )
    if handler is None:
      try:
        handler_to_register = checkpoint_args.get_registered_handler_cls(args)
      except ValueError as e:
        raise ValueError(
            'Provided a `None` `CheckpointHandler` when registering'
            f' item={item} and args_type={args_type}. Unable to find a globally'
            ' registered handler for the given args type. Provide a `handler`'
            ' or ensure the args type corresponds to a globally registered'
            ' handler.'
        ) from e
    else:
      handler_to_register = handler

    if isinstance(handler_to_register, type):
      try:
        handler_instance = handler_to_register()
      except TypeError as e:
        raise ValueError(
            'Provided a `None` `CheckpointHandler` when registering'
            f' item={item} and args_type={args_type}. The globally-registered'
            ' handler for this args type is not default-constructible. Please'
            ' a `handler` when adding this args type to the registry.'
        ) from e
    else:
      handler_instance = handler_to_register

    self._registry[(item, args_type)] = handler_instance

  def get(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> CheckpointHandler:
    """Returns the handler for the given item and args type.

    Args:
      item: The item name. If None, the entry will be added as a general type
        `args` entry.
      args: The args type to get the handler for. If a concerete type is
        provided, the type will be used to get the handler.

    Raises:
      NoEntryError: If no entry for the given item and args exists in the
        registry.
    """
    args_type = _get_args_type(args)

    if self.has(item, args_type):
      return self._registry[(item, args_type)]

    raise NoEntryError(
        f'No entry for item={item} and args_ty={args_type} in the registry.'
    )

  def has(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> bool:
    """Returns whether an entry for the given item and args type exists in the registry.

    Args:
      item: The item name or None.
      args: The args type to check for. If a concrete type is provided, the type
        will be used to check for the entry.
    """
    args_type = _get_args_type(args)
    return (
        item,
        args_type,
    ) in self._registry

  def get_all_entries(
      self,
  ) -> HandlerRegistryMapping:
    """Returns all entries in the registry."""
    return self._registry

  def __repr__(self):
    return f'DefaultCheckpointHandlerRegistry({self._registry})'

  def __str__(self):
    return f'DefaultCheckpointHandlerRegistry({self._registry})'


def create_default_handler_registry(
    **items_to_handlers: CheckpointHandler,
) -> CheckpointHandlerRegistry:
  """Creates a registry given a mapping of item names to handlers."""
  registry = DefaultCheckpointHandlerRegistry()
  for item_name, handler in items_to_handlers.items():
    save_args_cls, restore_args_cls = checkpoint_args.get_registered_args_cls(
        handler
    )
    registry.add(item_name, save_args_cls, handler)
    registry.add(item_name, restore_args_cls, handler)
  return registry

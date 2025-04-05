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

"""A global registry of checkpoint handler types."""

from typing import Type

from absl import logging
from orbax.checkpoint._src.handlers import checkpoint_handler


CheckpointHandler = checkpoint_handler.CheckpointHandler


class HandlerTypeRegistry:
  """A registry mapping handler type strings to handler types."""

  def __init__(self):
    self._registry = {}

  def add(self, handler_type: Type[CheckpointHandler]) -> None:
    """Adds an entry to the registry."""
    try:
      handler_typestr = handler_type.typestr()
    except AttributeError:
      handler_typestr = f'{handler_type.__module__}.{handler_type.__qualname__}'
      logging.vlog(
          1,
          'Handler class %s does not have a typestr method. '
          'Using the default typestr value "%s" instead.',
          handler_type,
          handler_typestr,
      )
    if handler_typestr in self._registry:
      previous_handler_type = self._registry[handler_typestr]
      # On Colab/notebook, it's very common to reload modules when iterating
      # on the code (e.g. with `importlib.reload` or by re-executing cells).
      # This re-creates and re-registers a new handler class everytime.
      has_same_fully_qualified_name = (
          previous_handler_type.__module__ == handler_type.__module__
          and previous_handler_type.__qualname__ == handler_type.__qualname__
      )
      # If the fully-qualified-name changes then raise error.
      if not has_same_fully_qualified_name:
        raise ValueError(
            f'Handler type string "{handler_typestr}" already exists in the '
            f'registry with type {previous_handler_type}. '
            f'Cannot add type {handler_type}.'
        )
      # If both fully-qualified-name and type ref are the same then skip.
      if previous_handler_type == handler_type:
        logging.vlog(
            1,
            'Handler "%s" already exists in the registry with associated type'
            ' %s. Skipping registration.',
            handler_typestr,
            handler_type,
        )
        return
      # If fully-qualified-name is the same but type ref has changed then
      # it is okay to overwrite registry with the new type ref.
      logging.vlog(
          1,
          'Handler "%s" already exists in the registry with associated type'
          ' %s. Overwriting it as the module was recreated (likely from '
          'Colab reload',
          handler_typestr,
          handler_type,
      )
    self._registry[handler_typestr] = handler_type

  def get(
      self,
      handler_typestr: str,
  ) -> Type[CheckpointHandler]:
    """Gets an entry from the registry."""
    if handler_typestr not in self._registry:
      raise KeyError(
          f'Handler type string "{handler_typestr}" not found in the registry.'
      )
    return self._registry[handler_typestr]


_GLOBAL_HANDLER_TYPE_REGISTRY = HandlerTypeRegistry()


def register_handler_type(handler_cls):
  """Registers a checkpoint handler type in the global registry.

  The registry is keyed by the handler's typestr. If the handler does not
  provide a typestr, the default typestr is resolved from the handler's
  module and class name.

  Args:
    handler_cls: The checkpoint handler class to register.

  Returns:
    The registered checkpoint handler class.
  """
  _GLOBAL_HANDLER_TYPE_REGISTRY.add(handler_cls)
  return handler_cls


def get_handler_type(handler_typestr: str) -> Type[CheckpointHandler]:
  return _GLOBAL_HANDLER_TYPE_REGISTRY.get(handler_typestr)

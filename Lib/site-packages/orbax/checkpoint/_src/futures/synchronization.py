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

"""Synchronization utilities for futures."""

import enum
import itertools


class HandlerAwaitableSignal(enum.Enum):
  """Defines signals that may be awaited by a `CheckpointHandler`.

  Signals may be passed from `CheckpointManager` or `Checkpointer` layers to
  `CheckpointHandler or below.`

  Attributes:
    AWAITABLE_SIGNALS_CONTRACT: Contract that contains a list of signals that
      may be sent and can be awaited by the handlers.
    STEP_DIRECTORY_CREATION: When recieved, indicates that the step directory
      has been created. The handler should not attempt to write files before the
      directory is created.
    ITEM_DIRECTORY_CREATION: When recieved, indicates that the item directory
      has been created. The handler should not attempt to write files before the
      directory is created.
  """

  AWAITABLE_SIGNALS_CONTRACT = "awaitable_signals_contract"
  STEP_DIRECTORY_CREATION = "step_directory_creation"
  ITEM_DIRECTORY_CREATION = "item_directory_creation"


class HandlerAwaitableSignalOperationIdGenerator:
  """A unique operation id generator for `HandlerAwaitableSignal`."""

  _operation_id_counter = itertools.count()
  _operation_id = next(_operation_id_counter)

  @classmethod
  def next_operation_id(cls) -> int:
    """Increments the operation id counter and returns the new value."""
    cls._operation_id = next(cls._operation_id_counter)
    return cls._operation_id

  @classmethod
  def get_current_operation_id(cls) -> str:
    """Returns the current operation id."""
    return str(cls._operation_id)

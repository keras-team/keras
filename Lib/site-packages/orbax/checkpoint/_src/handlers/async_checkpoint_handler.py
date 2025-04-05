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

"""AsyncCheckpointHandler interface."""

import abc
from typing import List, Optional

from etils import epath
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import checkpoint_handler


class AsyncCheckpointHandler(checkpoint_handler.CheckpointHandler):
  """An interface providing async methods that can be used with CheckpointHandler."""

  @abc.abstractmethod
  async def async_save(
      self, directory: epath.Path, *args, **kwargs
  ) -> Optional[List[future.Future]]:
    """Constructs a save operation.

    Synchronously awaits a copy of the item, before returning commit futures
    necessary to save the item.

    Args:
      directory: the directory to save to.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """
    pass

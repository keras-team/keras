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

"""Shorthand for `Checkpointer(PyTreeCheckpointHandler())`."""

from typing import Optional
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler


class PyTreeCheckpointer(checkpointer.Checkpointer):
  """Shorthand class.

  Instead of::
    ckptr = Checkpointer(PyTreeCheckpointHandler())

  we can use::
    ckptr = PyTreeCheckpointer()
  """

  def __init__(
      self,
      primary_host: Optional[int] = 0,
      use_ocdbt: bool = True,
      use_zarr3=False,
  ):
    super().__init__(
        pytree_checkpoint_handler.PyTreeCheckpointHandler(
            use_ocdbt=use_ocdbt, use_zarr3=use_zarr3
        ),
        multiprocessing_options=options_lib.MultiprocessingOptions(
            primary_host=primary_host
        ),
    )

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

"""Defines types for `CheckpointableHandler`."""

from typing import Awaitable, Protocol, TypeVar
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types


T = TypeVar('T')
AbstractT = TypeVar('AbstractT')
MetadataT = TypeVar('MetadataT')

PathLike = path_types.PathLike
AsyncResponse = async_types.AsyncResponse


class CheckpointableHandler(Protocol[T, AbstractT, MetadataT]):
  """An interface that defines save/load logic for a `checkpointable` object.

  NOTE: Prefer to use `Checkpointable` interface when possible.

  A "checkpointable" is a fundamental concept in Orbax. A “checkpointable”
  refers to a logical piece of the checkpoint that is distinct in some way from
  other pieces. Checkpointables are separable; they may or may not be loaded
  concurrently and some may be omitted from the checkpoint entirely.
  Checkpointables are often represented by different types, and have different
  representations on disk. The quintessential example is model params vs.
  dataset.

  A PyTree of arrays, representing model parameters, is the most basic
  "checkpointable". A singular array is also a checkpointable.

  In most contexts, when dealing with just a PyTree, the API of choice is::

    ocp.save_pytree(directory, pytree)

  The concept of "checkpointable" is not so obvious in this case. When dealing
  with multiple objects, we can use::

    ocp.save_checkpointables(
        directory,
        dict(
            pytree=model_params,
            dataset=dataset_iterator,
            # other checkpointables, e.g. extra metadata, etc.
        ),
    )

  Now, it is easy to simply skip loading the dataset, as is commonly desired
  when running evals or inference::

    ocp.load_checkpointables(
        directory,
        dict(
            pytree=abstract_model_params,
        ),
    )
    # Equivalently,
    ocp.load_pytree(directory, abstract_model_params)

  With the methods defined in this Protocol (`save`, `load`),
  logic within the method itself is executed in the main thread,
  in a blocking fashion. Additional logic can be executed in the background by
  returning an `Awaitable` function (which itself may return a result).

  TODO(b/398249409) Include more details on implementing this Protocol.
  """

  async def save(
      self, directory: path_types.PathLike, checkpointable: T
  ) -> Awaitable[None]:
    """Saves the given `checkpointable` to the given `directory`.

    Save should perform any operations that need to block the main thread, such
    as device-to-host copying of on-device arrays. It then creates a background
    operation to continue writing the object to the storage location.

    Args:
      directory: The directory to save the checkpoint to.
      checkpointable: The checkpointable object to save.

    Returns:
      An `Awaitable`. This object represents the result of the save
      operation running in the background.
    """
    ...

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: AbstractT | None = None,
  ) -> Awaitable[T]:
    """Loads the checkpointable from the given `directory`.

    Args:
      directory: The directory to load the checkpoint from.
      abstract_checkpointable: An optional abstract representation of the
        checkpointable to load. If provided, this is used to provide properties
        to guide the restoration logic of the checkpoint. In the case of arrays,
        for example, this conveys properties like shape and dtype, for casting
        and reshaping. In some cases, no information is needed, and `AbstractT`
        may always be None. In other cases, the abstract representation may be a
        hard requirement for loading.

    Returns:
      An `Awaitable` that continues to load the checkpointable in the background
      and returns the loaded checkpointable when complete.
    """
    ...

  async def metadata(
      self, directory: path_types.PathLike
  ) -> MetadataT:
    """Returns the metadata for the given `directory`.

    The logic in this method must be executed fully in the main thread; metadata
    access is expected to be cheap and fast.

    Args:
      directory: The directory where the checkpoint is located.

    Returns:
      MetadataT: The metadata is an `MetadataT`,
      which is the abstract representation of
      the checkpointable. `MetadataT` differs from `AbstractT` in that it may
      contain additional properties that cannot be directly consumed to
      customize loading behavior, but are nevertheless present and useful
      to know about in some cases.
    """
    ...

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

"""Defines free-function interface for loading."""

import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def load_pytree(
    directory: path_types.PathLike,
    abstract_pytree: (
        tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
    ) = None,
) -> tree_types.PyTreeOf[tree_types.LeafType]:
  """Loads a PyTree.

  The operation blocks until complete. For improved performance, consider using
  `load_async` instead.

  Args:
    directory: The directory to load the checkpoint from.
    abstract_pytree: Provides a tree structure for the checkpoint to be restored
      into. May be omitted to load exactly as saved, but this is much more
      brittle than providing the tree. Providing the `abstract_tree` guarantees
      two things: (1) The restored tree will exactly match the structure of
      `abstract_pytree` (or raise an error if it is impossible to guarantee
      this). For example, if `abstract_pytree` is a custom object registered as
      a PyTree, the checkpoint will be restored as the same object, if possible.
      (2) The leaves of the restored tree will be restored with the properties
      indicated by the abstract leaves. For example, if a leaf in
      `abstract_pytree` is a `jax.ShapeDtypeStruct`, the restored leaf will be a
      `jax.Array` with the same shape and dtype. Each `AbstractLeafType` has a

  Returns:
    The restored PyTree.
  """
  context = context_lib.get_context()

  if context.pytree_options.loading.partial_load:
    raise NotImplementedError('Partial loading is not yet supported.')


  handler_registry = ocp.handlers.create_default_handler_registry(
      pytree=pytree_handler.create_v0_handler(context)
  )
  ckptr = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
  )
  args = ocp.args.Composite(
      pytree=pytree_handler.create_v0_restore_args(context, abstract_pytree)
  )
  return ckptr.restore(directory, args=args).pytree


def load_pytree_async(
    directory: path_types.PathLike,
    abstract_pytree: (
        tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
    ) = None,
) -> async_types.AsyncResponse[tree_types.PyTreeOf[tree_types.LeafType]]:
  """Loads a PyTree asynchronously. Not yet implemented."""
  del directory, abstract_pytree
  raise NotImplementedError('Asynchronous loading is not yet supported.')

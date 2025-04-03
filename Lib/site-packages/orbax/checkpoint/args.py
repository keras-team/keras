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

"""Defines exported :py:class:`CheckpointArgs` classes.

:py:class:`CheckpointHandler` subclasses define logic used to save and restore
an object to and from a checkpoint. Each :py:class:`CheckpointHandler`
has corresponding :py:class:`SaveArgs` and :py:class:`RestoreArgs`
classes that define the arguments used to call the handler.

The `ocp.args` module provides a complete definition of these classes. Refer to
`ocp.handlers` for more information on the handlers themselves.
"""

# pylint: disable=g-importing-member, g-bad-import-order, unused-import

# Built-in CheckpointArgs.
from orbax.checkpoint._src.handlers.array_checkpoint_handler import ArrayRestoreArgs as ArrayRestore
from orbax.checkpoint._src.handlers.array_checkpoint_handler import ArraySaveArgs as ArraySave
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeArgs as Composite
from orbax.checkpoint._src.handlers.json_checkpoint_handler import JsonRestoreArgs as JsonRestore
from orbax.checkpoint._src.handlers.json_checkpoint_handler import JsonSaveArgs as JsonSave
from orbax.checkpoint._src.handlers.proto_checkpoint_handler import ProtoRestoreArgs as ProtoRestore
from orbax.checkpoint._src.handlers.proto_checkpoint_handler import ProtoSaveArgs as ProtoSave
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeRestoreArgs as PyTreeRestore
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeSaveArgs as PyTreeSave
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardRestoreArgs as StandardRestore
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardSaveArgs as StandardSave
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import JaxRandomKeySaveArgs as JaxRandomKeySave
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import JaxRandomKeyRestoreArgs as JaxRandomKeyRestore
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import NumpyRandomKeySaveArgs as NumpyRandomKeySave
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import NumpyRandomKeyRestoreArgs as NumpyRandomKeyRestore

# For defining custom CheckpointArgs.
from orbax.checkpoint.checkpoint_args import CheckpointArgs
from orbax.checkpoint.checkpoint_args import get_registered_handler_cls
from orbax.checkpoint.checkpoint_args import get_registered_args_cls
from orbax.checkpoint.checkpoint_args import has_registered_args
from orbax.checkpoint.checkpoint_args import register_with_handler

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

"""Defines exported :py:class:`CheckpointHandler`s and their arguments.

:py:class:`CheckpointHandler` subclasses define logic used to save and restore
an object
to and from a checkpoint. Each :py:class:`CheckpointHandler` has corresponding
:py:class:`SaveArgs`
and :py:class:`RestoreArgs` classes that define the arguments used to call the
handler.
Prefer to use `ocp.args` to reference these objects.
"""

# pylint: disable=g-importing-member, unused-import, g-bad-import-order

from orbax.checkpoint._src.handlers.checkpoint_handler import CheckpointHandler
from orbax.checkpoint._src.handlers.async_checkpoint_handler import AsyncCheckpointHandler

from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers.handler_registration import DefaultCheckpointHandlerRegistry
from orbax.checkpoint._src.handlers.handler_registration import CheckpointHandlerRegistry
from orbax.checkpoint._src.handlers.handler_registration import create_default_handler_registry

from orbax.checkpoint._src.handlers.array_checkpoint_handler import ArrayCheckpointHandler
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeCheckpointHandler
from orbax.checkpoint._src.handlers.json_checkpoint_handler import JsonCheckpointHandler
from orbax.checkpoint._src.handlers.proto_checkpoint_handler import ProtoCheckpointHandler
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeCheckpointHandler
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import JaxRandomKeyCheckpointHandler
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import NumpyRandomKeyCheckpointHandler
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardCheckpointHandler

from orbax.checkpoint._src.handlers.array_checkpoint_handler import ArraySaveArgs
from orbax.checkpoint._src.handlers.array_checkpoint_handler import ArrayRestoreArgs
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeArgs
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeResults
from orbax.checkpoint._src.handlers.json_checkpoint_handler import JsonSaveArgs
from orbax.checkpoint._src.handlers.json_checkpoint_handler import JsonRestoreArgs
from orbax.checkpoint._src.handlers.proto_checkpoint_handler import ProtoSaveArgs
from orbax.checkpoint._src.handlers.proto_checkpoint_handler import ProtoRestoreArgs
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeSaveArgs
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeRestoreArgs
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import JaxRandomKeySaveArgs
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import JaxRandomKeyRestoreArgs
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import NumpyRandomKeySaveArgs
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import NumpyRandomKeyRestoreArgs
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardSaveArgs
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardRestoreArgs

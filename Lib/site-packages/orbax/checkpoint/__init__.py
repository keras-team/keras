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

"""Defines exported symbols for the namespace package `orbax.checkpoint`."""

# pylint: disable=g-importing-member, g-bad-import-order

import contextlib
import functools

from orbax.checkpoint import arrays
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import checkpointers
from orbax.checkpoint import checkpoint_managers
from orbax.checkpoint import handlers
from orbax.checkpoint import logging
from orbax.checkpoint import metadata
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import options
from orbax.checkpoint import path
from orbax.checkpoint import serialization
from orbax.checkpoint import transform_utils
from orbax.checkpoint import tree
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint import version
# TODO(cpgaffney): Import the public multihost API.
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import step

from orbax.checkpoint.future import Future

from orbax.checkpoint.transform_utils import apply_transformations
from orbax.checkpoint.transform_utils import merge_trees
from orbax.checkpoint.transform_utils import RestoreTransform
from orbax.checkpoint.transform_utils import Transform

from orbax.checkpoint.abstract_checkpoint_manager import AbstractCheckpointManager
from orbax.checkpoint.checkpointers import *
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.checkpoint_manager import AsyncOptions
from orbax.checkpoint.checkpoint_manager import CheckpointManagerOptions

from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import RestoreArgs
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import ArrayRestoreArgs
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import SaveArgs

# Important handlers.
from orbax.checkpoint._src.handlers.checkpoint_handler import CheckpointHandler
from orbax.checkpoint._src.handlers.async_checkpoint_handler import AsyncCheckpointHandler

from orbax.checkpoint._src.handlers.array_checkpoint_handler import ArrayCheckpointHandler
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeCheckpointHandler
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeOptions
from orbax.checkpoint._src.handlers.json_checkpoint_handler import JsonCheckpointHandler
from orbax.checkpoint._src.handlers.proto_checkpoint_handler import ProtoCheckpointHandler
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeCheckpointHandler
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import JaxRandomKeyCheckpointHandler
from orbax.checkpoint._src.handlers.random_key_checkpoint_handler import NumpyRandomKeyCheckpointHandler
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardCheckpointHandler

from orbax.checkpoint._src.handlers.handler_registration import CheckpointHandlerRegistry
from orbax.checkpoint._src.handlers.handler_registration import DefaultCheckpointHandlerRegistry

# This class should be regarded as internal-only, and may be removed without
# warning.
from orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler import BasePyTreeCheckpointHandler

# Test utils.
from orbax.checkpoint import test_utils

# A new PyPI release will be pushed everytime `__version__` is increased.
__version__ = version.__version__
del version

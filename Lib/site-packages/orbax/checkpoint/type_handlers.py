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

"""Public symbols for type_handlers module."""

# pylint: disable=g-importing-member, unused-import, g-bad-import-order

from orbax.checkpoint._src.serialization.type_handlers import ArrayHandler
from orbax.checkpoint._src.serialization.type_handlers import ArrayRestoreArgs
from orbax.checkpoint._src.serialization.type_handlers import NumpyHandler
from orbax.checkpoint._src.serialization.type_handlers import ParamInfo
from orbax.checkpoint._src.serialization.type_handlers import RestoreArgs
from orbax.checkpoint._src.serialization.type_handlers import SaveArgs
from orbax.checkpoint._src.serialization.type_handlers import ScalarHandler
from orbax.checkpoint._src.serialization.type_handlers import SingleReplicaArrayHandler
from orbax.checkpoint._src.serialization.type_handlers import SingleReplicaArrayRestoreArgs
from orbax.checkpoint._src.serialization.type_handlers import StringHandler
from orbax.checkpoint._src.serialization.type_handlers import TypeHandler

# TypeHandler Registry
from orbax.checkpoint._src.serialization.type_handlers import TypeHandlerRegistry
from orbax.checkpoint._src.serialization.type_handlers import create_type_handler_registry
from orbax.checkpoint._src.serialization.type_handlers import get_type_handler
from orbax.checkpoint._src.serialization.type_handlers import has_type_handler
from orbax.checkpoint._src.serialization.type_handlers import register_standard_handlers_with_options
from orbax.checkpoint._src.serialization.type_handlers import register_type_handler
from orbax.checkpoint._src.serialization.type_handlers import supported_types

from orbax.checkpoint._src.serialization.type_handlers import is_ocdbt_checkpoint

# BEGIN OF DEPRECATED FUNCTIONS
# DON'T USE UNLESS YOU KNOW WHAT YOU'RE DOING
from orbax.checkpoint._src.serialization.type_handlers import _assert_parameter_files_exist
from orbax.checkpoint._src.serialization.type_handlers import _get_json_tspec
from orbax.checkpoint._src.serialization.type_handlers import check_input_arguments
from orbax.checkpoint._src.serialization.type_handlers import get_cast_tspec_deserialize
from orbax.checkpoint._src.serialization.type_handlers import get_cast_tspec_serialize
from orbax.checkpoint._src.serialization.type_handlers import get_json_tspec_read
from orbax.checkpoint._src.serialization.type_handlers import get_json_tspec_write
from orbax.checkpoint._src.serialization.type_handlers import get_ts_context
from orbax.checkpoint._src.serialization.type_handlers import is_supported_empty_value
from orbax.checkpoint._src.serialization.type_handlers import is_supported_type
from orbax.checkpoint._src.serialization.type_handlers import merge_ocdbt_per_process_files
# END OF DEPRECATED FUNCTIONS

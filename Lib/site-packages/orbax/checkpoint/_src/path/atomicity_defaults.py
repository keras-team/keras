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

"""Utilities for interacting with the atomicity module.

Mainly provides helper functions like `get_default_temporary_path_class` that
help interacting with the `atomicity` module, but can have larger dependencies
than we would want to introduce into the base `atomicity` module.
"""

from typing import Type
from etils import epath
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import step as step_lib


def get_item_default_temporary_path_class(
    final_path: epath.Path,
) -> Type[atomicity_types.TemporaryPath]:
  """Returns the default temporary path class for a given sub-item path."""
  if step_lib.is_gcs_path(final_path):
    return atomicity.CommitFileTemporaryPath
  else:
    return atomicity.AtomicRenameTemporaryPath


def get_default_temporary_path_class(
    final_path: epath.Path,
) -> Type[atomicity_types.TemporaryPath]:
  """Returns the default temporary path class for a given checkpoint path."""
  if step_lib.is_gcs_path(final_path):
    return atomicity.CommitFileTemporaryPath
  else:
    return atomicity.AtomicRenameTemporaryPath

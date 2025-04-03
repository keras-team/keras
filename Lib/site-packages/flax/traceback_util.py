# Copyright 2024 The Flax Authors.
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

"""Flax specific traceback_util functions."""

from jax._src import traceback_util as jax_traceback_util

from flax import config

# pylint: disable=protected-access

# Globals:
# Whether to filter flax frames from traceback.
_flax_filter_tracebacks = config.flax_filter_frames
# Flax specific set of paths to exclude from tracebacks.
_flax_exclusions = set()


# re-import JAX symbol for convenience.
api_boundary = jax_traceback_util.api_boundary


def register_exclusion(path):
  """Marks a Flax source file for exclusion."""
  global _flax_exclusions, _flax_filter_tracebacks
  # Record flax exclusions so we can dynamically add and remove them.
  _flax_exclusions.add(path)
  if _flax_filter_tracebacks:
    jax_traceback_util.register_exclusion(path)


def hide_flax_in_tracebacks():
  """Hides Flax internal stack frames in tracebacks."""
  global _flax_exclusions, _flax_filter_tracebacks
  _flax_filter_tracebacks = True
  for exclusion in _flax_exclusions:
    if exclusion not in jax_traceback_util._exclude_paths:
      jax_traceback_util._exclude_paths.append(exclusion)


def show_flax_in_tracebacks():
  """Shows Flax internal stack frames in tracebacks."""
  global _flax_exclusions, _flax_filter_tracebacks
  _flax_filter_tracebacks = False
  for exclusion in _flax_exclusions:
    if exclusion in jax_traceback_util._exclude_paths:
      jax_traceback_util._exclude_paths.remove(exclusion)

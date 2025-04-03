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

"""Provides async variants of path functions."""

from typing import Any

from etils import epath
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.path import step as step_lib

# TODO(b/360190539): Why not use just asyncio.to_thread?


# TODO(b/360190539): This functionality should be provided by either an external
# library or Orbax should subclass epath.Path.
def async_makedirs(
    path: epath.Path,
    *args,
    parents: bool = False,
    exist_ok: bool = True,
    **kwargs,
):
  return asyncio_utils.as_async_function(path.mkdir)(
      *args, parents=parents, exist_ok=exist_ok, **kwargs
  )




def async_write_bytes(path: epath.Path, data: Any):
  return asyncio_utils.as_async_function(path.write_bytes)(data)


def async_exists(path: epath.Path):
  return asyncio_utils.as_async_function(path.exists)()


def async_stat(path: epath.Path):
  return asyncio_utils.as_async_function(path.stat)()


def async_rmtree(path: epath.Path):
  return asyncio_utils.as_async_function(path.rmtree)()


def async_is_tmp_checkpoint(path: epath.Path):
  return asyncio_utils.as_async_function(step_lib.is_tmp_checkpoint)(path)

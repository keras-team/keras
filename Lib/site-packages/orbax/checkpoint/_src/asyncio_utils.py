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

"""Provides helper async functions."""

import asyncio
import functools
from typing import Any, Coroutine, TypeVar
import nest_asyncio


_T = TypeVar('_T')


def as_async_function(func):
  """Wraps a function to make it async."""

  @functools.wraps(func)
  async def run(*args, loop=None, executor=None, **kwargs):
    if loop is None:
      loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, partial_func)

  return run


def run_sync(
    coro: Coroutine[Any, Any, _T],
    enable_nest_asyncio: bool = True,  # For testing.
) -> _T:
  """Runs a coroutine and returns the result."""
  try:
    asyncio.get_running_loop()  # no event loop: ~0.001s, otherwise: ~0.182s
    if enable_nest_asyncio:
      nest_asyncio.apply()  # patch asyncio globally in a runtime (idempotent).
  except RuntimeError:
    pass
  return asyncio.run(coro)

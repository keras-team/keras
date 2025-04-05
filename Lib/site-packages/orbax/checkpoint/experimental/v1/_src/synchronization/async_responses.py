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

"""Defines implementations of AsyncResponse."""

from typing import Coroutine, Sequence, TypeVar

from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import types


AsyncResponse = types.AsyncResponse

T = TypeVar('T')


class _DefaultAsyncResponse(AsyncResponse[T]):
  """Represents the result of a background function call.

  May send signals to indicate that the commit has completed. Can also receive
  signals to indicate that the commit should proceed.
  """

  def __init__(
      self,
      coroutine: Coroutine[None, None, T],
      *,
      name: str | None = None,
      send_signals: (
          Sequence[synchronization.HandlerAwaitableSignal] | None
      ) = None,
      receive_signals: (
          Sequence[synchronization.HandlerAwaitableSignal] | None
      ) = None,
      timeout_secs: int = 600,
      operation_id: str | None = None,
  ):
    """Constructor.

    Args:
      coroutine: An coroutine (defined with e.g. `async def foo(a, b)`). The
        awaitable function is passed as `foo(a, b)`, without awaiting, since
        this will be done in the background thread.
      name: The name of the thread.
      send_signals: Signals to send to indicate that the commit has completed.
      receive_signals: Signals to wait for before proceeding with the commit.
      timeout_secs: Timeout in seconds for waiting for signals.
      operation_id: The operation id to use for the barrier keys. If None, the
        current operation id is used.
    """
    send_signals = send_signals or []
    receive_signals = receive_signals or []
    self._t = future._SignalingThread(
        send_signals=send_signals,
        receive_signals=receive_signals,
        timeout_secs=timeout_secs,
        operation_id=operation_id,
        target=lambda: asyncio_utils.run_sync(coroutine),
        name=name,
    )
    self._t.start()

  def result(self, timeout: float | None = None) -> T:
    """Waits for the commit to complete."""
    return self._t.result(timeout=timeout)


class _AsyncResponseAwaitingContractedSignals(AsyncResponse[T]):
  """Represents the result of a background function call.

  May send signals to indicate that the function has completed. Waits for all
  awaitable signals in the `AWAITABLE_SIGNALS_CONTRACT` to be set before
  proceeding with the commit.
  """

  def __init__(
      self,
      coroutine: Coroutine[None, None, T],
      *,
      name: str | None = None,
      send_signals: (
          Sequence[synchronization.HandlerAwaitableSignal] | None
      ) = None,
      timeout_secs: int = 600,
      operation_id: str | None = None,
  ):
    """Constructor.

    Synchronously gets all awaitable signals in the contract and waits to
    receive them in background before proceeding with the commit if JAX
    distributed client is initialized.

    Args:
      coroutine: An awaitable (defined with e.g. `async def foo(a, b)`). The
        awaitable function is passed as `foo(a, b)`, without awaiting, since
        this will be done in the background thread.
      name: The name of the thread.
      send_signals: Signals to send to indicate that the commit has completed.
      timeout_secs: Timeout in seconds for waiting for signals.
      operation_id: The operation id to use for the barrier keys. If None, the
        current operation id is used.
    """
    receive_signals = []
    if multihost.is_jax_distributed_client_initialized():
      receive_signals = future.get_awaitable_signals_from_contract()
    self._response = _DefaultAsyncResponse[T](
        coroutine,
        name=name,
        send_signals=send_signals,
        receive_signals=receive_signals,
        timeout_secs=timeout_secs,
        operation_id=operation_id,
    )

  def result(self, timeout: float | None = None) -> T:
    return self._response.result(timeout=timeout)

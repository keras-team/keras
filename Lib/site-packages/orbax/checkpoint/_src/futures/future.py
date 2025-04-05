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

"""Futures that can be used for signaling for synchronization."""

import threading
import time
from typing import Any, Callable, Coroutine, Optional, Sequence

from absl import logging
import jax
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost
from typing_extensions import Protocol

_SIGNAL_ACTION_SUCCESS = 'signal_action_success'


def _get_unique_barrier_key(
    signal: synchronization.HandlerAwaitableSignal, operation_id: str
) -> str:
  """Returns a unique barrier key for the signal.

  Args:
    signal: The signal to generate a barrier key for.
    operation_id: The operation id to use as a suffix for the barrier key.
  """
  return multihost.unique_barrier_key(signal.value, suffix=operation_id)


def get_awaitable_signals_from_contract() -> (
    Sequence[synchronization.HandlerAwaitableSignal]
):
  """Gets the awaitable signals that may be sent for the current operation id.

  Returns:
    A list of awaitable signals that may be sent for the current operation id.
  """
  client = multihost.get_jax_distributed_client()
  barrier_key = _get_unique_barrier_key(
      synchronization.HandlerAwaitableSignal.AWAITABLE_SIGNALS_CONTRACT,
      synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id(),
  )
  try:
    values_str = str(client.key_value_try_get(barrier_key))
    return [
        synchronization.HandlerAwaitableSignal(value)
        for value in values_str.split(',')
    ]
  except jax.errors.JaxRuntimeError:
    # If the key is not found, then there are no awaitable signals yet.
    return []


def add_to_awaitable_signals_contract(
    signals: Sequence[synchronization.HandlerAwaitableSignal],
):
  """Adds awaitable signals to `AWAITABLE_SIGNALS_CONTRACT` for lower checkpointing layers to wait on.

  These signals are added to the list of awaitable signals for the current
  operation id in `HandlerAwaitableSignalOperationIdGenerator`.

  Args:
    signals: The signals to add to the list of awaitable signals.
  """
  if not signals:
    return

  current_signals = list(get_awaitable_signals_from_contract())
  current_signals.extend(signals)
  keys = ','.join([current_signal.value for current_signal in current_signals])
  client = multihost.get_jax_distributed_client()
  barrier_key = _get_unique_barrier_key(
      synchronization.HandlerAwaitableSignal.AWAITABLE_SIGNALS_CONTRACT,
      synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id(),
  )
  client.key_value_set(barrier_key, keys, allow_overwrite=True)


class Future(Protocol):
  """Abstracted Orbax Future class.

  This is used to represent the return value of
  AsyncCheckpointHandler.async_save. This method may return multiple related,
  but potentially distinct, future objects. Common examples may include
  tensorstore.Future or concurrent.futures.Future. Since these types are not
  strictly related to one another, we merely enforce that any returned future
  must have a `result` method which blocks until the future's operation
  completes. Importantly, calling `result` should not *start* execution of the
  future, but merely wait for an ongoing operation to complete.
  """

  def result(self, timeout: Optional[int] = None) -> Any:
    """Waits for the future to complete its operation."""
    ...


class NoopFuture:

  def result(self, timeout: Optional[int] = None) -> Any:
    del timeout
    return None


class ChainedFuture:
  """A future representing a sequence of multiple futures."""

  def __init__(self, futures: Sequence[Future], cb: Callable[[], None]):
    self._futures = futures
    self._cb = cb

  def result(self, timeout: Optional[int] = None) -> Any:
    """Waits for all futures to complete."""
    n = len(self._futures)
    start = time.time()
    time_remaining = timeout
    for k, f in enumerate(self._futures):
      f.result(timeout=time_remaining)
      if time_remaining is not None:
        time_elapsed = time.time() - start
        time_remaining -= time_elapsed
        if time_remaining <= 0:
          raise TimeoutError(
              'ChainedFuture completed {:d}/{:d} futures but timed out after'
              ' {:.2f} seconds.'.format(k, n, time_elapsed)
          )
    time_elapsed = time.time() - start
    logging.info(
        'ChainedFuture completed %d/%d futures in %.2f seconds.',
        n,
        n,
        time_elapsed,
    )
    self._cb()


class _SignalingThread(threading.Thread):
  """Thread that raises an exception if it encounters an error.

  Waits for signals to be received for the current operation id before
  proceeding with the target function. Then sends signals to indicate that the
  target function has completed with the same operation id.
  """

  _exception: Optional[Exception] = None

  def __init__(
      self,
      *,
      target: Callable[[], Any],
      send_signals: Sequence[synchronization.HandlerAwaitableSignal],
      receive_signals: Sequence[synchronization.HandlerAwaitableSignal],
      timeout_secs: int = 600,
      operation_id: str | None = None,
      **kwargs,
  ):
    """Constructor.

    Args:
      target: The target function to run. Takes no arguments.
      send_signals: Signals to send to indicate that the target function has
        completed.
      receive_signals: Signals to wait for before proceeding with the target
        function.
      timeout_secs: Timeout in seconds for waiting for signals.
      operation_id: The operation id to use for the barrier keys. If None, the
        current operation id is used.
      **kwargs: Keyword arguments passed to the base class.
    """
    self._result = None

    def _target_setting_result():
      self._result = target()

    super().__init__(target=_target_setting_result, **kwargs)
    self._send_signals = send_signals
    self._receive_signals = receive_signals
    self._timeout_secs = timeout_secs
    # Capture the current operation id synchronously.
    self._operation_id = (
        operation_id
        or synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id()
    )

  def _wait_for_signals(self):
    """Waits for signals to be set."""
    for signal in self._receive_signals:
      logging.vlog(
          1,
          '[process=%d][thread=%s] Waiting for <%s> timeout: %d secs to be set',
          multihost.process_index(),
          threading.current_thread().name,
          signal.value,
          self._timeout_secs,
      )
      barrier_key = _get_unique_barrier_key(signal, self._operation_id)
      client = multihost.get_jax_distributed_client()
      client.blocking_key_value_get(barrier_key, self._timeout_secs * 1000)

  def _set_signals(self):
    """Sets the barrier keys for the signals using send_signals."""
    for signal in self._send_signals:
      logging.vlog(
          1,
          '[process=%d][thread=%s] Signalling completion of <%s>.',
          multihost.process_index(),
          threading.current_thread().name,
          signal.value,
      )
      barrier_key = _get_unique_barrier_key(signal, self._operation_id)
      client = multihost.get_jax_distributed_client()
      client.key_value_set(barrier_key, _SIGNAL_ACTION_SUCCESS)

  def run(self):
    """Runs the target function after waiting for signals."""
    try:
      self._wait_for_signals()
      super().run()
      self._set_signals()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._exception = e

  def join(self, timeout: Optional[float] = None):
    """Waits for the target function to complete."""
    super().join(timeout=timeout)
    if self.is_alive():
      raise TimeoutError(
          f'Thread {self.name} did not complete within {timeout} seconds.'
      )
    if self._exception is not None:
      raise self._exception

  def result(self, timeout: Optional[float] = None):
    self.join(timeout=timeout)
    return self._result


class CommitFuture(Future):
  """Represents the result of a background commit.

  May send signals to indicate that the commit has completed. Can also receive
  signals to indicate that the commit should proceed.
  """

  def __init__(
      self,
      coro: Coroutine[Any, Any, None],
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
      coro: The coroutine to run.
      name: The name of the thread.
      send_signals: Signals to send to indicate that the commit has completed.
      receive_signals: Signals to wait for before proceeding with the commit.
      timeout_secs: Timeout in seconds for waiting for signals.
      operation_id: The operation id to use for the barrier keys. If None, the
        current operation id is used.
    """
    super().__init__()
    send_signals = send_signals or []
    receive_signals = receive_signals or []
    self._t = _SignalingThread(
        send_signals=send_signals,
        receive_signals=receive_signals,
        timeout_secs=timeout_secs,
        operation_id=operation_id,
        target=lambda: asyncio_utils.run_sync(coro),
        name=name,
    )
    self._t.start()

  def result(self, timeout: Optional[float] = None) -> Any:
    """Waits for the commit to complete."""
    return self._t.join(timeout=timeout)


class CommitFutureAwaitingContractedSignals(Future):
  """Represents the result of a background commit.

  May send signals to indicate that the commit has completed. Waits for all
  awaitable signals in the `AWAITABLE_SIGNALS_CONTRACT` to be set before
  proceeding with the commit.
  """

  def __init__(
      self,
      coro: Coroutine[Any, Any, None],
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
      coro: The coroutine to run.
      name: The name of the thread.
      send_signals: Signals to send to indicate that the commit has completed.
      timeout_secs: Timeout in seconds for waiting for signals.
      operation_id: The operation id to use for the barrier keys. If None, the
        current operation id is used.
    """
    super().__init__()
    receive_signals = []
    if multihost.is_jax_distributed_client_initialized():
      receive_signals = get_awaitable_signals_from_contract()
    self._f = CommitFuture(
        coro,
        name=name,
        send_signals=send_signals,
        receive_signals=receive_signals,
        timeout_secs=timeout_secs,
        operation_id=operation_id,
    )

  def result(self, timeout: Optional[float] = None) -> Any:
    return self._f.result(timeout=timeout)

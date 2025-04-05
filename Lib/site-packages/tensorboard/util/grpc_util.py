# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for working with python gRPC stubs."""


import enum
import functools
import random
import threading
import time

import grpc

from tensorboard import version
from tensorboard.util import tb_logging

logger = tb_logging.get_logger()

# Default RPC timeout.
_GRPC_DEFAULT_TIMEOUT_SECS = 30

# Max number of times to attempt an RPC, retrying on transient failures.
_GRPC_RETRY_MAX_ATTEMPTS = 5

# Parameters to control the exponential backoff behavior.
_GRPC_RETRY_EXPONENTIAL_BASE = 2
_GRPC_RETRY_JITTER_FACTOR_MIN = 1.1
_GRPC_RETRY_JITTER_FACTOR_MAX = 1.5

# Status codes from gRPC for which it's reasonable to retry the RPC.
_GRPC_RETRYABLE_STATUS_CODES = frozenset(
    [
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.UNAVAILABLE,
    ]
)

# gRPC metadata key whose value contains the client version.
_VERSION_METADATA_KEY = "tensorboard-version"


class AsyncCallFuture:
    """Encapsulates the future value of a retriable async gRPC request.

    Abstracts over the set of futures returned by a set of gRPC calls
    comprising a single logical gRPC request with retries.  Communicates
    to the caller the result or exception resulting from the request.

    Args:
      completion_event: The constructor should provide a `threding.Event` which
        will be used to communicate when the set of gRPC requests is complete.
    """

    def __init__(self, completion_event):
        self._active_grpc_future = None
        self._active_grpc_future_lock = threading.Lock()
        self._completion_event = completion_event

    def _set_active_future(self, grpc_future):
        if grpc_future is None:
            raise RuntimeError(
                "_set_active_future invoked with grpc_future=None."
            )
        with self._active_grpc_future_lock:
            self._active_grpc_future = grpc_future

    def result(self, timeout):
        """Analogous to `grpc.Future.result`. Returns the value or exception.

        This method will wait until the full set of gRPC requests is complete
        and then act as `grpc.Future.result` for the single gRPC invocation
        corresponding to the first successful call or final failure, as
        appropriate.

        Args:
          timeout: How long to wait in seconds before giving up and raising.

        Returns:
          The result of the future corresponding to the single gRPC
          corresponding to the successful call.

        Raises:
          * `grpc.FutureTimeoutError` if timeout seconds elapse before the gRPC
          calls could complete, including waits and retries.
          * The exception corresponding to the last non-retryable gRPC request
          in the case that a successful gRPC request was not made.
        """
        if not self._completion_event.wait(timeout):
            raise grpc.FutureTimeoutError(
                f"AsyncCallFuture timed out after {timeout} seconds"
            )
        with self._active_grpc_future_lock:
            if self._active_grpc_future is None:
                raise RuntimeError("AsyncFuture never had an active future set")
            return self._active_grpc_future.result()


def async_call_with_retries(api_method, request, clock=None):
    """Initiate an asynchronous call to a gRPC stub, with retry logic.

    This is similar to the `async_call` API, except that the call is handled
    asynchronously, and the completion may be handled by another thread. The
    caller must provide a `done_callback` argument which will handle the
    result or exception rising from the gRPC completion.

    Retries are handled with jittered exponential backoff to spread out failures
    due to request spikes.

    This only supports unary-unary RPCs: i.e., no streaming on either end.

    Args:
      api_method: Callable for the API method to invoke.
      request: Request protocol buffer to pass to the API method.
      clock: an interface object supporting `time()` and `sleep()` methods
        like the standard `time` module; if not passed, uses the normal module.

    Returns:
      An `AsyncCallFuture` which will encapsulate the `grpc.Future`
      corresponding to the gRPC call which either completes successfully or
      represents the final try.
    """
    if clock is None:
        clock = time
    logger.debug("Async RPC call %s with request: %r", api_method, request)

    completion_event = threading.Event()
    async_future = AsyncCallFuture(completion_event)

    def async_call(handler):
        """Invokes the gRPC future and orchestrates it via the AsyncCallFuture."""
        future = api_method.future(
            request,
            timeout=_GRPC_DEFAULT_TIMEOUT_SECS,
            metadata=version_metadata(),
        )
        # Ensure we set the active future before invoking the done callback, to
        # avoid the case where the done callback completes immediately and
        # triggers completion event while async_future still holds the old
        # future.
        async_future._set_active_future(future)
        future.add_done_callback(handler)

    # retry_handler is the continuation of the `async_call`.  It should:
    #   * If the grpc call succeeds: trigger the `completion_event`.
    #   * If there are no more retries: trigger the `completion_event`.
    #   * Otherwise, invoke a new async_call with the same
    #     retry_handler.
    def retry_handler(future, num_attempts):
        e = future.exception()
        if e is None:
            completion_event.set()
            return
        else:
            logger.info("RPC call %s got error %s", api_method, e)
            # If unable to retry, proceed to completion.
            if e.code() not in _GRPC_RETRYABLE_STATUS_CODES:
                completion_event.set()
                return
            if num_attempts >= _GRPC_RETRY_MAX_ATTEMPTS:
                completion_event.set()
                return
            # If able to retry, wait then do so.
            backoff_secs = _compute_backoff_seconds(num_attempts)
            clock.sleep(backoff_secs)
            async_call(
                functools.partial(retry_handler, num_attempts=num_attempts + 1)
            )

    async_call(functools.partial(retry_handler, num_attempts=1))
    return async_future


def _compute_backoff_seconds(num_attempts):
    """Compute appropriate wait time between RPC attempts."""
    jitter_factor = random.uniform(
        _GRPC_RETRY_JITTER_FACTOR_MIN, _GRPC_RETRY_JITTER_FACTOR_MAX
    )
    backoff_secs = (_GRPC_RETRY_EXPONENTIAL_BASE**num_attempts) * jitter_factor
    return backoff_secs


def call_with_retries(api_method, request, clock=None):
    """Call a gRPC stub API method, with automatic retry logic.

    This only supports unary-unary RPCs: i.e., no streaming on either end.
    Streamed RPCs will generally need application-level pagination support,
    because after a gRPC error one must retry the entire request; there is no
    "retry-resume" functionality.

    Retries are handled with jittered exponential backoff to spread out failures
    due to request spikes.

    Args:
      api_method: Callable for the API method to invoke.
      request: Request protocol buffer to pass to the API method.
      clock: an interface object supporting `time()` and `sleep()` methods
        like the standard `time` module; if not passed, uses the normal module.

    Returns:
      Response protocol buffer returned by the API method.

    Raises:
      grpc.RpcError: if a non-retryable error is returned, or if all retry
        attempts have been exhausted.
    """
    if clock is None:
        clock = time
    # We can't actually use api_method.__name__ because it's not a real method,
    # it's a special gRPC callable instance that doesn't expose the method name.
    rpc_name = request.__class__.__name__.replace("Request", "")
    logger.debug("RPC call %s with request: %r", rpc_name, request)
    num_attempts = 0
    while True:
        num_attempts += 1
        try:
            return api_method(
                request,
                timeout=_GRPC_DEFAULT_TIMEOUT_SECS,
                metadata=version_metadata(),
            )
        except grpc.RpcError as e:
            logger.info("RPC call %s got error %s", rpc_name, e)
            if e.code() not in _GRPC_RETRYABLE_STATUS_CODES:
                raise
            if num_attempts >= _GRPC_RETRY_MAX_ATTEMPTS:
                raise
        backoff_secs = _compute_backoff_seconds(num_attempts)
        logger.info(
            "RPC call %s attempted %d times, retrying in %.1f seconds",
            rpc_name,
            num_attempts,
            backoff_secs,
        )
        clock.sleep(backoff_secs)


def version_metadata():
    """Creates gRPC invocation metadata encoding the TensorBoard version.

    Usage: `stub.MyRpc(request, metadata=version_metadata())`.

    Returns:
      A tuple of key-value pairs (themselves 2-tuples) to be passed as the
      `metadata` kwarg to gRPC stub API methods.
    """
    return ((_VERSION_METADATA_KEY, version.VERSION),)


def extract_version(metadata):
    """Extracts version from invocation metadata.

    The argument should be the result of a prior call to `metadata` or the
    result of combining such a result with other metadata.

    Returns:
      The TensorBoard version listed in this metadata, or `None` if none
      is listed.
    """
    return dict(metadata).get(_VERSION_METADATA_KEY)


@enum.unique
class ChannelCredsType(enum.Enum):
    LOCAL = "local"
    SSL = "ssl"
    SSL_DEV = "ssl_dev"

    def channel_config(self):
        """Create channel credentials and options.

        Returns:
          A tuple `(channel_creds, channel_options)`, where `channel_creds`
          is a `grpc.ChannelCredentials` and `channel_options` is a
          (potentially empty) list of `(key, value)` tuples. Both results
          may be passed to `grpc.secure_channel`.
        """

        options = []
        if self == ChannelCredsType.LOCAL:
            creds = grpc.local_channel_credentials()
        elif self == ChannelCredsType.SSL:
            creds = grpc.ssl_channel_credentials()
        elif self == ChannelCredsType.SSL_DEV:
            # Configure the dev cert to use by passing the environment variable
            # GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=path/to/cert.crt
            creds = grpc.ssl_channel_credentials()
            options.append(("grpc.ssl_target_name_override", "localhost"))
        else:
            raise AssertionError("unhandled ChannelCredsType: %r" % self)
        return (creds, options)

    @classmethod
    def choices(cls):
        return cls.__members__.values()

    def __str__(self):
        # Use user-facing string, because this is shown for flag choices.
        return self.value

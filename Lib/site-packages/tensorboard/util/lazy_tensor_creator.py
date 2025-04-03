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
"""Provides a lazy wrapper for deferring Tensor creation."""

import threading

from tensorboard.compat import tf2 as tf


# Sentinel used for LazyTensorCreator._tensor to indicate that a value is
# currently being computed, in order to fail hard on reentrancy.
_CALL_IN_PROGRESS_SENTINEL = object()


class LazyTensorCreator:
    """Lazy auto-converting wrapper for a callable that returns a `tf.Tensor`.

    This class wraps an arbitrary callable that returns a `Tensor` so that it
    will be automatically converted to a `Tensor` by any logic that calls
    `tf.convert_to_tensor()`. This also memoizes the callable so that it is
    called at most once.

    The intended use of this class is to defer the construction of a `Tensor`
    (e.g. to avoid unnecessary wasted computation, or ensure any new ops are
    created in a context only available later on in execution), while remaining
    compatible with APIs that expect to be given an already materialized value
    that can be converted to a `Tensor`.

    This class is thread-safe.
    """

    def __init__(self, tensor_callable):
        """Initializes a LazyTensorCreator object.

        Args:
          tensor_callable: A callable that returns a `tf.Tensor`.
        """
        if not callable(tensor_callable):
            raise ValueError("Not a callable: %r" % tensor_callable)
        self._tensor_callable = tensor_callable
        self._tensor = None
        self._tensor_lock = threading.RLock()
        _register_conversion_function_once()

    def __call__(self):
        if self._tensor is None or self._tensor is _CALL_IN_PROGRESS_SENTINEL:
            with self._tensor_lock:
                if self._tensor is _CALL_IN_PROGRESS_SENTINEL:
                    raise RuntimeError(
                        "Cannot use LazyTensorCreator with reentrant callable"
                    )
                elif self._tensor is None:
                    self._tensor = _CALL_IN_PROGRESS_SENTINEL
                    self._tensor = self._tensor_callable()
        return self._tensor


def _lazy_tensor_creator_converter(value, dtype=None, name=None, as_ref=False):
    del name  # ignored
    if not isinstance(value, LazyTensorCreator):
        raise RuntimeError("Expected LazyTensorCreator, got %r" % value)
    if as_ref:
        raise RuntimeError("Cannot use LazyTensorCreator to create ref tensor")
    tensor = value()
    if dtype not in (None, tensor.dtype):
        raise RuntimeError(
            "Cannot convert LazyTensorCreator returning dtype %s to dtype %s"
            % (tensor.dtype, dtype)
        )
    return tensor


# Use module-level bit and lock to ensure that registration of the
# LazyTensorCreator conversion function happens only once.
_conversion_registered = False
_conversion_registered_lock = threading.Lock()


def _register_conversion_function_once():
    """Performs one-time registration of `_lazy_tensor_creator_converter`.

    This helper can be invoked multiple times but only registers the conversion
    function on the first invocation, making it suitable for calling when
    constructing a LazyTensorCreator.

    Deferring the registration is necessary because doing it at at module import
    time would trigger the lazy TensorFlow import to resolve, and that in turn
    would break the delicate `tf.summary` import cycle avoidance scheme.
    """
    global _conversion_registered
    if not _conversion_registered:
        with _conversion_registered_lock:
            if not _conversion_registered:
                _conversion_registered = True
                tf.register_tensor_conversion_function(
                    base_type=LazyTensorCreator,
                    conversion_func=_lazy_tensor_creator_converter,
                    priority=0,
                )

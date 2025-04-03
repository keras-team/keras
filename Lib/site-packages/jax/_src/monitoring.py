# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for instrumenting code.

Code points can be marked as a named event. Every time an event is reached
during program execution, the registered listeners will be invoked.

A typical listener callback is to send an event to a metrics collector for
aggregation/exporting.
"""

from __future__ import annotations

from typing import Protocol


class EventListenerWithMetadata(Protocol):

  def __call__(self, event: str, **kwargs: str | int) -> None:
    ...


class EventDurationListenerWithMetadata(Protocol):

  def __call__(self, event: str, duration_secs: float,
               **kwargs: str | int) -> None:
    ...


class EventTimeSpanListenerWithMetadata(Protocol):

  def __call__(
      self, event: str, start_time: float, end_time: float, **kwargs: str | int
  ) -> None:
    ...


_event_listeners: list[EventListenerWithMetadata] = []
_event_duration_secs_listeners: list[EventDurationListenerWithMetadata] = []
_event_time_span_listeners: list[EventTimeSpanListenerWithMetadata] = []


def record_event(event: str, **kwargs: str | int) -> None:
  """Record an event.

  If **kwargs are specified, all of the named arguments have to be passed in the
  same order across all invocations of this method for the same event.
  """
  for callback in _event_listeners:
    callback(event, **kwargs)


def record_event_duration_secs(event: str, duration: float,
                               **kwargs: str | int) -> None:
  """Record an event duration in seconds (float).

  If **kwargs are specified, all of the named arguments have to be passed in the
  same order across all invocations of this method for the same event.
  """
  for callback in _event_duration_secs_listeners:
    callback(event, duration, **kwargs)


def record_event_time_span(
    event: str, start_time: float, end_time: float, **kwargs: str | int
) -> None:
  """Record an event start and end time in seconds (float)."""
  for callback in _event_time_span_listeners:
    callback(event, start_time, end_time, **kwargs)


def register_event_listener(
    callback: EventListenerWithMetadata,
) -> None:
  """Register a callback to be invoked during record_event()."""
  _event_listeners.append(callback)


def register_event_time_span_listener(
    callback: EventTimeSpanListenerWithMetadata,
) -> None:
  """Register a callback to be invoked during record_event_time_span()."""
  _event_time_span_listeners.append(callback)


def register_event_duration_secs_listener(
    callback : EventDurationListenerWithMetadata) -> None:
  """Register a callback to be invoked during record_event_duration_secs()."""
  _event_duration_secs_listeners.append(callback)

def get_event_duration_listeners() -> list[EventDurationListenerWithMetadata]:
  """Get event duration listeners."""
  return list(_event_duration_secs_listeners)


def get_event_time_span_listeners() -> list[EventTimeSpanListenerWithMetadata]:
  """Get event time span listeners."""
  return list(_event_time_span_listeners)


def get_event_listeners() -> list[EventListenerWithMetadata]:
  """Get event listeners."""
  return list(_event_listeners)

def clear_event_listeners():
  """Clear event listeners."""
  global _event_listeners, _event_duration_secs_listeners, _event_time_span_listeners
  _event_listeners = []
  _event_duration_secs_listeners = []
  _event_time_span_listeners = []

def _unregister_event_duration_listener_by_callback(
    callback: EventDurationListenerWithMetadata) -> None:
  """Unregister an event duration listener by callback.

  This function is supposed to be called for testing only.
  """
  assert callback in _event_duration_secs_listeners
  _event_duration_secs_listeners.remove(callback)

def _unregister_event_duration_listener_by_index(index: int) -> None:
  """Unregister an event duration listener by index.

  This function is supposed to be called for testing only.
  """
  size = len(_event_duration_secs_listeners)
  assert -size <= index < size
  del _event_duration_secs_listeners[index]


def _unregister_event_time_span_listener_by_callback(
    callback: EventTimeSpanListenerWithMetadata,
) -> None:
  """Unregister an event time span listener by callback.

  This function is supposed to be called for testing only.
  """
  assert callback in _event_time_span_listeners
  _event_time_span_listeners.remove(callback)


def _unregister_event_listener_by_callback(
    callback: EventListenerWithMetadata) -> None:
  """Unregister an event listener by callback.

  This function is supposed to be called for testing only.
  """
  assert callback in _event_listeners
  _event_listeners.remove(callback)

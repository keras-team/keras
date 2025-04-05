# -*- coding: utf-8 -*-
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
"""Converts trace events to JSON format consumed by catapult trace viewer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six

# Values for type (ph) and s (scope) parameters in catapult trace format.
_TYPE_METADATA = 'M'
_TYPE_COMPLETE = 'X'
_TYPE_INSTANT = 'i'
_SCOPE_THREAD = 't'


class TraceEventsJsonStream(object):
  """A streaming trace file in the format expected by catapult trace viewer.

  Iterating over this yields a sequence of string chunks, so it is suitable for
  returning in a werkzeug Response.
  """

  def __init__(self, proto):
    """Create an iterable JSON stream over the supplied Trace.

    Args:
      proto: a tensorboard.profile.Trace protobuf
    """
    self._proto = proto

  def _events(self):
    """Iterator over all catapult trace events, as python values."""
    for did, device in sorted(six.iteritems(self._proto.devices)):
      if device.name:
        yield dict(
            ph=_TYPE_METADATA,
            pid=did,
            name='process_name',
            args=dict(name=device.name))
      yield dict(
          ph=_TYPE_METADATA,
          pid=did,
          name='process_sort_index',
          args=dict(sort_index=did))
      for rid, resource in sorted(six.iteritems(device.resources)):
        if resource.name:
          yield dict(
              ph=_TYPE_METADATA,
              pid=did,
              tid=rid,
              name='thread_name',
              args=dict(name=resource.name))
        yield dict(
            ph=_TYPE_METADATA,
            pid=did,
            tid=rid,
            name='thread_sort_index',
            args=dict(sort_index=rid))
    for event in self._proto.trace_events:
      yield self._event(event)

  def _event(self, event):
    """Converts a TraceEvent proto into a catapult trace event python value."""
    result = dict(
        pid=event.device_id,
        tid=event.resource_id,
        name=event.name,
        ts=event.timestamp_ps / 1000000.0)
    if event.duration_ps:
      result['ph'] = _TYPE_COMPLETE
      result['dur'] = event.duration_ps / 1000000.0
    else:
      result['ph'] = _TYPE_INSTANT
      result['s'] = _SCOPE_THREAD
    for key in dict(event.args):
      if 'args' not in result:
        result['args'] = {}
      result['args'][key] = event.args[key]
    return result

  def __iter__(self):
    """Returns an iterator of string chunks of a complete JSON document."""
    yield '{"displayTimeUnit":"ns","metadata":{"highres-ticks":true},\n'
    yield '"traceEvents":[\n'
    for event in self._events():
      yield json.dumps(event, sort_keys=True)
      yield ',\n'
    # Add one fake event to avoid dealing with no-trailing-comma rule.
    yield '{}]}\n'

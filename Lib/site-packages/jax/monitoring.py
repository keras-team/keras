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

from jax._src.monitoring import (
    clear_event_listeners as clear_event_listeners,
    record_event_duration_secs as record_event_duration_secs,
    record_event_time_span as record_event_time_span,
    record_event as record_event,
    register_event_duration_secs_listener as register_event_duration_secs_listener,
    register_event_listener as register_event_listener,
    register_event_time_span_listener as register_event_time_span_listener,
)

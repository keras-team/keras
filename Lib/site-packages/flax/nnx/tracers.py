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

# Taken from flax/core/tracer.py 🏴‍☠️


import jax
import jax.core
import treescope  # type: ignore[import-not-found,import-untyped]

from flax.nnx import reprlib


def current_jax_trace():
  """Returns the Jax tracing state."""
  if jax.__version_info__ <= (0, 4, 33):
    return jax.core.thread_local_state.trace_state.trace_stack.dynamic
  return jax.core.get_opaque_trace_state(convention="nnx")


class TraceState(reprlib.Representable):
  __slots__ = ['_jax_trace']

  def __init__(self):
    self._jax_trace = current_jax_trace()

  @property
  def jax_trace(self):
    return self._jax_trace

  def is_valid(self) -> bool:
    # TODO: re-enable when we update nnx to use stackless trace context
    return True

  def __nnx_repr__(self):
    yield reprlib.Object(f'{type(self).__name__}')
    yield reprlib.Attr('jax_trace', self._jax_trace)

  def __treescope_repr__(self, path, subtree_renderer):
    return treescope.repr_lib.render_object_constructor(
        object_type=type(self),
        attributes={'jax_trace': self._jax_trace},
        path=path,
        subtree_renderer=subtree_renderer,
    )

  def __eq__(self, other):
    if jax.__version_info__ <= (0, 4, 33):
      return isinstance(other, TraceState) and self._jax_trace is other._jax_trace

    return isinstance(other, TraceState) and self._jax_trace == other._jax_trace

  # pickle support
  def __getstate__(self):
    return {}

  def __setstate__(self, state):
    self._jax_trace = current_jax_trace()

def _flatten_trace_state(trace_state: TraceState):
  return (), None


def _unflatten_trace_state(_1, _2):
  return TraceState()


jax.tree_util.register_pytree_node(
  TraceState,
  _flatten_trace_state,
  _unflatten_trace_state,
)
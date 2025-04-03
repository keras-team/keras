# Copyright 2024 The JAX Authors.
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

from __future__ import annotations
import threading
from contextlib import contextmanager
from jax._src import config


class ComputeOnContext(threading.local):

  def __init__(self):
    self.stack = []

compute_on_context = ComputeOnContext()


@contextmanager
def extend_compute_type(c_type: str):
  compute_on_context.stack.append(c_type)
  config.compute_on_context_manager.set_local(
      tuple(compute_on_context.stack))
  try:
    if len(set(filter(lambda x: x is not None, set(compute_on_context.stack)))) > 1:
      raise NotImplementedError(
          'Nesting `compute_on` with different compute types is not supported'
          f' yet. Current stack: {compute_on_context.stack}')
    yield compute_on_context.stack[-1]
  finally:
    compute_on_context.stack.pop()
    config.compute_on_context_manager.set_local(tuple(compute_on_context.stack))

def current_compute_type() -> str | None:
  return compute_on_context.stack[-1] if compute_on_context.stack else None

def _check_valid(c_type: str):
  if c_type not in {'device_host', 'device', 'tpu_sparsecore'}:
    raise ValueError(
        'Invalid compute type received. Current supported values '
        f'are `device_host`, `device` and `tpu_sparsecore`. Got {c_type}')

@contextmanager
def compute_on(compute_type: str):
  if not isinstance(compute_type, str):
    raise TypeError("`compute_on`'s compute_type argument must be a string.")
  _check_valid(compute_type)

  with extend_compute_type(compute_type):
    yield

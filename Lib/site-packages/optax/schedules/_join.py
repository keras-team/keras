# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Utilities to join schedules."""

from typing import Sequence

import chex
import jax.numpy as jnp
from optax._src import base


def join_schedules(
    schedules: Sequence[base.Schedule], boundaries: Sequence[int]
) -> base.Schedule:
  """Sequentially apply multiple schedules.

  Args:
    schedules: A list of callables (expected to be optax schedules). Each
      schedule will receive a step count indicating the number of steps since
      the previous boundary transition.
    boundaries: A list of integers (of length one less than schedules) that
      indicate when to transition between schedules.

  Returns:
    schedule: A function that maps step counts to values.
  """

  def schedule(step: chex.Numeric) -> chex.Numeric:
    output = schedules[0](step)
    for boundary, schedule in zip(boundaries, schedules[1:]):
      output = jnp.where(step < boundary, output, schedule(step - boundary))
    return output

  return schedule

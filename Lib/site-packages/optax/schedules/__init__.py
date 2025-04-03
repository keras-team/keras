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
"""Utilities for creating schedules."""

# pylint: disable=g-importing-member

from optax._src.base import Schedule
from optax._src.base import StatefulSchedule
from optax.schedules._inject import inject_hyperparams
from optax.schedules._inject import inject_stateful_hyperparams
from optax.schedules._inject import InjectHyperparamsState
from optax.schedules._inject import InjectStatefulHyperparamsState
from optax.schedules._inject import WrappedSchedule
from optax.schedules._join import join_schedules
from optax.schedules._schedule import constant_schedule
from optax.schedules._schedule import cosine_decay_schedule
from optax.schedules._schedule import cosine_onecycle_schedule
from optax.schedules._schedule import exponential_decay
from optax.schedules._schedule import linear_onecycle_schedule
from optax.schedules._schedule import linear_schedule
from optax.schedules._schedule import piecewise_constant_schedule
from optax.schedules._schedule import piecewise_interpolate_schedule
from optax.schedules._schedule import polynomial_schedule
from optax.schedules._schedule import sgdr_schedule
from optax.schedules._schedule import warmup_constant_schedule
from optax.schedules._schedule import warmup_cosine_decay_schedule
from optax.schedules._schedule import warmup_exponential_decay_schedule

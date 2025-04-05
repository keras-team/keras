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
"""Hyper-parameters Schedules.

Schedules may be used to anneal the value of a hyper-parameter over time; for
instance, they may be used to anneal the learning rate used to update an agent's
parameters or the exploration factor used to select actions.
"""

from optax import schedules


# TODO(mtthss): remove schedules alises from flat namespaces after user updates.
constant_schedule = schedules.constant_schedule
cosine_decay_schedule = schedules.cosine_decay_schedule
cosine_onecycle_schedule = schedules.cosine_onecycle_schedule
exponential_decay = schedules.exponential_decay
inject_hyperparams = schedules.inject_hyperparams
InjectHyperparamsState = schedules.InjectHyperparamsState
join_schedules = schedules.join_schedules
linear_onecycle_schedule = schedules.linear_onecycle_schedule
linear_schedule = schedules.linear_schedule
piecewise_constant_schedule = schedules.piecewise_constant_schedule
piecewise_interpolate_schedule = schedules.piecewise_interpolate_schedule
polynomial_schedule = schedules.polynomial_schedule
sgdr_schedule = schedules.sgdr_schedule
warmup_constant_schedule = schedules.warmup_constant_schedule
warmup_cosine_decay_schedule = schedules.warmup_cosine_decay_schedule
warmup_exponential_decay_schedule = schedules.warmup_exponential_decay_schedule

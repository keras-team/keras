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
"""Utilities for efficient monte carlo gradient estimation."""

# pylint:disable=g-importing-member

from optax.monte_carlo.control_variates import control_delta_method
from optax.monte_carlo.control_variates import control_variates_jacobians
from optax.monte_carlo.control_variates import moving_avg_baseline
from optax.monte_carlo.stochastic_gradient_estimators import measure_valued_jacobians
from optax.monte_carlo.stochastic_gradient_estimators import pathwise_jacobians
from optax.monte_carlo.stochastic_gradient_estimators import score_function_jacobians

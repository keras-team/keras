# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""The projections sub-package."""

# pylint: disable=g-importing-member

from optax.projections._projections import projection_box
from optax.projections._projections import projection_hypercube
from optax.projections._projections import projection_l1_ball
from optax.projections._projections import projection_l1_sphere
from optax.projections._projections import projection_l2_ball
from optax.projections._projections import projection_l2_sphere
from optax.projections._projections import projection_linf_ball
from optax.projections._projections import projection_non_negative
from optax.projections._projections import projection_simplex

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
"""Contributed optimizers in Optax."""

# pylint: disable=g-importing-member

from optax.contrib._acprop import acprop
from optax.contrib._acprop import scale_by_acprop
from optax.contrib._cocob import cocob
from optax.contrib._cocob import COCOBState
from optax.contrib._cocob import scale_by_cocob
from optax.contrib._complex_valued import split_real_and_imaginary
from optax.contrib._complex_valued import SplitRealAndImaginaryState
from optax.contrib._dadapt_adamw import dadapt_adamw
from optax.contrib._dadapt_adamw import DAdaptAdamWState
from optax.contrib._dog import dog
from optax.contrib._dog import DoGState
from optax.contrib._dog import dowg
from optax.contrib._dog import DoWGState
from optax.contrib._mechanic import MechanicState
from optax.contrib._mechanic import mechanize
from optax.contrib._momo import momo
from optax.contrib._momo import momo_adam
from optax.contrib._momo import MomoAdamState
from optax.contrib._momo import MomoState
from optax.contrib._privacy import differentially_private_aggregate
from optax.contrib._privacy import DifferentiallyPrivateAggregateState
from optax.contrib._privacy import dpsgd
from optax.contrib._prodigy import prodigy
from optax.contrib._prodigy import ProdigyState
from optax.contrib._reduce_on_plateau import reduce_on_plateau
from optax.contrib._reduce_on_plateau import ReduceLROnPlateauState
from optax.contrib._sam import normalize
from optax.contrib._sam import NormalizeState
from optax.contrib._sam import sam
from optax.contrib._sam import SAMState
from optax.contrib._schedule_free import schedule_free
from optax.contrib._schedule_free import schedule_free_adamw
from optax.contrib._schedule_free import schedule_free_eval_params
from optax.contrib._schedule_free import schedule_free_sgd
from optax.contrib._schedule_free import ScheduleFreeState

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
"""Optax: composable gradient processing and optimization, in JAX."""

# pylint: disable=wrong-import-position
# pylint: disable=g-importing-member

from optax import assignment
from optax import contrib
from optax import losses
from optax import monte_carlo
from optax import perturbations
from optax import projections
from optax import schedules
from optax import second_order
from optax import transforms
from optax import tree_utils
from optax._src.alias import adabelief
from optax._src.alias import adadelta
from optax._src.alias import adafactor
from optax._src.alias import adagrad
from optax._src.alias import adam
from optax._src.alias import adamax
from optax._src.alias import adamaxw
from optax._src.alias import adamw
from optax._src.alias import adan
from optax._src.alias import amsgrad
from optax._src.alias import fromage
from optax._src.alias import lamb
from optax._src.alias import lars
from optax._src.alias import lbfgs
from optax._src.alias import lion
from optax._src.alias import MaskOrFn
from optax._src.alias import nadam
from optax._src.alias import nadamw
from optax._src.alias import noisy_sgd
from optax._src.alias import novograd
from optax._src.alias import optimistic_adam
from optax._src.alias import optimistic_gradient_descent
from optax._src.alias import polyak_sgd
from optax._src.alias import radam
from optax._src.alias import rmsprop
from optax._src.alias import rprop
from optax._src.alias import sgd
from optax._src.alias import sign_sgd
from optax._src.alias import sm3
from optax._src.alias import yogi
from optax._src.base import EmptyState
from optax._src.base import GradientTransformation
from optax._src.base import GradientTransformationExtraArgs
from optax._src.base import identity
from optax._src.base import OptState
from optax._src.base import Params
from optax._src.base import ScalarOrSchedule
from optax._src.base import Schedule
from optax._src.base import set_to_zero
from optax._src.base import stateless
from optax._src.base import stateless_with_tree_map
from optax._src.base import TransformInitFn
from optax._src.base import TransformUpdateExtraArgsFn
from optax._src.base import TransformUpdateFn
from optax._src.base import Updates
from optax._src.base import with_extra_args_support
from optax._src.clipping import adaptive_grad_clip
from optax._src.clipping import AdaptiveGradClipState
from optax._src.clipping import clip
from optax._src.clipping import clip_by_block_rms
from optax._src.clipping import clip_by_global_norm
from optax._src.clipping import ClipByGlobalNormState
from optax._src.clipping import ClipState
from optax._src.clipping import per_example_global_norm_clip
from optax._src.clipping import per_example_layer_norm_clip
from optax._src.combine import chain
from optax._src.combine import multi_transform
from optax._src.combine import MultiTransformState
from optax._src.combine import named_chain
from optax._src.constrain import keep_params_nonnegative
from optax._src.constrain import NonNegativeParamsState
from optax._src.constrain import zero_nans
from optax._src.constrain import ZeroNansState
from optax._src.factorized import FactoredState
from optax._src.factorized import scale_by_factored_rms
from optax._src.linear_algebra import global_norm
from optax._src.linear_algebra import matrix_inverse_pth_root
from optax._src.linear_algebra import power_iteration
from optax._src.linesearch import scale_by_backtracking_linesearch
from optax._src.linesearch import scale_by_zoom_linesearch
from optax._src.linesearch import ScaleByBacktrackingLinesearchState
from optax._src.linesearch import ScaleByZoomLinesearchState
from optax._src.linesearch import ZoomLinesearchInfo
from optax._src.lookahead import lookahead
from optax._src.lookahead import LookaheadParams
from optax._src.lookahead import LookaheadState
from optax._src.numerics import safe_increment
from optax._src.numerics import safe_int32_increment
from optax._src.numerics import safe_norm
from optax._src.numerics import safe_root_mean_squares
from optax._src.transform import add_decayed_weights
from optax._src.transform import add_noise
from optax._src.transform import AddDecayedWeightsState
from optax._src.transform import AddNoiseState
from optax._src.transform import apply_every
from optax._src.transform import ApplyEvery
from optax._src.transform import centralize
from optax._src.transform import ema
from optax._src.transform import EmaState
from optax._src.transform import normalize_by_update_norm
from optax._src.transform import scale
from optax._src.transform import scale_by_adadelta
from optax._src.transform import scale_by_adam
from optax._src.transform import scale_by_adamax
from optax._src.transform import scale_by_adan
from optax._src.transform import scale_by_amsgrad
from optax._src.transform import scale_by_belief
from optax._src.transform import scale_by_distance_over_gradients
from optax._src.transform import scale_by_lbfgs
from optax._src.transform import scale_by_learning_rate
from optax._src.transform import scale_by_lion
from optax._src.transform import scale_by_novograd
from optax._src.transform import scale_by_optimistic_gradient
from optax._src.transform import scale_by_param_block_norm
from optax._src.transform import scale_by_param_block_rms
from optax._src.transform import scale_by_polyak
from optax._src.transform import scale_by_radam
from optax._src.transform import scale_by_rms
from optax._src.transform import scale_by_rprop
from optax._src.transform import scale_by_rss
from optax._src.transform import scale_by_schedule
from optax._src.transform import scale_by_sign
from optax._src.transform import scale_by_sm3
from optax._src.transform import scale_by_stddev
from optax._src.transform import scale_by_trust_ratio
from optax._src.transform import scale_by_yogi
from optax._src.transform import ScaleByAdaDeltaState
from optax._src.transform import ScaleByAdamState
from optax._src.transform import ScaleByAdanState
from optax._src.transform import ScaleByAmsgradState
from optax._src.transform import ScaleByBeliefState
from optax._src.transform import ScaleByLBFGSState
from optax._src.transform import ScaleByLionState
from optax._src.transform import ScaleByNovogradState
from optax._src.transform import ScaleByRmsState
from optax._src.transform import ScaleByRpropState
from optax._src.transform import ScaleByRssState
from optax._src.transform import ScaleByRStdDevState
from optax._src.transform import ScaleByScheduleState
from optax._src.transform import ScaleBySM3State
from optax._src.transform import ScaleByTrustRatioState
from optax._src.transform import ScaleState
from optax._src.transform import trace
from optax._src.transform import TraceState
from optax._src.update import apply_updates
from optax._src.update import incremental_update
from optax._src.update import periodic_update
from optax._src.utils import multi_normal
from optax._src.utils import scale_gradient
from optax._src.utils import value_and_grad_from_state
from optax._src.wrappers import apply_if_finite
from optax._src.wrappers import ApplyIfFiniteState
from optax._src.wrappers import conditionally_mask
from optax._src.wrappers import conditionally_transform
from optax._src.wrappers import ConditionallyMaskState
from optax._src.wrappers import ConditionallyTransformState
from optax._src.wrappers import flatten
from optax._src.wrappers import masked
from optax._src.wrappers import MaskedNode
from optax._src.wrappers import MaskedState
from optax._src.wrappers import maybe_update
from optax._src.wrappers import MaybeUpdateState
from optax._src.wrappers import MultiSteps
from optax._src.wrappers import MultiStepsState
from optax._src.wrappers import ShouldSkipUpdateFunction
from optax._src.wrappers import skip_large_updates
from optax._src.wrappers import skip_not_finite


# TODO(mtthss): remove tree_utils aliases after updates.
tree_map_params = tree_utils.tree_map_params
bias_correction = tree_utils.tree_bias_correction
update_infinity_moment = tree_utils.tree_update_infinity_moment
update_moment = tree_utils.tree_update_moment
update_moment_per_elem_norm = tree_utils.tree_update_moment_per_elem_norm

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
inject_stateful_hyperparams = schedules.inject_stateful_hyperparams
InjectStatefulHyperparamsState = schedules.InjectStatefulHyperparamsState
WrappedSchedule = schedules.WrappedSchedule

# TODO(mtthss): remove loss aliases from flat namespace once users have updated.
convex_kl_divergence = losses.convex_kl_divergence
cosine_distance = losses.cosine_distance
cosine_similarity = losses.cosine_similarity
ctc_loss = losses.ctc_loss
ctc_loss_with_forward_probs = losses.ctc_loss_with_forward_probs
hinge_loss = losses.hinge_loss
huber_loss = losses.huber_loss
kl_divergence = losses.kl_divergence
l2_loss = losses.l2_loss
log_cosh = losses.log_cosh
ntxent = losses.ntxent
sigmoid_binary_cross_entropy = losses.sigmoid_binary_cross_entropy
smooth_labels = losses.smooth_labels
safe_softmax_cross_entropy = losses.safe_softmax_cross_entropy
softmax_cross_entropy = losses.softmax_cross_entropy
softmax_cross_entropy_with_integer_labels = (
    losses.softmax_cross_entropy_with_integer_labels
)
squared_error = losses.squared_error
sigmoid_focal_loss = losses.sigmoid_focal_loss

# pylint: disable=g-import-not-at-top
# TODO(mtthss): remove contrib aliases from flat namespace once users updated.
# Deprecated modules
from optax.contrib import differentially_private_aggregate as _deprecated_differentially_private_aggregate
from optax.contrib import DifferentiallyPrivateAggregateState as _deprecated_DifferentiallyPrivateAggregateState
from optax.contrib import dpsgd as _deprecated_dpsgd

_deprecations = {
    # Added Apr 2024
    "differentially_private_aggregate": (
        (
            "optax.differentially_private_aggregate is deprecated: use"
            " optax.contrib.differentially_private_aggregate (optax v0.1.8 or"
            " newer)."
        ),
        _deprecated_differentially_private_aggregate,
    ),
    "DifferentiallyPrivateAggregateState": (
        (
            "optax.DifferentiallyPrivateAggregateState is deprecated: use"
            " optax.contrib.DifferentiallyPrivateAggregateState (optax v0.1.8"
            " or newer)."
        ),
        _deprecated_DifferentiallyPrivateAggregateState,
    ),
    "dpsgd": (
        (
            "optax.dpsgd is deprecated: use optax.contrib.dpsgd (optax v0.1.8"
            " or newer)."
        ),
        _deprecated_dpsgd,
    ),
}
# pylint: disable=g-bad-import-order
import typing as _typing

if _typing.TYPE_CHECKING:
  # pylint: disable=reimported
  from optax.contrib import differentially_private_aggregate
  from optax.contrib import DifferentiallyPrivateAggregateState
  from optax.contrib import dpsgd
  # pylint: enable=reimported

else:
  from optax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
# pylint: enable=g-importing-member


__version__ = "0.2.4"

__all__ = (
    "adabelief",
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamaxw",
    "adamw",
    "adan",
    "adaptive_grad_clip",
    "AdaptiveGradClipState",
    "add_decayed_weights",
    "add_noise",
    "AddDecayedWeightsState",
    "AddNoiseState",
    "amsgrad",
    "apply_every",
    "apply_if_finite",
    "apply_updates",
    "ApplyEvery",
    "ApplyIfFiniteState",
    "assignment",
    "centralize",
    "chain",
    "clip_by_block_rms",
    "clip_by_global_norm",
    "clip",
    "ClipByGlobalNormState",
    "ClipState",
    "conditionally_mask",
    "ConditionallyMaskState",
    "conditionally_transform",
    "ConditionallyTransformState",
    "constant_schedule",
    "ctc_loss",
    "ctc_loss_with_forward_probs",
    "convex_kl_divergence",
    "cosine_decay_schedule",
    "cosine_distance",
    "cosine_onecycle_schedule",
    "cosine_similarity",
    "differentially_private_aggregate",
    "DifferentiallyPrivateAggregateState",
    "dpsgd",
    "ema",
    "EmaState",
    "EmptyState",
    "exponential_decay",
    "FactoredState",
    "flatten",
    "fromage",
    "global_norm",
    "GradientTransformation",
    "GradientTransformationExtraArgs",
    "hinge_loss",
    "huber_loss",
    "identity",
    "incremental_update",
    "inject_hyperparams",
    "InjectHyperparamsState",
    "join_schedules",
    "keep_params_nonnegative",
    "kl_divergence",
    "l2_loss",
    "lamb",
    "lars",
    "lbfgs",
    "lion",
    "linear_onecycle_schedule",
    "linear_schedule",
    "log_cosh",
    "lookahead",
    "LookaheadParams",
    "LookaheadState",
    "masked",
    "MaskOrFn",
    "MaskedState",
    "matrix_inverse_pth_root",
    "maybe_update",
    "MaybeUpdateState",
    "multi_normal",
    "multi_transform",
    "MultiSteps",
    "MultiStepsState",
    "MultiTransformState",
    "nadam",
    "nadamw",
    "noisy_sgd",
    "novograd",
    "NonNegativeParamsState",
    "ntxent",
    "OptState",
    "Params",
    "periodic_update",
    "per_example_global_norm_clip",
    "per_example_layer_norm_clip",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "polynomial_schedule",
    "power_iteration",
    "polyak_sgd",
    "radam",
    "rmsprop",
    "rprop",
    "safe_increment",
    "safe_int32_increment",
    "safe_norm",
    "safe_root_mean_squares",
    "ScalarOrSchedule",
    "scale_by_adadelta",
    "scale_by_adam",
    "scale_by_adamax",
    "scale_by_adan",
    "scale_by_amsgrad",
    "scale_by_backtracking_linesearch",
    "scale_by_belief",
    "scale_by_lbfgs",
    "scale_by_lion",
    "scale_by_factored_rms",
    "scale_by_novograd",
    "scale_by_param_block_norm",
    "scale_by_param_block_rms",
    "scale_by_polyak",
    "scale_by_radam",
    "scale_by_rms",
    "scale_by_rprop",
    "scale_by_rss",
    "scale_by_schedule",
    "scale_by_sign",
    "scale_by_sm3",
    "scale_by_stddev",
    "scale_by_trust_ratio",
    "scale_by_yogi",
    "scale_by_zoom_linesearch",
    "scale_gradient",
    "scale",
    "ScaleByAdaDeltaState",
    "ScaleByAdamState",
    "ScaleByAdanState",
    "ScaleByAmsgradState",
    "ScaleByBacktrackingLinesearchState",
    "ScaleByBeliefState",
    "ScaleByLBFGSState",
    "ScaleByLionState",
    "ScaleByNovogradState",
    "ScaleByRmsState",
    "ScaleByRpropState",
    "ScaleByRssState",
    "ScaleByRStdDevState",
    "ScaleByScheduleState",
    "ScaleBySM3State",
    "ScaleByTrustRatioState",
    "ScaleByZoomLinesearchState",
    "ScaleState",
    "Schedule",
    "set_to_zero",
    "sgd",
    "sgdr_schedule",
    "ShouldSkipUpdateFunction",
    "sigmoid_binary_cross_entropy",
    "sign_sgd",
    "skip_large_updates",
    "skip_not_finite",
    "sm3",
    "smooth_labels",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
    "stateless",
    "stateless_with_tree_map",
    "trace",
    "TraceState",
    "TransformInitFn",
    "TransformUpdateFn",
    "TransformUpdateExtraArgsFn",
    "Updates",
    "value_and_grad_from_state",
    "warmup_cosine_decay_schedule",
    "warmup_exponential_decay_schedule",
    "yogi",
    "zero_nans",
    "ZeroNansState",
    "ZoomLinesearchInfo",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Optax public API.   /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#

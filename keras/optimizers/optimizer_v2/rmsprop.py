# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""RMSprop optimizer implementation."""

import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend_config
from keras.optimizers.optimizer_v2 import optimizer_v2

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.optimizers.legacy.RMSprop",
    v1=["keras.optimizers.RMSprop", "keras.optimizers.legacy.RMSprop"],
)
class RMSprop(optimizer_v2.OptimizerV2):
    r"""Optimizer that implements the RMSprop algorithm.

    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average

    This implementation of RMSprop uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.001.
      rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
      momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory. Defaults to `False`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to `"RMSprop"`.
      **kwargs: keyword arguments. Allowed arguments are `clipvalue`,
        `clipnorm`, `global_clipnorm`.
        If `clipvalue` (float) is set, the gradient of each weight
        is clipped to be no higher than this value.
        If `clipnorm` (float) is set, the gradient of each weight
        is individually clipped so that its norm is no higher than this value.
        If `global_clipnorm` (float) is set the gradient of all weights is
        clipped so that their global norm is no higher than this value.

    Note that in the dense implementation of this algorithm, variables and their
    corresponding accumulators (momentum, gradient moving average, square
    gradient moving average) will be updated even if the gradient is zero
    (i.e. accumulators will decay, momentum will be applied). The sparse
    implementation (used when the gradient is an `IndexedSlices` object,
    typically because of `tf.gather` or an embedding lookup in the forward pass)
    will not update variable slices or their accumulators unless those slices
    were used in the forward pass (nor is there an "eventual" correction to
    account for these omitted updates). This leads to more efficient updates for
    large embedding lookup tables (where most of the slices are not accessed in
    a particular graph execution), but differs from the published algorithm.

    Usage:

    >>> opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0    # d(loss) / d(var1) = var1
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> var1.numpy()
    9.683772

    Reference:
      - [Hinton, 2012](
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        name="RMSprop",
        **kwargs,
    ):
        """Construct a new RMSprop optimizer.

        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is
            a `tf.keras.optimizers.schedules.LearningRateSchedule`, or a
            callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to 0.001.
          rho: Discounting factor for the history/coming gradient. Defaults to
            0.9.
          momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
            to 1e-7.
          centered: Boolean. If `True`, gradients are normalized by the
            estimated variance of the gradient; if False, by the uncentered
            second moment.  Setting this to `True` may help with training, but
            is slightly more expensive in terms of computation and memory.
            Defaults to `False`.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "RMSprop".
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
            `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is
            clip gradients by value, `decay` is included for backward
            compatibility to allow time inverse decay of learning rate. `lr` is
            included for backward compatibility, recommended to use
            `learning_rate` instead.

        @compatibility(eager)
        When eager execution is enabled, `learning_rate`, `decay`, `momentum`,
        and `epsilon` can each be a callable that takes no arguments and returns
        the actual value to use. This can be useful for changing these values
        across different invocations of optimizer functions.
        @end_compatibility
        """
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("rho", rho)

        self._momentum = False
        if (
            isinstance(momentum, tf.Tensor)
            or callable(momentum)
            or momentum > 0
        ):
            self._momentum = True
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError(
                "`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)})."
            )
        self._set_hyper("momentum", momentum)

        self.epsilon = epsilon or backend_config.epsilon()
        self.centered = centered

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.centered:
            for var in var_list:
                self.add_slot(var, "mg")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        rho = tf.identity(self._get_hyper("rho", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                rho=rho,
                momentum=tf.identity(self._get_hyper("momentum", var_dtype)),
                one_minus_rho=1.0 - rho,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return tf.raw_ops.ResourceApplyCenteredRMSProp(
                    var=var.handle,
                    mg=mg.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    use_locking=self._use_locking,
                )
            else:
                return tf.raw_ops.ResourceApplyRMSProp(
                    var=var.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    use_locking=self._use_locking,
                )
        else:
            rms_t = coefficients["rho"] * rms + coefficients[
                "one_minus_rho"
            ] * tf.square(grad)
            rms_t = tf.compat.v1.assign(
                rms, rms_t, use_locking=self._use_locking
            )
            denom_t = rms_t
            if self.centered:
                mg = self.get_slot(var, "mg")
                mg_t = (
                    coefficients["rho"] * mg
                    + coefficients["one_minus_rho"] * grad
                )
                mg_t = tf.compat.v1.assign(
                    mg, mg_t, use_locking=self._use_locking
                )
                denom_t = rms_t - tf.square(mg_t)
            var_t = var - coefficients["lr_t"] * grad / (
                tf.sqrt(denom_t) + coefficients["epsilon"]
            )
            return tf.compat.v1.assign(
                var, var_t, use_locking=self._use_locking
            ).op

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return tf.raw_ops.ResourceSparseApplyCenteredRMSProp(
                    var=var.handle,
                    mg=mg.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    indices=indices,
                    use_locking=self._use_locking,
                )
            else:
                return tf.raw_ops.ResourceSparseApplyRMSProp(
                    var=var.handle,
                    ms=rms.handle,
                    mom=mom.handle,
                    lr=coefficients["lr_t"],
                    rho=coefficients["rho"],
                    momentum=coefficients["momentum"],
                    epsilon=coefficients["epsilon"],
                    grad=grad,
                    indices=indices,
                    use_locking=self._use_locking,
                )
        else:
            rms_scaled_g_values = (grad * grad) * coefficients["one_minus_rho"]
            rms_t = tf.compat.v1.assign(
                rms, rms * coefficients["rho"], use_locking=self._use_locking
            )
            with tf.control_dependencies([rms_t]):
                rms_t = self._resource_scatter_add(
                    rms, indices, rms_scaled_g_values
                )
                rms_slice = tf.gather(rms_t, indices)
            denom_slice = rms_slice
            if self.centered:
                mg = self.get_slot(var, "mg")
                mg_scaled_g_values = grad * coefficients["one_minus_rho"]
                mg_t = tf.compat.v1.assign(
                    mg, mg * coefficients["rho"], use_locking=self._use_locking
                )
                with tf.control_dependencies([mg_t]):
                    mg_t = self._resource_scatter_add(
                        mg, indices, mg_scaled_g_values
                    )
                    mg_slice = tf.gather(mg_t, indices)
                    denom_slice = rms_slice - tf.square(mg_slice)
            var_update = self._resource_scatter_add(
                var,
                indices,
                coefficients["neg_lr_t"]
                * grad
                / (tf.sqrt(denom_slice) + coefficients["epsilon"]),
            )
            if self.centered:
                return tf.group(*[var_update, rms_t, mg_t])
            return tf.group(*[var_update, rms_t])

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._initial_decay,
                "rho": self._serialize_hyperparameter("rho"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config


RMSProp = RMSprop

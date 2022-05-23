# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Nadam optimizer implementation."""

import tensorflow.compat.v2 as tf
from keras import backend_config
from keras.optimizers.schedules import learning_rate_schedule
from keras.optimizers.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export


# pylint: disable=g-classes-have-attributes
@keras_export("keras.optimizers.Nadam")
class Nadam(optimizer_v2.OptimizerV2):
    r"""Optimizer that implements the NAdam algorithm.
    Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
    Nesterov momentum.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the exponentially weighted infinity norm.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to `"Nadam"`.
      **kwargs: keyword arguments. Allowed arguments are `clipvalue`,
        `clipnorm`, `global_clipnorm`.
        If `clipvalue` (float) is set, the gradient of each weight
        is clipped to be no higher than this value.
        If `clipnorm` (float) is set, the gradient of each weight
        is individually clipped so that its norm is no higher than this value.
        If `global_clipnorm` (float) is set the gradient of all weights is
        clipped so that their global norm is no higher than this value.

    Usage Example:
      >>> opt = tf.keras.optimizers.Nadam(learning_rate=0.2)
      >>> var1 = tf.Variable(10.0)
      >>> loss = lambda: (var1 ** 2) / 2.0
      >>> step_count = opt.minimize(loss, [var1]).numpy()
      >>> "{:.1f}".format(var1.numpy())
      9.8

    Reference:
      - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        name="Nadam",
        **kwargs
    ):
        # Backwards compatibility with keras NAdam optimizer.
        kwargs["decay"] = kwargs.pop("schedule_decay", 0.004)
        learning_rate = kwargs.get("lr", learning_rate)
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            raise ValueError(
                "The Nadam optimizer does not support "
                "tf.keras.optimizers.LearningRateSchedules as the "
                "learning rate."
            )

        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self._m_cache = None

    def _create_slots(self, var_list):
        var_dtype = var_list[0].dtype.base_dtype
        if self._m_cache is None:
            self._m_cache = self.add_weight(
                "momentum_cache",
                shape=[],
                dtype=var_dtype,
                initializer="ones",
                trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            self._weights.append(self._m_cache)
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            # Create slots for the first moments.
            self.add_slot(var, "m")
        for var in var_list:
            # Create slots for the second moments.
            self.add_slot(var, "v")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        lr_t = tf.identity(self._get_hyper("learning_rate", var_dtype))
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        local_step = tf.cast(self.iterations + 1, var_dtype)
        next_step = tf.cast(self.iterations + 2, var_dtype)

        decay_base = tf.cast(0.96, var_dtype)

        m_t = beta_1_t * (
            1.0 - 0.5 * (tf.pow(decay_base, self._initial_decay * local_step))
        )
        m_t_1 = beta_1_t * (
            1.0 - 0.5 * (tf.pow(decay_base, self._initial_decay * next_step))
        )

        m_schedule_new = tf.cast(self._m_cache_read, var_dtype) * m_t
        if var_dtype is self._m_cache.dtype:
            m_schedule_new = tf.identity(
                tf.compat.v1.assign(
                    self._m_cache, m_schedule_new, use_locking=self._use_locking
                )
            )
        m_schedule_next = m_schedule_new * m_t_1

        apply_state[(var_device, var_dtype)] = dict(
            lr_t=lr_t,
            neg_lr_t=-lr_t,  # pylint: disable=invalid-unary-operand-type
            epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_2_t=beta_2_t,
            m_t=m_t,
            m_t_1=m_t_1,
            one_minus_beta_1_t=1 - beta_1_t,
            one_minus_beta_2_t=1 - beta_2_t,
            one_minus_m_t=1.0 - m_t,
            one_minus_m_schedule_new=1.0 - m_schedule_new,
            one_minus_m_schedule_next=1.0 - m_schedule_next,
            v_t_prime_denominator=1.0 - tf.pow(beta_2_t, local_step),
        )

    def _prepare(self, var_list):
        # Get the value of the momentum cache before starting to apply gradients.
        self._m_cache_read = tf.identity(self._m_cache)
        return super()._prepare(var_list)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        g_prime = grad / coefficients["one_minus_m_schedule_new"]
        m_t = (
            coefficients["beta_1_t"] * m
            + coefficients["one_minus_beta_1_t"] * grad
        )
        m_t = tf.compat.v1.assign(m, m_t, use_locking=self._use_locking)
        m_t_prime = m_t / coefficients["one_minus_m_schedule_next"]
        v_t = coefficients["beta_2_t"] * v + coefficients[
            "one_minus_beta_2_t"
        ] * tf.square(grad)
        v_t = tf.compat.v1.assign(v, v_t, use_locking=self._use_locking)
        v_t_prime = v_t / coefficients["v_t_prime_denominator"]
        m_t_bar = (
            coefficients["one_minus_m_t"] * g_prime
            + coefficients["m_t_1"] * m_t_prime
        )
        var_t = var - coefficients["lr_t"] * m_t_bar / (
            tf.sqrt(v_t_prime) + coefficients["epsilon"]
        )
        return tf.compat.v1.assign(var, var_t, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        g_prime = grad / coefficients["one_minus_m_schedule_new"]

        # m_t = beta1 * m + (1 - beta1) * g_t
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = tf.compat.v1.assign(
            m, m * coefficients["beta_1_t"], use_locking=self._use_locking
        )

        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
            m_t_slice = tf.gather(m_t, indices)

        m_t_prime = m_t_slice / coefficients["one_minus_m_schedule_next"]
        m_t_bar = (
            coefficients["one_minus_m_t"] * g_prime
            + coefficients["m_t_1"] * m_t_prime
        )

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v_scaled_g_values = (grad * grad) * coefficients["one_minus_beta_2_t"]
        v_t = tf.compat.v1.assign(
            v, v * coefficients["beta_2_t"], use_locking=self._use_locking
        )

        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
            v_t_slice = tf.gather(v_t, indices)

        v_t_prime = v_t_slice / coefficients["v_t_prime_denominator"]
        v_prime_sqrt_plus_eps = tf.sqrt(v_t_prime) + coefficients["epsilon"]

        var_update = self._resource_scatter_add(
            var,
            indices,
            coefficients["neg_lr_t"] * m_t_bar / v_prime_sqrt_plus_eps,
        )
        return tf.group(*[var_update, m_t_bar, v_t])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._initial_decay,
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
            }
        )
        return config

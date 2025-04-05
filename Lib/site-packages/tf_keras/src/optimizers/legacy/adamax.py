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
"""Adamax optimizer implementation."""

import tensorflow.compat.v2 as tf

from tf_keras.src import backend_config
from tf_keras.src.optimizers.legacy import optimizer_v2

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.optimizers.legacy.Adamax",
    v1=["keras.optimizers.Adamax", "keras.optimizers.legacy.Adamax"],
)
class Adamax(optimizer_v2.OptimizerV2):
    """Optimizer that implements the Adamax algorithm.

    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.
    Adamax is sometimes superior to adam, specially in models with embeddings.

    Initialization:

    ```python
    m = 0  # Initialize initial 1st moment vector
    v = 0  # Initialize the exponentially weighted infinity norm
    t = 0  # Initialize timestep
    ```

    The update rule for parameter `w` with gradient `g` is
    described at the end of section 7.1 of the paper:

    ```python
    t += 1
    m = beta1 * m + (1 - beta) * g
    v = max(beta2 * v, abs(g))
    current_lr = learning_rate / (1 - beta1 ** t)
    w = w - current_lr * m / (v + epsilon)
    ```

    Similarly to `Adam`, the epsilon is added for numerical stability
    (especially to get rid of division by zero when `v_t == 0`).

    In contrast to `Adam`, the sparse implementation of this algorithm
    (used when the gradient is an IndexedSlices object, typically because of
    `tf.gather` or an embedding lookup in the forward pass) only updates
    variable slices and corresponding `m_t`, `v_t` terms when that part of
    the variable was used in the forward pass. This means that the sparse
    behavior is contrast to the dense behavior (similar to some momentum
    implementations which ignore momentum unless a variable slice was actually
    used).

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the exponentially weighted infinity norm.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to `"Adamax"`.
      **kwargs: keyword arguments. Allowed arguments are `clipvalue`,
        `clipnorm`, `global_clipnorm`.
        If `clipvalue` (float) is set, the gradient of each weight
        is clipped to be no higher than this value.
        If `clipnorm` (float) is set, the gradient of each weight
        is individually clipped so that its norm is no higher than this value.
        If `global_clipnorm` (float) is set the gradient of all weights is
        clipped so that their global norm is no higher than this value.

    Reference:
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        name="Adamax",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")  # Create slots for the first moments.
        for var in var_list:
            self.add_slot(var, "v")  # Create slots for the second moments.

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        lr_t = apply_state[(var_device, var_dtype)]["lr_t"]

        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_scaled_lr=-lr_t / (1 - beta_1_power),
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                zero=tf.zeros((), dtype=tf.int64),
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return tf.raw_ops.ResourceApplyAdaMax(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=coefficients["beta_1_power"],
            lr=coefficients["lr_t"],
            beta1=coefficients["beta_1_t"],
            beta2=coefficients["beta_2_t"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            use_locking=self._use_locking,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_slice = tf.gather(m, indices, axis=coefficients["zero"])
        m_t_slice = (
            m_slice * coefficients["beta_1_t"]
            + grad * coefficients["one_minus_beta_1_t"]
        )
        with tf.control_dependencies([m_t_slice]):
            m_t = self._resource_scatter_update(m, indices, m_t_slice)

        # u_t = max(beta2 * u, abs(g_t))
        v = self.get_slot(var, "v")
        v_slice = tf.gather(v, indices, axis=coefficients["zero"])
        v_t_slice = tf.maximum(v_slice * coefficients["beta_2_t"], tf.abs(grad))
        with tf.control_dependencies([v_t_slice]):
            v_t = self._resource_scatter_update(v, indices, v_t_slice)
        # theta_t = theta - lr / (1 - beta1^t) * m_t / u_t
        var_slice = coefficients["neg_scaled_lr"] * (
            m_t_slice / (v_t_slice + coefficients["epsilon"])
        )
        with tf.control_dependencies([var_slice]):
            var_update = self._resource_scatter_add(var, indices, var_slice)
        return tf.group(*[var_update, m_t, v_t])

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


# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Adagrad optimizer implementation."""

import tensorflow.compat.v2 as tf

from tf_keras.src.optimizers import optimizer
from tf_keras.src.optimizers.schedules import learning_rate_schedule
from tf_keras.src.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export(
    "keras.optimizers.Adafactor",
    "keras.optimizers.experimental.Adafactor",
    v1=[],
)
class Adafactor(optimizer.Optimizer):
    """Optimizer that implements the Adafactor algorithm.

    Adafactor is commonly used in NLP tasks, and has the advantage
    of taking less memory because it only saves partial information of previous
    gradients.

    The default argument setup is based on the original paper (see reference).
    When gradients are of dimension > 2, Adafactor optimizer will delete the
    last 2 dimensions separately in its accumulator variables.

    Args:
        learning_rate: Initial value for the learning rate:
            either a floating point value,
            or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
            Defaults to 0.001.
        beta_2_decay: float, defaults to -0.8. The decay rate of `beta_2`.
        epsilon_1: float, defaults to 1e-30. A small offset to keep denominator
            away from 0.
        epsilon_2: float, defaults to 1e-3. A small offset to avoid learning
            rate becoming too small by time.
        clip_threshold: float, defaults to 1.0. Clipping threshold. This is a
            part of Adafactor algorithm, independent from `clipnorm`,
            `clipvalue` and `global_clipnorm`.
        relative_step: bool, defaults to True. If `learning_rate` is a
            constant and `relative_step=True`, learning rate will be adjusted
            based on current iterations. This is a default learning rate decay
            in Adafactor.
      {{base_optimizer_keyword_args}}

    Reference:
        - [Shazeer, Noam et al., 2018](https://arxiv.org/abs/1804.04235).

    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_2_decay=-0.8,
        epsilon_1=1e-30,
        epsilon_2=1e-3,
        clip_threshold=1.0,
        relative_step=True,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adafactor",
        **kwargs,
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs,
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_2_decay = beta_2_decay
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.clip_threshold = clip_threshold
        self.relative_step = relative_step

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._r = []
        self._c = []
        self._v = []
        for var in var_list:
            if len(var.shape) < 2:
                # Don't factor if variable is of dimension < 2, but we still
                # need to create dummy variables as placeholder.
                self._r.append(tf.Variable(0, name=f"r/{var._shared_name}"))
                self._c.append(tf.Variable(0, name=f"r/{var._shared_name}"))
            else:
                # Always factor the last 2 dimenstions.
                r_shape = var.shape[:-1]
                c_shape = var.shape[:-2] + var.shape[-1]
                self._r.append(
                    self.add_variable(
                        shape=r_shape,
                        dtype=var.dtype,
                        name=f"r/{var._shared_name}",
                    )
                )
                self._c.append(
                    self.add_variable(
                        shape=c_shape,
                        dtype=var.dtype,
                        name=f"c/{var._shared_name}",
                    )
                )
            self._v.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )

    def _rms(self, x):
        return tf.sqrt(tf.reduce_mean(tf.square(x)))

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        lr = tf.cast(self.learning_rate, variable.dtype)
        epsilon_2 = tf.cast(self.epsilon_2, variable.dtype)
        one = tf.cast(1.0, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        if (
            not isinstance(
                self._learning_rate, learning_rate_schedule.LearningRateSchedule
            )
            and self.relative_step
        ):
            # If `relative_step=True` and learning rate is a constant, we
            # apply the relative step algorithm.
            lr = tf.minimum(lr, tf.math.rsqrt(local_step))

        var_key = self._var_key(variable)
        r = self._r[self._index_dict[var_key]]
        c = self._c[self._index_dict[var_key]]
        v = self._v[self._index_dict[var_key]]

        rho_t = tf.minimum(lr, tf.math.rsqrt(local_step))
        alpha_t = tf.maximum(epsilon_2, self._rms(variable)) * rho_t
        regulated_grad_square = tf.square(gradient) + self.epsilon_1
        beta_2_t = 1 - tf.pow(local_step, self.beta_2_decay)

        if len(variable.shape) >= 2:
            # `r` deletes the last dimension of gradient, so it is of shape
            # `gradient.shape[:-1]`.
            r.assign(
                beta_2_t * r
                + (1 - beta_2_t)
                * tf.reduce_mean(regulated_grad_square, axis=-1)
            )
            # `c` deletes the second last dimension of gradient, so it is of
            # shape `gradient.shape[:-2] + gradient.shape[-1]`.
            c.assign(
                beta_2_t * c
                + (1 - beta_2_t)
                * tf.reduce_mean(regulated_grad_square, axis=-2)
            )
            v.assign(
                tf.expand_dims(
                    r / tf.reduce_mean(r, axis=-1, keepdims=True), axis=-1
                )
                * tf.expand_dims(c, -2)
            )
        else:
            v.assign(beta_2_t * v + (1 - beta_2_t) * regulated_grad_square)

        # `convert_to_tensor` unifies the handling of sparse and dense grads.
        u_t = tf.convert_to_tensor(gradient) * tf.math.rsqrt(v)
        u_t_hat = u_t / tf.maximum(one, (self._rms(u_t) / self.clip_threshold))
        variable.assign_add(-alpha_t * u_t_hat)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_2_decay": self.beta_2_decay,
                "epsilon_1": self.epsilon_1,
                "epsilon_2": self.epsilon_2,
                "clip_threshold": self.clip_threshold,
                "relative_step": self.relative_step,
            }
        )
        return config


Adafactor.__doc__ = Adafactor.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)


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
"""RMSprop optimizer implementation."""

import tensorflow.compat.v2 as tf

from keras.optimizers.optimizer_experimental import optimizer
from keras.utils import generic_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@generic_utils.register_keras_serializable()
@keras_export("keras.optimizers.experimental.RMSprop", v1=[])
class RMSprop(optimizer.Optimizer):
    r"""Optimizer that implements the RMSprop algorithm.

    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average

    This implementation of RMSprop uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance.

    Args:
      learning_rate: Initial value for the learning rate:
        either a floating point value,
        or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
        Defaults to 0.001.
      rho: float, defaults to 0.9. Discounting factor for the old gradients.
      momentum: float, defaults to 0.0. If not 0.0., the optimizer tracks the
        momentum value, with a decay rate equals to `1 - momentum`.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory. Defaults to `False`.
      {{base_optimizer_keyword_args}}

    Usage:

    >>> opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0    # d(loss) / d(var1) = var1
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> var1.numpy()
    9.683772

    Reference:
      - [Hinton, 2012](
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    """

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=100,
        jit_compile=True,
        name="RMSprop",
        **kwargs
    ):
        super().__init__(
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            name=name,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        self._velocities = []
        for var in var_list:
            self._velocities.append(
                self.add_variable_from_reference(var, "velocity")
            )

        self._momentums = []
        if self.momentum > 0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, "momentum")
                )

        self._average_gradients = []
        if self.centered:
            for var in var_list:
                self._average_gradients.append(
                    self.add_variable_from_reference(var, "average_gradient")
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        var_key = self._var_key(variable)
        velocity = self._velocities[self._index_dict[var_key]]
        momentum = None
        if self.momentum > 0:
            momentum = self._momentums[self._index_dict[var_key]]
        average_grad = None
        if self.centered:
            average_grad = self._average_gradients[self._index_dict[var_key]]

        rho = self.rho

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            velocity.assign(rho * velocity)
            velocity.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - rho), gradient.indices
                )
            )
            if self.centered:
                average_grad.assign(rho * average_grad)
                average_grad.scatter_add(
                    tf.IndexedSlices(
                        tf.square(gradient.values) * (1 - rho), gradient.indices
                    )
                )
                velocity.assign_add(-tf.square(average_grad))
            velocity_value = tf.gather(velocity, gradient.indices)
            transformed_grad = tf.IndexedSlices(
                gradient.values / (tf.sqrt(velocity_value) + self.epsilon),
                gradient.indices,
            )

            if self.momentum > 0:
                momentum.assign(self.momentum * momentum)
                momentum.scatter_add(transformed_grad)
                variable.assign_add(-lr * momentum)
            else:
                variable.scatter_add(
                    tf.IndexedSlices(
                        -lr * transformed_grad.values, transformed_grad.indices
                    )
                )
        else:
            # Dense gradients.
            velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient))
            if self.centered:
                average_grad.assign(
                    rho * average_grad + (1 - rho) * tf.square(gradient)
                )
                velocity.assign_add(-tf.square(average_grad))
            transformed_grad = gradient / (tf.sqrt(velocity) + self.epsilon)
            if self.momentum > 0:
                momentum.assign(self.momentum * momentum + transformed_grad)
                variable.assign_add(-lr * momentum)
            else:
                variable.assign_add(-lr * transformed_grad)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config


RMSprop.__doc__ = RMSprop.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)

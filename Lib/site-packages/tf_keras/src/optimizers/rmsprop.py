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

from tf_keras.src.optimizers import optimizer
from tf_keras.src.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export(
    "keras.optimizers.RMSprop",
    "keras.optimizers.experimental.RMSprop",
    "keras.dtensor.experimental.optimizers.RMSprop",
    v1=[],
)
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
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to `1e-7`.
        centered: Boolean. If `True`, gradients are normalized by the estimated
            variance of the gradient; if False, by the uncentered second moment.
            Setting this to `True` may help with training, but is slightly more
            expensive in terms of computation and memory. Defaults to `False`.
        {{base_optimizer_keyword_args}}

    Usage:

    >>> opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0  # d(loss) / d(var1) = var1
    >>> opt.minimize(loss, [var1])
    >>> var1.numpy()
    9.683772

    Reference:
        - [Hinton, 2012](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) # noqa: E501
    """

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        weight_decay=None,
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
            weight_decay=weight_decay,
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
                        gradient.values * (1 - rho), gradient.indices
                    )
                )
                denominator = velocity - tf.square(average_grad) + self.epsilon
            else:
                denominator = velocity + self.epsilon
            denominator_slices = tf.gather(denominator, gradient.indices)
            increment = tf.IndexedSlices(
                lr * gradient.values * tf.math.rsqrt(denominator_slices),
                gradient.indices,
            )

            if self.momentum > 0:
                momentum.assign(self.momentum * momentum)
                momentum.scatter_add(increment)
                variable.assign_add(-momentum)
            else:
                variable.scatter_add(-increment)
        else:
            # Dense gradients.
            velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient))
            if self.centered:
                average_grad.assign(rho * average_grad + (1 - rho) * gradient)
                denominator = velocity - tf.square(average_grad) + self.epsilon
            else:
                denominator = velocity + self.epsilon
            increment = lr * gradient * tf.math.rsqrt(denominator)
            if self.momentum > 0:
                momentum.assign(self.momentum * momentum + increment)
                variable.assign_add(-momentum)
            else:
                variable.assign_add(-increment)

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


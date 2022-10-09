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
"""SGD optimizer implementation."""

import tensorflow.compat.v2 as tf

from keras.optimizers.optimizer_experimental import optimizer
from keras.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export(
    "keras.optimizers.experimental.SGD", "keras.optimizers.SGD", v1=[]
)
class SGD(optimizer.Optimizer):
    r"""Gradient descent (with momentum) optimizer.

    Update rule for parameter `w` with gradient `g` when `momentum` is 0:

    ```python
    w = w - learning_rate * g
    ```

    Update rule when `momentum` is larger than 0:

    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + velocity
    ```

    When `nesterov=True`, this rule becomes:

    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + momentum * velocity - learning_rate * g
    ```

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.001.
      momentum: float hyperparameter >= 0 that accelerates gradient descent in
        the relevant direction and dampens oscillations. Defaults to 0, i.e.,
        vanilla gradient descent.
      nesterov: boolean. Whether to apply Nesterov momentum.
        Defaults to `False`.
      {{base_optimizer_keyword_args}}

    Usage:

    >>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
    >>> var = tf.Variable(1.0)
    >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
    >>> opt.minimize(loss, [var])
    >>> # Step is `- learning_rate * grad`
    >>> var.numpy()
    0.9

    >>> opt = tf.keras.optimizers.experimental.SGD(0.1, momentum=0.9)
    >>> var = tf.Variable(1.0)
    >>> val0 = var.value()
    >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
    >>> # First step is `- learning_rate * grad`
    >>> opt.minimize(loss, [var])
    >>> val1 = var.value()
    >>> (val0 - val1).numpy()
    0.1
    >>> # On later steps, step-size increases because of momentum
    >>> opt.minimize(loss, [var])
    >>> val2 = var.value()
    >>> (val1 - val2).numpy()
    0.18

    Reference:
        - For `nesterov=True`, See [Sutskever et al., 2013](
          http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
    """

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="SGD",
        **kwargs
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
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]

        # TODO(b/204321487): Add nesterov acceleration.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            add_value = tf.IndexedSlices(
                -gradient.values * lr, gradient.indices
            )
            if m is not None:
                m.assign(m * momentum)
                m.scatter_add(add_value)
                if self.nesterov:
                    variable.scatter_add(add_value)
                    variable.assign_add(m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.scatter_add(add_value)
        else:
            # Dense gradients
            if m is not None:
                m.assign(-gradient * lr + m * momentum)
                if self.nesterov:
                    variable.assign_add(-gradient * lr + m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.assign_add(-gradient * lr)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            }
        )
        return config


SGD.__doc__ = SGD.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)

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
"""Adadelta optimizer implementation."""

import tensorflow.compat.v2 as tf

from keras.optimizers import optimizer
from keras.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export(
    "keras.optimizers.experimental.Adadelta", "keras.optimizers.Adadelta", v1=[]
)
class Adadelta(optimizer.Optimizer):
    r"""Optimizer that implements the Adadelta algorithm.

    Adadelta optimization is a stochastic gradient descent method that is based
    on adaptive learning rate per dimension to address two drawbacks:

    - The continual decay of learning rates throughout training.
    - The need for a manually selected global learning rate.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many updates
    have been done. Compared to Adagrad, in the original version of Adadelta you
    don't have to set an initial learning rate. In this version, the initial
    learning rate can be set, as in most other Keras optimizers.

    Args:
      learning_rate: Initial value for the learning rate: either a floating
        point value, or a `tf.keras.optimizers.schedules.LearningRateSchedule`
        instance. Defaults to 0.001. Note that `Adadelta` tends to benefit from
        higher initial learning rate values compared to other optimizers. To
        match the exact form in the original paper, use 1.0.
      rho: A `Tensor` or a floating point value. The decay rate. Defaults to
        0.95.
      epsilon: Small floating point value used to maintain numerical stability.
        Defaults to 1e-7.
      {{base_optimizer_keyword_args}}

    Reference:
      - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
    """

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.95,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adadelta",
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
        self.epsilon = epsilon

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._accumulated_grads = []
        self._accumulated_delta_vars = []
        for var in var_list:
            self._accumulated_grads.append(
                self.add_variable_from_reference(var, "accumulated_grad")
            )
            self._accumulated_delta_vars.append(
                self.add_variable_from_reference(var, "accumulated_delta_var")
            )

    def update_step(self, grad, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        var_key = self._var_key(variable)
        rho = self.rho
        accumulated_grad = self._accumulated_grads[self._index_dict[var_key]]
        accumulated_delta_var = self._accumulated_delta_vars[
            self._index_dict[var_key]
        ]

        def rms(x):
            return tf.sqrt(x + self.epsilon)

        if isinstance(grad, tf.IndexedSlices):
            # Sparse gradients.
            accumulated_grad.assign_add((rho - 1) * accumulated_grad)
            accumulated_grad.scatter_add(
                tf.IndexedSlices(
                    (1 - rho) * tf.square(grad.values), grad.indices
                )
            )
            delta_var = (
                -rms(accumulated_delta_var) * grad / rms(accumulated_grad)
            )
            accumulated_delta_var.assign(
                rho * accumulated_delta_var + (1 - rho) * delta_var * delta_var
            )
        else:
            # Dense gradients.
            accumulated_grad.assign(
                rho * accumulated_grad + (1 - rho) * grad * grad
            )
            delta_var = (
                -rms(accumulated_delta_var) * grad / rms(accumulated_grad)
            )
            accumulated_delta_var.assign(
                rho * accumulated_delta_var + (1 - rho) * delta_var * delta_var
            )
        variable.assign_add(lr * delta_var)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "rho": self.rho,
                "epsilon": self.epsilon,
            }
        )
        return config


Adadelta.__doc__ = Adadelta.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)

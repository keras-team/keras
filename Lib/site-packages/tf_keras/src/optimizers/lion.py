# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Lion optimizer implementation."""

import tensorflow.compat.v2 as tf

from tf_keras.src.optimizers import optimizer
from tf_keras.src.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export("keras.optimizers.Lion", v1=[])
class Lion(optimizer.Optimizer):
    """Optimizer that implements the Lion algorithm.

    The Lion optimizer is a stochastic-gradient-descent method that uses the
    sign operator to control the magnitude of the update, unlike other adaptive
    optimizers such as Adam that rely on second-order moments. This make
    Lion more memory-efficient as it only keeps track of the momentum. According
    to the authors (see reference), its performance gain over Adam grows with
    the batch size. Because the update of Lion is produced through the sign
    operation, resulting in a larger norm, a suitable learning rate for Lion is
    typically 3-10x smaller than that for AdamW. The weight decay for Lion
    should be in turn 3-10x larger than that for AdamW to maintain a
    similar strength (lr * wd).

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.0001.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            rate to combine the current gradient and the 1st moment estimate.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimate.
        {{base_optimizer_keyword_args}}

    References:
        - [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
        - [Authors' implementation](
            http://github.com/google/automl/tree/master/lion)

    """

    def __init__(
        self,
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Lion",
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        if beta_1 <= 0 or beta_1 > 1:
            raise ValueError(
                f"`beta_1`={beta_1} must be between ]0, 1]. Otherwise, "
                "the optimizer degenerates to SignSGD."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        Lion optimizer has one variable `momentums`.

        Args:
            var_list: list of model variables to build Lion variables on.
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
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients (use m as a buffer)
            m.assign(m * beta_1)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1.0 - beta_1), gradient.indices
                )
            )
            variable.assign_sub(lr * tf.math.sign(m))

            m.assign(m * beta_2 / beta_1)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1.0 - beta_2 / beta_1), gradient.indices
                )
            )
        else:
            # Dense gradients
            variable.assign_sub(
                lr * tf.math.sign(m * beta_1 + gradient * (1.0 - beta_1))
            )
            m.assign(m * beta_2 + gradient * (1.0 - beta_2))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
            }
        )
        return config


Lion.__doc__ = Lion.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)


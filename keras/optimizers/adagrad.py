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

from keras import initializers
from keras.optimizers import optimizer
from keras.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export(
    "keras.optimizers.experimental.Adagrad", "keras.optimizers.Adagrad", v1=[]
)
class Adagrad(optimizer.Optimizer):
    r"""Optimizer that implements the Adagrad algorithm.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    Args:
      learning_rate: Initial value for the learning rate:
        either a floating point value,
        or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
        Defaults to 0.001.
        Note that `Adagrad` tends to benefit from higher initial learning rate
        values compared to other optimizers.
        To match the exact form in the original paper, use 1.0.
      initial_accumulator_value: Floating point value.
        Starting value for the accumulators (per-parameter momentum values).
        Must be non-negative.
      epsilon: Small floating point value used to maintain numerical stability.
      {{base_optimizer_keyword_args}}

    Reference:
      - [Duchi et al., 2011](
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
    """

    def __init__(
        self,
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adagrad",
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
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._accumulators = []
        initializer = initializers.Constant(self.initial_accumulator_value)
        for var in var_list:
            self._accumulators.append(
                self.add_variable_from_reference(
                    var,
                    "accumulator",
                    initial_value=initializer(shape=var.shape, dtype=var.dtype),
                )
            )

    def update_step(self, grad, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        var_key = self._var_key(variable)
        accumulator = self._accumulators[self._index_dict[var_key]]

        if isinstance(grad, tf.IndexedSlices):
            # Sparse gradients.
            accumulator.scatter_add(
                tf.IndexedSlices(grad.values * grad.values, grad.indices)
            )
            sparse_accumulator = tf.gather(accumulator, indices=grad.indices)
            sparse_denominator = tf.sqrt(sparse_accumulator + self.epsilon)
            variable.scatter_add(
                tf.IndexedSlices(
                    -lr * grad.values / sparse_denominator, grad.indices
                )
            )
        else:
            # Dense gradients.
            accumulator.assign_add(grad * grad)
            variable.assign_sub(lr * grad / tf.sqrt(accumulator + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
            }
        )
        return config


Adagrad.__doc__ = Adagrad.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)

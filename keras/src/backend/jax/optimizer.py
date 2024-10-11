"""A class for JAX specific optimizer logic.

Its purpose is to route around statelessness
requirements in cond ops used for EMA handling
and gradient accumulation handling. We do this
by skipping conditionals entirely.
"""

import jax
from jax import numpy as jnp

from keras.src.optimizers import base_optimizer


class JaxOptimizer(base_optimizer.BaseOptimizer):

    def _backend_apply_gradients(self, grads, trainable_variables):
        if self.gradient_accumulation_steps:
            is_update_step = (
                self._iterations + 1
            ) % self.gradient_accumulation_steps == 0
            steps = self.gradient_accumulation_steps

            current_trainable_vars_value = [
                v.value for v in trainable_variables
            ]
            current_optimizer_vars_value = [v.value for v in self.variables]

            # `trainable_variables` might have been filtered in previous
            # processing steps, so we need to ensure the correct mapping between
            # `self._accumulated_gradients` and `trainable_variables`
            acc_grads = [
                self._accumulated_gradients[self._get_variable_index(v)]
                for v in trainable_variables
            ]

            new_g_accs = jax.lax.cond(
                is_update_step,
                lambda: [jnp.zeros(g.shape, dtype=g.dtype) for g in acc_grads],
                lambda: [g + acc_g for g, acc_g in zip(grads, acc_grads)],
            )

            grads = jax.lax.cond(
                is_update_step,
                lambda: [
                    (g + acc_g) / steps for g, acc_g in zip(grads, acc_grads)
                ],
                lambda: list(grads),
            )

            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            self._backend_update_step(
                grads, trainable_variables, self.learning_rate
            )
            new_trainable_vars = jax.lax.cond(
                is_update_step,
                lambda: [v.value for v in trainable_variables],
                lambda: current_trainable_vars_value,
            )
            new_opt_vars = jax.lax.cond(
                is_update_step,
                lambda: [v.value for v in self.variables],
                lambda: current_optimizer_vars_value,
            )

            for value, v in zip(new_trainable_vars, trainable_variables):
                v.assign(value)

            for value, v in zip(new_opt_vars, self.variables):
                v.assign(value)

            for n_g_acc, g_acc in zip(new_g_accs, acc_grads):
                g_acc.assign(n_g_acc)

        else:
            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            self._backend_update_step(
                grads, trainable_variables, self.learning_rate
            )

        if self.use_ema:
            self._update_model_variables_moving_average(
                self._trainable_variables
            )
            if self.ema_overwrite_frequency is not None:
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                should_overwrite_model_vars_int = (
                    should_overwrite_model_vars.astype("int32")
                )
                should_not_overwrite_model_vars_int = jnp.logical_not(
                    should_overwrite_model_vars
                ).astype("int32")
                current_trainable_vars_value = [
                    v.value for v in self._trainable_variables
                ]
                for var, average_var in zip(
                    self._trainable_variables,
                    self._model_variables_moving_average,
                ):
                    var.assign(
                        average_var * should_overwrite_model_vars_int
                        + var.value * should_not_overwrite_model_vars_int
                    )

        self._iterations.assign_add(1)

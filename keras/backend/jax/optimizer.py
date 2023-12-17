from jax import numpy as jnp

from keras.optimizers import base_optimizer


class JaxOptimizer(base_optimizer.BaseOptimizer):
    """A class for JAX specific optimizer logic.

    Its purpose is to route around statelessness
    requirements in cond ops used for EMA handling
    and gradient accumulation handling. We do this
    by skipping conditionals entirely.
    """

    def _backend_apply_gradients(self, grads, trainable_variables):
        if self.gradient_accumulation_steps:
            is_update_step = (
                self.iterations + 1
            ) % self.gradient_accumulation_steps == 0
            is_update_step_int = is_update_step.astype("int32")
            is_not_update_step_int = jnp.logical_not(is_update_step).astype(
                "int32"
            )
            steps = self.gradient_accumulation_steps

            current_trainable_vars_value = [
                v.value for v in trainable_variables
            ]
            current_optimizer_vars_value = [v.value for v in self.variables]
            new_g_accs = [
                is_not_update_step_int
                * (grads[i] + self._accumulated_gradients[i])
                for i in range(len(grads))
            ]
            grads = [
                is_update_step_int
                * (grads[i] + self._accumulated_gradients[i])
                / steps
                for i in range(len(grads))
            ]

            self._backend_update_step(
                grads, trainable_variables, self.learning_rate
            )

            for curr_v, v in zip(
                current_trainable_vars_value, trainable_variables
            ):
                v.assign(
                    v.value * is_update_step_int
                    + curr_v * is_not_update_step_int
                )
            for curr_v, v in zip(current_optimizer_vars_value, self.variables):
                v.assign(
                    v.value * is_update_step_int
                    + curr_v * is_not_update_step_int
                )
            for n_g_acc, g_acc in zip(new_g_accs, self._accumulated_gradients):
                g_acc.assign(n_g_acc)
        else:
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

        self.iterations.assign_add(1)

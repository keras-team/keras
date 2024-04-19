import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Adam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adam):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)
        local_step = ops.cast(self.iterations + 1, dtype)

        beta_1_power = ops.power(ops.cast(self.beta_1, dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, dtype), local_step)
        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        m_list = [
            self._momentums[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        v_list = [
            self._velocities[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        torch._foreach_mul_(m_list, self.beta_1)
        torch._foreach_add_(m_list, grads, alpha=1 - self.beta_1)

        torch._foreach_mul_(v_list, self.beta_2)
        torch._foreach_add_(
            v_list, torch._foreach_mul(grads, grads), alpha=1 - self.beta_2
        )

        if self.amsgrad:
            v_hat_list = [
                self._velocity_hats[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]
            torch._foreach_maximum_(v_hat_list, v_list)
            v_list = v_hat_list

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, alpha),
                torch._foreach_add(torch._foreach_sqrt(v_list), self.epsilon),
            ),
            alpha=-1,
        )

import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch import core
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Nadam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Nadam):
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
        next_step = ops.cast(self.iterations + 2, dtype)
        decay = ops.cast(0.96, dtype)
        beta_1 = ops.cast(self.beta_1, dtype)
        beta_2 = ops.cast(self.beta_2, dtype)
        u_t = beta_1 * (1.0 - 0.5 * (ops.power(decay, local_step)))
        u_t_1 = beta_1 * (1.0 - 0.5 * (ops.power(decay, next_step)))
        u_product_t = self._u_product.value * u_t
        u_product_t_1 = u_product_t * u_t_1
        beta_2_power = ops.power(beta_2, local_step)

        self._u_product.assign(u_product_t)

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

        m_hat_list = torch._foreach_add(
            torch._foreach_div(
                torch._foreach_mul(m_list, u_t_1),
                1 - core.convert_to_numpy(u_product_t_1),
            ),
            torch._foreach_div(
                torch._foreach_mul(grads, 1 - u_t),
                1 - core.convert_to_numpy(u_product_t),
            ),
        )

        v_hat_list = torch._foreach_div(v_list, 1 - beta_2_power)

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_hat_list, lr),
                torch._foreach_add(
                    torch._foreach_sqrt(v_hat_list), self.epsilon
                ),
            ),
            alpha=-1,
        )

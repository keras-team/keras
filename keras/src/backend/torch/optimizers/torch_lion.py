import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Lion(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Lion):
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

        m_list = [
            self._momentums[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        c_t = torch._foreach_mul(m_list, self.beta_1)
        torch._foreach_add_(c_t, grads, alpha=1 - self.beta_1)
        c_t = [c.sign() for c in c_t]

        torch._foreach_add_(
            variables,
            torch._foreach_mul(c_t, lr),
            alpha=-1,
        )

        torch._foreach_mul_(m_list, self.beta_2)
        torch._foreach_add_(m_list, grads, alpha=1 - self.beta_2)

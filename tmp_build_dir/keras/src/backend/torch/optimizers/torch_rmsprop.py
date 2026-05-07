import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class RMSprop(
    torch_parallel_optimizer.TorchParallelOptimizer, optimizers.RMSprop
):
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

        velocities = [
            self._velocities[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        rho = self.rho

        torch._foreach_mul_(velocities, rho)
        torch._foreach_add_(
            velocities, torch._foreach_mul(grads, grads), alpha=1 - rho
        )

        denominators = torch._foreach_add(velocities, self.epsilon)
        if self.centered:
            average_grads = [
                self._average_gradients[
                    self._get_variable_index(variable)
                ].value
                for variable in keras_variables
            ]
            torch._foreach_mul_(average_grads, rho)
            torch._foreach_add_(average_grads, grads, alpha=1 - rho)
            torch._foreach_add_(
                denominators,
                torch._foreach_mul(average_grads, average_grads),
                alpha=-1,
            )
        torch._foreach_sqrt_(denominators)
        increments = torch._foreach_div(
            torch._foreach_mul(grads, lr), denominators
        )

        if self.momentum > 0:
            momentum_list = [
                self._momentums[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]
            torch._foreach_mul_(momentum_list, self.momentum)
            torch._foreach_add_(momentum_list, increments)
            torch._foreach_add_(variables, momentum_list, alpha=-1)
        else:
            torch._foreach_add_(variables, increments, alpha=-1)

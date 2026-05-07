import torch

from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class SGD(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.SGD):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]
        if self.momentum != 0:
            bufs = [
                self.momentums[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]

            for i in range(len(bufs)):
                if bufs[i] is None:
                    bufs[i] = torch.clone(grads[i]).detach()

            torch._foreach_mul_(bufs, self.momentum)
            torch._foreach_add_(bufs, grads, alpha=-learning_rate)

            if self.nesterov:
                torch._foreach_add_(variables, grads, alpha=-learning_rate)
                torch._foreach_add_(variables, bufs, alpha=self.momentum)
            else:
                torch._foreach_add_(variables, bufs)

        else:
            torch._foreach_add_(variables, grads, alpha=-learning_rate)

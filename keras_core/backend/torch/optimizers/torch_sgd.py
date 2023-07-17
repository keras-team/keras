import torch

from keras_core import optimizers


class SGD(optimizers.SGD):
    def _internal_apply_gradients(self, grads_and_vars):
        grads, trainable_variables = zip(*grads_and_vars)

        self._parallel_update_step(
            grads,
            [v.value for v in trainable_variables],
            self._get_current_learning_rate(),
        )
        self.iterations.assign(self.iterations + 1)

    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        if self.momentum != 0:
            bufs = [
                self.momentums[self._get_variable_index(variable.value)]
                for variable in variables
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

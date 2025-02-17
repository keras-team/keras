import torch

from keras.src.optimizers.base_optimizer import BaseOptimizer
from keras.src.utils import torch_utils


class TorchParallelOptimizer(BaseOptimizer):
    @torch_utils.no_grad
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        self._parallel_update_step(
            grads,
            trainable_variables,
            learning_rate,
        )

    @torch_utils.no_grad
    def _backend_reset_gradient_accumulators(self):
        acc_list = [v.value for v in self._accumulated_gradients]
        torch._foreach_mul_(acc_list, 0.0)

    @torch_utils.no_grad
    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        acc_list = [v.value for v in acc_grads]
        torch._foreach_add_(acc_list, grads, alpha=1.0)

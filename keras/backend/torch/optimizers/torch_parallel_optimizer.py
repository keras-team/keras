from keras.optimizers.base_optimizer import BaseOptimizer
from keras.utils import torch_utils


class TorchParallelOptimizer(BaseOptimizer):
    @torch_utils.no_grad
    def _internal_apply_gradients(self, grads_and_vars):
        grads, trainable_variables = zip(*grads_and_vars)

        self._parallel_update_step(
            grads,
            trainable_variables,
            self._get_current_learning_rate(),
        )
        self.iterations.assign(self.iterations + 1)

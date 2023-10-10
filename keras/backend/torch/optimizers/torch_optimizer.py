import torch

from keras import optimizers
from keras.optimizers.base_optimizer import BaseOptimizer
from keras.utils import torch_utils


class TorchOptimizer(BaseOptimizer):
    def __new__(cls, *args, **kwargs):
        # Import locally to avoid circular imports.
        from keras.backend.torch.optimizers import torch_adadelta
        from keras.backend.torch.optimizers import torch_adagrad
        from keras.backend.torch.optimizers import torch_adam
        from keras.backend.torch.optimizers import torch_adamax
        from keras.backend.torch.optimizers import torch_adamw
        from keras.backend.torch.optimizers import torch_lion
        from keras.backend.torch.optimizers import torch_nadam
        from keras.backend.torch.optimizers import torch_rmsprop
        from keras.backend.torch.optimizers import torch_sgd

        OPTIMIZERS = {
            optimizers.Adadelta: torch_adadelta.Adadelta,
            optimizers.Adagrad: torch_adagrad.Adagrad,
            optimizers.Adam: torch_adam.Adam,
            optimizers.Adamax: torch_adamax.Adamax,
            optimizers.AdamW: torch_adamw.AdamW,
            optimizers.Lion: torch_lion.Lion,
            optimizers.Nadam: torch_nadam.Nadam,
            optimizers.RMSprop: torch_rmsprop.RMSprop,
            optimizers.SGD: torch_sgd.SGD,
        }

        if cls in OPTIMIZERS:
            return OPTIMIZERS[cls](*args, **kwargs)
        return super().__new__(cls)

    @torch_utils.no_grad
    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return

        torch._foreach_mul_(
            [v.value for v in variables if self._use_weight_decay(v)],
            1 - self.weight_decay * self._get_current_learning_rate(),
        )

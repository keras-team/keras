import torch

from keras_core import optimizers
from keras_core.optimizers.base_optimizer import BaseOptimizer


class TorchOptimizer(BaseOptimizer):
    def __new__(cls, *args, **kwargs):
        # Import locally to avoid circular imports.
        from keras_core.backend.torch.optimizers import torch_adadelta
        from keras_core.backend.torch.optimizers import torch_adagrad
        from keras_core.backend.torch.optimizers import torch_adam
        from keras_core.backend.torch.optimizers import torch_adamax
        from keras_core.backend.torch.optimizers import torch_adamw
        from keras_core.backend.torch.optimizers import torch_nadam
        from keras_core.backend.torch.optimizers import torch_rmsprop
        from keras_core.backend.torch.optimizers import torch_sgd

        OPTIMIZERS = {
            optimizers.Adadelta: torch_adadelta.Adadelta,
            optimizers.Adagrad: torch_adagrad.Adagrad,
            optimizers.Adam: torch_adam.Adam,
            optimizers.Adamax: torch_adamax.Adamax,
            optimizers.AdamW: torch_adamw.AdamW,
            optimizers.Nadam: torch_nadam.Nadam,
            optimizers.RMSprop: torch_rmsprop.RMSprop,
            optimizers.SGD: torch_sgd.SGD,
        }

        if cls in OPTIMIZERS:
            return OPTIMIZERS[cls](*args, **kwargs)
        return super().__new__(cls)

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return

        torch._foreach_mul_(
            [v.value for v in variables if self._use_weight_decay(v)],
            1 - self.weight_decay * self._get_current_learning_rate(),
        )

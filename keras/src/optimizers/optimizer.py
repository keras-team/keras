from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.optimizer import (
        TFOptimizer as BackendOptimizer,
    )
elif backend.backend() == "torch":
    from keras.src.backend.torch.optimizers import (
        TorchOptimizer as BackendOptimizer,
    )
elif backend.backend() == "jax":
    from keras.src.backend.jax.optimizer import JaxOptimizer as BackendOptimizer
else:

    class BackendOptimizer(base_optimizer.BaseOptimizer):
        pass


@keras_export(["keras.Optimizer", "keras.optimizers.Optimizer"])
class Optimizer(BackendOptimizer, base_optimizer.BaseOptimizer):
    pass


Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__
base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args

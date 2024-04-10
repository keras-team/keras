from keras import backend
from keras.api_export import keras_export
from keras.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras.backend.tensorflow.optimizer import (
        TFOptimizer as BackendOptimizer,
    )
elif backend.backend() == "torch":
    from keras.backend.torch.optimizers import (
        TorchOptimizer as BackendOptimizer,
    )
elif backend.backend() == "jax":
    from keras.backend.jax.optimizer import JaxOptimizer as BackendOptimizer
else:

    class BackendOptimizer(base_optimizer.BaseOptimizer):
        pass


@keras_export(["keras.Optimizer", "keras.optimizers.Optimizer"])
class Optimizer(BackendOptimizer, base_optimizer.BaseOptimizer):
    pass


base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args

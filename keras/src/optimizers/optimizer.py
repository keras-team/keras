from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.optimizer import TFOptimizer

    BackendOptimizer = TFOptimizer
elif backend.backend() == "torch":
    from keras.src.backend.torch.optimizers import TorchOptimizer

    BackendOptimizer = TorchOptimizer
elif backend.backend() == "jax":
    from keras.src.backend.jax.optimizer import JaxOptimizer

    BackendOptimizer = JaxOptimizer
else:
    BackendOptimizer = base_optimizer.BaseOptimizer


@keras_export(["keras.Optimizer", "keras.optimizers.Optimizer"])
class Optimizer(BackendOptimizer):
    pass


base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args
Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__

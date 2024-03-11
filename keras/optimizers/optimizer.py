from keras import backend
from keras.api_export import keras_export
from keras.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras.backend.tensorflow.optimizer import TFOptimizer

    BackendOptimizer = TFOptimizer
elif backend.backend() == "torch":
    from keras.backend.torch.optimizers import TorchOptimizer

    BackendOptimizer = TorchOptimizer
elif backend.backend() == "jax":
    from keras.backend.jax.optimizer import JaxOptimizer

    BackendOptimizer = JaxOptimizer
else:
    BackendOptimizer = base_optimizer.BaseOptimizer


@keras_export(["keras.Optimizer", "keras.optimizers.Optimizer"])
class Optimizer(BackendOptimizer):
    pass


base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args
Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__

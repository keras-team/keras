from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras_core.backend.tensorflow import optimizer as tf_optimizer

    BackendOptimizer = tf_optimizer.TFOptimizer
else:
    BackendOptimizer = base_optimizer.BaseOptimizer


keras_core_export(["keras_core.Optimizer", "keras_core.optimizers.Optimizer"])


class Optimizer(BackendOptimizer):
    pass


base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args
Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__

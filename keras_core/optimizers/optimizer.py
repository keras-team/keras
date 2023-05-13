from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras_core.backend.tensorflow import optimizer as tf_optimizer

    Optimizer = tf_optimizer.TFOptimizer
else:
    Optimizer = base_optimizer.Optimizer


keras_core_export(["keras_core.Optimizer", "keras_core.optimizers.Optimizer"])(
    Optimizer
)


base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args

from keras_core import backend
from keras_core.api_export import keras_core_export

if backend.backend() == "tensorflow":
    BackendVariable = backend.tensorflow.core.Variable
elif backend.backend() == "jax":
    BackendVariable = backend.jax.core.Variable
elif backend.backend() == "torch":
    BackendVariable = backend.torch.core.Variable
elif backend.backend() == "numpy":
    from keras_core.backend.numpy.core import Variable as NumpyVariable

    BackendVariable = NumpyVariable
else:
    raise RuntimeError(f"Invalid backend: {backend.backend()}")


@keras_core_export("keras_core.backend.Variable")
class Variable(BackendVariable):
    pass

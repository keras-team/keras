from keras.src import backend
from keras.src.api_export import keras_export

if backend.backend() == "tensorflow":
    BackendVariable = backend.tensorflow.core.Variable
    backend_name_scope = backend.tensorflow.core.name_scope
elif backend.backend() == "jax":
    BackendVariable = backend.jax.core.Variable
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "torch":
    BackendVariable = backend.torch.core.Variable
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "numpy":
    from keras.src.backend.numpy.core import Variable as NumpyVariable

    BackendVariable = NumpyVariable
    backend_name_scope = backend.common.name_scope.name_scope
else:
    raise RuntimeError(f"Invalid backend: {backend.backend()}")


@keras_export("keras.Variable")
class Variable(BackendVariable):
    pass


@keras_export("keras.name_scope")
class name_scope(backend_name_scope):
    pass


@keras_export("keras.device")
def device(device_name):
    return backend.device_scope(device_name)

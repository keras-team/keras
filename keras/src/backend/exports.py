from keras.src import backend
from keras.src.api_export import keras_export

if backend.backend() == "tensorflow":
    backend_name_scope = backend.tensorflow.core.name_scope
elif backend.backend() == "jax":
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "torch":
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "numpy":
    backend_name_scope = backend.common.name_scope.name_scope
else:
    raise RuntimeError(f"Invalid backend: {backend.backend()}")


@keras_export("keras.name_scope")
class name_scope(backend_name_scope):
    pass


@keras_export("keras.device")
def device(device_name):
    return backend.device_scope(device_name)

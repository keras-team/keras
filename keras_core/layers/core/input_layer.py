from keras_core import backend
from keras_core.layers.layer import Layer
from keras_core.operations.node import Node


class InputLayer(Layer):
    def __init__(
        self, shape, batch_size=None, dtype=None, input_tensor=None, name=None
    ):
        # TODO: support for sparse, ragged.
        super().__init__(name=name)
        self.shape = backend.standardize_shape(shape)
        self._dtype = backend.standardize_dtype(dtype)
        self.batch_size = batch_size
        if input_tensor is not None:
            if not isinstance(input_tensor, backend.KerasTensor):
                raise ValueError(
                    "Argument `input_tensor` must be a KerasTensor. "
                    f"Received invalid type: input_tensor={input_tensor} (of type {type(input_tensor)})"
                )
        else:
            input_tensor = backend.KerasTensor(
                shape=(batch_size,) + shape, dtype=dtype, name=name
            )
        self._input_tensor = input_tensor
        Node(operation=self, call_args=(), call_kwargs={}, outputs=input_tensor)
        self.built = True

    def call(self):
        return

    @property
    def dtype(self):
        return self._dtype

    def get_config(self):
        return {
            "shape": self.shape,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "name": self.name,
        }


def Input(shape=None, batch_size=None, dtype=None, name=None):
    layer = InputLayer(
        shape=shape, batch_size=batch_size, dtype=dtype, name=name
    )
    return layer.output

import base64
import io

from packaging.version import parse

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers import Layer
from keras.src.ops import convert_to_numpy
from keras.src.ops import convert_to_tensor
from keras.src.saving.serialization_lib import in_safe_mode


@keras_export("keras.layers.TorchModuleWrapper")
class TorchModuleWrapper(Layer):
    """Torch module wrapper layer.

    `TorchModuleWrapper` is a wrapper class that can turn any
    `torch.nn.Module` into a Keras layer, in particular by making its
    parameters trackable by Keras.

    `TorchModuleWrapper` is only compatible with the PyTorch backend and
    cannot be used with the TensorFlow or JAX backends.

    Args:
        module: `torch.nn.Module` instance. If it's a `LazyModule`
            instance, then its parameters must be initialized before
            passing the instance to `TorchModuleWrapper` (e.g. by calling
            it once).
        output_shape :The shape of the output of this layer. It helps Keras
            perform automatic shape inference.
        name: The name of the layer (string).

    Example:

    Here's an example of how the `TorchModuleWrapper` can be used with vanilla
    PyTorch modules.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import keras
    from keras.layers import TorchModuleWrapper

    class Classifier(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Wrap `torch.nn.Module`s with `TorchModuleWrapper`
            # if they contain parameters
            self.conv1 = TorchModuleWrapper(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
            )
            self.conv2 = TorchModuleWrapper(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
            )
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=0.5)
            self.fc = TorchModuleWrapper(nn.Linear(1600, 10))

        def call(self, inputs):
            x = F.relu(self.conv1(inputs))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.fc(x)
            return F.softmax(x, dim=1)


    model = Classifier()
    model.build((1, 28, 28))
    print("Output shape:", model(torch.ones(1, 1, 28, 28).to("cuda")).shape)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(train_loader, epochs=5)
    ```
    """

    def __init__(self, module, name=None, output_shape=None, **kwargs):
        super().__init__(name=name, **kwargs)
        import torch.nn as nn

        from keras.src.backend.torch.core import get_device

        if (
            isinstance(module, nn.modules.lazy.LazyModuleMixin)
            and module.has_uninitialized_params()
        ):
            raise ValueError(
                "LazyModules are not supported unless they "
                "are already initialized. "
                f"Received uninitialized LazyModule: module={module}"
            )

        self.module = module.to(get_device())
        self._track_module_parameters()
        self.output_shape = output_shape

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def _track_module_parameters(self):
        for param in self.module.parameters():
            # The Variable will reuse the raw `param`
            # and simply wrap it.
            variable = backend.Variable(
                initializer=param, trainable=param.requires_grad
            )
            self._track_variable(variable)
        self.built = True

    def call(self, *args, training=None, **kwargs):
        if training is False:
            self.eval()
        else:
            self.train()
        return self.module(*args, **kwargs)

    def save_own_variables(self, store):
        """Saves model's state from `state_dict`.
        `model.parameters` excludes some of model's state like
        `BatchNorm` mean and variance. So, use `state_dict` to obtain
        all of model's state.
        """
        state_dict = self.module.state_dict()
        for key in state_dict.keys():
            store[key] = convert_to_numpy(state_dict[key])

    def load_own_variables(self, store):
        """Loads model's state via `state_dict`."""
        state_dict = {}
        for key in store.keys():
            if isinstance(key, bytes):
                key = key.decode()
            state_dict[key] = convert_to_tensor(store[key])
        self.module.load_state_dict(state_dict)

    def compute_output_shape(self, input_shape):
        if self.output_shape is None:
            return super().compute_output_shape(input_shape)
        return self.output_shape

    def get_config(self):
        base_config = super().get_config()
        import torch

        buffer = io.BytesIO()
        torch.save(self.module, buffer)
        # Encode the buffer using base64 to ensure safe serialization
        buffer_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        config = {
            "module": buffer_b64,
            "output_shape": self.output_shape,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        import torch

        if "module" in config:
            if in_safe_mode():
                raise ValueError(
                    "Requested the deserialization of a `torch.nn.Module` "
                    "object via `torch.load()`. This carries a potential risk "
                    "of arbitrary code execution and thus it is disallowed by "
                    "default. If you trust the source of the artifact, you can "
                    "override this error by passing `safe_mode=False` to the "
                    "loading function, or calling "
                    "`keras.config.enable_unsafe_deserialization()."
                )

            # Decode the base64 string back to bytes
            buffer_bytes = base64.b64decode(config["module"].encode("ascii"))
            buffer = io.BytesIO(buffer_bytes)
            config["module"] = torch.load(buffer, weights_only=False)
        return cls(**config)


def no_grad(orig_func):
    import torch

    if parse(torch.__version__) >= parse("2.1.0"):
        return torch.no_grad(orig_func)
    else:
        return orig_func

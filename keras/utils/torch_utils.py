from keras.api_export import keras_export
from keras.layers import Layer
from keras.ops import convert_to_numpy
from keras.ops import convert_to_tensor


@keras_export("keras.layers.TorchModuleWrapper")
class TorchModuleWrapper(Layer):
    """Torch module wrapper layer.

    `TorchModuleWrapper` is a wrapper class that can turn any
    `torch.nn.Module` into a Keras layer, in particular by making its
    parameters trackable by Keras.

    Args:
        module: `torch.nn.Module` instance. If it's a `LazyModule`
            instance, then its parameters must be initialized before
            passing the instance to `TorchModuleWrapper` (e.g. by calling
            it once).
        name: The name of the layer (string).

    Examples:

    Here's an example of how the `TorchModuleWrapper` can be used with vanilla
    PyTorch modules.

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    import keras
    from keras.layers import TorchModuleWrapper

    class Classifier(keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
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

    def __init__(self, module, name="torch_module_wrapper"):
        if name is None:
            raise ValueError(
                "The `name` argument is required for `TorchModuleWrapper`. "
                "This helps save the `torch` state dictionary with an unique "
                "name that's consistent during saving and loading of model."
            )
        super().__init__(name=name)
        import torch.nn as nn

        if (
            isinstance(module, nn.modules.lazy.LazyModuleMixin)
            and module.has_uninitialized_params()
        ):
            raise ValueError(
                "LazyModules are not supported unless they "
                "are already initialized. "
                f"Received uninitialized LazyModule: module={module}"
            )

        from keras.backend.torch.core import get_device

        self.module = module.to(get_device())
        self._track_module_parameters()

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def _track_module_parameters(self):
        from keras.backend.torch import Variable

        for param in self.module.parameters():
            variable = Variable(
                initializer=param, trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def call(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def save_own_variables(self, store):
        """Saves model's state from `state_dict`.
        `model.parameters` excludes some of model's state like
        `BatchNorm` mean and variance. So, use `state_dict` to obtain
        all of model's state.
        """
        state_dict = self.module.state_dict()
        # Save all layer's state keys
        store[self.name + "._keys"] = list(state_dict.keys())
        for key in state_dict.keys():
            store[self.name + "." + key] = convert_to_numpy(state_dict[key])

    def load_own_variables(self, store):
        """Loads model's state via `state_dict`.
        """
        keys_name = self.name + "._keys"
        if keys_name not in store:
            raise ValueError(
                f"Weights file is missing state for {self.name} layer."
            )
        state_dict = {}
        for key in store[keys_name]:
            if isinstance(key, bytes):
                key = key.decode()
            try:
                state_dict[key] = convert_to_tensor(
                    store[self.name + "." + key]
                )
            except KeyError:
                raise ValueError(
                    f"Weights file is missing state for {self.name}.{key}"
                )
        self.module.load_state_dict(state_dict)

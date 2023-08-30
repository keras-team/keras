from keras_core.layers import Layer


class TorchModuleWrapper(Layer):
    """Torch module wrapper layer.

    `TorchModuleWrapper` is an abstraction that can be wrapped around a
    `torch.nn.Module` to make its parameters trackable as a
    `keras_core.layers.Layer`. It works with both vanilla and lazy PyTorch
    modules.

    Args:
        module: torch.nn.Module, A vanilla or lazy PyTorch neural network
            module.
        name: The name of the layer (string).

    References:
    - [PyTorch docs for `torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) # noqa: E501
    - [PyTorch docs for `LazyModuleMixin`](https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html) # noqa: E501

    Examples:

    Here's an example of how the `TorchModuleWrapper` can be used with vanilla
    PyTorch modules.

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    import keras_core
    from keras_core.backend.torch import TorchModuleWrapper


    class Classifier(keras_core.Model):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Wrap all `torch.nn.Module`s with `TorchModuleWrapper`
            self.conv1 = TorchModuleWrapper(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
            )
            self.conv2 = TorchModuleWrapper(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
            )
            self.pool = TorchModuleWrapper(
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.flatten = TorchModuleWrapper(nn.Flatten())
            self.dropout = TorchModuleWrapper(nn.Dropout(p=0.5))
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

    Here's an example of how the `TorchModuleWrapper` can be used with PyTorch
    Lazy modules.

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    import keras_core
    from keras_core.backend.torch import TorchModuleWrapper


    class LazyClassifier(keras.Model):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # You can wrap all `torch.nn.Module`s with `TorchModuleWrapper`
            # irrespective of whether they are lazy or not.
            self.conv1 = TorchModuleWrapper(
                nn.LazyConv2d(out_channels=32, kernel_size=(3, 3))
            )
            self.conv2 = TorchModuleWrapper(
                nn.LazyConv2d(out_channels=64, kernel_size=(3, 3))
            )
            self.pool = TorchModuleWrapper(nn.MaxPool2d(kernel_size=(2, 2)))
            self.flatten = TorchModuleWrapper(nn.Flatten())
            self.dropout = TorchModuleWrapper(nn.Dropout(p=0.5))
            self.fc = TorchModuleWrapper(nn.LazyLinear(10))

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

    def __init__(self, module, name=None):
        super().__init__(name=name)
        self.module = module
        import torch.nn as nn

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.module = module.to(self.device)
        self._is_lazy_module = isinstance(
            self.module, nn.modules.lazy.LazyModuleMixin
        )
        if not self._is_lazy_module:
            self._track_module_parameters()

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def _track_module_parameters(self):
        from keras_core.backend.torch import Variable

        for param in self.module.parameters():
            variable = Variable(
                initializer=param, trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self, *args, **kwargs):
        if self._is_lazy_module:
            self._build_by_run(*args, **kwargs)
        self.track_module_parameters()

    def call(self, inputs, **kwargs):
        return self.module.forward(inputs, **kwargs)

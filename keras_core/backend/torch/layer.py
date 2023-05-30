import torch

from keras_core.operations.operation import Operation


class TorchLayer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: find a good place to add the params. It should be added right
        # after the variables are initialized.
        self.params = torch.nn.ParameterList(
            [variable.value for variable in self.variables]
        )
        return Operation.__call__(self, *args, **kwargs)

import torch

from keras_core.operations.operation import Operation


class TorchLayer(torch.nn.Module):
    def parameters(self, recurse=True):
        if not hasattr(self, "torch_params"):
            self.torch_params = torch.nn.ParameterList(
                [variable.value for variable in self.variables]
            )
        return torch.nn.Module.parameters(self, recurse=recurse)

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

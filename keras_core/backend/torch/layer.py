import torch

from keras_core.operations.operation import Operation


class TorchLayer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: find a good place to add the params. It should be added right
        # after the variables are initialized.
        if not hasattr(self, "torch_params"):
            self.torch_params = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        variable.value, requires_grad=variable.trainable
                    )
                    for variable in self.variables
                ]
            )
        return Operation.__call__(self, *args, **kwargs)

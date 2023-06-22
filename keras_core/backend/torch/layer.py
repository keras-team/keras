import torch

from keras_core.backend.common.stateless_scope import in_stateless_scope
from keras_core.operations.operation import Operation


class TorchLayer(torch.nn.Module):
    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return
        self._track_variables()

    def _track_variables(self):
        self.torch_params = torch.nn.ParameterList(
            [variable.value for variable in self.variables]
        )

    def parameters(self, recurse=True):
        if not hasattr(self, "torch_params"):
            self._track_variables()
        return torch.nn.Module.parameters(self, recurse=recurse)

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation


class TorchLayer(torch.nn.Module):
    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return
        self._track_variables()

    def _track_variables(self):
        # Index given to ParameterDict must be a string
        self.torch_params = torch.nn.ParameterDict(
            {str(id(variable)): variable.value for variable in self.variables}
        )

    def parameters(self, recurse=True):
        if not hasattr(self, "torch_params"):
            self._track_variables()
        return torch.nn.Module.parameters(self, recurse=recurse)

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

    def _setattr_hook(self, name, value):
        from keras.src.layers import Layer

        if (
            isinstance(value, torch.nn.Module)
            and not isinstance(value, Layer)
            and not name == "torch_params"
        ):
            from keras.src.utils.torch_utils import TorchModuleWrapper

            if not isinstance(self, TorchModuleWrapper):
                value = TorchModuleWrapper(value)
        return name, value

    def _post_track_variable(self, variable):
        if hasattr(self, "torch_params"):
            # Index given to ParameterDict must be a string
            key = str(id(variable))
            if key not in self.torch_params:
                self.torch_params[key] = variable.value

    def _post_untrack_variable(self, variable):
        if hasattr(self, "torch_params"):
            # Index given to ParameterDict must be a string
            key = str(id(variable))
            self.torch_params.pop(key)

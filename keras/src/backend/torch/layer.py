import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation


class TorchLayer(torch.nn.Module):
    @property
    def torch_params(self):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        return self._torch_params

    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return
        self._track_variables()

    def _track_variables(self):
        # set torch_params attribute will have module automatically track
        # parameters.
        from keras.src.backend.torch.core import get_device

        params = {}
        device = get_device()
        for variable in self.variables:
            value = variable.value
            if not isinstance(value, torch.nn.Parameter):
                requires_grad = variable.trainable and torch.is_floating_point(
                    value
                )
                from torch.distributed.tensor import DTensor
                if isinstance(value, DTensor):
                    value = torch.nn.Parameter(value, requires_grad=requires_grad)
                else:
                    value = torch.nn.Parameter(value, requires_grad=requires_grad).to(device)
            params[variable.path] = value
        self._torch_params = torch.nn.ParameterDict(params)

    def named_parameters(
        self,
        prefix="",
        recurse=True,
        remove_duplicate=True,
    ):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        return torch.nn.Module.named_parameters(
            self, prefix, recurse, remove_duplicate
        )

    def forward(self, *args, **kwargs):
        from keras.src.backend.common import global_state
        from keras.src.backend.torch.core import get_device

        distribution = global_state.get_global_attribute("distribution")
        if distribution is not None and str(get_device()) != "meta":
            from keras.src.backend.torch import distribution_lib

            args = distribution_lib._maybe_distribute_input(args, distribution)
            kwargs = distribution_lib._maybe_distribute_input(kwargs, distribution)

        return Operation.__call__(self, *args, **kwargs)

    def _setattr_hook(self, name, value):
        from keras.src.layers import Layer

        if (
            isinstance(value, torch.nn.Module)
            and not isinstance(value, Layer)
            and not name == "_torch_params"
        ):
            from keras.src.utils.torch_utils import TorchModuleWrapper

            if not isinstance(self, TorchModuleWrapper):
                value = TorchModuleWrapper(value)
        return name, value

    def _post_track_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path not in self.torch_params:
                self.torch_params[variable.path] = variable.value

    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                self.torch_params.pop(variable.path)
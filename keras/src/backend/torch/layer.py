import warnings

import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation


class TorchLayer(torch.nn.Module):
    @property
    def torch_params(self):
        warnings.warn(
            "`layer.torch_param` is deprecated and will be removed in a future "
            "release. Please use `layer.variables` (Keras style) or "
            "`layer.named_parameters()` (Torch style) instead."
        )
        return list(self.named_parameters())

    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return

    def forward(self, *args, **kwargs):
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
        self._parameters[variable.path] = variable.value

    def _post_untrack_variable(self, variable):
        if variable.path in self._parameters:
            del self._parameters[variable.path]

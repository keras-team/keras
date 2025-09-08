import warnings
from collections import OrderedDict

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

    def _setattr_hook(self, name, value):
        from keras.src.layers import Layer

        if isinstance(value, torch.nn.Module) and not isinstance(value, Layer):
            from keras.src.utils.torch_utils import TorchModuleWrapper

            if not isinstance(self, TorchModuleWrapper):
                value = TorchModuleWrapper(value)
        return name, value

    # Overrided torch.nn.Module methods.

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for layer in self._flatten_layers(
            include_self=False, recursive=recurse
        ):
            for variable in layer.variables:
                if remove_duplicate and id(variable) in seen:
                    continue
                seen.add(id(variable))
                name = prefix + variable.path
                yield name, variable.value
        for variable in self.variables:
            if remove_duplicate and id(variable) in seen:
                continue
            seen.add(id(variable))
            name = prefix + variable.path
            yield name, variable.value

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        raise NotImplementedError(
            "Keras with the Torch backend does not support `load_state_dict`. "
            "Please use `layer.set_weights` instead."
        )

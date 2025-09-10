import collections

import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation


class TorchLayer(torch.nn.Module):
    @property
    def torch_params(self):
        if len(self._parameters) != len(self.variables):
            self._track_variables()
        return self._parameters

    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return
        self._track_variables()

    def _track_variables(self):
        self._parameters.clear()
        for variable in self.variables:
            self._parameters[variable.path] = variable.value

    def _setattr_hook(self, name, value):
        from keras.src.layers import Layer

        if isinstance(value, torch.nn.Module) and not isinstance(value, Layer):
            from keras.src.utils.torch_utils import TorchModuleWrapper

            if not isinstance(self, TorchModuleWrapper):
                value = TorchModuleWrapper(value)
        return name, value

    def _post_track_variable(self, variable):
        if len(self._parameters) > 0:
            self._track_variables()

    def _post_untrack_variable(self, variable):
        if len(self._parameters) > 0:
            self._track_variables()

    def _post_quantize(self, mode, **kwargs):
        # Re-track variables after quantization.
        self._track_variables()

    # Override torch.nn.Module methods. The reason for this is that in Keras, we
    # use recursive `self.variables` to track Torch parameters, and the path
    # attribute is different from the one in Torch.

    def named_parameters(
        self,
        prefix="",
        recurse=True,
        remove_duplicate=True,
    ):
        if recurse is not True:
            raise ValueError("recurse must be True in Keras.")
        if remove_duplicate is not True:
            raise ValueError("remove_duplicate must be True in Keras.")

        if len(self._parameters) != len(self.variables):
            self._track_variables()
        for k, v in self._parameters.items():
            yield prefix + k, v

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if keep_vars is not False:
            raise ValueError("keep_vars must be False in Keras.")

        if len(self._parameters) != len(self.variables):
            self._track_variables()
        if destination is None:
            destination = collections.OrderedDict()
        for k, v in self._parameters.items():
            destination[prefix + k] = v
        return destination

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

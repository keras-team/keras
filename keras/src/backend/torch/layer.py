import torch

from keras.src.backend.common.keras_tensor import any_symbolic_tensors
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
        self._torch_params = torch.nn.ParameterDict(
            {variable.path: variable.value for variable in self.variables}
        )

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
        # Fast path: when built, inputs are real tensors, and no special
        # execution modes (remat/quantization), bypass Operation.__call__.
        if (
            self.built
            and not any_symbolic_tensors(args, kwargs)
            and getattr(self, "_remat_mode", None) is None
            and getattr(self, "quantization_mode", None) is None
        ):
            return self.call(*args, **kwargs)
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

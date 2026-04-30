import torch
import torch._dynamo as dynamo

from keras.src.backend.common.keras_tensor import is_keras_tensor
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.layers.input_spec import assert_input_compatibility
from keras.src.ops.operation import Operation


@dynamo.disable()
def _has_symbolic_arg(args, kwargs=None):
    """Quick check if the first arg (or its contents) is symbolic."""
    for a in list(args) + list((kwargs or {}).values()):
        if is_keras_tensor(a):
            return True
        if isinstance(a, (dict, list, tuple)):
            for v in a.values() if isinstance(a, dict) else a:
                if is_keras_tensor(v):
                    return True
    return False


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
        # Fast path: built layer, real tensors, no special features
        if (
            self.built
            and self.quantization_mode is None
            and getattr(self, "_remat_mode", None) is None
            and not in_stateless_scope()
            and not _has_symbolic_arg(args, kwargs)
        ):
            if self.input_spec is not None and args:
                assert_input_compatibility(self.input_spec, args[0], self.name)
            return self.call(*args, **kwargs)
        return Operation.__call__(self, *args, **kwargs)

    @dynamo.disable()
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

    @dynamo.disable()
    def _post_track_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path not in self.torch_params:
                self.torch_params[variable.path] = variable.value

    @dynamo.disable()
    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                self.torch_params.pop(variable.path)

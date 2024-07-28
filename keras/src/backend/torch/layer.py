import torch

from keras.src.ops.operation import Operation


class TorchLayer(torch.nn.Module):
    """Adaptation layer to make sure keras.layers.Layer works well with
    torch.nn.Module. Currently, the main modification are on parameter/module
    tracking and pointing torch.nn.Module.forward() to the right keras call.

    Module tracking:
      All sublayers are tracked as modules in Module._modules. All module level
    api with recurse=True should work properly just like a torch.nn.Module.

    Variable tracking:
      Since keras has a different variable tracking mechanism, unlike modules,
    Modules._parameter doesn't automatically tracks variables created for torch
    layers.
      This is currently manually populated through _track_torch_params() that
    does following work:
        1. Populate all sublayers torch params by calling _track_torch_params()
        2. Create a single torch.nn.ParameterList() parameter with trainable,
           non trainable and seed generator states belongs to the current layer.

    Few additional points that user should be aware of:
    1. When torch backend is enabled KerasVariable.value is torch.nn.Parameter,
       this is not visible to torch since it is separately tracked in keras
       tracker.
    2. When torch parameter is exposed with _track_torch_params(), no copy is
       made to the torch parameter in keras tracker; so both keras tracker and
       torch module sees the same object it is just present in 2 different
       member variables. This also means any modification to keras variable,
       for instance, setting trainable is automatically populated to torch
       parameters.
    3. Since keras creates variables in a deterministic order, resulted torch
       parameter list will also in deterministic order with the order of
       trainable->non_trainable->seed_generator_states. Changing variable from
       trainable to non trainable won't move keras variable from one tracker to
       the another, so does the final populated torch_params.
    4. It is recommended for user to alternate variables through keras variable
       apis instead of alternate with torch_params since it is simpler with the
       keras variable api and it is backend agnostic.
    5. Any torch module operation should in theory works; for example
       state_dict() and load_state_dict() works if you want a more torch way of
       saving variables.
    6. Although not recommended, but you can use below code snippet to find the
       corresponding parameter in torch_params from a keras variable:
       parameters = [(pname, p) for pname, p in layer.named_parameters() \
                      if id(p) == id(variable.value)]
    7. For non trainable varialbes like mean and var in BatchNormalization, this
       is registered as part of torch_params as parameters instead of buffers.
       This is not really torch best practices but it is not really possible in
       keras to track since keras doesn't distinguish a variable that is a stats
       or just have gradient skipped.
    """

    def _track_torch_params(self):
        for layer in self._layers:
            layer._track_torch_params()
        torch_params = []
        for v in self._trainable_variables + self._non_trainable_variables:
            torch_params.append(v.value)
        for sg in self._seed_generators:
            torch_params.append(sg.state.value)

        # set torch_params attribute will have module automatically track
        # parameters.
        self.torch_params = torch.nn.ParameterList(torch_params)

    def _all_layers_built(self):
        sublayers_built = all(
            layer._all_layers_built() for layer in self._layers
        )
        return self.built and sublayers_built

    def _torch_params_tracked(self):
        return hasattr(self, "torch_params")

    def _populate_torch_params(self):
        if not self._all_layers_built():
            raise RuntimeError(
                "Torch parameters are not tracked since all layers are not "
                "built. Did you forget to call model once?"
            )

        if not self._torch_params_tracked():
            self._track_torch_params()

    def named_modules(
        self,
        memo=None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        # named_modules is the root of all torch parameters/module calls.
        self._populate_torch_params()
        return torch.nn.Module.named_modules(
            self, memo, prefix, remove_duplicate
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self._populate_torch_params()
        return torch.nn.Module.state_dict(
            self,
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._populate_torch_params()
        return torch.nn.Module.load_state_dict(self, state_dict, strict, assign)

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
        # Torch module don't register list[Module] in its __setattr__, it uses
        # nn.ModuleList normally. In Keras3, we only need a way for the module
        # class to be tracked by torch since keras3 user can still do
        # self._layers to reference all layers instead of using
        # torch.nn.Module.named_members().
        if (
            isinstance(value, list)
            and all(isinstance(v, Layer) for v in value)
            and len(value) > 0
        ):
            for idx, v in enumerate(value):
                self.add_module(f"{name}_{idx}", v)

        return name, value

    def _post_track_variable(self, _):
        if self._torch_params_tracked():
            if not self._all_layers_built():
                raise ValueError(
                    "Torch parameters are tracked but not all "
                    "layers are built. This is an invalid state "
                    "in pytorch backend and please raise an "
                    "issue in github repo."
                )
            self._track_torch_params()

    def _post_untrack_variable(self, _):
        if self._torch_params_tracked():
            if not self._all_layers_built():
                raise ValueError(
                    "Torch parameters are tracked but not all "
                    "layers are built. This is an invalid state "
                    "in pytorch backend and please raise an "
                    "issue in github repo."
                )
            self._track_torch_params()

# Porting your custom layers to Keras 1.0

- make sure your layer inherits from `Layer` and not `MaskedLayer`. `MaskedLayer` has been removed.
- First, instead of the `build` method reading the layer's input via `self.input_shape`, it now takes
    it explicitly as an argument. So convert `build(self)` to `build(self, input_shape)`. 
- Convert the `output_shape` property to a mnethod `get_output_shape_for(self, input_shape)`. Make sure to remove the old `output_shape`.
- The layer's logic now lives in a `call` method (note: NOT `__call__`. Your layer should NOT implement a `__call__` method). Convert the layer's `get_output(self, train=False)` method
    to `call(self, x, mask=None)`. Make sure to remove the old `get_output` method.
- If your layer used a different behavior at training and test time: Keras does not rely on the boolean flag `train` anymore.
    Instead, put phase-specific statements inside calls to `x = K.in_train_phase(train_x, test_x)`.
    E.g. here's dropout (inside of the `call` method of the `Dropout` layer):
    `return K.in_train_phase(K.dropout(x, level=self.p), x)`
- The config dictionary returns by your layer's `get_config` probably included the class name as `name`.
    Just remove that `name` entry from `get_config`.

If you were using masking:
- implement a method `compute_mask(input_tensor, input_mask)` return `output_mask`
- make sure that your layer sets `self.supports_masking = True` in `__init__()`

Optionally:
- if your layer requires more to be instantiated than just calling it with `config` as kwargs:
    implement a `from_config` classmethod.
- if you want Keras to perform input compatibility checks when the layer gets connected to inbound layers,
    set `self.input_specs` in `__init__` (or implement an `input_specs()` @property). It should be
    a list of `engine.InputSpec` instances.

Reserved method names that you should NOT override:
- __call__
- add_input
- assert_input_compatibility
- get_input
- get_output
- input_shape
- output_shape
- input_mask
- output_mask
- get_input_at
- get_output_at
- get_input_shape_at
- get_output_shape_at
- get_input_mask_at
- get_output_mask_at
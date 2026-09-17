"""Shared building blocks for the built-in quantization modes.

The helpers here are pure code motion: each one emits exactly the op
sequence its call sites emitted inline, so modes that adopt them keep
producing identical traced programs.
"""

from keras.src import ops
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class GeometryDispatchStrategy(QuantizationStrategy):
    """A mode whose math is written once per geometry family.

    `build`, `call` and `quantize` resolve the layer's geometry and hand
    off to the matching `_<verb>_<family>` method. Subclasses implement
    only the families they support; anything else reports the mode as
    unsupported for that layer.
    """

    def build(self, layer, input_shape, config):
        geometry = self.require_geometry(layer)
        handler = self._handler("build", geometry.family, layer)
        handler(layer, geometry, input_shape, config)

    def call(self, layer, *args, **kwargs):
        geometry = self.require_geometry(layer)
        family = geometry.family
        handler = self._handler("call", family, layer)
        return handler(layer, *args, **kwargs)

    def quantize(self, layer, config):
        geometry = self.require_geometry(layer)
        handler = self._handler("quantize", geometry.family, layer)
        handler(layer, geometry, config)

    def _handler(self, verb, family, layer):
        """Returns this mode's implementation for one geometry family."""
        handler = getattr(self, f"_{verb}_{family}", None)
        if handler is None:
            raise NotImplementedError(
                f"Quantization mode '{self.name}' does not support the "
                f"'{family}' quantization geometry of layer "
                f"{layer.__class__.__name__}."
            )
        return handler


def apply_bias_activation(layer, x):
    """Adds the layer's bias and applies its activation, when present."""
    if layer.bias is not None:
        x = ops.add(x, layer.bias)
    if layer.activation is not None:
        x = layer.activation(x)
    return x

"""Built-in quantization mode descriptors.

Importing this package registers the built-in modes. They register here,
explicitly, rather than by decorating each descriptor: registration order
is the canonical `QUANTIZATION_MODES` order, validation error messages
render the registered-names tuple, and decorating would instead tie that
order to the (alphabetical) import order.
"""

from keras.src.quantizers.mode_registry import register_quantization_mode
from keras.src.quantizers.modes.awq import AWQMode
from keras.src.quantizers.modes.float8 import Float8Mode
from keras.src.quantizers.modes.gptq import GPTQMode
from keras.src.quantizers.modes.int4 import Int4Mode
from keras.src.quantizers.modes.int8 import Int8Mode
from keras.src.quantizers.modes.ternary import TernaryMode

register_quantization_mode(Int8Mode)
register_quantization_mode(Float8Mode)
register_quantization_mode(Int4Mode)
register_quantization_mode(TernaryMode)
register_quantization_mode(GPTQMode)
register_quantization_mode(AWQMode)

"""Built-in quantization strategies.

Importing this package registers the built-in modes. They register here,
explicitly, rather than by decorating each strategy: registration order
is the canonical `QUANTIZATION_MODES` order, validation error messages
render the registered-names tuple, and decorating would instead tie that
order to the (alphabetical) import order.
"""

from keras.src.quantizers.modes.awq import AWQStrategy
from keras.src.quantizers.modes.float8 import Float8Strategy
from keras.src.quantizers.modes.gptq import GPTQStrategy
from keras.src.quantizers.modes.int4 import Int4Strategy
from keras.src.quantizers.modes.int8 import Int8Strategy
from keras.src.quantizers.modes.ternary import TernaryStrategy
from keras.src.quantizers.strategy_registry import (
    register_quantization_strategy,
)

register_quantization_strategy(Int8Strategy)
register_quantization_strategy(Float8Strategy)
register_quantization_strategy(Int4Strategy)
register_quantization_strategy(TernaryStrategy)
register_quantization_strategy(GPTQStrategy)
register_quantization_strategy(AWQStrategy)

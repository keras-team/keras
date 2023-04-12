import types

from keras_core.saving import register_keras_core_serializable

try:
    import namex
except ImportError:
    namex = None


def maybe_register_serializable(symbol):
    if isinstance(symbol, types.FunctionType) or hasattr(symbol, "get_config"):
        register_keras_core_serializable(symbol)


if namex:

    class keras_core_export(namex.export):
        def __init__(self, path):
            super().__init__(package="keras_core", path=path)

        def __call__(self, symbol):
            maybe_register_serializable(symbol)
            return super().__call__(symbol)

else:

    class keras_core_export:
        def __init__(self, path):
            pass

        def __call__(self, symbol):
            maybe_register_serializable(symbol)
            return symbol

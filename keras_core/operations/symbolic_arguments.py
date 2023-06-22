from tensorflow import nest

from keras_core.backend import KerasTensor


class SymbolicArguments:
    def __init__(self, *args, **kwargs):
        # TODO: validation
        self.args = nest.map_structure(lambda x: x, args)
        self.kwargs = nest.map_structure(lambda x: x, kwargs)
        self._flat_arguments = nest.flatten((self.args, self.kwargs))

        # Used to avoid expensive `nest` operations in the most common case.
        if (
            not self.kwargs
            and len(self.args) == 1
            and isinstance(self.args[0], KerasTensor)
        ):
            self._single_positional_tensor = self.args[0]
        else:
            self._single_positional_tensor = None

        self.keras_tensors = []
        for arg in self._flat_arguments:
            if isinstance(arg, KerasTensor):
                self.keras_tensors.append(arg)

    def convert(self, conversion_fn):
        args = nest.map_structure(conversion_fn, self.args)
        kwargs = nest.map_structure(conversion_fn, self.kwargs)
        return args, kwargs

    def fill_in(self, tensor_dict):
        """Maps KerasTensors to computed values using `tensor_dict`.

        `tensor_dict` maps `KerasTensor` instances to their current values.
        """
        if self._single_positional_tensor is not None:
            # Performance optimization for most common case.
            # Approx. 70x faster.
            return (tensor_dict[id(self._single_positional_tensor)],), {}

        def switch_fn(x):
            if isinstance(x, KerasTensor):
                val = tensor_dict.get(id(x), None)
                if val is not None:
                    return val
            return x

        return self.convert(switch_fn)

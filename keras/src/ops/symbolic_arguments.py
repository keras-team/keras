from keras.src import tree
from keras.src.backend import KerasTensor


class SymbolicArguments:
    def __init__(self, *args, **kwargs):
        self.args = tree.map_structure(lambda x: x, args)
        self.kwargs = tree.map_structure(lambda x: x, kwargs)
        self._flat_arguments = tree.flatten((self.args, self.kwargs))

        # Pre-detect common call patterns to avoid expensive tree ops
        # in fill_in().  The fast paths cover ~95% of inference calls.
        self._single_positional_tensor = None
        self._dual_positional_tensors = None
        self._dual_tensors_static_kwargs = None

        if not self.kwargs and len(self.args) == 1:
            if isinstance(self.args[0], KerasTensor):
                self._single_positional_tensor = self.args[0]
        elif not self.kwargs and len(self.args) == 2:
            if isinstance(self.args[0], KerasTensor) and isinstance(
                self.args[1], KerasTensor
            ):
                self._dual_positional_tensors = (self.args[0], self.args[1])
        elif (
            len(self.args) == 2
            and isinstance(self.args[0], KerasTensor)
            and isinstance(self.args[1], KerasTensor)
            and self.kwargs
            and not any(
                isinstance(v, KerasTensor) for v in self.kwargs.values()
            )
        ):
            self._dual_tensors_static_kwargs = (
                self.args[0],
                self.args[1],
                self.kwargs,
            )

        self.keras_tensors = []
        for arg in self._flat_arguments:
            if isinstance(arg, KerasTensor):
                self.keras_tensors.append(arg)

    def convert(self, conversion_fn):
        args = tree.map_structure(conversion_fn, self.args)
        kwargs = tree.map_structure(conversion_fn, self.kwargs)
        return args, kwargs

    def fill_in(self, tensor_dict):
        """Maps KerasTensors to computed values using `tensor_dict`.

        `tensor_dict` maps `KerasTensor` instances to their current values.
        """
        if self._single_positional_tensor is not None:
            # Performance optimization for most common case.
            # Approx. 70x faster.
            return (tensor_dict[id(self._single_positional_tensor)],), {}

        if self._dual_positional_tensors is not None:
            t0, t1 = self._dual_positional_tensors
            return (tensor_dict[id(t0)], tensor_dict[id(t1)]), {}

        if self._dual_tensors_static_kwargs is not None:
            t0, t1, kw = self._dual_tensors_static_kwargs
            return (tensor_dict[id(t0)], tensor_dict[id(t1)]), kw

        def switch_fn(x):
            if isinstance(x, KerasTensor):
                return tensor_dict.get(id(x), None)
            return x

        return self.convert(switch_fn)

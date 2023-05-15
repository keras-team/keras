import numpy as np
import torch

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import get_autocast_scope
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common import standardize_shape
from keras_core.backend.common.stateless_scope import get_stateless_scope
from keras_core.backend.common.stateless_scope import in_stateless_scope


DYNAMIC_SHAPES_OK = True


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = convert_to_tensor(value, dtype=self._dtype)

    def assign(self, value):
        value = convert_to_tensor(value, dtype=self.dtype)
        if value.shape != self.shape:
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                "`variable.assign(value)` must match. "
                f"Received: value.shape={value.shape}; "
                f"variable.shape={self.value.shape}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            # torch `as_tensor` by default, doesn't copy if tensor is same type
            self._value = convert_to_tensor(value, dtype=self.dtype)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            # Unitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # during shape inference in a scratch graph
            # (anything else would be a bug, to be fixed.)
            return self._maybe_autocast(
                convert_to_tensor(
                    self._initializer(self._shape, dtype=self._dtype),
                    dtype=self._dtype,
                )
            )
        return self._maybe_autocast(self._value)

    def numpy(self):
        return np.array(self.value)

    # Overload native accessor.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)


def convert_to_tensor(x, dtype=None):
    # TODO: Need to address device placement arg of `as_tensor`
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            # TODO: fix below
            return x.value.astype(dtype)
        return x.value
    if dtype == "float32":
        dtype = torch.float32
    return torch.as_tensor(x, dtype=dtype)


def is_tensor(x):
    return torch.is_tensor(x)


def shape(x):
    return x.shape


def cast(x, dtype):
    return x.to(dtype)


def name_scope(name):
    raise NotImplementedError


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    raise NotImplementedError(
        "`compute_output_spec` not implemented for PyTorch backend"
    )


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    raise NotImplementedError


def scatter(*args, **kwargs):
    raise NotImplementedError

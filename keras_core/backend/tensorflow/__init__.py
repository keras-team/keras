import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope
from keras_core.backend.common.stateless_scope import get_stateless_scope
from keras_core.backend.common.stateless_scope import in_stateless_scope
from keras_core.backend.tensorflow import core
from keras_core.backend.tensorflow import image
from keras_core.backend.tensorflow import math
from keras_core.backend.tensorflow import nn
from keras_core.backend.tensorflow import numpy
from keras_core.backend.tensorflow import random
from keras_core.backend.tensorflow.rnn import gru
from keras_core.backend.tensorflow.rnn import lstm
from keras_core.backend.tensorflow.rnn import rnn
from keras_core.utils.naming import auto_name

DYNAMIC_SHAPES_OK = True


class Variable(KerasVariable, tf.__internal__.types.Tensor):
    _should_act_as_resource_variable = True

    @property
    def handle(self):
        return self.value.handle

    def _initialize(self, value):
        self._value = tf.Variable(
            value, dtype=self._dtype, trainable=self.trainable
        )

    def assign(self, value):
        value = convert_to_tensor(value, dtype=self.dtype)
        if value.shape != self.value.shape:
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
            self.value.assign(value)

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
            init_val = self._initializer(self._shape, dtype=self._dtype)
            return self._maybe_autocast(init_val)
        return self._maybe_autocast(self._value)

    def numpy(self):  # noqa: F811
        return self.value.numpy()

    @property
    def shape(self):
        return tf.TensorShape(super().shape)

    # Overload native accessor.
    def __tf_tensor__(self, dtype=None, name=None):
        return tf.convert_to_tensor(self.value, dtype=dtype, name=name)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)


def convert_to_tensor(x, dtype=None):
    if dtype is not None:
        dtype = standardize_dtype(dtype)
        if tf.is_tensor(x):
            return tf.cast(x, dtype=dtype)
    return tf.convert_to_tensor(x, dtype=dtype)


def is_tensor(x):
    return tf.is_tensor(x)


def shape(x):
    return tf.shape(x)


def cast(x, dtype):
    dtype = standardize_dtype(dtype)
    return tf.cast(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    return tf.cond(pred, true_fn=true_fn, false_fn=false_fn)


def name_scope(name):
    return tf.name_scope(name)


def vectorized_map(function, elements):
    return tf.vectorized_map(function, elements)


def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():
        graph_name = auto_name("scratch_graph")
        with tf.__internal__.FuncGraph(graph_name).as_default():

            def convert_keras_tensor_to_tf(x):
                if isinstance(x, KerasTensor):
                    return tf.compat.v1.placeholder(
                        shape=x.shape, dtype=x.dtype
                    )
                return x

            args, kwargs = tf.nest.map_structure(
                convert_keras_tensor_to_tf, (args, kwargs)
            )
            tf_out = fn(*args, **kwargs)

            def convert_tf_to_keras_tensor(x):
                if tf.is_tensor(x):
                    return KerasTensor(x.shape, x.dtype)
                return x

            output_shape = tf.nest.map_structure(
                convert_tf_to_keras_tensor, tf_out
            )
    return output_shape

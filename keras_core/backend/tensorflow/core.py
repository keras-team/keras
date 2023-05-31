import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope
from keras_core.utils.naming import auto_name

DYNAMIC_SHAPES_OK = True


class Variable(
    KerasVariable,
    tf.__internal__.types.Tensor,
    tf.__internal__.tracking.AutoTrackable,
):
    _should_act_as_resource_variable = True

    @property
    def handle(self):
        return self.value.handle

    def _initialize(self, value):
        self._value = tf.Variable(
            value, dtype=self._dtype, trainable=self.trainable, name=self.name
        )

    def _direct_assign(self, value):
        self.value.assign(value)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def numpy(self):  # noqa: F811
        return self.value.numpy()

    @property
    def shape(self):
        return tf.TensorShape(super().shape)

    # Overload native accessor.
    def __tf_tensor__(self, dtype=None, name=None):
        return tf.convert_to_tensor(self.value, dtype=dtype, name=name)

    # For SavedModel
    def _write_object_proto(self, *args, **kwargs):
        return self.value._write_object_proto(*args, **kwargs)


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


def name_scope(name):
    return tf.name_scope(name)


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

            output_spec = tf.nest.map_structure(
                convert_tf_to_keras_tensor, tf_out
            )
    return output_spec


def cond(pred, true_fn, false_fn):
    return tf.cond(pred, true_fn=true_fn, false_fn=false_fn)


def vectorized_map(function, elements):
    return tf.vectorized_map(function, elements)


def scatter(indices, values, shape):
    return tf.scatter_nd(indices, values, shape)


def scatter_update(inputs, indices, updates):
    return tf.tensor_scatter_nd_update(inputs, indices, updates)


def block_update(inputs, start_indices, updates):
    return dynamic_update_slice(inputs, updates, start_indices)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    return tf.while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=maximum_iterations,
    )

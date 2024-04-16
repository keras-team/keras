import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.tensorflow.sparse import sparse_to_dense
from keras.src.utils.naming import auto_name

SUPPORTS_SPARSE_TENSORS = True


class Variable(
    KerasVariable,
    tf.__internal__.types.Tensor,
    tf.__internal__.tracking.Trackable,
):
    _should_act_as_resource_variable = True

    @property
    def handle(self):
        return self.value.handle

    def _initialize(self, value):
        self._value = tf.Variable(
            value, dtype=self._dtype, trainable=self.trainable, name=self.name
        )

    def _deferred_initialize(self):
        if self._value is not None:
            raise ValueError(f"Variable {self.path} is already initialized.")

        if in_stateless_scope():
            raise ValueError(
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed. "
                "Make sure that all variables are initialized "
                "before you start using your layer/model objects."
            )
        with tf.init_scope():
            value = self._initializer(self._shape, dtype=self._dtype)
            self._initialize(value)

    def _direct_assign(self, value):
        self._value.assign(tf.cast(value, self._value.dtype))

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

    # Methods below are for SavedModel support
    @property
    def _shared_name(self):
        return self.value._shared_name

    def _serialize_to_tensors(self):
        try:
            return self.value._serialize_to_tensors()
        except NotImplementedError:
            return {"VARIABLE_VALUE": self.value}

    def _restore_from_tensors(self, restored_tensors):
        try:
            return self.value._restore_from_tensors(restored_tensors)
        except NotImplementedError:
            self.assign(restored_tensors["VARIABLE_VALUE"])
            return self.value

    def _copy_trackable_to_cpu(self, object_map):
        self.value._copy_trackable_to_cpu(object_map)
        object_map[self] = tf.Variable(object_map[self.value])

    def _export_to_saved_model_graph(
        self, object_map, tensor_map, options, **kwargs
    ):
        resource_list = self.value._export_to_saved_model_graph(
            object_map, tensor_map, options, **kwargs
        )
        object_map[self] = tf.Variable(object_map[self.value])
        return resource_list

    def _write_object_proto(self, proto, options):
        return self.value._write_object_proto(proto, options)


def convert_to_tensor(x, dtype=None, sparse=None):
    if isinstance(x, tf.SparseTensor) and sparse is not None and not sparse:
        x = sparse_to_dense(x)
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if not tf.is_tensor(x):
        if dtype == "bool":
            # TensorFlow boolean conversion is stricter than other backends.
            # It does not allow ints. We convert without dtype and cast instead.
            x = tf.convert_to_tensor(x)
            return tf.cast(x, dtype)
        return tf.convert_to_tensor(x, dtype=dtype)
    elif dtype is not None and not x.dtype == dtype:
        if isinstance(x, tf.SparseTensor):
            x_shape = x.shape
            x = tf.cast(x, dtype)
            x.set_shape(x_shape)
            return x
        return tf.cast(x, dtype=dtype)
    return x


def convert_to_numpy(x):
    if isinstance(x, tf.SparseTensor):
        x = sparse_to_dense(x)
    elif isinstance(x, tf.IndexedSlices):
        x = tf.convert_to_tensor(x)
    elif isinstance(x, tf.RaggedTensor):
        x = x.to_tensor()
    return np.asarray(x)


def is_tensor(x):
    return tf.is_tensor(x)


def shape(x):
    """Always return a tuple shape.

    `tf.shape` will return a `tf.Tensor`, which differs from the tuple return
    type on the torch and jax backends. We write our own method instead which
    always returns a tuple, with integer values when the shape is known, and
    tensor values when the shape is unknown (this is tf specific, as dynamic
    shapes do not apply in other backends).
    """
    if isinstance(x, KerasTensor):
        return x.shape
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    dynamic = tf.shape(x)
    if x.shape == tf.TensorShape(None):
        raise ValueError(
            "All tensors passed to `ops.shape` must have a statically known "
            f"rank. Received: x={x} with unknown rank."
        )
    static = x.shape.as_list()
    return tuple(dynamic[i] if s is None else s for i, s in enumerate(static))


def cast(x, dtype):
    dtype = standardize_dtype(dtype)
    if isinstance(x, tf.SparseTensor):
        x_shape = x.shape
        x = tf.cast(x, dtype)
        x.set_shape(x_shape)
        return x
    else:
        return tf.cast(x, dtype=dtype)


def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():
        graph_name = auto_name("scratch_graph")
        with tf.__internal__.FuncGraph(graph_name).as_default():

            def convert_keras_tensor_to_tf(x):
                if isinstance(x, KerasTensor):
                    if x.sparse:
                        return tf.compat.v1.sparse_placeholder(
                            shape=x.shape, dtype=x.dtype
                        )
                    else:
                        return tf.compat.v1.placeholder(
                            shape=x.shape, dtype=x.dtype
                        )
                return x

            args, kwargs = tree.map_structure(
                convert_keras_tensor_to_tf, (args, kwargs)
            )
            tf_out = fn(*args, **kwargs)

            def convert_tf_to_keras_tensor(x):
                if tf.is_tensor(x):
                    return KerasTensor(
                        x.shape, x.dtype, sparse=isinstance(x, tf.SparseTensor)
                    )
                return x

            output_spec = tree.map_structure(convert_tf_to_keras_tensor, tf_out)
    return output_spec


def cond(pred, true_fn, false_fn):
    return tf.cond(pred, true_fn=true_fn, false_fn=false_fn)


def vectorized_map(function, elements):
    return tf.vectorized_map(function, elements)


def scatter(indices, values, shape):
    return tf.scatter_nd(indices, values, shape)


def scatter_update(inputs, indices, updates):
    return tf.tensor_scatter_nd_update(inputs, indices, updates)


def slice(inputs, start_indices, shape):
    return tf.slice(inputs, start_indices, shape)


def slice_update(inputs, start_indices, updates):
    return dynamic_update_slice(inputs, updates, start_indices)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)

    def _body(*args):
        outputs = body(*args)
        return tuple(outputs) if is_tuple else (outputs,)

    outputs = tf.while_loop(
        cond,
        _body,
        loop_vars,
        maximum_iterations=maximum_iterations,
    )
    return outputs if is_tuple else outputs[0]


def fori_loop(lower, upper, body_fun, init_val):
    return tf.while_loop(
        lambda i, val: i < upper,
        lambda i, val: (i + 1, body_fun(i, val)),
        (lower, init_val),
    )[1]


def stop_gradient(variable):
    return tf.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    return tf.unstack(x, num=num, axis=axis)


def custom_gradient(fun):
    return tf.custom_gradient(f=fun)


class name_scope(base_name_scope):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._tf_name_scope = tf.name_scope(name)

    def __enter__(self):
        name_scope_stack = global_state.get_global_attribute(
            "name_scope_stack", default=[], set_to_default=True
        )
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if (
                self.caller is not None
                and self.caller is parent_caller
                and self.name == parent_name
            ):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        self._tf_name_scope.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        if self._pop_on_exit:
            self._tf_name_scope.__exit__(*args, **kwargs)


def device_scope(device_name):
    return tf.device(device_name)

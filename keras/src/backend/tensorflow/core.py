import builtins

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import is_int_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.tensorflow.sparse import sparse_to_dense
from keras.src.utils.naming import auto_name

SUPPORTS_SPARSE_TENSORS = True
# https://github.com/tensorflow/tensorflow/issues/78338
IS_THREAD_SAFE = False


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
            value,
            dtype=self._dtype,
            trainable=self.trainable,
            name=self.name,
            aggregation=self._map_aggregation(self.aggregation),
        )

    def _initialize_with_initializer(self, initializer):
        self._value = tf.Variable(
            lambda: initializer(self._shape, dtype=self._dtype),
            dtype=self._dtype,
            trainable=self.trainable,
            name=self.name,
            aggregation=self._map_aggregation(self.aggregation),
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
            self._initialize_with_initializer(self._initializer)
            self._initializer = None

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

    def _map_aggregation(self, aggregation):
        mapping = {
            "none": tf.VariableAggregation.NONE,
            "sum": tf.VariableAggregation.SUM,
            "mean": tf.VariableAggregation.MEAN,
            "only_first_replica": tf.VariableAggregation.ONLY_FIRST_REPLICA,
        }
        return mapping[aggregation]


def convert_to_tensor(x, dtype=None, sparse=None):
    if isinstance(x, tf.SparseTensor) and sparse is not None and not sparse:
        x = sparse_to_dense(x)
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if not tf.is_tensor(x):
        if dtype == "bool" or is_int_dtype(dtype):
            # TensorFlow conversion is stricter than other backends, it does not
            # allow ints for bools or floats for ints. We convert without dtype
            # and cast instead.
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
    return np.array(x)


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
    if x.shape == tf.TensorShape(None):
        raise ValueError(
            "All tensors passed to `ops.shape` must have a statically known "
            f"rank. Received: x={x} with unknown rank."
        )
    shape = x.shape.as_list()
    dynamic = tf.shape(x)
    for i in range(len(shape)):
        if shape[i] is None:
            try:
                shape[i] = dynamic[i]
            except:
                # With RaggedTensors, accessing a ragged dimension will fail,
                # we leave it as None.
                pass
    return tuple(shape)


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
    with StatelessScope(), SymbolicScope():
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
    if isinstance(pred, tf.Variable):
        return tf.cond(pred, true_fn=true_fn, false_fn=false_fn)
    return tf.__internal__.smart_cond.smart_cond(
        pred, true_fn=true_fn, false_fn=false_fn
    )


def vectorized_map(function, elements):
    return tf.vectorized_map(function, elements)


def map(f, xs):
    xs = tree.map_structure(convert_to_tensor, xs)

    def get_fn_output_signature(x):
        out = f(x)
        return tree.map_structure(tf.TensorSpec.from_tensor, out)

    if tree.is_nested(xs):
        input = tree.pack_sequence_as(xs, [x[0] for x in tree.flatten(xs)])
        fn_output_signature = get_fn_output_signature(input)
        return tf.map_fn(f, xs, fn_output_signature=fn_output_signature)
    else:
        fn_output_signature = get_fn_output_signature(xs[0])
        return tf.map_fn(f, xs, fn_output_signature=fn_output_signature)


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    # We have reimplemented `scan` to match the behavior of `jax.lax.scan`
    # Ref: tf.scan, jax.lax.scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        # xs_flat = flatten_input(xs)
        xs_flat = tree.flatten(xs)
        xs_flat = [tf.convert_to_tensor(elem) for elem in xs_flat]
        n = int(length) if length is not None else tf.shape(xs_flat[0])[0]

    # TensorArrays are always flat
    xs_array = [
        tf.TensorArray(
            dtype=x.dtype,
            size=n,
            dynamic_size=False,
            element_shape=x.shape[1:],
            infer_shape=True,
        )
        for x in xs_flat
    ]
    xs_array = [x_a.unstack(x) for x_a, x in zip(xs_array, xs_flat)]

    init_flat = tree.flatten(init)
    carry_flat = [tf.convert_to_tensor(init) for init in init_flat]

    # Store the intermediate values
    # Note: there is a constraint that the output of `f` must have the same
    # shape and dtype as carry (`init`).
    ys_array = [
        tf.TensorArray(
            dtype=carry.dtype,
            size=n,
            dynamic_size=False,
            element_shape=carry.shape,
            infer_shape=True,
        )
        for carry in carry_flat
    ]
    carry_array = [
        tf.TensorArray(
            dtype=carry.dtype,
            size=1,
            dynamic_size=False,
            clear_after_read=False,
            element_shape=carry.shape,
            infer_shape=True,
        )
        for carry in carry_flat
    ]
    carry_array = [
        carry.write(0, c) for (carry, c) in zip(carry_array, carry_flat)
    ]

    def loop_body(i, carry_array, ys_array):
        packed_xs = (
            pack_input([xs.read(i) for xs in xs_array])
            if len(xs_array) > 0
            else None
        )
        packed_carry = pack_output([carry.read(0) for carry in carry_array])

        carry, ys = f(packed_carry, packed_xs)

        if ys is not None:
            flat_ys = tree.flatten(ys)
            ys_array = [ys.write(i, v) for (ys, v) in zip(ys_array, flat_ys)]
        if carry is not None:
            flat_carry = tree.flatten(carry)
            carry_array = [
                carry.write(0, v) for (carry, v) in zip(carry_array, flat_carry)
            ]
        next_i = i + 1 if not reverse else i - 1
        return (next_i, carry_array, ys_array)

    if isinstance(unroll, bool):
        unroll = max(n, 1) if unroll else 1

    _, carry_array, ys_array = tf.while_loop(
        lambda i, _1, _2: i >= 0 if reverse else i < n,
        loop_body,
        (n - 1 if reverse else 0, carry_array, ys_array),
        parallel_iterations=unroll,
    )

    ys_flat = [ys.stack() for ys in ys_array]
    carry_flat = [carry.read(0) for carry in carry_array]
    if xs is not None:
        n_static = xs_flat[0].get_shape().with_rank_at_least(1)[0]
        if not isinstance(n_static, int):
            for x in xs_flat[1:]:
                n_static.assert_is_compatible_with(
                    x.get_shape().with_rank_at_least(1)[0]
                )
        for r in ys_flat:
            r.set_shape(tf.TensorShape(n_static).concatenate(r.get_shape()[1:]))
    return pack_output(carry_flat), pack_output(ys_flat)


def associative_scan(f, elems, reverse=False, axis=0):
    # Implementation is the same as tfp.math.scan_associative
    # with additional checks to ensure similar behavior with jax
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [tf.convert_to_tensor(elem) for elem in elems_flat]
    if reverse:
        elems_flat = [tf.reverse(elem, [axis]) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        c_flat = tree.flatten(c)
        return c_flat

    def _get_dim(x):
        return shape(x)[axis]

    # TODO add constant dim check
    num_elems = _get_dim(elems_flat[0])
    if not all(_get_dim(elem) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [tf.shape(elem) for elem in elems_flat]
            )
        )

    def _interleave(a, b, axis):
        # [a b c ...] [d e f ...] -> [a d b e c f ...]
        num_elems_a = _get_dim(a)
        num_elems_b = _get_dim(b)

        # Note that interleaving implies rank(a)==rank(b).
        axis = tf.where(axis >= 0, axis, tf.rank(a) + axis)
        axis = (
            int(axis)  # Avoid ndarray values.
            if tf.get_static_value(axis) is not None
            else axis
        )

        def _interleave_with_b(a):
            return tf.reshape(
                # Work around lack of support for Tensor axes in
                # `tf.stack` by using `concat` and `expand_dims` instead.
                tf.concat(
                    [
                        tf.expand_dims(a, axis=axis + 1),
                        tf.expand_dims(b, axis=axis + 1),
                    ],
                    axis=axis + 1,
                ),
                tf.concat(
                    [
                        a.get_shape()[:axis],
                        [2 * num_elems_b],
                        a.get_shape()[axis + 1 :],
                    ],
                    axis=0,
                ),
            )

        return tf.cond(
            tf.equal(num_elems_a, num_elems_b + 1),
            lambda: tf.concat(
                [
                    _interleave_with_b(
                        slice_along_axis(a, None, -1, axis=axis)
                    ),
                    slice_along_axis(a, -1, None, axis=axis),
                ],
                axis=axis,
            ),
            lambda: _interleave_with_b(a),
        )

    def _scan(elems):
        elem_length = _get_dim(elems[0])
        a = [slice_along_axis(elem, 0, -1, step=2, axis=axis) for elem in elems]
        b = [
            slice_along_axis(elem, 1, None, step=2, axis=axis) for elem in elems
        ]
        reduced_elems = _combine(a, b)

        def _handle_base_case_elem_length_two():
            return [
                tf.concat(
                    [slice_along_axis(elem, 0, 1, axis=axis), reduced_elem],
                    axis=axis,
                )
                for (reduced_elem, elem) in zip(reduced_elems, elems)
            ]

        def _handle_base_case_elem_length_three():
            reduced_reduced_elems = _combine(
                reduced_elems,
                [slice_along_axis(elem, 2, 3, axis=axis) for elem in elems],
            )
            return [
                tf.concat(
                    [
                        slice_along_axis(elem, 0, 1, axis=axis),
                        reduced_elem,
                        reduced_reduced_elem,
                    ],
                    axis=axis,
                )
                for (reduced_reduced_elem, reduced_elem, elem) in zip(
                    reduced_reduced_elems, reduced_elems, elems
                )
            ]

        at_base_case = tf.logical_or(
            tf.equal(elem_length, 2), tf.equal(elem_length, 3)
        )

        def _base_case():
            return tf.cond(
                tf.equal(elem_length, 2),
                _handle_base_case_elem_length_two,
                _handle_base_case_elem_length_three,
            )

        def _recursive_case():
            odd_elems = _scan(reduced_elems)

            def _even_length_case():
                return _combine(
                    [
                        slice_along_axis(odd_elem, 0, -1, axis=axis)
                        for odd_elem in odd_elems
                    ],
                    [
                        slice_along_axis(elem, 2, None, 2, axis=axis)
                        for elem in elems
                    ],
                )

            def _odd_length_case():
                return _combine(
                    [odd_elem for odd_elem in odd_elems],
                    [
                        slice_along_axis(elem, 2, None, 2, axis=axis)
                        for elem in elems
                    ],
                )

            results = tf.cond(
                tf.equal(elem_length % 2, 0),
                _even_length_case,
                _odd_length_case,
            )

            even_elems = [
                tf.concat(
                    [slice_along_axis(elem, 0, 1, axis=axis), result], axis=axis
                )
                for (elem, result) in zip(elems, results)
            ]
            return list(
                builtins.map(
                    lambda a, b: _interleave(a, b, axis=axis),
                    even_elems,
                    odd_elems,
                )
            )

        return tf.cond(at_base_case, _base_case, _recursive_case)

    scans = _scan(elems_flat)
    if reverse:
        scans = [tf.reverse(scanned, [axis]) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


def scatter(indices, values, shape):
    return tf.scatter_nd(indices, values, shape)


def scatter_update(inputs, indices, updates):
    return tf.tensor_scatter_nd_update(inputs, indices, updates)


def slice(inputs, start_indices, shape):
    return tf.slice(inputs, start_indices, shape)


def slice_update(inputs, start_indices, updates):
    return dynamic_update_slice(inputs, updates, start_indices)


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = tf.clip_by_value(index, 0, len(branches) - 1)

    # Workaround to deal with python closures. More details:
    # https://github.com/tensorflow/tensorflow/issues/8776#issuecomment-311383887
    def gen_fn(i):
        return lambda: branches[i](*operands)

    branch_fns = [gen_fn(i) for i in range(len(branches))]
    return tf.switch_case(index, branch_fns)


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


def random_seed_dtype():
    # tensorflow random operation only works on int32/int64, not uint32.
    return "int64"


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

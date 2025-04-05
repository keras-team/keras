# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lazily initialized variables, useful for creating symbolic TF-Keras model."""

import threading

# isort: off
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib

_DISABLE_LAZY_VARIABLE_INIT = threading.local()


def _infer_shape_dtype_and_create_handle(initial_value, shape, dtype, name):
    """Infer shape and dtype from initial_value and create a variable handle."""
    with ops.name_scope(name, "Variable", skip_on_eager=False) as name:
        handle_name = ops.name_from_scope_name(name)
        unique_id = "%s_%d" % (handle_name, ops.uid())

        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        device_context_manager = ops.NullContextmanager
        attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                s=[compat.as_bytes(f"loc:@{handle_name}")]
            )
        )
        with ops.get_default_graph()._attr_scope({"_class": attr}):
            with ops.name_scope("Initializer"), device_context_manager(None):
                if not callable(initial_value):
                    if isinstance(
                        initial_value, trackable.CheckpointInitialValue
                    ):
                        raise NotImplementedError(
                            "CheckpointInitialValue is not supported to be the "
                            "initial value of a lazy variable."
                        )
                    initial_value = ops.convert_to_tensor(
                        initial_value, name="initial_value", dtype=dtype
                    )
                    assert not callable(initial_value)

                    assert initial_value.shape.is_compatible_with(shape)
                    dtype = dtype or initial_value.dtype.base_dtype
                    shape = shape or initial_value.shape

            assert dtype
            assert shape
            handle = (
                resource_variable_ops._variable_handle_from_shape_and_dtype(
                    shape=shape,
                    dtype=dtype,
                    shared_name=None,  # Never shared
                    name=name,
                    graph_mode=False,
                    initial_value=None,
                )
            )
            # initial_value=initial_value if not callable(initial_value) else
            # None)
    return initial_value, shape, dtype, handle, handle_name, unique_id


class LazyInitVariable(resource_variable_ops.BaseResourceVariable):
    """Lazily initialized variables.

    The major use case for this class is to serve as a memory efficient
    alternative for tf.Variable. The resource handle of this class is point to
    nothing, which mean it will raise error when its value is fetched in a eager
    context. Having said that, it will perform like a normal tf.Variable when
    using with graph tensor, like KerasTensor produced from tf.keras.Input.
    """

    def __init__(
        self,
        initial_value=None,
        trainable=None,
        collections=None,
        validate_shape=True,
        caching_device=None,
        name=None,
        dtype=None,
        variable_def=None,
        import_scope=None,
        constraint=None,
        distribute_strategy=None,
        synchronization=None,
        aggregation=None,
        shape=None,
        **kwargs,
    ):
        assert context.executing_eagerly()  # To simplify the logic
        assert variable_def is None  # Not supported yet.
        assert caching_device is None  # Not supported yet

        if initial_value is None:
            raise ValueError(
                "The `initial_value` arg to `tf.Variable` must "
                "be specified except when you are not providing a "
                "`variable_def`. You provided neither."
            )

        if (
            isinstance(initial_value, tensor.Tensor)
            and hasattr(initial_value, "graph")
            and initial_value.graph.building_function
        ):
            raise ValueError(
                f"Argument `initial_value` ({initial_value}) could not "
                "be lifted out of a `tf.function`. "
                f"(Tried to create variable with name='{name}'). "
                "To avoid this error, when constructing `tf.Variable`s "
                "inside of `tf.function` you can create the "
                "`initial_value` tensor in a "
                "`tf.init_scope` or pass a callable `initial_value` "
                "(e.g., `tf.Variable(lambda : "
                "tf.truncated_normal([10, 40]))`). "
                "Please file a feature request if this "
                "restriction inconveniences you."
            )

        if constraint is not None and not callable(constraint):
            raise ValueError(
                "Argument `constraint` must be None or a callable. "
                f"a callable. Got a {type(constraint)}:  {constraint}"
            )

        self._name = name
        (
            initial_value,
            shape,
            dtype,
            handle,
            handle_name,
            unique_id,
        ) = _infer_shape_dtype_and_create_handle(
            initial_value, shape, dtype, name
        )

        super().__init__(
            distribute_strategy=distribute_strategy,
            initial_value=initial_value,
            shape=shape,
            dtype=dtype,
            name=name,
            unique_id=unique_id,
            handle_name=handle_name,
            constraint=constraint,
            handle=handle,
            graph_element=None,
            trainable=trainable,
            synchronization=synchronization,
            aggregation=aggregation,
            in_graph_mode=False,
        )

    # TODO(scottzhu): This method and create_and_initialize might be removed if
    # we decide to just use the tf.Variable to replace this class.
    def initialize(self):
        with ops.name_scope(self._name, "Variable", skip_on_eager=False):
            with ops.colocate_with(self._handle), ops.name_scope("Initializer"):
                if callable(self._initial_value):
                    initial_value = self._initial_value()
                else:
                    initial_value = self._initial_value

                if not initial_value.shape.is_compatible_with(self._shape):
                    raise ValueError(
                        "In this `tf.Variable` creation, the initial value's "
                        f"shape ({initial_value.shape}) is not compatible with "
                        "the explicitly supplied `shape` "
                        f"argument ({self._shape})."
                    )
                assert self._dtype is initial_value.dtype.base_dtype
            gen_resource_variable_ops.assign_variable_op(
                self._handle, initial_value
            )

    def create_and_initialize(self):
        if callable(self._initial_value):
            initial_value = self._initial_value()

        with ops.device(initial_value.device):
            (
                initial_value,
                shape,
                dtype,
                handle,
                handle_name,
                unique_id,
            ) = _infer_shape_dtype_and_create_handle(
                initial_value, self._shape, self._dtype, self._name
            )
            self.initialize()

        super().__init__(
            trainable=self._trainable,
            shape=shape,
            dtype=dtype,
            handle=handle,
            synchronization=self._synchronization,
            constraint=self._constraint,
            aggregation=self._aggregation,
            distribute_strategy=self._distribute_strategy,
            name=self._name,
            unique_id=unique_id,
            handle_name=handle_name,
            graph_element=None,
            initial_value=initial_value,
            initializer_op=None,
            is_initialized_op=None,
            cached_value=None,
            caching_device=None,
        )


def _lazy_init_variable_creator(next_creator, **kwargs):
    if getattr(_DISABLE_LAZY_VARIABLE_INIT, "disabled", False):
        return next_creator(**kwargs)
    else:
        return LazyInitVariable(**kwargs)


@tf_contextlib.contextmanager
def lazy_init_scope():
    with variable_scope.variable_creator_scope(_lazy_init_variable_creator):
        yield


@tf_contextlib.contextmanager
def disable_init_variable_creator():
    try:
        global _DISABLE_LAZY_VARIABLE_INIT
        existing_value = getattr(_DISABLE_LAZY_VARIABLE_INIT, "disabled", False)
        _DISABLE_LAZY_VARIABLE_INIT.disabled = True
        yield
    finally:
        _DISABLE_LAZY_VARIABLE_INIT.disabled = existing_value


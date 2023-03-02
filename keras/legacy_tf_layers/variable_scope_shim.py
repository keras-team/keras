# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Contains a shim to allow using TF1 get_variable code in TF2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools

import tensorflow.compat.v2 as tf

from keras.engine import base_layer
from keras.utils import layer_utils
from keras.utils import tf_inspect

# isort: off
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


def as_shape(shape):
    """Converts the given object to a TensorShape."""
    if isinstance(shape, tf.TensorShape):
        return shape
    else:
        return tf.TensorShape(shape)


def _is_callable_object(obj):
    return hasattr(obj, "__call__") and tf_inspect.ismethod(obj.__call__)


def _has_kwargs(fn):
    """Returns whether the passed callable has **kwargs in its signature.

    Args:
      fn: Function, or function-like object (e.g., result of
        `functools.partial`).

    Returns:
      `bool`: if `fn` has **kwargs in its signature.

    Raises:
       `TypeError`: If fn is not a Function, or function-like object.
    """
    if isinstance(fn, functools.partial):
        fn = fn.func
    elif _is_callable_object(fn):
        fn = fn.__call__
    elif not callable(fn):
        raise TypeError(
            f"fn should be a function-like object, but is of type {type(fn)}."
        )
    return tf_inspect.getfullargspec(fn).varkw is not None


def fn_args(fn):
    """Get argument names for function-like object.

    Args:
      fn: Function, or function-like object (e.g., result of
        `functools.partial`).

    Returns:
      `tuple` of string argument names.

    Raises:
      ValueError: if partial function has positionally bound arguments
    """
    if isinstance(fn, functools.partial):
        args = fn_args(fn.func)
        args = [a for a in args[len(fn.args) :] if a not in (fn.keywords or [])]
    else:
        if hasattr(fn, "__call__") and tf_inspect.ismethod(fn.__call__):
            fn = fn.__call__
        args = tf_inspect.getfullargspec(fn).args
        if _is_bound_method(fn) and args:
            # If it's a bound method, it may or may not have a self/cls first
            # argument; for example, self could be captured in *args.
            # If it does have a positional argument, it is self/cls.
            args.pop(0)
    return tuple(args)


def _is_bound_method(fn):
    _, fn = tf.__internal__.decorator.unwrap(fn)
    return tf_inspect.ismethod(fn) and (fn.__self__ is not None)


def validate_synchronization_aggregation_trainable(
    synchronization, aggregation, trainable, name
):
    """Given user-provided variable properties, sets defaults and validates."""
    if aggregation is None:
        aggregation = tf.compat.v1.VariableAggregation.NONE
    else:
        if not isinstance(
            aggregation,
            (tf.compat.v1.VariableAggregation, tf.VariableAggregation),
        ):
            try:
                aggregation = tf.VariableAggregation(aggregation)
            except ValueError:
                raise ValueError(
                    "Invalid variable aggregation mode: {} "
                    "for variable: {}".format(aggregation, name)
                )
    if synchronization is None:
        synchronization = tf.VariableSynchronization.AUTO
    else:
        try:
            synchronization = tf.VariableSynchronization(synchronization)
        except ValueError:
            raise ValueError(
                "Invalid variable synchronization mode: {} "
                "for variable: {}".format(synchronization, name)
            )
    if trainable is None:
        trainable = synchronization != tf.VariableSynchronization.ON_READ
    return synchronization, aggregation, trainable


class _EagerVariableStore(tf.Module):
    """TF2-safe VariableStore that avoids collections & tracks regularizers.

    New variable names and new variables can be created; all stored
    variables are initialized with the initializer passed to __init__.

    All variables get created in `tf.init_scope.` to avoid a bad
    interaction between `tf.function` `FuncGraph` internals, Keras
    Functional Models, and TPUStrategy variable initialization.

    Also, it always acts as if reuse is set to either "TRUE" or
    tf.compat.v1.AUTO_REUSE

    Attributes:
      vars: a dictionary with string names (same as passed in GetVar) as keys
        and the corresponding TensorFlow Variables as values.
      regularizers: a dictionary with string names as keys and the corresponding
        callables that return losses as values.
      layers: a dictionary with string names as keys and the corresponding
        nested keras layers as values.
    """

    def __init__(self):
        """Create a variable store."""
        self._vars = {}  # A dictionary of the stored TensorFlow variables.
        self._regularizers = (
            {}
        )  # A dict mapping var names to their regularizers.
        self._layers = {}  # A dictionary of stored keras layers.
        self._store_eager_variables = True

    @contextlib.contextmanager
    def scope(self):
        with vs.with_variable_store(self):
            yield

    def get_variable(
        self,
        name,
        shape=None,
        dtype=tf.float32,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=None,
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        custom_getter=None,
        constraint=None,
        synchronization=tf.VariableSynchronization.AUTO,
        aggregation=tf.compat.v1.VariableAggregation.NONE,
    ):
        """Gets an existing variable with these parameters or create a new one.

        If a variable with the given name is already stored, we return the
        stored variable. Otherwise, we create a new one.

        Set `reuse` to `True` when you only want to reuse existing Variables.
        Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you
        want variables to be created if they don't exist or returned if they do.
        In this shim, `reuse` of `False` will be treated as auto-reuse.

        If initializer is `None` (the default), the default initializer passed
        in the constructor is used. If that one is `None` too, we use a new
        `glorot_uniform_initializer`. If initializer is a Tensor, we use it as a
        value and derive the shape from the initializer.

        If a partitioner is provided, a `PartitionedVariable` is returned.
        Accessing this object as a `Tensor` returns the shards concatenated
        along the partition axis.

        Some useful partitioners are available.  See, e.g.,
        `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

        Args:
          name: The name of the new or existing variable.
          shape: Shape of the new or existing variable.
          dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
          initializer: Initializer for the variable.
          regularizer: A (Tensor -> Tensor or None) function; the result of
            applying it on a newly created variable will be added to the
            collection GraphKeys.REGULARIZATION_LOSSES and can be used for
            regularization.
          reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation
            of variables. When eager execution is enabled  this argument is
            always forced to be False.
          trainable: If `True` also add the variable to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
            defaults to `True`, unless `synchronization` is set to `ON_READ`, in
            which case it defaults to `False`.
          collections: List of graph collections keys to add the `Variable` to.
            Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
          caching_device: Optional device string or function describing where
            the Variable should be cached for reading.  Defaults to the
            Variable's device.  If not `None`, caches on another device.
            Typical use is to cache on the device where the Ops using the
            `Variable` reside, to deduplicate copying through `Switch` and other
            conditional statements.
          partitioner: Optional callable that accepts a fully defined
            `TensorShape` and dtype of the `Variable` to be created, and returns
            a list of partitions for each axis (currently only one axis can be
            partitioned).
          validate_shape: If False, allows the variable to be initialized with a
            value of unknown shape. If True, the default, the shape of
            initial_value must be known.
          use_resource: If False, creates a regular Variable. If True, creates
            instead an experimental ResourceVariable which has well-defined
            semantics. Defaults to False (will later change to True). When eager
            execution is enabled this argument is always forced to be true.
          custom_getter: Callable that takes as a first argument the true
            getter, and allows overwriting the internal get_variable method. The
            signature of `custom_getter` should match that of this method, but
            the most future-proof version will allow for changes:
            `def custom_getter(getter, *args, **kwargs)`.
            Direct access to all `get_variable` parameters is also allowed:
            `def custom_getter(getter, name, *args, **kwargs)`.
            A simple identity custom getter that simply creates variables with
            modified names is:
            ```python
            def custom_getter(getter, name, *args, **kwargs):
              return getter(name + '_suffix', *args, **kwargs)
            ```
          constraint: An optional projection function to be applied to the
            variable after being updated by an `Optimizer` (e.g. used to
            implement norm constraints or value constraints for layer weights).
            The function must take as input the unprojected Tensor representing
            the value of the variable and return the Tensor for the projected
            value (which must have the same shape). Constraints are not safe to
            use when doing asynchronous distributed training.
          synchronization: Indicates when a distributed a variable will be
            aggregated. Accepted values are constants defined in the class
            `tf.VariableSynchronization`. By default the synchronization is set
            to `AUTO` and the current `DistributionStrategy` chooses when to
            synchronize.
          aggregation: Indicates how a distributed variable will be aggregated.
            Accepted values are constants defined in the class
            `tf.VariableAggregation`.

        Returns:
          The created or existing `Variable` (or `PartitionedVariable`, if a
          partitioner was used).

        Raises:
          ValueError: when creating a new variable and shape is not declared,
            when reusing a variable and specifying a conflicting shape,
            or when violating reuse during variable creation.
          RuntimeError: when eager execution is enabled and not called from an
            EagerVariableStore.
        """
        if custom_getter is not None and not callable(custom_getter):
            raise ValueError(
                f"Passed a custom_getter which is not callable: {custom_getter}"
            )

        with tf.init_scope():
            if tf.executing_eagerly():
                # Variable creation and initialization takes place in
                # `init_scope`s; as such, if an `init_scope` lifts us into the
                # eager context, then we need to use `ResourceVariable`s.
                use_resource = True

        # Note that it's fine to reuse eager variables whose initialization was
        # lifted from a function-building graph into the eager context (that's
        # why the following clause is not wrapped in an `init_scope`); lifted
        # variables are tracked by the graph's `VariableStore`.
        if not reuse:
            reuse = tf.compat.v1.AUTO_REUSE

        # If a *_ref type is passed in an error would be triggered further down
        # the stack. We prevent this using base_dtype to get a non-ref version
        # of the type, before doing anything else. When _ref types are removed
        # in favor of resources, this line can be removed.
        try:
            dtype = dtype.base_dtype
        except AttributeError:
            # .base_dtype not existing means that we will try and use the raw
            # dtype which was passed in - this might be a NumPy type which is
            # valid.
            pass

        # This is the main logic of get_variable.  However, custom_getter
        # may override this logic.  So we save it as a callable and pass
        # it to custom_getter.
        # Note: the parameters of _true_getter, and their documentation, match
        # *exactly* item-for-item with the docstring of this method.
        def _true_getter(
            name,
            shape=None,
            dtype=tf.float32,
            initializer=None,
            regularizer=None,
            reuse=None,
            trainable=None,
            collections=None,
            caching_device=None,
            partitioner=None,
            validate_shape=True,
            use_resource=None,
            constraint=None,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.compat.v1.VariableAggregation.NONE,
        ):
            # Partitioned variable currently unsupported w/ the shim
            if partitioner is not None:
                raise ValueError(
                    "`partitioner` arg for `get_variable` is unsupported in "
                    "TF2. File a bug if you need help. "
                    "You passed %s" % partitioner
                )

            # Single variable case
            if f"{name}/part_0" in self._vars:
                raise ValueError(
                    "No partitioner was provided, but a partitioned version of "
                    "the variable was found: %s/part_0. Perhaps a variable of "
                    "the same name was already created with "
                    "partitioning?" % name
                )

            return self._get_single_variable(
                name=name,
                shape=shape,
                dtype=dtype,
                initializer=initializer,
                regularizer=regularizer,
                reuse=reuse,
                trainable=trainable,
                caching_device=caching_device,
                validate_shape=validate_shape,
                constraint=constraint,
                synchronization=synchronization,
                aggregation=aggregation,
            )

        (
            synchronization,
            aggregation,
            trainable,
        ) = validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name
        )

        if custom_getter is not None:
            # Handle backwards compatibility with getter arguments that were
            # added to the API after users started writing custom getters.
            custom_getter_kwargs = {
                "getter": _true_getter,
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "initializer": initializer,
                "regularizer": regularizer,
                "reuse": reuse,
                "trainable": trainable,
                "collections": collections,
                "caching_device": caching_device,
                "partitioner": partitioner,
                "validate_shape": validate_shape,
                "use_resource": use_resource,
                "synchronization": synchronization,
                "aggregation": aggregation,
            }
            # `fn_args` and `has_kwargs` can handle functions,
            # `functools.partial`, `lambda`.
            if "constraint" in fn_args(custom_getter) or _has_kwargs(
                custom_getter
            ):
                custom_getter_kwargs["constraint"] = constraint
            return custom_getter(**custom_getter_kwargs)
        else:
            return _true_getter(
                name,
                shape=shape,
                dtype=dtype,
                initializer=initializer,
                regularizer=regularizer,
                reuse=reuse,
                trainable=trainable,
                collections=collections,
                caching_device=caching_device,
                partitioner=partitioner,
                validate_shape=validate_shape,
                use_resource=use_resource,
                constraint=constraint,
                synchronization=synchronization,
                aggregation=aggregation,
            )

    def _get_single_variable(
        self,
        name,
        shape=None,
        dtype=tf.float32,
        initializer=None,
        regularizer=None,
        partition_info=None,
        reuse=None,
        trainable=None,
        caching_device=None,
        validate_shape=True,
        constraint=None,
        synchronization=tf.VariableSynchronization.AUTO,
        aggregation=tf.compat.v1.VariableAggregation.NONE,
    ):
        """Get or create a single Variable (e.g. a shard or entire variable).

        See the documentation of get_variable above (ignore partitioning
        components) for details.

        Args:
          name: see get_variable.
          shape: see get_variable.
          dtype: see get_variable.
          initializer: see get_variable.
          regularizer: see get_variable.
          partition_info: _PartitionInfo object.
          reuse: see get_variable.
          trainable: see get_variable.
          caching_device: see get_variable.
          validate_shape: see get_variable.
          constraint: see get_variable.
          synchronization: see get_variable.
          aggregation: see get_variable.

        Returns:
          A Variable.  See documentation of get_variable above.

        Raises:
          ValueError: See documentation of get_variable above.
        """
        # Set to true if initializer is a constant.
        initializing_from_value = False
        if initializer is not None and not callable(initializer):
            initializing_from_value = True
        if shape is not None and initializing_from_value:
            raise ValueError(
                "If initializer is a constant, do not specify shape."
            )

        dtype = tf.as_dtype(dtype)
        shape = as_shape(shape)

        if name in self._vars:
            # Here we handle the case when returning an existing variable.
            found_var = self._vars[name]
            if not shape.is_compatible_with(found_var.get_shape()):
                raise ValueError(
                    "Trying to share variable %s, but specified shape %s"
                    " and found shape %s."
                    % (name, shape, found_var.get_shape())
                )
            if not dtype.is_compatible_with(found_var.dtype):
                dtype_str = dtype.name
                found_type_str = found_var.dtype.name
                raise ValueError(
                    "Trying to share variable %s, but specified dtype %s"
                    " and found dtype %s." % (name, dtype_str, found_type_str)
                )
            return found_var

        # The code below handles only the case of creating a new variable.
        if reuse is True:
            raise ValueError(
                "Variable %s does not exist, or was not created with "
                "tf.get_variable(). Did you mean to set "
                "reuse=tf.AUTO_REUSE in VarScope?" % name
            )

        # Create the tensor to initialize the variable with default value.
        if initializer is None:
            (
                initializer,
                initializing_from_value,
            ) = self._get_default_initializer(
                name=name, shape=shape, dtype=dtype
            )
        # Enter an init scope when creating the initializer.
        with tf.init_scope():
            if initializing_from_value:
                init_val = initializer
                variable_dtype = None
            else:
                # Instantiate initializer if provided initializer is a type
                # object.
                if tf_inspect.isclass(initializer):
                    initializer = initializer()
                if shape.is_fully_defined():
                    if (
                        "partition_info"
                        in tf_inspect.getargspec(initializer).args
                    ):
                        init_val = functools.partial(
                            initializer,
                            shape.as_list(),
                            dtype=dtype,
                            partition_info=partition_info,
                        )
                    else:
                        init_val = functools.partial(
                            initializer, shape.as_list(), dtype=dtype
                        )
                    variable_dtype = dtype.base_dtype
                else:
                    init_val = initializer
                    variable_dtype = None

        # Create the variable (Always eagerly as a workaround for a strange
        # tpu / funcgraph / keras functional model interaction )
        with tf.init_scope():
            v = tf.Variable(
                initial_value=init_val,
                name=name,
                trainable=trainable,
                caching_device=caching_device,
                dtype=variable_dtype,
                validate_shape=validate_shape,
                constraint=constraint,
                synchronization=synchronization,
                aggregation=aggregation,
            )

        self._vars[name] = v
        logging.vlog(
            1,
            "Created variable %s with shape %s and init %s",
            v.name,
            format(shape),
            initializer,
        )

        # Run the regularizer if requested and save the resulting loss.
        if regularizer:
            self.add_regularizer(v, regularizer)

        return v

    def get_or_create_layer(self, name, create_layer_method):
        if name not in self._layers:
            layer = create_layer_method()
            self._layers[name] = layer
            if isinstance(layer, base_layer.Layer):
                self._regularizers[name] = lambda: tf.math.reduce_sum(
                    layer.losses
                )
        return self._layers[name]

    def add_regularizer(self, var, regularizer):
        self._regularizers[var.name] = functools.partial(regularizer, var)

    # Initialize variable when no initializer provided
    def _get_default_initializer(self, name, shape=None, dtype=tf.float32):
        """Provide a default initializer and a corresponding value.

        Args:
          name: see get_variable.
          shape: see get_variable.
          dtype: see get_variable.

        Returns:
          initializer and initializing_from_value. See get_variable above.

        Raises:
          ValueError: When giving unsupported dtype.
        """
        del shape
        # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
        if dtype.is_floating:
            initializer = tf.compat.v1.glorot_uniform_initializer()
            initializing_from_value = False
        # If dtype is DT_INT/DT_UINT, provide a default value `zero`
        # If dtype is DT_BOOL, provide a default value `FALSE`
        elif (
            dtype.is_integer
            or dtype.is_unsigned
            or dtype.is_bool
            or dtype == tf.string
        ):
            initializer = tf.compat.v1.zeros_initializer()
            initializing_from_value = False
        # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX
        # here?
        else:
            raise ValueError(
                "An initializer for variable %s of %s is required"
                % (name, dtype.base_dtype)
            )

        return initializer, initializing_from_value


@keras_export(v1=["keras.utils.track_tf1_style_variables"])
def track_tf1_style_variables(method):
    """Wrap layer & module methods in this decorator to capture tf1-style
    weights.

    Decorating a `tf.keras.Layer`'s  or `tf.Module`'s methods with this
    decorator will cause the layer/module to track weights created/used
    via `tf.compat.v1.get_variable` (and by extension `tf.compat.v1.layers`)
    inside the decorated method.

    In addition to tracking the weights themselves under the standard
    `layer.variable`/`module.variable`/etc. properties, if the method belongs
    to a `tf.keras.Layer` then any regularization losses specified via the
    `get_variable` or `tf.compat.v1.layers` regularizer arguments will get
    tracked by the layer under the standard `layer.losses` property.

    This tracking enables using large classes of TF1-style model-forward-pass
    code inside of Keras layers or `tf.Modules` in TF2 with TF2 behaviors
    enabled.

    Example of capturing tf.compat.v1.layer-based modeling code as a Keras
    layer:

    ```python
    class WrappedDoubleDenseLayer(tf.keras.layers.Layer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      @tf.compat.v1.keras.utils.track_tf1_style_variables
      def call(self, inputs):
        with tf.compat.v1.variable_scope("double_dense_layer"):
          out = tf.compat.v1.layers.dense(
              inputs, self.units, name="dense_one",
              kernel_initializer=tf.compat.v1.random_normal_initializer,
              kernel_regularizer="l2")
          out = tf.compat.v1.layers.dense(
              out, self.units, name="dense_two",
              kernel_initializer=tf.compat.v1.random_normal_initializer(),
              kernel_regularizer="l2")
        return out

    # Create a layer that can be used as a standard keras layer
    layer = WrappedDoubleDenseLayer(10)

    # call the layer on inputs
    layer(...)

    # Variables created/used within the scope will be tracked by the layer
    layer.weights
    layer.trainable_variables

    # Regularization losses will be captured in layer.losses after a call,
    # just like any other Keras layer
    reg_losses = layer.losses
    ```

    Example of capturing tf.compat.v1.get_variable-based modeling code as
    a Keras layer:

    ```python
    class WrappedDoubleDenseLayer(tf.keras.layers.Layer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      @tf.compat.v1.keras.utils.track_tf1_style_variables
      def call(self, inputs):
        out = inputs
        with tf.compat.v1.variable_scope("double_dense_layer"):
          with tf.compat.v1.variable_scope("dense_one"):
            # The weights are created with a `regularizer`,
            # so the layer should track their regularization losses
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = tf.compat.v1.math.matmul(out, kernel)
            out = tf.compat.v1.nn.bias_add(out, bias)
          with tf.compat.v1.variable_scope("dense_two"):
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = tf.compat.v1.math.matmul(out, kernel)
            out = tf.compat.v1.nn.bias_add(out, bias)
        return out

    # Create a layer that can be used as a standard keras layer
    layer = WrappedDoubleDenseLayer(10)

    # call the layer on inputs
    layer(...)

    # Variables created/used within the scope will be tracked by the layer
    layer.weights
    layer.trainable_variables

    # Regularization losses will be captured in layer.losses after a call,
    # just like any other Keras layer
    reg_losses = layer.losses
    ```

    Regularization losses:
      Any regularizers specified in the `get_variable` calls or
      `compat.v1.layer` creations will get captured if they occur in your
      decorated method and the method belongs to a
      `tf.keras.Layer`/`tf.keras.Module`. Regularization losses
      are accessible in `layer.losses` after a call just like in a standard
      Keras layer, and will be captured by any model that includes this layer.
      Regularization losses attached to Keras layers/models set as attributes
      of your layer will also get captured in the standard Keras regularization
      loss tracking.

      (While Modules have no `losses` property, no-arg callables to compute
       the regularization losses may be tracked as dict values in a private
       `module._tf1_style_var_store._regularizers` property, but only for
       `tf.compat.v1.layers` and `get_variable` weights and not for any other
       nested Keras layers/tf.Modules)

    Variable scope / variable reuse:
      variable-scope based reuse in your decorated method will be respected,
      and work like variable-scope based reuse in TF1.

    Variable Names/Pre-trained checkpoint loading:
      Variable naming from get_variable and `compat.v1.layer` layers will match
      the TF1 names, so you should be able to re-use your old name-based
      checkpoints. Variable naming for Keras layers/models or for variables
      created by `tf.Variable` may change when going to eager execution.

    Training Arg if you decorate `layer.call`:
      Keras will pass a `training` arg to this layer if `call` contains
      a `training` arg or a `**kwargs` varargs in its call signature,
      similarly to how keras passes `training` to other layers in TF2 that have
      similar signatures in their `call` implementations.
      See more details in the docs
      on `tf.keras.layers.Layer` to understand what will be passed and when.
      Note: tf.compat.v1.layers are usually not called with `training=None`,
      so the training arg to `forward_pass` might not feed through to them
      unless you pass it to their calls explicitly.

    Caveats:
      * TF2 will not prune unused variable updates (or unused outputs). You may
        need to adjust your forward pass code to avoid computations or variable
        updates that you don't intend to use.
      * Avoid Nesting variable creation in tf.function inside of
        methods decorated with `track_tf1_style_variables`
        While the method may safely be used from inside a `tf.function`, using
        a function inside of a decorated method may break the variable scoping.
      * This decorator only adds implicit tracking for legacy tf1-style
        get_variable / compat.v1.layers usage.
        If you would like to use nested Keras layers/models
        inside the decorated method, you need to
        assign them as attributes of your layer so that Keras/Module's standard
        object-oriented weights (and loss tracking for layers) will kick in.
        See the intro to modules, layers, and models
        [guide](https://www.tensorflow.org/guide/intro_to_modules) for more
        info.  As a backup, the `compat.v1.keras.utils.get_or_create_layer`
        method will ease tracking nested keras model weights and losses for
        existing TF1 code, but new code should use explicit tracking.

    Args:
      method: The method to decorate. This should belong to a custom tf.Module,
      tf.keras.layers.Layer, or tf.keras.Model.

    Returns:
      The decorated method.
    """

    def _method_wrapper(self, *args, **kwargs):
        var_store = getattr(self, "_tf1_style_var_store", None)
        if not var_store:
            if not isinstance(self, tf.Module):
                # Raise an error if you incorrectly decorate a method
                # that is not a method of a Module, Layer, or Model:
                raise ValueError(
                    "`@tf.compat.v1.keras.utils.track_tf1_layers_and_variables`"
                    " must be applied to a method of a subclassed `tf.Module`, "
                    "`tf.keras.layers.Layer`, or `tf.keras.Model` and which "
                    "takes `self` as the first argument. But, the first "
                    "argument passed to the decorated method was {}, which "
                    "does not extend Module, Layer, or Model.".format(self)
                )
            var_store = _EagerVariableStore()
            self._tf1_style_var_store = var_store

        existing_regularized_variables = set(var_store._regularizers.keys())
        with var_store.scope():
            out = method(self, *args, **kwargs)

        # If this is a layer method, add the regularization losses
        # to the layer for any newly-created regularized variables
        if isinstance(self, base_layer.Layer):
            for (
                var_name,
                regularizer,
            ) in var_store._regularizers.items():
                if var_name not in existing_regularized_variables:
                    self.add_loss(regularizer)

        return out

    return tf.__internal__.decorator.make_decorator(
        target=method, decorator_func=_method_wrapper
    )


class VariableScopeLayer(base_layer.Layer):
    """Wrapper Layer to capture `compat.v1.get_variable` and `compat.v1.layers`.

    This shim layer allows using large sets of TF1 model-forward-pass code as a
    Keras layer that works in TF2 with TF2 behaviors enabled. It will capture
    both weights and regularization losses of your forward-pass code. To use it,
    override this class and put your TF1 model's forward pass inside your
    implementation for `forward_pass`. (Unlike standard custom Keras layers,
    do not override `call`.)

    Below are some examples, and then more details on the functionality of this
    shim layer to wrap TF1 model forward passes.

    Example of capturing tf.compat.v1.layer-based modeling code as a Keras
    layer:

    ```python
    class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs):
        with variable_scope.variable_scope("double_dense_layer"):
          out = tf.compat.v1.layers.dense(
              inputs, self.units, name="dense_one",
              kernel_initializer=tf.compat.v1.random_normal_initializer,
              kernel_regularizer="l2")
          out = tf.compat.v1.layers.dense(
              out, self.units, name="dense_two",
              kernel_initializer=tf.compat.v1.random_normal_initializer(),
              kernel_regularizer="l2")
        return out

    # Create a layer that can be used as a standard keras layer
    layer = WrappedDoubleDenseLayer(10)

    # call the layer on inputs
    layer(...)

    # Variables created/used within the scope will be tracked by the layer
    layer.weights
    layer.trainable_variables

    # Regularization losses will be captured in layer.losses after a call,
    # just like any other Keras layer
    reg_losses = layer.losses
    ```

    Example of capturing tf.compat.v1.get_variable-based modeling code as
    a Keras layer:

    ```python
    class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs):
        out = inputs
        with tf.compat.v1.variable_scope("double_dense_layer"):
          with tf.compat.v1.variable_scope("dense_one"):
            # The weights are created with a `regularizer`,
            # so the layer should track their regularization losses
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = tf.compat.v1.math.matmul(out, kernel)
            out = tf.compat.v1.nn.bias_add(out, bias)
          with tf.compat.v1.variable_scope("dense_two"):
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = tf.compat.v1.math.matmul(out, kernel)
            out = tf.compat.v1.nn.bias_add(out, bias)
        return out

    # Create a layer that can be used as a standard keras layer
    layer = WrappedDoubleDenseLayer(10)

    # call the layer on inputs
    layer(...)

    # Variables created/used within the scope will be tracked by the layer
    layer.weights
    layer.trainable_variables

    # Regularization losses will be captured in layer.losses after a call,
    # just like any other Keras layer
    reg_losses = layer.losses
    ```

    Regularization losses:
      Any regularizers specified in the `get_variable` calls or
      `compat.v1.layer` creations will get captured by this wrapper layer.
      Regularization losses are accessible in `layer.losses` after a call just
      like in a standard Keras layer, and will be captured by any model that
      includes this layer.  Regularization losses attached to Keras
      layers/models set as attributes of your layer will also get captured in
      the standard Keras regularization loss tracking.

    Variable scope / variable reuse:
      variable-scope based reuse in the `forward_pass` will be respected,
      and work like variable-scope based reuse in TF1.

    Variable Names/Pre-trained checkpoint loading:
      Variable naming from get_variable and `compat.v1.layer` layers will match
      the TF1 names, so you should be able to re-use your old name-based
      checkpoints. Variable naming for Keras layers/models or for variables
      created by `tf.Variable` may change when going to eager execution.

    Training Arg in `forward_pass`:
      Keras will pass a `training` arg to this layer if `forward_pass` contains
      a `training` arg or a `**kwargs` varargs in its call signature,
      similarly to how keras passes `training` to other layers in TF2 that have
      similar signatures in their `call` implementations.
      See more details in the docs
      on `tf.keras.layers.Layer` to understand what will be passed and when.
      Note: tf.compat.v1.layers are usually not called with `training=None`,
      so the training arg to `forward_pass` might not feed through to them
      unless you pass it to their calls explicitly.

    Call signature of the forward pass:
      The semantics of the forward pass signature match the standard
      Keras layer `call` signature, including how Keras decides when
      to pass in a `training` arg., and the semantics applied to
      the first positional arg in the call signature.

    Caveats:
      * TF2 will not prune unused variable updates (or unused outputs). You may
        need to adjust your forward pass code to avoid computations or variable
        updates that you don't intend to use. (E.g. by adding a flag to the
        `forward_pass` call signature and branching on it).
      * Avoid Nesting variable creation in tf.function inside of `forward_pass`
        While the layer may safely be used from inside a `tf.function`, using
        a function inside of `forward_pass` will break the variable scoping.
      * If you would like to nest Keras layers/models or other
        `VariableScopeLayer`s directly in `forward_pass`, you need to
        assign them as attributes of your layer so that Keras's standard
        object-oriented weights and loss tracking will kick in.
        See the intro to modules, layers, and models
        [guide](https://www.tensorflow.org/guide/intro_to_modules) for more info
    """

    @property
    @layer_utils.cached_per_instance
    def _call_full_argspec(self):
        # Argspec inspection is expensive and the call spec is used often, so it
        # makes sense to cache the result.
        return tf_inspect.getfullargspec(self.forward_pass)

    def forward_pass(self, *args, **kwargs):
        """Implement this method. It should include your model forward pass."""
        raise NotImplementedError

    @track_tf1_style_variables
    def call(self, *args, **kwargs):
        return self.forward_pass(*args, **kwargs)


@keras_export(v1=["keras.utils.get_or_create_layer"])
def get_or_create_layer(name, create_layer_method):
    """Use this method to track nested keras models in a shim-decorated method.

    This method can be used within a `tf.keras.Layer`'s methods decorated by
    the`track_tf1_style_variables` shim, to additionally track inner keras Model
    objects created within the same method. The inner model's variables and
    losses will be accessible via the outer model's `variables` and `losses`
    attributes.

    This enables tracking of inner keras models using TF2 behaviors, with
    minimal changes to existing TF1-style code.

    Example:

    ```python
    class NestedLayer(tf.keras.layers.Layer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def build_model(self):
        inp = tf.keras.Input(shape=(5, 5))
        dense_layer = tf.keras.layers.Dense(
            10, name="dense", kernel_regularizer="l2",
            kernel_initializer=tf.compat.v1.ones_initializer())
        model = tf.keras.Model(inputs=inp, outputs=dense_layer(inp))
        return model

      @tf.compat.v1.keras.utils.track_tf1_style_variables
      def call(self, inputs):
        model = tf.compat.v1.keras.utils.get_or_create_layer(
            "dense_model", self.build_model)
        return model(inputs)
    ```
    The inner model creation should be confined to its own zero-arg function,
    which should be passed into this method. In TF1, this method will
    immediately create and return the desired model, without any tracking.

    Args:
      name: A name to give the nested layer to track.
      create_layer_method: a Callable that takes no args and returns the nested
      layer.

    Returns:
      The created layer.
    """
    store = vs._get_default_variable_store()
    if not isinstance(store, _EagerVariableStore):
        if not tf.compat.v1.executing_eagerly_outside_functions():
            # tf1 case; just create and return layer
            return create_layer_method()
        else:
            raise ValueError(
                "Tried to call get_or_create_layer in eager mode from a method "
                "notdecorated with "
                "@tf.compat.v1.keras.utils.track_tf1_style_variables."
            )
    vs_name = tf.compat.v1.get_variable_scope().name
    name = f"{vs_name}/{name}"
    return store.get_or_create_layer(name, create_layer_method)

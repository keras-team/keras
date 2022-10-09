# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the Lambda layer."""

import sys
import textwrap
import types as python_types
import warnings

import numpy as np
import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer
from keras.saving.legacy import serialization
from keras.utils import generic_utils
from keras.utils import tf_inspect
from keras.utils import tf_utils

# isort: off
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Lambda")
class Lambda(Layer):
    """Wraps arbitrary expressions as a `Layer` object.

    The `Lambda` layer exists so that arbitrary expressions can be used
    as a `Layer` when constructing `Sequential`
    and Functional API models. `Lambda` layers are best suited for simple
    operations or quick experimentation. For more advanced use cases, follow
    [this guide](
    https://www.tensorflow.org/guide/keras/custom_layers_and_models)
    for subclassing `tf.keras.layers.Layer`.

    WARNING: `tf.keras.layers.Lambda` layers have (de)serialization limitations!

    The main reason to subclass `tf.keras.layers.Layer` instead of using a
    `Lambda` layer is saving and inspecting a Model. `Lambda` layers
    are saved by serializing the Python bytecode, which is fundamentally
    non-portable. They should only be loaded in the same environment where
    they were saved. Subclassed layers can be saved in a more portable way
    by overriding their `get_config` method. Models that rely on
    subclassed Layers are also often easier to visualize and reason about.

    Examples:

    ```python
    # add a x -> x^2 layer
    model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
    # add a layer that returns the concatenation
    # of the positive part of the input and
    # the opposite of the negative part

    def antirectifier(x):
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)

    model.add(Lambda(antirectifier))
    ```

    Variables:
      While it is possible to use Variables with Lambda layers, this practice is
      discouraged as it can easily lead to bugs. For instance, consider the
      following layer:

    ```python
      scale = tf.Variable(1.)
      scale_layer = tf.keras.layers.Lambda(lambda x: x * scale)
    ```

      Because scale_layer does not directly track the `scale` variable, it will
      not appear in `scale_layer.trainable_weights` and will therefore not be
      trained if `scale_layer` is used in a Model.

      A better pattern is to write a subclassed Layer:

    ```python
      class ScaleLayer(tf.keras.layers.Layer):
        def __init__(self):
          super(ScaleLayer, self).__init__()
          self.scale = tf.Variable(1.)

        def call(self, inputs):
          return inputs * self.scale
    ```

      In general, Lambda layers can be convenient for simple stateless
      computation, but anything more complex should use a subclass Layer
      instead.

    Args:
      function: The function to be evaluated. Takes input tensor as first
        argument.
      output_shape: Expected output shape from function. This argument can be
        inferred if not explicitly provided. Can be a tuple or function. If a
        tuple, it only specifies the first dimension onward;
        sample dimension is assumed either the same as the input:
        `output_shape = (input_shape[0], ) + output_shape` or, the input is
        `None` and the sample dimension is also `None`:
        `output_shape = (None, ) + output_shape` If a function, it specifies the
        entire shape as a function of the input shape:
        `output_shape = f(input_shape)`
      mask: Either None (indicating no masking) or a callable with the same
        signature as the `compute_mask` layer method, or a tensor that will be
        returned as output mask regardless of what the input is.
      arguments: Optional dictionary of keyword arguments to be passed to the
        function.
    Input shape: Arbitrary. Use the keyword argument input_shape (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.
    Output shape: Specified by `output_shape` argument
    """

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(
        self, function, output_shape=None, mask=None, arguments=None, **kwargs
    ):
        super().__init__(**kwargs)

        self.arguments = arguments or {}
        self.function = function

        if mask is not None:
            self.supports_masking = True
        self.mask = mask
        self._output_shape = output_shape

        # Warning on every invocation will be quite irksome in Eager mode.
        self._already_warned = False

        function_args = tf_inspect.getfullargspec(function).args
        self._fn_expects_training_arg = "training" in function_args
        self._fn_expects_mask_arg = "mask" in function_args

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self._output_shape is None:
            # Make use of existing autocomputation but provide Lambda-specific
            # error message. This is always safe to run even when the outer
            # context is Graph mode because Lambda layers don't have side
            # effects such as `add_loss`.
            with tf.__internal__.eager_context.eager_mode():
                try:
                    return super().compute_output_shape(input_shape)
                except NotImplementedError:
                    raise NotImplementedError(
                        "We could not automatically infer the shape of "
                        "the Lambda's output. Please specify `output_shape` "
                        "for this Lambda."
                    )

        if callable(self._output_shape):
            output_shapes = self._output_shape(input_shape)
            return tf_utils.convert_shapes(output_shapes, to_tuples=False)

        # Output shapes are passed directly and don't include batch dimension.
        input_tensor_shape = tf_utils.convert_shapes(
            input_shape, to_tuples=False
        )
        batch_size = (
            tf.nest.flatten(input_tensor_shape)[0][0] if input_shape else None
        )

        def _add_batch(shape):
            return tf.TensorShape([batch_size] + shape.as_list())

        output_shapes = tf_utils.convert_shapes(
            self._output_shape, to_tuples=False
        )
        return tf.nest.map_structure(_add_batch, output_shapes)

    def call(self, inputs, mask=None, training=None):
        # We must copy for thread safety, but it only needs to be a shallow
        # copy.
        kwargs = {k: v for k, v in self.arguments.items()}
        if self._fn_expects_mask_arg:
            kwargs["mask"] = mask
        if self._fn_expects_training_arg:
            kwargs["training"] = training

        created_variables = []

        def _variable_creator(next_creator, **kwargs):
            var = next_creator(**kwargs)
            created_variables.append(var)
            return var

        with tf.GradientTape(
            watch_accessed_variables=True
        ) as tape, tf.variable_creator_scope(_variable_creator):
            result = self.function(inputs, **kwargs)
        self._check_variables(created_variables, tape.watched_variables())
        return result

    def _check_variables(self, created_variables, accessed_variables):
        if not created_variables and not accessed_variables:
            # In the common case that a Lambda layer does not touch a Variable,
            # we don't want to incur the runtime cost of assembling any state
            # used for checking only to immediately discard it.
            return

        # Filter out the state variable in the tf.random.Generator, which is
        # commonly used for initializer or droput. The variable is intentionally
        # not tracked and it is not a trainable variable.
        created_variables = [
            v for v in created_variables if "StateVar" not in v.name
        ]

        tracked_weights = set(v.ref() for v in self.weights)
        untracked_new_vars = [
            v for v in created_variables if v.ref() not in tracked_weights
        ]
        if untracked_new_vars:
            variable_str = "\n".join(f"  {i}" for i in untracked_new_vars)
            error_str = textwrap.dedent(
                """
          The following Variables were created within a Lambda layer ({name})
          but are not tracked by said layer:
          {variable_str}
          The layer cannot safely ensure proper Variable reuse across multiple
          calls, and consequently this behavior is disallowed for safety. Lambda
          layers are not well suited to stateful computation; instead, writing a
          subclassed Layer is the recommend way to define layers with
          Variables."""
            ).format(name=self.name, variable_str=variable_str)
            raise ValueError(error_str)

        untracked_used_vars = [
            v for v in accessed_variables if v.ref() not in tracked_weights
        ]
        if untracked_used_vars and not self._already_warned:
            variable_str = "\n".join(f"  {i}" for i in untracked_used_vars)
            self._warn(
                textwrap.dedent(
                    """
          The following Variables were used a Lambda layer's call ({name}), but
          are not present in its tracked objects:
          {variable_str}
          It is possible that this is intended behavior, but it is more likely
          an omission. This is a strong indication that this layer should be
          formulated as a subclassed Layer rather than a Lambda layer."""
                ).format(name=self.name, variable_str=variable_str)
            )
            self._already_warned = True

    def _warn(self, msg):
        # This method will be overridden in a unit test to raise an error,
        # because self.assertWarns is not universally implemented.
        return tf_logging.warning(msg)

    def compute_mask(self, inputs, mask=None):
        if callable(self.mask):
            return self.mask(inputs, mask)
        return self.mask

    def get_config(self):
        function_config = self._serialize_function_to_config(self.function)
        output_shape_config = self._serialize_function_to_config(
            self._output_shape, allow_raw=True
        )
        config = {
            "function": function_config[0],
            "function_type": function_config[1],
            "module": function_config[2],
            "output_shape": output_shape_config[0],
            "output_shape_type": output_shape_config[1],
            "output_shape_module": output_shape_config[2],
        }
        if self.mask is not None:
            mask_config = self._serialize_function_to_config(self.mask)
            config.update(
                {
                    "mask": mask_config[0],
                    "mask_type": mask_config[1],
                    "mask_module": mask_config[2],
                }
            )
        config["arguments"] = self.arguments

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _serialize_function_to_config(self, inputs, allow_raw=False):
        if isinstance(inputs, python_types.LambdaType):
            output = generic_utils.func_dump(inputs)
            output_type = "lambda"
            module = inputs.__module__
        elif callable(inputs):
            output = inputs.__name__
            output_type = "function"
            module = inputs.__module__
        elif allow_raw:
            output = inputs
            output_type = "raw"
            module = None
        else:
            raise ValueError(
                f"Invalid input for serialization, type: {type(inputs)} "
            )

        return output, output_type, module

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        function = cls._parse_function_from_config(
            config, custom_objects, "function", "module", "function_type"
        )

        output_shape = cls._parse_function_from_config(
            config,
            custom_objects,
            "output_shape",
            "output_shape_module",
            "output_shape_type",
        )
        if "mask" in config:
            mask = cls._parse_function_from_config(
                config, custom_objects, "mask", "mask_module", "mask_type"
            )
        else:
            mask = None

        config["function"] = function
        config["output_shape"] = output_shape
        config["mask"] = mask

        # If arguments were numpy array, they have been saved as
        # list. We need to recover the ndarray
        if "arguments" in config:
            for key in config["arguments"]:
                if isinstance(config["arguments"][key], dict):
                    arg_dict = config["arguments"][key]
                    if "type" in arg_dict and arg_dict["type"] == "ndarray":
                        # Overwrite the argument with its numpy translation
                        config["arguments"][key] = np.array(arg_dict["value"])

        return cls(**config)

    @classmethod
    def _parse_function_from_config(
        cls,
        config,
        custom_objects,
        func_attr_name,
        module_attr_name,
        func_type_attr_name,
    ):
        globs = globals().copy()
        module = config.pop(module_attr_name, None)
        if module in sys.modules:
            globs.update(sys.modules[module].__dict__)
        elif module is not None:
            # Note: we don't know the name of the function if it's a lambda.
            warnings.warn(
                "{} is not loaded, but a Lambda layer uses it. "
                "It may cause errors.".format(module),
                UserWarning,
                stacklevel=2,
            )
        if custom_objects:
            globs.update(custom_objects)
        function_type = config.pop(func_type_attr_name)
        if function_type == "function":
            # Simple lookup in custom objects
            function = serialization.deserialize_keras_object(
                config[func_attr_name],
                custom_objects=custom_objects,
                printable_module_name="function in Lambda layer",
            )
        elif function_type == "lambda":
            # Unsafe deserialization from bytecode
            function = generic_utils.func_load(
                config[func_attr_name], globs=globs
            )
        elif function_type == "raw":
            function = config[func_attr_name]
        else:
            supported_types = ["function", "lambda", "raw"]
            raise TypeError(
                "Unsupported value for `function_type` argument. Received: "
                f"function_type={function_type}. "
                f"Expected one of {supported_types}"
            )
        return function

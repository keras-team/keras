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
"""Keras Utilities for DTensor related API."""

import inspect

import tensorflow.compat.v2 as tf

from keras.dtensor import dtensor_api as dtensor

# All the variable names in the default keras layers. We will use those to map
# against the args in the __init__ method to find corresponding layout args.
# See allow_layout() for more details.
KERAS_VARIABLE_NAMES = [
    "alpha",
    "beta",
    "bias",
    "depthwise",
    "embeddings",
    "gamma",
    "kernel",
    "moving_mean",
    "moving_variance",
    "pointwise",
    "recurrent",
]


def allow_initializer_layout(init_method):
    """A decorator for injecting layout information to layer.__init__.

    Layout will be a new param for any of the weights for all the keras layers.
    Adding the param to all the __init__ method will be a big/duplicated work.

    This decorator is design to reduce and code duplication and make it easy to
    add/remove the dtensor feature if needed.

    Sample usage:
    ```python
    class Dense(tf.keras.layer.Layer):

      @allow_initializer_layout
      def __init__(self, units,
                   kernel_initializer='zeros',
                   bias_initializer='zeros',
                   **kwargs):
         super().__init__(**kwargs)

    d = Dense(units=8, kernel_layout=layout1, bias_layout=layout2)
    d.kernel_layout == layout1
    d.bias_layout == layout2
    ```

    By adding this annotation, it will:

    1. Filter out the kwargs based on some keywords, eg if the
      'kernel_initialzer' appears in method signature, then it will try to pop
      the 'kernel_layout' if it presents. Same for "bias" and
      "recurrent_kernel", etc. This will make sure the layout related param is
      not passed to `BaseLayer.__init__`, which will raise error about unexpect
      keyword args.
    2. Set the self.kernel/bias_layout attribute after the `__init__` method is
       called. Keras framework will use those fields to create weights down the
       stream.

    Args:
      init_method: the `__init__` method of the Keras layer to annotate.

    Returns:
      the annotated __init__ method.
    """

    def _wrap_function(layer_instance, *args, **kwargs):
        signature = inspect.signature(init_method)
        layout_args = {}
        # Check args like 'kernel_initializer' and pop the 'kernel_layout' if it
        # presents.
        for variable_name in KERAS_VARIABLE_NAMES:
            if variable_name + "_initializer" in signature.parameters:
                layout = kwargs.pop(variable_name + "_layout", None)
                if layout:
                    layout_args[variable_name + "_layout"] = layout

        init_method(layer_instance, *args, **kwargs)

        # Inject the layout parameter after the invocation of __init__()
        for layout_param_name, layout in layout_args.items():
            setattr(layer_instance, layout_param_name, layout)

    # return decorated
    return tf.__internal__.decorator.make_decorator(
        target=init_method, decorator_func=_wrap_function
    )


def inject_mesh(init_method):
    """Inject DTensor mesh information to an object.

    This is useful for keras object like `Metric` and `Optimizer` which need
    DTensor mesh to create the weights, but doesn't want to change the current
    public API interface.

    This is for temporary usage and eventually the mesh/layout information will
    be public arguments in the `__init__` method.

    Sample usage:
    ```python
    class Accuracy(tf.keras.metrics.Metric):

      @inject_mesh
      def __init__(self, name='accuracy', dtype=None):
         super().__init__(**kwargs)

      acc = Accuracy(mesh=mesh)
      assert acc._mesh == mesh
    ```

    Args:
      init_method: the `__init__` method of the Keras class to annotate.

    Returns:
      the annotated __init__ method.
    """

    def _wrap_function(instance, *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        # Note that the injection of _mesh need to happen before the invocation
        # of __init__, since the class might need the mesh to create weights in
        # the __init__.
        if mesh is not None:
            instance._mesh = mesh
        init_method(instance, *args, **kwargs)

    return tf.__internal__.decorator.make_decorator(
        target=init_method, decorator_func=_wrap_function
    )


def call_with_layout(fn, layout, *args, **kwargs):
    """Invoke the function with inputs and relayout the result.

    Args:
      fn: the function to invoke.
      layout: if not None, the output of the fn will be relayout with this.
      *args: positional arguments to be called with fn.
      **kwargs: keyword arguments to be called with fn.

    Returns:
      The output of fn, with potential relayout with the layout specified.
    """
    if layout:
        with dtensor.run_on(layout.mesh):
            result = fn(*args, **kwargs)
            return dtensor.relayout(result, layout)
    return fn(*args, **kwargs)

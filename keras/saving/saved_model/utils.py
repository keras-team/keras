# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions shared between SavedModel saving/loading
implementations."""

import copy
import itertools
import threading
import types

import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine import base_layer_utils
from keras.utils import control_flow_util
from keras.utils import layer_utils
from keras.utils import tf_contextlib
from keras.utils.generic_utils import LazyLoader

training_lib = LazyLoader("training_lib", globals(), "keras.engine.training")


def use_wrapped_call(
    layer, call_fn, call_spec, default_training_value=None, return_method=False
):
    """Creates fn that adds the losses returned by call_fn & returns the outputs.

    Args:
      layer: A Keras layer object
      call_fn: tf.function that takes layer inputs (and possibly a training
        arg), and returns a tuple of (outputs, list of losses).
      call_spec: The `CallFunctionSpec` for the layer's call function.
      default_training_value: Default value of the training kwarg. If `None`,
        the default is `tf.keras.backend.learning_phase()`.
      return_method: Whether to return a method bound to the layer.

    Returns:
      function that calls call_fn and returns the outputs. Losses returned by
      call_fn are added to the layer losses.
    """
    expects_training_arg = layer_uses_training_bool(layer)

    fn, arg_spec = maybe_add_training_arg(
        call_spec, call_fn, expects_training_arg, default_training_value
    )

    def return_outputs_and_add_losses(*args, **kwargs):
        """Returns the outputs from the layer call function, and adds the
        losses."""
        if return_method:
            args = args[1:]

        outputs, losses = fn(*args, **kwargs)
        layer.add_loss(losses)

        # TODO(kathywu): This is a temporary hack. When a network of layers is
        # revived from SavedModel, only the top-level layer will have losses.
        # This causes issues in eager mode because the child layers may have
        # graph losses (thus model.losses returns a mix of Eager and graph
        # tensors). To fix this, whenever eager losses are added to one layer,
        # add eager losses to all child layers. This causes `.losses` to only
        # return eager losses.

        if tf.executing_eagerly():
            for i in layer._flatten_layers():
                if i is not layer:
                    i._eager_losses = [
                        base_layer_utils.REVIVED_LOSS_PLACEHOLDER
                    ]

        return outputs

    decorated = tf.__internal__.decorator.make_decorator(
        target=call_fn,
        decorator_func=return_outputs_and_add_losses,
        decorator_argspec=arg_spec,
    )

    if return_method:
        return types.MethodType(decorated, layer)
    else:
        return decorated


def layer_uses_training_bool(layer):
    """Returns whether this layer or any of its children uses the training
    arg."""
    if layer._expects_training_arg:
        return True
    visited = {layer}
    to_visit = list_all_layers(layer)
    while to_visit:
        layer = to_visit.pop()
        if layer in visited:
            continue
        if getattr(layer, "_expects_training_arg", True):
            return True
        visited.add(layer)
        to_visit.extend(list_all_layers(layer))
    return False


def list_all_layers(obj):
    if isinstance(obj, training_lib.Model):
        # Handle special case of Sequential, which doesn't return
        # the `Input` layer.
        return obj.layers
    else:
        return list(obj._flatten_layers(include_self=False, recursive=False))


def list_all_layers_and_sublayers(obj):
    s = set([obj])
    s.update(
        itertools.chain.from_iterable(
            list_all_layers_and_sublayers(layer)
            for layer in list_all_layers(obj)
        )
    )
    return s


def maybe_add_training_arg(
    call_spec, wrapped_call, expects_training_arg, default_training_value
):
    """Decorate call and optionally adds training argument.

    If a layer expects a training argument, this function ensures that
    'training' is present in the layer args or kwonly args, with the default
    training value.

    Args:
      call_spec: CallFunctionSpec of the layer.
      wrapped_call: Wrapped call function.
      expects_training_arg: Whether to include 'training' argument.
      default_training_value: Default value of the training kwarg to include in
        the arg spec. If `None`, the default is
        `tf.keras.backend.learning_phase()`.

    Returns:
      Tuple of (
        function that calls `wrapped_call` and sets the training arg,
        Argspec of returned function or `None` if the argspec is unchanged)
    """
    if not expects_training_arg:
        return wrapped_call, None

    arg_spec = set_training_arg_spec(
        call_spec.full_argspec, default_training_value
    )
    call_spec = layer_utils.CallFunctionSpec(arg_spec)

    def wrap_with_training_arg(*args, **kwargs):
        """Wrap the `wrapped_call` function, and set training argument."""
        try:
            training = call_spec.get_arg_value(
                "training", args, kwargs, inputs_in_args=True
            )
        except KeyError:
            training = None

        if training is None:
            training = (
                default_training_value
                or base_layer_utils.call_context().training
                or backend.learning_phase()
            )

        args = list(args)
        kwargs = kwargs.copy()

        def replace_training_and_call(training):
            new_args, new_kwargs = call_spec.set_arg_value(
                "training", training, args, kwargs, inputs_in_args=True
            )
            return wrapped_call(*new_args, **new_kwargs)

        return control_flow_util.smart_cond(
            training,
            lambda: replace_training_and_call(True),
            lambda: replace_training_and_call(False),
        )

    return wrap_with_training_arg, arg_spec


def set_training_arg_spec(arg_spec, default_training_value):
    """Set `training=DEFAULT` argument in an ArgSpec."""
    if "training" in arg_spec.args:
        # If `training` is already in the args list, try to set the default
        # value.
        index = arg_spec.args.index("training")
        training_default_index = len(arg_spec.args) - index
        defaults = (
            list(arg_spec.defaults) if arg_spec.defaults is not None else []
        )
        if (
            arg_spec.defaults
            and len(arg_spec.defaults) >= training_default_index
            and defaults[-training_default_index] is None
        ):
            defaults[-training_default_index] = default_training_value
            return arg_spec._replace(defaults=defaults)
    elif "training" not in arg_spec.kwonlyargs:
        kwonlyargs = arg_spec.kwonlyargs + ["training"]
        kwonlydefaults = copy.copy(arg_spec.kwonlydefaults) or {}
        kwonlydefaults["training"] = default_training_value
        return arg_spec._replace(
            kwonlyargs=kwonlyargs, kwonlydefaults=kwonlydefaults
        )

    return arg_spec


class SaveOptionsContext(threading.local):
    def __init__(self):
        super().__init__()
        self.save_traces = True


_save_options_context = SaveOptionsContext()


@tf_contextlib.contextmanager
def keras_option_scope(save_traces):
    previous_value = _save_options_context.save_traces
    try:
        _save_options_context.save_traces = save_traces
        yield
    finally:
        _save_options_context.save_traces = previous_value


def should_save_traces():
    """Whether to trace layer functions-can be disabled in the save_traces
    arg."""
    return _save_options_context.save_traces


@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj):
    """A context that disables automatic dependency tracking when assigning attrs.

    Objects that inherit from Autotrackable automatically creates dependencies
    to trackable objects through attribute assignments, and wraps data
    structures (lists or dicts) with trackable classes. This scope may be used
    to temporarily disable this behavior. This works similar to the decorator
    `no_automatic_dependency_tracking`.

    Example usage:
    ```
    model = tf.keras.Model()
    model.arr1 = []  # Creates a ListWrapper object
    with no_automatic_dependency_tracking_scope(model):
      model.arr2 = []  # Creates a regular, untracked python list
    ```

    Args:
      obj: A trackable object.

    Yields:
      a scope in which the object doesn't track dependencies.
    """
    previous_value = getattr(obj, "_setattr_tracking", True)
    obj._setattr_tracking = False
    try:
        yield
    finally:
        obj._setattr_tracking = previous_value

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

"""Utilities related to layer/model functionality."""

import copy
import functools
import re
import weakref

import numpy as np
import tensorflow.compat.v2 as tf

from keras import initializers
from keras.utils import io_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.utils.get_source_inputs")
def get_source_inputs(tensor, layer=None, node_index=None):
    """Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    Args:
        tensor: The tensor to start from.
        layer: Origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: Origin node index of the tensor.

    Returns:
        List of input tensors.
    """
    if not hasattr(tensor, "_keras_history"):
        return tensor

    if layer is None or node_index:
        layer, node_index, _ = tensor._keras_history
    if not layer._inbound_nodes:
        return [tensor]
    else:
        node = layer._inbound_nodes[node_index]
        if node.is_input:
            # Reached an Input layer, stop recursion.
            return tf.nest.flatten(node.input_tensors)
        else:
            source_tensors = []
            for layer, node_index, _, tensor in node.iterate_inbound():
                previous_sources = get_source_inputs(tensor, layer, node_index)
                # Avoid input redundancy.
                for x in previous_sources:
                    if all(x is not t for t in source_tensors):
                        source_tensors.append(x)
            return source_tensors


def validate_string_arg(
    input_data,
    allowable_strings,
    layer_name,
    arg_name,
    allow_none=False,
    allow_callables=False,
):
    """Validates the correctness of a string-based arg."""
    if allow_none and input_data is None:
        return
    elif allow_callables and callable(input_data):
        return
    elif isinstance(input_data, str) and input_data in allowable_strings:
        return
    else:
        allowed_args = "`None`, " if allow_none else ""
        allowed_args += "a `Callable`, " if allow_callables else ""
        allowed_args += f"or one of the following values: {allowable_strings}"
        if allow_callables:
            callable_note = (
                f"If restoring a model and `{arg_name}` is a custom callable, "
                "please ensure the callable is registered as a custom object. "
                "See https://www.tensorflow.org/guide/keras/save_and_serialize"
                "#registering_the_custom_object for details. "
            )
        else:
            callable_note = ""
        raise ValueError(
            f"Unkown value for `{arg_name}` argument of layer {layer_name}. "
            f"{callable_note}Allowed values are: {allowed_args}. Received: "
            f"{input_data}"
        )


def count_params(weights):
    """Count the total number of scalars composing the weights.

    Args:
        weights: An iterable containing the weights on which to compute params

    Returns:
        The total number of scalars composing the weights
    """
    unique_weights = {id(w): w for w in weights}.values()
    # Ignore TrackableWeightHandlers, which will not have a shape defined.
    unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))


def get_layer_index_bound_by_layer_name(model, layer_range=None):
    """Get the layer indexes from the model based on layer names.

    The layer indexes can be used to slice the model into sub models for
    display.

    Args:
        model: `tf.keras.Model` instance.
        layer_names: a list or tuple of 2 strings, the starting layer name and
            ending layer name (both inclusive) for the result. All layers will
            be included when `None` is provided.

    Returns:
        The index value of layer based on its unique name (layer_names).
        Output will be [first_layer_index, last_layer_index + 1].
    """
    if layer_range is not None:
        if len(layer_range) != 2:
            raise ValueError(
                "layer_range must be a list or tuple of length 2. Received: "
                f"layer_range = {layer_range} of length {len(layer_range)}"
            )
        if not isinstance(layer_range[0], str) or not isinstance(
            layer_range[1], str
        ):
            raise ValueError(
                "layer_range should contain string type only. "
                f"Received: {layer_range}"
            )
    else:
        return [0, len(model.layers)]

    lower_index = [
        idx
        for idx, layer in enumerate(model.layers)
        if re.match(layer_range[0], layer.name)
    ]
    upper_index = [
        idx
        for idx, layer in enumerate(model.layers)
        if re.match(layer_range[1], layer.name)
    ]

    if not lower_index or not upper_index:
        raise ValueError(
            "Passed layer_names do not match the layer names in the model. "
            f"Received: {layer_range}"
        )

    if min(lower_index) > max(upper_index):
        return [min(upper_index), max(lower_index) + 1]
    return [min(lower_index), max(upper_index) + 1]


def print_summary(
    model,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
):
    """Prints a summary of a model.

    Args:
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
        expand_nested: Whether to expand the nested models.
            If not provided, defaults to `False`.
        show_trainable: Whether to show if a layer is trainable.
            If not provided, defaults to `False`.
        layer_range: List or tuple containing two strings,
            the starting layer name and ending layer name (both inclusive),
            indicating the range of layers to be printed in the summary. The
            strings could also be regexes instead of an exact name. In this
             case, the starting layer will be the first layer that matches
            `layer_range[0]` and the ending layer will be the last element that
            matches `layer_range[1]`. By default (`None`) all
            layers in the model are included in the summary.
    """
    if print_fn is None:
        print_fn = io_utils.print_msg

    if model.__class__.__name__ == "Sequential":
        sequential_like = True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(tf.nest.flatten(v[0].keras_inputs)) > 1
            ):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [0.45, 0.85, 1.0]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ["Layer (type)", "Output Shape", "Param #"]
    else:
        line_length = line_length or 98
        positions = positions or [0.33, 0.55, 0.67, 1.0]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    if show_trainable:
        line_length += 11
        positions.append(line_length)
        to_display.append("Trainable")

    layer_range = get_layer_index_bound_by_layer_name(model, layer_range)

    def print_row(fields, positions, nested_level=0):
        left_to_print = [str(x) for x in fields]
        while any(left_to_print):
            line = ""
            for col in range(len(left_to_print)):
                if col > 0:
                    start_pos = positions[col - 1]
                else:
                    start_pos = 0
                end_pos = positions[col]
                # Leave room for 2 spaces to delineate columns
                # we don't need any if we are printing the last column
                space = 2 if col != len(positions) - 1 else 0
                cutoff = end_pos - start_pos - space
                fit_into_line = left_to_print[col][:cutoff]
                # For nicer formatting we line-break on seeing end of
                # tuple/dict etc.
                line_break_conditions = ("),", "},", "],", "',")
                candidate_cutoffs = [
                    fit_into_line.find(x) + len(x)
                    for x in line_break_conditions
                    if fit_into_line.find(x) >= 0
                ]
                if candidate_cutoffs:
                    cutoff = min(candidate_cutoffs)
                    fit_into_line = fit_into_line[:cutoff]

                if col == 0:
                    line += "|" * nested_level + " "
                line += fit_into_line
                line += " " * space if space else ""
                left_to_print[col] = left_to_print[col][cutoff:]

                # Pad out to the next position
                if nested_level:
                    line += " " * (positions[col] - len(line) - nested_level)
                else:
                    line += " " * (positions[col] - len(line))
            line += "|" * nested_level
            print_fn(line)

    print_fn(f'Model: "{model.name}"')
    print_fn("_" * line_length)
    print_row(to_display, positions)
    print_fn("=" * line_length)

    def print_layer_summary(layer, nested_level=0):
        """Prints a summary for a single layer.

        Args:
            layer: target layer.
            nested_level: level of nesting of the layer inside its parent layer
              (e.g. 0 for a top-level layer, 1 for a nested layer).
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        except RuntimeError:  # output_shape unknown in Eager mode.
            output_shape = "?"
        name = layer.name
        cls_name = layer.__class__.__name__
        if not layer.built and not getattr(layer, "_is_graph_network", False):
            # If a subclassed model has a layer that is not called in
            # Model.call, the layer will not be built and we cannot call
            # layer.count_params().
            params = "0 (unused)"
        else:
            params = layer.count_params()
        fields = [name + " (" + cls_name + ")", output_shape, params]

        if show_trainable:
            fields.append("Y" if layer.trainable else "N")

        print_row(fields, positions, nested_level)

    def print_layer_summary_with_connections(layer, nested_level=0):
        """Prints a summary for a single layer (including its connections).

        Args:
            layer: target layer.
            nested_level: level of nesting of the layer inside its parent layer
              (e.g. 0 for a top-level layer, 1 for a nested layer).
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue

            for (
                inbound_layer,
                node_index,
                tensor_index,
                _,
            ) in node.iterate_inbound():
                connections.append(
                    f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                )

        name = layer.name
        cls_name = layer.__class__.__name__
        fields = [
            name + " (" + cls_name + ")",
            output_shape,
            layer.count_params(),
            connections,
        ]

        if show_trainable:
            fields.append("Y" if layer.trainable else "N")

        print_row(fields, positions, nested_level)

    def print_layer(layer, nested_level=0, is_nested_last=False):
        if sequential_like:
            print_layer_summary(layer, nested_level)
        else:
            print_layer_summary_with_connections(layer, nested_level)

        if expand_nested and hasattr(layer, "layers") and layer.layers:
            print_fn(
                "|" * (nested_level + 1)
                + "¯" * (line_length - 2 * nested_level - 2)
                + "|" * (nested_level + 1)
            )

            nested_layer = layer.layers
            is_nested_last = False
            for i in range(len(nested_layer)):
                if i == len(nested_layer) - 1:
                    is_nested_last = True
                print_layer(nested_layer[i], nested_level + 1, is_nested_last)

            print_fn(
                "|" * nested_level
                + "¯" * (line_length - 2 * nested_level)
                + "|" * nested_level
            )

        if not is_nested_last:
            print_fn(
                "|" * nested_level
                + " " * (line_length - 2 * nested_level)
                + "|" * nested_level
            )

    for layer in model.layers[layer_range[0] : layer_range[1]]:
        print_layer(layer)
    print_fn("=" * line_length)

    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)

    print_fn(f"Total params: {trainable_count + non_trainable_count:,}")
    print_fn(f"Trainable params: {trainable_count:,}")
    print_fn(f"Non-trainable params: {non_trainable_count:,}")
    print_fn("_" * line_length)


def convert_dense_weights_data_format(
    dense, previous_feature_map_shape, target_data_format="channels_first"
):
    """Utility useful when changing a convnet's `data_format`.

    When porting the weights of a convnet from one data format to the other,
    if the convnet includes a `Flatten` layer
    (applied to the last convolutional feature map)
    followed by a `Dense` layer, the weights of that `Dense` layer
    should be updated to reflect the new dimension ordering.

    Args:
        dense: The target `Dense` layer.
        previous_feature_map_shape: A shape tuple of 3 integers,
            e.g. `(512, 7, 7)`. The shape of the convolutional
            feature map right before the `Flatten` layer that
            came before the target `Dense` layer.
        target_data_format: One of "channels_last", "channels_first".
            Set it "channels_last"
            if converting a "channels_first" model to "channels_last",
            or reciprocally.
    """
    assert target_data_format in {"channels_last", "channels_first"}
    kernel, bias = dense.get_weights()
    for i in range(kernel.shape[1]):
        if target_data_format == "channels_first":
            c, h, w = previous_feature_map_shape
            original_fm_shape = (h, w, c)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (2, 0, 1))  # last -> first
        else:
            h, w, c = previous_feature_map_shape
            original_fm_shape = (c, h, w)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (1, 2, 0))  # first -> last
        kernel[:, i] = np.reshape(ki, (np.prod(previous_feature_map_shape),))
    dense.set_weights([kernel, bias])


def is_builtin_layer(layer):
    if not getattr(layer, "_keras_api_names", None):
        return False

    # Subclasses of `Layer` that are not exported inherit the export name
    # of the base layer class.
    return layer._keras_api_names != (
        "keras.layers.Layer",
    ) and layer._keras_api_names_v1 != ("keras.layers.Layer",)


def cached_per_instance(f):
    """Lightweight decorator for caching lazily constructed properties.

    When to use:
    This decorator provides simple caching with minimal overhead. It is designed
    for properties which are expensive to compute and static over the life of a
    class instance, and provides no mechanism for cache invalidation. Thus it is
    best suited for lazily exposing derived properties of other static data.

    For classes with custom getattr / setattr behavior (such as trackable
    objects), storing cache results as object attributes is not performant.
    Instead, a specialized cache can significantly reduce property lookup
    overhead. (While still allowing the decorated property to be lazily
    computed.) Consider the following class:

    ```
    class MyClass:
      def __setattr__(self, key, value):
        # Some expensive class specific code
        # ...
        # ...

        super(MyClass, self).__setattr__(key, value)

      @property
      def thing(self):
        # `thing` is expensive to compute (and may not even be requested), so we
        # want to lazily compute it and then cache it.
        output = getattr(self, '_thing', None)
        if output is None:
          self._thing = output = compute_thing(self)
        return output
    ```

    It's also worth noting that ANY overriding of __setattr__, even something as
    simple as:
    ```
      def __setattr__(self, key, value):
        super(MyClass, self).__setattr__(key, value)
    ```

    Slows down attribute assignment by nearly 10x.

    By contrast, replacing the definition of `thing` with the following
    sidesteps the expensive __setattr__ altogether:

    '''
    @property
    @tracking.cached_per_instance
    def thing(self):
      # `thing` is expensive to compute (and may not even be requested), so we
      # want to lazily compute it and then cache it.
      return compute_thing(self)
    '''

    Performance:
    The overhead for this decorator is ~0.4 us / call. A much lower overhead
    implementation (~0.085 us / call) can be achieved by using a custom dict
    type:

    ```
    def dict_based_cache(f):
      class Cache(dict):
        __slots__ = ()
        def __missing__(self, key):
          self[key] = output = f(key)
          return output

      return property(Cache().__getitem__)
    ```

    However, that implementation holds class instances as keys, and as a result
    blocks garbage collection. (And modifying it to use weakref's as keys raises
    the lookup overhead to ~0.4 us) As a result, the WeakKeyDictionary
    implementation below turns out to be more prudent.

    Args:
      f: The function to cache.

    Returns:
      f decorated with simple caching behavior.
    """

    cache = weakref.WeakKeyDictionary()

    @functools.wraps(f)
    def wrapped(item):
        output = cache.get(item)
        if output is None:
            cache[item] = output = f(item)
        return output

    wrapped.cache = cache
    return wrapped


def filter_empty_layer_containers(layer_list):
    """Filter out empty Layer-like containers and uniquify."""
    # TODO(b/130381733): Make this an attribute in base_layer.Layer.
    existing = set()
    to_visit = layer_list[::-1]
    while to_visit:
        obj = to_visit.pop()
        if id(obj) in existing:
            continue
        existing.add(id(obj))
        if hasattr(obj, "_is_layer") and not isinstance(obj, type):
            yield obj
        else:
            sub_layers = getattr(obj, "layers", None) or []

            # Trackable data structures will not show up in ".layers" lists, but
            # the layers they contain will.
            to_visit.extend(sub_layers[::-1])


class CallFunctionSpec:
    """Caches the spec and provides utilities for handling call function
    args."""

    def __init__(self, full_argspec):
        """Initialies a `CallFunctionSpec`.

        Args:
          full_argspec: the FullArgSpec of a call function of a layer.
        """
        self._full_argspec = full_argspec

        self._arg_names = list(self._full_argspec.args)
        # Scrub `self` that appears if a decorator was applied.
        if self._arg_names and self._arg_names[0] == "self":
            self._arg_names = self._arg_names[1:]
        self._arg_names += self._full_argspec.kwonlyargs or []

        call_accepts_kwargs = self._full_argspec.varkw is not None
        self._expects_training_arg = (
            "training" in self._arg_names or call_accepts_kwargs
        )
        self._expects_mask_arg = (
            "mask" in self._arg_names or call_accepts_kwargs
        )

        call_fn_defaults = self._full_argspec.defaults or []
        defaults = dict()
        # The call arg defaults are an n-tuple of the last n elements of the
        # args list. (n = # of elements that have a default argument)
        for i in range(-1 * len(call_fn_defaults), 0):
            defaults[self._arg_names[i]] = call_fn_defaults[i]
        # The default training arg will be any (non-None) default specified in
        # the method signature, or None if no value is specified.
        defaults.update(self._full_argspec.kwonlydefaults or {})
        self._default_training_arg = defaults.get("training")

    @property
    def full_argspec(self):
        """Returns the FullArgSpec of the call function."""
        return self._full_argspec

    @property
    def arg_names(self):
        """List of names of args and kwonlyargs."""
        # `arg_names` is not accurate if the layer has variable positional args.
        return self._arg_names

    @arg_names.setter
    def arg_names(self, value):
        self._arg_names = value

    @property
    @cached_per_instance
    def arg_positions(self):
        """Returns a dict mapping arg names to their index positions."""
        # `arg_positions` is not accurate if the layer has variable positional
        # args.
        call_fn_arg_positions = dict()
        for pos, arg in enumerate(self._arg_names):
            call_fn_arg_positions[arg] = pos
        return call_fn_arg_positions

    @property
    def expects_training_arg(self):
        """Whether the call function uses 'training' as a parameter."""
        return self._expects_training_arg

    @expects_training_arg.setter
    def expects_training_arg(self, value):
        self._expects_training_arg = value

    @property
    def expects_mask_arg(self):
        """Whether the call function uses `mask` as a parameter."""
        return self._expects_mask_arg

    @expects_mask_arg.setter
    def expects_mask_arg(self, value):
        self._expects_mask_arg = value

    @property
    def default_training_arg(self):
        """The default value given to the "training" argument."""
        return self._default_training_arg

    def arg_was_passed(self, arg_name, args, kwargs, inputs_in_args=False):
        """Returns true if argument is present in `args` or `kwargs`.

        Args:
          arg_name: String name of the argument to find.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.

        Returns:
          True if argument with `arg_name` is present in `args` or `kwargs`.
        """
        # Performance optimization: do no work in most common case.
        if not args and not kwargs:
            return False

        if arg_name in kwargs:
            return True
        call_fn_args = self._arg_names
        if not inputs_in_args:
            # Ignore `inputs` arg.
            call_fn_args = call_fn_args[1:]
        return arg_name in dict(zip(call_fn_args, args))

    def get_arg_value(self, arg_name, args, kwargs, inputs_in_args=False):
        """Retrieves the value for the argument with name `arg_name`.

        Args:
          arg_name: String name of the argument to find.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.

        Returns:
          The value of the argument with name `arg_name`, extracted from `args`
          or `kwargs`.

        Raises:
          KeyError if the value of `arg_name` cannot be found.
        """
        if arg_name in kwargs:
            return kwargs[arg_name]
        call_fn_args = self._arg_names
        if not inputs_in_args:
            # Ignore `inputs` arg.
            call_fn_args = call_fn_args[1:]
        args_dict = dict(zip(call_fn_args, args))
        return args_dict[arg_name]

    def set_arg_value(
        self,
        arg_name,
        new_value,
        args,
        kwargs,
        inputs_in_args=False,
        pop_kwarg_if_none=False,
    ):
        """Sets the value of an argument into the given args/kwargs.

        Args:
          arg_name: String name of the argument to find.
          new_value: New value to give to the argument.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.
          pop_kwarg_if_none: If the new value is `None`, and this is `True`,
            then the argument is deleted from `kwargs`.

        Returns:
          The updated `(args, kwargs)`.
        """
        if self.full_argspec.varargs:
            try:
                arg_pos = self.full_argspec.args.index(arg_name)
                if self.full_argspec.args[0] == "self":
                    arg_pos -= 1
            except ValueError:
                arg_pos = None
        else:
            arg_pos = self.arg_positions.get(arg_name, None)

        if arg_pos is not None:
            if not inputs_in_args:
                # Ignore `inputs` arg.
                arg_pos = arg_pos - 1
            if len(args) > arg_pos:
                args = list(args)
                args[arg_pos] = new_value
                return tuple(args), kwargs
        if new_value is None and pop_kwarg_if_none:
            kwargs.pop(arg_name, None)
        else:
            kwargs[arg_name] = new_value
        return args, kwargs

    def split_out_first_arg(self, args, kwargs):
        """Splits (args, kwargs) into (inputs, args, kwargs)."""
        # Grab the argument corresponding to the first argument in the
        # layer's `call` method spec. This will either be the first positional
        # argument, or it will be provided as a keyword argument.
        if args:
            inputs = args[0]
            args = args[1:]
        elif self._arg_names[0] in kwargs:
            kwargs = copy.copy(kwargs)
            inputs = kwargs.pop(self._arg_names[0])
        else:
            raise ValueError(
                "The first argument to `Layer.call` must always be passed."
            )
        return inputs, args, kwargs


@keras_export("keras.utils.warmstart_embedding_matrix")
def warmstart_embedding_matrix(
    base_vocabulary,
    new_vocabulary,
    base_embeddings,
    new_embeddings_initializer="uniform",
):
    """Warm start embedding matrix with changing vocab.

    This util can be used to warmstart the embedding layer matrix when
    vocabulary changes between previously saved checkpoint and model.
    Vocabulary change could mean, the size of the new vocab is different or the
    vocabulary is reshuffled or new vocabulary has been added to old vocabulary.
    If the vocabulary size changes, size of the embedding layer matrix also
    changes. This util remaps the old vocabulary embeddings to the new embedding
    layer matrix.

    Example:
    Here is an example that demonstrates how to use the
    `warmstart_embedding_matrix` util.
    >>> import keras
    >>> vocab_base = tf.convert_to_tensor(["unk", "a", "b", "c"])
    >>> vocab_new = tf.convert_to_tensor(
    ...        ["unk", "unk", "a", "b", "c", "d", "e"])
    >>> vectorized_vocab_base = np.random.rand(vocab_base.shape[0], 3)
    >>> vectorized_vocab_new = np.random.rand(vocab_new.shape[0], 3)
    >>> warmstarted_embedding_matrix = warmstart_embedding_matrix(
    ...       base_vocabulary=vocab_base,
    ...       new_vocabulary=vocab_new,
    ...       base_embeddings=vectorized_vocab_base,
    ...       new_embeddings_initializer=keras.initializers.Constant(
    ...         vectorized_vocab_new))

    Here is an example that demonstrates how to get vocabulary and embedding
    weights from layers, use the `warmstart_embedding_matrix` util to remap the
    layer embeddings and continue with model training.
    ```
    # get old and new vocabulary by using layer.get_vocabulary()
    # for example assume TextVectorization layer is used
    base_vocabulary = old_text_vectorization_layer.get_vocabulary()
    new_vocabulary = new_text_vectorization_layer.get_vocabulary()
    # get previous embedding layer weights
    embedding_weights_base = model.get_layer('embedding').get_weights()[0]
    warmstarted_embedding = keras.utils.warmstart_embedding_matrix(
                                  base_vocabulary,
                                  new_vocabulary,
                                  base_embeddings=embedding_weights_base,
                                  new_embeddings_initializer="uniform")
    updated_embedding_variable = tf.Variable(warmstarted_embedding)

    # update embedding layer weights
    model.layers[1].embeddings = updated_embedding_variable
    model.fit(..)
    # continue with model training

    ```

    Args:
        base_vocabulary: The list of vocabulary terms that
          the preexisting embedding matrix `base_embeddings` represents.
          It can be either a 1D array/tensor or a tuple/list of vocabulary
          terms (strings), or a path to a vocabulary text file. If passing a
           file path, the file should contain one line per term in the
           vocabulary.
        new_vocabulary: The list of vocabulary terms for the new vocabulary
           (same format as above).
        base_embeddings: NumPy array or tensor representing the preexisting
          embedding matrix.
        new_embeddings_initializer: Initializer for embedding vectors for
          previously unseen terms to be added to the new embedding matrix (see
          `keras.initializers`). Defaults to "uniform". new_embedding matrix
          needs to be specified with "constant" initializer.
          matrix. Default value is None.

    Returns:
      tf.tensor of remapped embedding layer matrix

    """
    # convert vocab to list
    base_vocabulary = convert_vocab_to_list(base_vocabulary)
    new_vocabulary = convert_vocab_to_list(new_vocabulary)

    # Initialize the new embedding layer matrix
    new_embeddings_initializer = initializers.get(new_embeddings_initializer)
    new_embedding = new_embeddings_initializer(
        shape=(len(new_vocabulary), base_embeddings.shape[1]),
        dtype=base_embeddings.dtype,
    )

    # create mapping dict {vocab:index}
    base_vocabulary_dict = dict(
        zip(base_vocabulary, range(len(base_vocabulary)))
    )

    indices_base_vocabulary = []
    indices_new_vocabulary = []
    for index, key in enumerate(new_vocabulary):
        if key in base_vocabulary_dict:
            indices_base_vocabulary.append(base_vocabulary_dict[key])
            indices_new_vocabulary.append(int(index))

    # update embedding matrix
    if indices_base_vocabulary:
        values_to_update = tf.gather(base_embeddings, indices_base_vocabulary)
        new_embedding = tf.tensor_scatter_nd_update(
            new_embedding,
            tf.expand_dims(indices_new_vocabulary, axis=1),
            values_to_update,
        )
    return new_embedding


def convert_vocab_to_list(vocab):
    """Convert input vacabulary to list."""
    vocab_list = []
    if tf.is_tensor(vocab):
        vocab_list = list(vocab.numpy())
    elif isinstance(vocab, (np.ndarray, tuple, list)):
        vocab_list = list(vocab)
    elif isinstance(vocab, str):
        if not tf.io.gfile.exists(vocab):
            raise ValueError(f"Vocabulary file {vocab} does not exist.")
        with tf.io.gfile.GFile(vocab, "r") as vocabulary_file:
            vocab_list = vocabulary_file.read().splitlines()
    else:
        raise ValueError(
            "Vocabulary is expected to be either a NumPy array, "
            "list, 1D tensor or a vocabulary text file. Instead type "
            f"{type(vocab)} was received."
        )
    if len(vocab_list) == 0:
        raise ValueError(
            "Vocabulary is expected to be either a NumPy array, "
            "list, 1D tensor or a vocabulary text file with at least one token."
            " Received 0 instead."
        )
    return vocab_list

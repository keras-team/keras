# Implementation notes:
#
# - CompositeLayer encapsulates a functional subgraph of layers.
#   It is the equivalent of a Functional or Sequential model but
#   as a lighter-weight component, without the model-specific
#   functionality like .fit() etc.
#
# - A CompositeLayer can be cretaed either from a list of layers
#   or a function that defines a graph of layers. There is no
#   constructor similar to a functional Model(inputs, outputs)
#   because inputs and outputs are usually not known when creating
#   a layer. They will be created when the layer is built.
#
# - Functional and CompositeLayer only depend on Function. There is
#   no circular dependency between models and layers.
#
# - Note 1:
#   This is an intermediate implementation to make reviewing easier
#   It isolates 4 functions in functional.py that are used by both
#   CompositeLayer and Functional:
#      1) compute_input_spec
#      2) run_through_graph_with_training_and_mask
#      3) function_from_config
#      4) serialize_functional_config
#   With this approach, no changes are made to the Functional Model
#   class hierarchy.
#
#   The next step is to move these 4 functions to CompositeLayer, then
#   base Functional on CompositeLayer instead of Function. This will
#   also allow the unification of Functional and Sequential models since
#   both will be based on CompositeLayer and have a Function once build()
#   is called. Code explicitly testing for Functional or Sequential
#   can then be removed throughout the code base and replaced with
#   `isinstance(obj, CompositeLayer) and obj.built`
#
# - Note 2:
#   Once change potentially affecting Functional model:
#   in functional.serialize_functional_config, a line was
#   changed from
#       return [get_tensor_config(tensors)]
#   to
#       return get_tensor_config(tensors)
#   This looked like a bug but Model.assert_input_compatible
#   let it through. However, Function.assert_input_compatible
#   does not so the bug needed to be fixed.
#
#   Also in functional.compute_input_spec, a list with a single element
#   was returned when the input spec was a single tensor. This was changed
#   too.
#
# - Note 3:
#   In functional.py, there are two explicit tetst for "Function":
#   "if isinstance(layer, Function)" in function_from_config
#   and
#   "issubclass(operation.__class__, Function)" in serialize_functional_config
#   I have added a test that triggers these conditions in functions_test.py
#   This is supposed to handle the case when a Functional sub-model is used
#   in the computatio graph twice. However, if the two conditional clauses
#   in functional.py are removed, the tests still pass. I do not think they
#   are useful at all and I do not thing their explanatory comment "Functional
#   models start with a pre-existing node linking their input to output." is
#   correct. Therefore, I have not adapted these conditional clauses for
#   CompositeLayer. It does not seem to matter in tests.
#
# - Note 4:
#   Passing a list of inputs to a model expecting a dictionary
#   of inputs seems to be allowed, as long as flattening the dict does
#   not result in reordering. There is an explicit reordering test in
#   functional._standardize_inputs (look for "sort").
#   Changing this in Functional is not possible at this point but I would
#   consider disalowing this in CompositeLayer.
#   Tests covering thids behavior:
#     functional_test.test_list_input_with_dict_build
#     composite_layer_test.test_list_input_with_dict_build
#
# - TODOs:
#   - TODO: optional inputs for CompositeLayer

import typing

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.core.input_layer import Input
from keras.src.layers.layer import Layer
from keras.src.models.functional import compute_input_spec
from keras.src.models.functional import function_from_config
from keras.src.models.functional import run_through_graph_with_training_and_mask
from keras.src.models.functional import serialize_functional_config
from keras.src.ops.function import Function


@keras_export(["keras.layers.CompositeLayer"])
class CompositeLayer(Layer):
    """Layer that encapsulates a subgraph of layers into a single layer.

    `CompositeLayer` can be created in two ways:

    1. From a list of layers:

    ```python
    # Composite layer from a list of layers
    composite = layers.CompositeLayer([
        layers.Dense(64, activation='relu'),
        layers.Dense(32)
    ])
    ```

    2. Using a function that defines a graph of layers:
       This allows more complex computation graphs.

    ```python
    def layer_fn(x):
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(32)(x)
        return outputs

    # Create the composite layer using the function
    composite = layers.CompositeLayer(layer_fn)
    ```

    It is recommended to package reusable composite layers
    in a subclass of `CompositeLayer`:

    ```python
    # A reusable composite layer
    class MyCompositeLayer(CompositeLayer):

        @staticmethod
        def my_layer_fn(inputs):
            x = layers.Dense(5)(inputs)
            return layers.Dense(4)(x)

        def __init__(self, **kwargs):
            super().__init__(MyCompositeLayer.my_layer_fn, **kwargs)
    ```

    Args:
        layers: Either
            - a callable function that defines a computation graph
            - or a list of Layer objects to compose sequentially.
        name: Optional name for the layer.
    """

    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, layers, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # Init from either a function that defines the layer graph
        # or a sequence of layers.
        if not callable(layers) and not (
            isinstance(layers, (list, tuple)) and len(layers) > 0
        ):
            raise ValueError(
                f"Must provide a layers parameter that is either a callable "
                f"function that defines the layer's computation graph or a "
                f"list of layers. Got: {layers}"
            )

        self._arg_layers = layers
        self._function = None
        self.built = False
        self._convert_input_args = True
        self._allow_non_tensor_positional_args = True

    def build(self, input_shape):
        if callable(self._arg_layers):
            layer_fn = self._arg_layers
            # Create dummy inputs (remove batch size from shape)
            input = tree.map_shape_structure(
                lambda x: Input(shape=x[1:], dtype=self.input_dtype),
                input_shape,
            )
            outputs = layer_fn(input)
            self._function = Function(input, outputs, name=self.name)

        elif (
            isinstance(self._arg_layers, (list, tuple))
            and len(self._arg_layers) > 0
        ):
            layers_list = self._arg_layers
            # Create dummy inputs (remove batch size from shape)
            input = tree.map_shape_structure(
                lambda x: Input(shape=x[1:]), input_shape
            )
            x = input
            for layer in layers_list:
                x = layer(x)
            self._function = Function(input, x, name=self.name)

        # Store structure for get_config/serialization
        self._inputs_struct = self._function._inputs_struct
        self._outputs_struct = self._function._outputs_struct

        # remove input param references now that _function is built
        self._arg_layers = None

        # tracking
        self._layers = self.layers

    @property
    def layers(self):
        """Returns the list of layers contained in this composite layer."""
        # Ensure the function is built
        if not self.built and not self._function:
            raise ValueError(
                "This CompositeLayer has not been built yet. "
                "Call it on inputs to build it before accessing layers."
            )

        # Collect all Layer objects from operations
        layers = []
        for operation in self._function.operations:
            if isinstance(operation, Layer):
                layers.append(operation)
        return layers

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`CompositeLayer.layers` attribute is reserved and should not "
            "be used. Please use another name."
        )

    def call(self, inputs, training=None, mask=None):
        # Apply the function with training mode
        return run_through_graph_with_training_and_mask(
            self._function, inputs, training=training, mask=mask
        )

    def compute_output_shape(self, input_shape):
        return self._function.compute_output_shape(input_shape)

    def compute_output_spec(self, inputs, training=None, mask=None):
        return self._function.compute_output_spec(inputs)

    def get_config(self):
        config = super().get_config()
        self.layers  # accessing this checks the layer was built
        functional_config = serialize_functional_config(self, self._function)
        config.update(functional_config)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Extract CompositeLayer specific config
        layer_config = {}
        for key in ["trainable", "dtype"]:
            layer_config[key] = config.pop(key, None)
        for key in ["name"]:
            layer_config[key] = config.get(key, None)  # keep name for Function

        # Create instance without Function (dummy lambda fn)
        instance = cls.__new__(cls)
        CompositeLayer.__init__(instance, lambda x: x, **layer_config)

        # Initialize Function from config
        instance._function = function_from_config(
            Function, config, custom_objects
        )

        # Copy relevant attributes from Function
        instance._inputs_struct = instance._function._inputs_struct
        instance._outputs_struct = instance._function._outputs_struct

        instance._arg_layers = None
        instance.built = True
        return instance

    def get_layer(self, name=None, index=None):
        """Retrieves a layer based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.
        Indices are based on order of horizontal graph traversal (bottom-up).

        Args:
            name: String, name of layer.
            index: Integer, index of layer.

        Returns:
            A layer instance.
        """
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a layer name or a layer index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError(
                    f"Was asked to retrieve layer at index {index}"
                    f" but model only has {len(self.layers)}"
                    " layers."
                )
            else:
                return self.layers[index]

        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(
                f"No such layer: {name}. Existing layers are: "
                f"{list(layer.name for layer in self.layers)}."
            )
        raise ValueError(
            "Provide either a layer name or layer index at `get_layer`."
        )

    @property
    def input_spec(self):
        if hasattr(self, "_manual_input_spec"):
            return self._manual_input_spec
        elif self._function:
            return compute_input_spec(
                self._function._inputs_struct, self._function._inputs
            )
        else:
            return None

    @input_spec.setter
    def input_spec(self, value):
        self._manual_input_spec = value

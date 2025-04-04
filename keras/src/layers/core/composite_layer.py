import inspect
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
    """Layer that encapsulates a subgraph of layers into a single layer
       in a Keras functional way. This means that the subgraph of layers is
       programmatically accessible. Functional Models containing
       CompositeLayers can be plotted with `keras.utils.plot_model`
       or programmatically edited with 'keras.models.clone_model(call_fn)'.

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
       The first argument of the function will become
       the inputs of the composite layer.

    ```python
    def layer_fn(x):
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(32)(x)
        return outputs

    # Create the composite layer using the function
    composite = layers.CompositeLayer(layer_fn)
    ```

    Additional arguments in layer_fn can be used for configuration,
    but only the first argument represents the layer's runtime
    inputs. Use a dict or list as first argument if your layer
    requires multiple inputs.

    ```python
    # Additional args for configuration.
    # Multiple inputs passed as a list or dict to 'inputs' argument.
     def layer_fn(inputs, dense_size=64):
        x0 = inputs[0] # inputs is a list
        x1 = inputs[1]
        y0 = layers.Dense(dense_size, activation='relu')(x0)
        y1 = layers.Dense(dense_size, activation='relu')(x1)
        return y0 + y1

    composite = layers.CompositeLayer(layer_fn)
    ```

    Reusable composite layers can be packaged
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

        # Init from either a function that defines the
        # layer graph or a sequence of layers.
        # Internally, a CompositeLayer can also
        # be initialized from a Keras Function.
        if not isinstance(layers, Function):
            if not (
                (isinstance(layers, (list, tuple)) and len(layers) > 0)
                or (callable(layers))
            ):
                raise ValueError(
                    f"CompositeLayer requires a layers parameter that is "
                    f"either a function that defines the layer's computation "
                    f"graph or a non-empty list of layers. Got: {layers}"
                )
            # error out on wrong layer_fn signature
            if callable(layers):
                layer_fn = layers
                layer_fn_params = list(inspect.signature(layer_fn).parameters)
                if len(layer_fn_params) < 1:
                    raise ValueError(
                        f"The function used to initialize a CompositeLayer "
                        f"must take the layer's inputs as its first argument. "
                        f"Additional arguments may be used for configuration. "
                        f"If multiple inputs are required at runtime, use a "
                        f"list or a dictionary. "
                        f"Got: {layer_fn_params} for {layer_fn}"
                    )

        # Constructing from a Keras Function is useful
        # internally when deserializing or cloning the layer.
        if isinstance(layers, Function):
            self._build_from_function(function=layers)
            self._arg_layers = None
        # defer building until the first call to build()
        else:
            self._arg_layers = layers
            self._function = None
            self.built = False

        # Allow calling the layer on raw Python data (e.g list of numbers)
        # to be similar to what Functional does.
        self._convert_input_args = True
        # BUG: this is NOT useful and extra positional args are NOT allowed
        # but _convert_input_args=True won't work without this flag.
        self._allow_non_tensor_positional_args = False

    # Note: CompositeLayer does not have the following attributes:
    # _inputs_struct, _outputs_struct, _inputs, _outputs as in
    # Functional model since those are private attributes of Function.

    @property
    def inputs(self):
        return self._function._inputs

    @property
    def outputs(self):
        return self._function._outputs

    # Override Operation.input (as in Functional)
    @property
    def input(self):
        return self._function._inputs_struct

    # Override Operation.output (as in Functional)
    @property
    def output(self):
        return self._function._outputs_struct

    # Only call this from __init__ or build()
    # otherwise, must handle state locking/unlocking.
    def _build_from_function(self, function):
        self._function = function
        # tracking: compute list of layers from the new function
        self._layers = self.layers
        self.built = True

    def build(self, input_shape):
        # if __init__ from Function, build() should do nothing
        assert not isinstance(self._arg_layers, Function)

        def spec_to_input(spec):
            # InputSpec shapes have batch size as first
            # dimension but InputLayer shapes do not.
            return Input(
                shape=spec.shape[1:],
                dtype=spec.dtype,
                name=spec.name,
                optional=spec.optional,
            )

        # create appropriate inputs
        if hasattr(self, "_manual_input_spec"):
            # code path for a manual input spec which may contain
            # optional inputs (set with InputSpec(optional=True)
            inputs = tree.map_structure(spec_to_input, self.input_spec)
        else:
            # In this code path, there are no optional inputs and
            # input_shape cannot have None fields.
            inputs = tree.map_shape_structure(
                lambda x: Input(shape=x[1:], dtype=self.input_dtype),
                input_shape,
            )

        # if "layers" is a callable, call to create the layer graph
        if callable(self._arg_layers):
            layer_fn = self._arg_layers
            outputs = layer_fn(inputs)
            self._build_from_function(Function(inputs, outputs, name=self.name))
        # if "layers" is a list or tuple, create the layer graph sequantially
        elif (
            isinstance(self._arg_layers, (list, tuple))
            and len(self._arg_layers) > 0
        ):
            layers_list = self._arg_layers
            x = inputs
            for layer in layers_list:
                x = layer(x)
            self._build_from_function(Function(inputs, x, name=self.name))

        # remove input param references now that _function is built
        self._arg_layers = None

    @property
    def layers(self):
        """Returns the list of layers contained in this composite layer."""
        # Collect all Layer objects from operations
        layers = []
        if self._function:
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
        if not self.built:
            raise ValueError(
                "This CompositeLayer has not been built yet."
                "You need to call `build()` or call the layer on an input."
            )
        config = super().get_config()
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

        # Recreate the Keras Function
        function = function_from_config(Function, config, custom_objects)
        # Create instance from Function
        instance = cls.__new__(cls)
        CompositeLayer.__init__(instance, function, **layer_config)
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

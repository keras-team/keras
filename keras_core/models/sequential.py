import copy

from tensorflow import nest

from keras_core.api_export import keras_core_export
from keras_core.layers.core.input_layer import InputLayer
from keras_core.models.functional import Functional
from keras_core.models.model import Model
from keras_core.saving import serialization_lib
from keras_core.utils import tracking


@keras_core_export(["keras_core.Sequential", "keras_core.models.Sequential"])
class Sequential(Model):
    """`Sequential` groups a linear stack of layers into a `Model`.

    Examples:

    ```python
    model = keras_core.Sequential()
    model.add(keras_core.Input(shape=(16,)))
    model.add(keras_core.layers.Dense(8))

    # Note that you can also omit the initial `Input`.
    # In that case the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = keras_core.Sequential()
    model.add(keras_core.layers.Dense(8))
    model.add(keras_core.layers.Dense(4))
    # model.weights not created yet

    # Whereas if you specify an `Input`, the model gets built
    # continuously as you are adding layers:
    model = keras_core.Sequential()
    model.add(keras_core.Input(shape=(16,)))
    model.add(keras_core.layers.Dense(8))
    len(model.weights)  # Returns "2"

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = keras_core.Sequential()
    model.add(keras_core.layers.Dense(8))
    model.add(keras_core.layers.Dense(4))
    model.build((None, 16))
    len(model.weights)  # Returns "4"

    # Note that when using the delayed-build pattern (no input shape specified),
    # the model gets built the first time you call `fit`, `eval`, or `predict`,
    # or the first time you call the model on some input data.
    model = keras_core.Sequential()
    model.add(keras_core.layers.Dense(8))
    model.add(keras_core.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)
    ```
    """

    @tracking.no_automatic_dependency_tracking
    def __init__(self, layers=None, trainable=True, name=None):
        super().__init__(trainable=trainable, name=name)
        self._functional = None
        self._layers = []
        if layers:
            for layer in layers:
                self.add(layer, rebuild=False)
            self._maybe_rebuild()

    def add(self, layer, rebuild=True):
        # Legacy case: if the first layer has an input_shape arg,
        # use it to build an InputLayer.
        if not self._layers:
            if getattr(layer, "_input_shape_arg", None) is not None:
                self.add(InputLayer(shape=layer._input_shape_arg))

        # If we are passed a Keras tensor created by keras.Input(), we
        # extract the input layer from its keras history and use that.
        if hasattr(layer, "_keras_history"):
            origin_layer = layer._keras_history[0]
            if isinstance(origin_layer, InputLayer):
                layer = origin_layer
        if not self._is_layer_name_unique(layer):
            raise ValueError(
                "All layers added to a Sequential model "
                f"should have unique names. Name '{layer.name}' is already "
                "the name of a layer in this model. Update the `name` argument "
                "to pass a unique name."
            )
        if (
            isinstance(layer, InputLayer)
            and self._layers
            and isinstance(self._layers[0], InputLayer)
        ):
            raise ValueError(
                f"Sequential model '{self.name}' has already been configured "
                f"to use input shape {self._layers[0].batch_shape}. You cannot "
                f"add a different Input layer to it."
            )

        self._layers.append(layer)
        if rebuild:
            self._maybe_rebuild()
        else:
            self.built = False
            self._tracker.locked = False
            self._functional = None

    def pop(self, rebuild=True):
        layer = self._layers.pop()
        self.built = False
        self._tracker.locked = False
        self._functional = None
        if rebuild:
            self._maybe_rebuild()
        return layer

    def _maybe_rebuild(self):
        self.built = False
        self._functional = None
        self._tracker.locked = False
        if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
            input_shape = self._layers[0].batch_shape
            self.build(input_shape)

    def build(self, input_shape=None):
        if not isinstance(input_shape, (tuple, list)):
            # Do not attempt to build if the model does not have a single
            # input tensor.
            return
        if input_shape and not (
            isinstance(input_shape[0], int) or input_shape[0] is None
        ):
            # Do not attempt to build if the model does not have a single
            # input tensor.
            return
        if not self._layers:
            raise ValueError(
                f"Sequential model {self.name} cannot be built because it has "
                "no layers. Call `model.add(layer)`."
            )
        if isinstance(self._layers[0], InputLayer):
            if self._layers[0].batch_shape != input_shape:
                raise ValueError(
                    f"Sequential model '{self.name}' has already been "
                    "configured to use input shape "
                    f"{self._layers[0].batch_shape}. You cannot build it "
                    f"with input_shape {input_shape}"
                )
        else:
            self._layers = [InputLayer(batch_shape=input_shape)] + self._layers

        # Build functional model
        inputs = self._layers[0].output
        x = inputs
        for layer in self._layers[1:]:
            try:
                x = layer(x)
            except NotImplementedError:
                # Can happen if shape inference is not implemented.
                # TODO: consider reverting inbound nodes on layers processed.
                return
        outputs = x
        self._functional = Functional(inputs=inputs, outputs=outputs)
        self.built = True
        self._post_build()

    def call(self, inputs, training=None, mask=None):
        if self._functional:
            return self._functional.call(inputs, training=training, mask=mask)

        # Fallback: Just apply the layer sequence.
        # This typically happens if `inputs` is a nested struct.
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and
            # `outputs` are the outputs of `layer` applied to `inputs`. At the
            # end of each iteration `inputs` is set to `outputs` to prepare for
            # the next layer.
            kwargs = {}
            if layer._call_has_mask_arg():
                kwargs["mask"] = mask
            if layer._call_has_training_arg():
                kwargs["training"] = training
            outputs = layer(inputs, **kwargs)
            inputs = outputs

            def _get_mask_from_keras_tensor(kt):
                return getattr(kt, "_keras_mask", None)

            mask = nest.map_structure(_get_mask_from_keras_tensor, outputs)
        return outputs

    @property
    def layers(self):
        # Historically, `sequential.layers` only returns layers that were added
        # via `add`, and omits the auto-generated `InputLayer` that comes at the
        # bottom of the stack.
        layers = self._layers
        if layers and isinstance(layers[0], InputLayer):
            return layers[1:]
        return layers[:]

    def compute_output_spec(self, inputs, training=None, mask=None):
        if self._functional:
            return self._functional.compute_output_spec(
                inputs, training=training, mask=mask
            )
        # Direct application
        for layer in self.layers:
            outputs = layer.compute_output_spec(
                inputs, training=training
            )  # Ignore mask
            inputs = outputs
        return outputs

    @property
    def input_shape(self):
        if self._functional:
            return self._functional.input_shape
        raise ValueError(
            f"Sequential model '{self.name}' has no defined input shape yet."
        )

    @property
    def output_shape(self):
        if self._functional:
            return self._functional.output_shape
        raise ValueError(
            f"Sequential model '{self.name}' has no defined output shape yet."
        )

    @property
    def inputs(self):
        if self._functional:
            return self._functional.inputs
        raise ValueError(
            f"Sequential model '{self.name}' has no defined inputs yet."
        )

    @property
    def outputs(self):
        if self._functional:
            return self._functional.outputs
        raise ValueError(
            f"Sequential model '{self.name}' has no defined outputs yet."
        )

    def _is_layer_name_unique(self, layer):
        for ref_layer in self._layers:
            if layer.name == ref_layer.name and ref_layer is not layer:
                return False
        return True

    def get_config(self):
        layer_configs = []
        for layer in super().layers:
            # `super().layers` include the InputLayer if available (it is
            # filtered out of `self.layers`).
            layer_configs.append(
                serialization_lib.serialize_keras_object(layer)
            )
        config = Model.get_config(self)
        config["name"] = self.name
        config["layers"] = copy.deepcopy(layer_configs)
        if self._functional is not None:
            config["build_input_shape"] = self._layers[0].batch_shape
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "name" in config:
            name = config["name"]
            build_input_shape = config.get("build_input_shape")
            layer_configs = config["layers"]
        else:
            name = None
            layer_configs = config
        model = cls(name=name)
        for layer_config in layer_configs:
            layer = serialization_lib.deserialize_keras_object(
                layer_config,
                custom_objects=custom_objects,
            )
            model.add(layer)
        if (
            not model._functional
            and build_input_shape
            and isinstance(build_input_shape, (tuple, list))
        ):
            model.build(build_input_shape)
        return model

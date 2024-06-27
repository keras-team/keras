import inspect
import json
import typing
import warnings

from keras.src import backend
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.models.variable_mapping import map_saveable_variables
from keras.src.saving import saving_api
from keras.src.trainers import trainer as base_trainer
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.trainer import (
        TensorFlowTrainer as Trainer,
    )
elif backend.backend() == "jax":
    from keras.src.backend.jax.trainer import JAXTrainer as Trainer
elif backend.backend() == "torch":
    from keras.src.backend.torch.trainer import TorchTrainer as Trainer
elif backend.backend() == "numpy":
    from keras.src.backend.numpy.trainer import NumpyTrainer as Trainer
else:
    raise RuntimeError(
        f"Backend '{backend.backend()}' must implement the Trainer class."
    )


@keras_export(["keras.Model", "keras.models.Model"])
class Model(Trainer, base_trainer.Trainer, Layer):
    """A model grouping layers into an object with training/inference features.

    There are three ways to instantiate a `Model`:

    ## With the "Functional API"

    You start from `Input`,
    you chain layer calls to specify the model's forward pass,
    and finally, you create your model from inputs and outputs:

    ```python
    inputs = keras.Input(shape=(37,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    ```

    Note: Only dicts, lists, and tuples of input tensors are supported. Nested
    inputs are not supported (e.g. lists of list or dicts of dict).

    A new Functional API model can also be created by using the
    intermediate tensors. This enables you to quickly extract sub-components
    of the model.

    Example:

    ```python
    inputs = keras.Input(shape=(None, None, 3))
    processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
    conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
    pooling = keras.layers.GlobalAveragePooling2D()(conv)
    feature = keras.layers.Dense(10)(pooling)

    full_model = keras.Model(inputs, feature)
    backbone = keras.Model(processed, conv)
    activations = keras.Model(conv, feature)
    ```

    Note that the `backbone` and `activations` models are not
    created with `keras.Input` objects, but with the tensors that originate
    from `keras.Input` objects. Under the hood, the layers and weights will
    be shared across these models, so that user can train the `full_model`, and
    use `backbone` or `activations` to do feature extraction.
    The inputs and outputs of the model can be nested structures of tensors as
    well, and the created models are standard Functional API models that support
    all the existing APIs.

    ## By subclassing the `Model` class

    In that case, you should define your
    layers in `__init__()` and you should implement the model's forward pass
    in `call()`.

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation="relu")
            self.dense2 = keras.layers.Dense(5, activation="softmax")

        def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)

    model = MyModel()
    ```

    If you subclass `Model`, you can optionally have
    a `training` argument (boolean) in `call()`, which you can use to specify
    a different behavior in training and inference:

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation="relu")
            self.dense2 = keras.layers.Dense(5, activation="softmax")
            self.dropout = keras.layers.Dropout(0.5)

        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.dropout(x, training=training)
            return self.dense2(x)

    model = MyModel()
    ```

    Once the model is created, you can config the model with losses and metrics
    with `model.compile()`, train the model with `model.fit()`, or use the model
    to do prediction with `model.predict()`.

    ## With the `Sequential` class

    In addition, `keras.Sequential` is a special case of model where
    the model is purely a stack of single-input, single-output layers.

    ```python
    model = keras.Sequential([
        keras.Input(shape=(None, None, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=3),
    ])
    ```
    """

    def __new__(cls, *args, **kwargs):
        # Signature detection for usage of `Model` as a `Functional`
        if functional_init_arguments(args, kwargs) and cls == Model:
            from keras.src.models.functional import Functional

            return Functional.__new__(Functional, *args, **kwargs)
        return typing.cast(Model, super().__new__(cls))

    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        from keras.src.models import functional

        # Signature detection for usage of a `Model` subclass
        # as a `Functional` subclass
        if functional_init_arguments(args, kwargs):
            inject_functional_model_class(self.__class__)
            functional.Functional.__init__(self, *args, **kwargs)
        else:
            Layer.__init__(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    @property
    def layers(self):
        return list(self._flatten_layers(include_self=False, recursive=False))

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Model.layers` attribute is reserved and should not be used. "
            "Please use another name."
        )

    @traceback_utils.filter_traceback
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

    @traceback_utils.filter_traceback
    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
        """Prints a string summary of the network.

        Args:
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided, becomes
                `[0.3, 0.6, 0.70, 1.]`. Defaults to `None`.
            print_fn: Print function to use. By default, prints to `stdout`.
                If `stdout` doesn't work in your environment, change to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            expand_nested: Whether to expand the nested models.
                Defaults to `False`.
            show_trainable: Whether to show if a layer is trainable.
                Defaults to `False`.
            layer_range: a list or tuple of 2 strings,
                which is the starting layer name and ending layer name
                (both inclusive) indicating the range of layers to be printed
                in summary. It also accepts regex patterns instead of exact
                names. In this case, the start predicate will be
                the first element that matches `layer_range[0]`
                and the end predicate will be the last element
                that matches `layer_range[1]`.
                By default `None` considers all layers of the model.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        summary_utils.print_summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            layer_range=layer_range,
        )

    @traceback_utils.filter_traceback
    def save(self, filepath, overwrite=True, zipped=None, **kwargs):
        """Saves a model as a `.keras` file.

        Args:
            filepath: `str` or `pathlib.Path` object.
                The path where to save the model. Must end in `.keras`
                (unless saving the model as an unzipped directory
                via `zipped=False`).
            overwrite: Whether we should overwrite any existing model at
                the target location, or instead ask the user via
                an interactive prompt.
            zipped: Whether to save the model as a zipped `.keras`
                archive (default when saving locally), or as an
                unzipped directory (default when saving on the
                Hugging Face Hub).

        Example:

        ```python
        model = keras.Sequential(
            [
                keras.layers.Dense(5, input_shape=(3,)),
                keras.layers.Softmax(),
            ],
        )
        model.save("model.keras")
        loaded_model = keras.saving.load_model("model.keras")
        x = keras.random.uniform((10, 3))
        assert np.allclose(model.predict(x), loaded_model.predict(x))
        ```

        Note that `model.save()` is an alias for `keras.saving.save_model()`.

        The saved `.keras` file contains:

        - The model's configuration (architecture)
        - The model's weights
        - The model's optimizer's state (if any)

        Thus models can be reinstantiated in the exact same state.
        """
        return saving_api.save_model(
            self, filepath, overwrite=overwrite, zipped=zipped, **kwargs
        )

    @traceback_utils.filter_traceback
    def save_weights(self, filepath, overwrite=True):
        """Saves all layer weights to a `.weights.h5` file.

        Args:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the model. Must end in `.weights.h5`.
            overwrite: Whether we should overwrite any existing model
                at the target location, or instead ask the user
                via an interactive prompt.
        """
        return saving_api.save_weights(self, filepath, overwrite=overwrite)

    @traceback_utils.filter_traceback
    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        """Load weights from a file saved via `save_weights()`.

        Weights are loaded based on the network's
        topology. This means the architecture should be the same as when the
        weights were saved. Note that layers that don't have weights are not
        taken into account in the topological ordering, so adding or removing
        layers is fine as long as they don't have weights.

        **Partial weight loading**

        If you have modified your model, for instance by adding a new layer
        (with weights) or by changing the shape of the weights of a layer,
        you can choose to ignore errors and continue loading
        by setting `skip_mismatch=True`. In this case any layer with
        mismatching weights will be skipped. A warning will be displayed
        for each skipped layer.

        Args:
            filepath: String, path to the weights file to load.
                It can either be a `.weights.h5` file
                or a legacy `.h5` weights file.
            skip_mismatch: Boolean, whether to skip loading of layers where
                there is a mismatch in the number of weights, or a mismatch in
                the shape of the weights.
        """
        saving_api.load_weights(
            self, filepath, skip_mismatch=skip_mismatch, **kwargs
        )

    def quantize(self, mode):
        """Quantize the weights of the model.

        Note that the model must be built first before calling this method.
        `quantize` will recursively call `quantize(mode)` in all layers and
        will be skipped if the layer doesn't implement the function.

        Args:
            mode: The mode of the quantization. Only 'int8' is supported at this
                time.
        """
        from keras.src.dtype_policies import QUANTIZATION_MODES

        if not self.built:
            raise ValueError(
                "The model must be built first before calling `quantize()`."
            )
        if mode not in QUANTIZATION_MODES:
            raise ValueError(
                "Invalid quantization mode. "
                f"Expected one of {QUANTIZATION_MODES}. Received: mode={mode}"
            )
        mode_changed = False
        for layer in self._flatten_layers():
            list_of_sublayers = list(layer._flatten_layers())
            if len(list_of_sublayers) == 1:  # leaves of the model
                try:
                    layer.quantize(mode)
                    mode_changed = True
                except NotImplementedError as e:
                    warnings.warn(str(e))
        # We need to set these functions to `None` to remake them for changed
        # call function
        if mode_changed:
            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def build_from_config(self, config):
        if not config:
            return
        if "input_shape" in config:
            # Case: all inputs are in the first arg (possibly nested).
            if utils.is_default(self.build):
                status = self._build_by_run_for_single_pos_arg(
                    config["input_shape"]
                )
            else:
                try:
                    self.build(config["input_shape"])
                    status = True
                except:
                    status = False
            self._build_shapes_dict = config

        elif "shapes_dict" in config:
            # Case: inputs were recorded as multiple keyword arguments.
            if utils.is_default(self.build):
                status = self._build_by_run_for_kwargs(config["shapes_dict"])
            else:
                try:
                    self.build(**config["shapes_dict"])
                    status = True
                except:
                    status = False
            self._build_shapes_dict = config["shapes_dict"]

        if not status:
            warnings.warn(
                f"Model '{self.name}' had a build config, but the model "
                "cannot be built automatically in "
                "`build_from_config(config)`. "
                "You should implement "
                "`def build_from_config(self, config)`, "
                "and you might also want to implement the method "
                " that generates the config at saving time, "
                "`def get_build_config(self)`. "
                "The method `build_from_config()` is meant to "
                "create the state of the model (i.e. its variables) "
                "upon deserialization.",
                stacklevel=2,
            )

    def to_json(self, **kwargs):
        """Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={...})`.

        Args:
            **kwargs: Additional keyword arguments to be passed to
                `json.dumps()`.

        Returns:
            A JSON string.
        """
        from keras.src.saving import serialization_lib

        model_config = serialization_lib.serialize_keras_object(self)
        return json.dumps(model_config, **kwargs)

    def export(self, filepath, format="tf_saved_model"):
        """Create a TF SavedModel artifact for inference.

        **Note:** This can currently only be used with
        the TensorFlow or JAX backends.

        This method lets you export a model to a lightweight SavedModel artifact
        that contains the model's forward pass only (its `call()` method)
        and can be served via e.g. TF-Serving. The forward pass is registered
        under the name `serve()` (see example below).

        The original code of the model (including any custom layers you may
        have used) is *no longer* necessary to reload the artifact -- it is
        entirely standalone.

        Args:
            filepath: `str` or `pathlib.Path` object. Path where to save
                the artifact.

        Example:

        ```python
        # Create the artifact
        model.export("path/to/location")

        # Later, in a different process/environment...
        reloaded_artifact = tf.saved_model.load("path/to/location")
        predictions = reloaded_artifact.serve(input_data)
        ```

        If you would like to customize your serving endpoints, you can
        use the lower-level `keras.export.ExportArchive` class. The
        `export()` method relies on `ExportArchive` internally.
        """
        from keras.src.export import export_lib

        export_lib.export_model(self, filepath)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.src.models.functional import Functional

        functional_config_keys = [
            "name",
            "layers",
            "input_layers",
            "output_layers",
        ]
        is_functional_config = all(
            key in config for key in functional_config_keys
        )
        argspec = inspect.getfullargspec(cls.__init__)
        functional_init_args = inspect.getfullargspec(Functional.__init__).args[
            1:
        ]
        revivable_as_functional = (
            cls in {Functional, Model}
            or argspec.args[1:] == functional_init_args
            or (argspec.varargs == "args" and argspec.varkw == "kwargs")
        )
        if is_functional_config and revivable_as_functional:
            # Revive Functional model
            # (but not Functional subclasses with a custom __init__)
            from keras.src.models.functional import functional_from_config

            return functional_from_config(
                cls, config, custom_objects=custom_objects
            )

        # Either the model has a custom __init__, or the config
        # does not contain all the information necessary to
        # revive a Functional model. This happens when the user creates
        # subclassed models where `get_config()` is returning
        # insufficient information to be considered a Functional model.
        # In this case, we fall back to provide all config into the
        # constructor of the class.
        try:
            return cls(**config)
        except TypeError as e:
            raise TypeError(
                "Unable to revive model from config. When overriding "
                "the `get_config()` method, make sure that the "
                "returned config contains all items used as arguments "
                f"in the  constructor to {cls}, "
                "which is the default behavior. "
                "You can override this default behavior by defining a "
                "`from_config(cls, config)` class method to specify "
                "how to create an "
                f"instance of {cls.__name__} from its config.\n\n"
                f"Received config={config}\n\n"
                f"Error encountered during deserialization: {e}"
            )

    def _get_variable_map(self):
        store = {}
        map_saveable_variables(self, store=store, visited_saveables=set())
        return store


@keras_export("keras.models.model_from_json")
def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.

    Example:

    >>> model = keras.Sequential([
    ...     keras.layers.Dense(5, input_shape=(3,)),
    ...     keras.layers.Softmax()])
    >>> config = model.to_json()
    >>> loaded_model = keras.models.model_from_json(config)

    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A Keras model instance (uncompiled).
    """
    from keras.src.saving import serialization_lib

    model_config = json.loads(json_string)
    return serialization_lib.deserialize_keras_object(
        model_config, custom_objects=custom_objects
    )


def functional_init_arguments(args, kwargs):
    return (
        (len(args) == 2)
        or (len(args) == 1 and "outputs" in kwargs)
        or ("inputs" in kwargs and "outputs" in kwargs)
    )


def inject_functional_model_class(cls):
    """Inject `Functional` into the hierarchy of this class if needed."""
    from keras.src.models import functional

    if cls == Model:
        return functional.Functional
    # In case there is any multiple inheritance, we stop injecting the
    # class if keras model is not in its class hierarchy.
    if cls == object:
        return object

    cls.__bases__ = tuple(
        inject_functional_model_class(base) for base in cls.__bases__
    )
    # Trigger any `__new__` class swapping that needed to happen on `Functional`
    # but did not because functional was not in the class hierarchy.
    cls.__new__(cls)

    return cls

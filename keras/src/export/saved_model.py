"""Library for exporting SavedModel for Keras models/layers."""

from keras.src import backend
from keras.src import layers
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_tf_tensor_spec
from keras.src.utils import io_utils
from keras.src.utils.module_utils import tensorflow as tf

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.export import (
        TFExportArchive as BackendExportArchive,
    )
elif backend.backend() == "jax":
    from keras.src.backend.jax.export import (
        JaxExportArchive as BackendExportArchive,
    )
elif backend.backend() == "torch":
    from keras.src.backend.torch.export import (
        TorchExportArchive as BackendExportArchive,
    )
elif backend.backend() == "numpy":
    from keras.src.backend.numpy.export import (
        NumpyExportArchive as BackendExportArchive,
    )
elif backend.backend() == "openvino":
    from keras.src.backend.openvino.export import (
        OpenvinoExportArchive as BackendExportArchive,
    )
else:
    raise RuntimeError(
        f"Backend '{backend.backend()}' must implement ExportArchive."
    )


DEFAULT_ENDPOINT_NAME = "serve"


@keras_export("keras.export.ExportArchive")
class ExportArchive(BackendExportArchive):
    """ExportArchive is used to write SavedModel artifacts (e.g. for inference).

    If you have a Keras model or layer that you want to export as SavedModel for
    serving (e.g. via TensorFlow-Serving), you can use `ExportArchive`
    to configure the different serving endpoints you need to make available,
    as well as their signatures. Simply instantiate an `ExportArchive`,
    use `track()` to register the layer(s) or model(s) to be used,
    then use the `add_endpoint()` method to register a new serving endpoint.
    When done, use the `write_out()` method to save the artifact.

    The resulting artifact is a SavedModel and can be reloaded via
    `tf.saved_model.load`.

    Examples:

    Here's how to export a model for inference.

    ```python
    export_archive = ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(
        name="serve",
        fn=model.call,
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
    )
    export_archive.write_out("path/to/location")

    # Elsewhere, we can reload the artifact and serve it.
    # The endpoint we added is available as a method:
    serving_model = tf.saved_model.load("path/to/location")
    outputs = serving_model.serve(inputs)
    ```

    Here's how to export a model with one endpoint for inference and one
    endpoint for a training-mode forward pass (e.g. with dropout on).

    ```python
    export_archive = ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(
        name="call_inference",
        fn=lambda x: model.call(x, training=False),
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
    )
    export_archive.add_endpoint(
        name="call_training",
        fn=lambda x: model.call(x, training=True),
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
    )
    export_archive.write_out("path/to/location")
    ```

    **Note on resource tracking:**

    `ExportArchive` is able to automatically track all `keras.Variables` used
    by its endpoints, so most of the time calling `.track(model)`
    is not strictly required. However, if your model uses lookup layers such
    as `IntegerLookup`, `StringLookup`, or `TextVectorization`,
    it will need to be tracked explicitly via `.track(model)`.

    Explicit tracking is also required if you need to be able to access
    the properties `variables`, `trainable_variables`, or
    `non_trainable_variables` on the revived archive.
    """

    def __init__(self):
        super().__init__()
        if backend.backend() not in ("tensorflow", "jax", "torch"):
            raise NotImplementedError(
                "`ExportArchive` is only compatible with TensorFlow, JAX and "
                "Torch backends."
            )

        self._endpoint_names = []
        self._endpoint_signatures = {}
        self.tensorflow_version = tf.__version__

        self._tf_trackable = tf.__internal__.tracking.AutoTrackable()
        self._tf_trackable.variables = []
        self._tf_trackable.trainable_variables = []
        self._tf_trackable.non_trainable_variables = []

    @property
    def variables(self):
        return self._tf_trackable.variables

    @property
    def trainable_variables(self):
        return self._tf_trackable.trainable_variables

    @property
    def non_trainable_variables(self):
        return self._tf_trackable.non_trainable_variables

    def track(self, resource):
        """Track the variables (of a layer or model) and other assets.

        By default, all variables used by an endpoint function are automatically
        tracked when you call `add_endpoint()`. However, non-variables assets
        such as lookup tables need to be tracked manually. Note that lookup
        tables used by built-in Keras layers (`TextVectorization`,
        `IntegerLookup`, `StringLookup`) are automatically tracked by
        `add_endpoint()`.

        Args:
            resource: A layer, model or a TensorFlow trackable resource.
        """
        if isinstance(resource, layers.Layer) and not resource.built:
            raise ValueError(
                "The layer provided has not yet been built. "
                "It must be built before export."
            )

        # Note: with the TensorFlow backend, Layers and Models fall into both
        # the Layer case and the Trackable case. The Trackable case is needed
        # for preprocessing layers in order to track lookup tables.
        if isinstance(resource, tf.__internal__.tracking.Trackable):
            if not hasattr(self, "_tracked"):
                self._tracked = []
            self._tracked.append(resource)

        if isinstance(resource, layers.Layer):
            self._track_layer(resource)
        elif not isinstance(resource, tf.__internal__.tracking.Trackable):
            raise ValueError(
                "Invalid resource type. Expected a Keras `Layer` or `Model` "
                "or a TensorFlow `Trackable` object. "
                f"Received object {resource} of type '{type(resource)}'. "
            )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        """Register a new serving endpoint.

        Args:
            name: `str`. The name of the endpoint.
            fn: A callable. It should only leverage resources
                (e.g. `keras.Variable` objects or `tf.lookup.StaticHashTable`
                objects) that are available on the models/layers tracked by the
                `ExportArchive` (you can call `.track(model)` to track a new
                model).
                The shape and dtype of the inputs to the function must be
                known. For that purpose, you can either 1) make sure that `fn`
                is a `tf.function` that has been called at least once, or 2)
                provide an `input_signature` argument that specifies the shape
                and dtype of the inputs (see below).
            input_signature: Optional. Specifies the shape and dtype of `fn`.
                Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
                `backend.KerasTensor`, or backend tensor (see below for an
                example showing a `Functional` model with 2 input arguments). If
                not provided, `fn` must be a `tf.function` that has been called
                at least once. Defaults to `None`.
            **kwargs: Additional keyword arguments:
                - Specific to the JAX backend:
                    - `is_static`: Optional `bool`. Indicates whether `fn` is
                        static. Set to `False` if `fn` involves state updates
                        (e.g., RNG seeds).
                    - `jax2tf_kwargs`: Optional `dict`. Arguments for
                        `jax2tf.convert`. See [`jax2tf.convert`](
                            https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                        If `native_serialization` and `polymorphic_shapes` are
                        not provided, they are automatically computed.

        Returns:
            The `tf.function` wrapping `fn` that was added to the archive.

        Example:

        Adding an endpoint using the `input_signature` argument when the
        model has a single input argument:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
        )
        ```

        Adding an endpoint using the `input_signature` argument when the
        model has two positional input arguments:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[
                keras.InputSpec(shape=(None, 3), dtype="float32"),
                keras.InputSpec(shape=(None, 4), dtype="float32"),
            ],
        )
        ```

        Adding an endpoint using the `input_signature` argument when the
        model has one input argument that is a list of 2 tensors (e.g.
        a Functional model with 2 inputs):

        ```python
        model = keras.Model(inputs=[x1, x2], outputs=outputs)

        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[
                [
                    keras.InputSpec(shape=(None, 3), dtype="float32"),
                    keras.InputSpec(shape=(None, 4), dtype="float32"),
                ],
            ],
        )
        ```

        This also works with dictionary inputs:

        ```python
        model = keras.Model(inputs={"x1": x1, "x2": x2}, outputs=outputs)

        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[
                {
                    "x1": keras.InputSpec(shape=(None, 3), dtype="float32"),
                    "x2": keras.InputSpec(shape=(None, 4), dtype="float32"),
                },
            ],
        )
        ```

        Adding an endpoint that is a `tf.function`:

        ```python
        @tf.function()
        def serving_fn(x):
            return model(x)

        # The function must be traced, i.e. it must be called at least once.
        serving_fn(tf.random.normal(shape=(2, 3)))

        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(name="serve", fn=serving_fn)
        ```

        Combining a model with some TensorFlow preprocessing, which can use
        TensorFlow resources:

        ```python
        lookup_table = tf.lookup.StaticHashTable(initializer, default_value=0.0)

        export_archive = ExportArchive()
        model_fn = export_archive.track_and_add_endpoint(
            "model_fn",
            model,
            input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32)],
        )
        export_archive.track(lookup_table)

        @tf.function()
        def serving_fn(x):
            x = lookup_table.lookup(x)
            return model_fn(x)

        export_archive.add_endpoint(name="serve", fn=serving_fn)
        ```
        """
        if name in self._endpoint_names:
            raise ValueError(f"Endpoint name '{name}' is already taken.")

        if backend.backend() != "jax":
            if "jax2tf_kwargs" in kwargs or "is_static" in kwargs:
                raise ValueError(
                    "'jax2tf_kwargs' and 'is_static' are only supported with "
                    f"the jax backend. Current backend: {backend.backend()}"
                )

        # The fast path if `fn` is already a `tf.function`.
        if input_signature is None:
            if isinstance(fn, tf.types.experimental.GenericFunction):
                if not fn._list_all_concrete_functions():
                    raise ValueError(
                        f"The provided tf.function '{fn}' "
                        "has never been called. "
                        "To specify the expected shape and dtype "
                        "of the function's arguments, "
                        "you must either provide a function that "
                        "has been called at least once, or alternatively pass "
                        "an `input_signature` argument in `add_endpoint()`."
                    )
                decorated_fn = fn
            else:
                raise ValueError(
                    "If the `fn` argument provided is not a `tf.function`, "
                    "you must provide an `input_signature` argument to "
                    "specify the shape and dtype of the function arguments. "
                    "Example:\n\n"
                    "export_archive.add_endpoint(\n"
                    "    name='call',\n"
                    "    fn=model.call,\n"
                    "    input_signature=[\n"
                    "        keras.InputSpec(\n"
                    "            shape=(None, 224, 224, 3),\n"
                    "            dtype='float32',\n"
                    "        )\n"
                    "    ],\n"
                    ")"
                )
            setattr(self._tf_trackable, name, decorated_fn)
            self._endpoint_names.append(name)
            return decorated_fn

        input_signature = tree.map_structure(
            make_tf_tensor_spec, input_signature
        )
        decorated_fn = super().add_endpoint(name, fn, input_signature, **kwargs)
        self._endpoint_signatures[name] = input_signature
        setattr(self._tf_trackable, name, decorated_fn)
        self._endpoint_names.append(name)
        return decorated_fn

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        """Track the variables and register a new serving endpoint.

        This function combines the functionality of `track` and `add_endpoint`.
        It tracks the variables of the `resource` (either a layer or a model)
        and registers a serving endpoint using `resource.__call__`.

        Args:
            name: `str`. The name of the endpoint.
            resource: A trackable Keras resource, such as a layer or model.
            input_signature: Optional. Specifies the shape and dtype of `fn`.
                Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
                `backend.KerasTensor`, or backend tensor (see below for an
                example showing a `Functional` model with 2 input arguments). If
                not provided, `fn` must be a `tf.function` that has been called
                at least once. Defaults to `None`.
            **kwargs: Additional keyword arguments:
                - Specific to the JAX backend:
                    - `is_static`: Optional `bool`. Indicates whether `fn` is
                        static. Set to `False` if `fn` involves state updates
                        (e.g., RNG seeds).
                    - `jax2tf_kwargs`: Optional `dict`. Arguments for
                        `jax2tf.convert`. See [`jax2tf.convert`](
                            https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                        If `native_serialization` and `polymorphic_shapes` are
                        not provided, they are automatically computed.

        """
        if name in self._endpoint_names:
            raise ValueError(f"Endpoint name '{name}' is already taken.")
        if not isinstance(resource, layers.Layer):
            raise ValueError(
                "Invalid resource type. Expected an instance of a Keras "
                "`Layer` or `Model`. "
                f"Received: resource={resource} (of type {type(resource)})"
            )
        if not resource.built:
            raise ValueError(
                "The layer provided has not yet been built. "
                "It must be built before export."
            )
        if backend.backend() != "jax":
            if "jax2tf_kwargs" in kwargs or "is_static" in kwargs:
                raise ValueError(
                    "'jax2tf_kwargs' and 'is_static' are only supported with "
                    f"the jax backend. Current backend: {backend.backend()}"
                )

        input_signature = tree.map_structure(
            make_tf_tensor_spec, input_signature
        )

        if not hasattr(BackendExportArchive, "track_and_add_endpoint"):
            # Default behavior.
            self.track(resource)
            return self.add_endpoint(
                name, resource.__call__, input_signature, **kwargs
            )
        else:
            # Special case for the torch backend.
            decorated_fn = super().track_and_add_endpoint(
                name, resource, input_signature, **kwargs
            )
            self._endpoint_signatures[name] = input_signature
            setattr(self._tf_trackable, name, decorated_fn)
            self._endpoint_names.append(name)
            return decorated_fn

    def add_variable_collection(self, name, variables):
        """Register a set of variables to be retrieved after reloading.

        Arguments:
            name: The string name for the collection.
            variables: A tuple/list/set of `keras.Variable` instances.

        Example:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        # Register an endpoint
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
        )
        # Save a variable collection
        export_archive.add_variable_collection(
            name="optimizer_variables", variables=model.optimizer.variables)
        export_archive.write_out("path/to/location")

        # Reload the object
        revived_object = tf.saved_model.load("path/to/location")
        # Retrieve the variables
        optimizer_variables = revived_object.optimizer_variables
        ```
        """
        if not isinstance(variables, (list, tuple, set)):
            raise ValueError(
                "Expected `variables` to be a list/tuple/set. "
                f"Received instead object of type '{type(variables)}'."
            )
        # Ensure that all variables added are either tf.Variables
        # or Variables created by Keras 3 with the TF or JAX backends.
        if not all(
            isinstance(v, (tf.Variable, backend.Variable)) for v in variables
        ):
            raise ValueError(
                "Expected all elements in `variables` to be "
                "`tf.Variable` instances. Found instead the following types: "
                f"{list(set(type(v) for v in variables))}"
            )
        if backend.backend() == "jax":
            variables = tree.flatten(
                tree.map_structure(self._convert_to_tf_variable, variables)
            )
        setattr(self._tf_trackable, name, list(variables))

    def write_out(self, filepath, options=None, verbose=True):
        """Write the corresponding SavedModel to disk.

        Arguments:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the artifact.
            options: `tf.saved_model.SaveOptions` object that specifies
                SavedModel saving options.
            verbose: whether to print all the variables of an
                exported SavedModel.

        **Note on TF-Serving**: all endpoints registered via `add_endpoint()`
        are made visible for TF-Serving in the SavedModel artifact. In addition,
        the first endpoint registered is made visible under the alias
        `"serving_default"` (unless an endpoint with the name
        `"serving_default"` was already registered manually),
        since TF-Serving requires this endpoint to be set.
        """
        if not self._endpoint_names:
            raise ValueError(
                "No endpoints have been set yet. Call add_endpoint()."
            )
        self._filter_and_track_resources()

        signatures = {}
        for name in self._endpoint_names:
            signatures[name] = self._get_concrete_fn(name)
        # Add "serving_default" signature key for TFServing
        if "serving_default" not in self._endpoint_names:
            signatures["serving_default"] = self._get_concrete_fn(
                self._endpoint_names[0]
            )

        tf.saved_model.save(
            self._tf_trackable,
            filepath,
            options=options,
            signatures=signatures,
        )

        # Print out available endpoints
        if verbose:
            endpoints = "\n\n".join(
                _print_signature(
                    getattr(self._tf_trackable, name), name, verbose=verbose
                )
                for name in self._endpoint_names
            )
            io_utils.print_msg(
                f"Saved artifact at '{filepath}'. "
                "The following endpoints are available:\n\n"
                f"{endpoints}"
            )

    def _convert_to_tf_variable(self, backend_variable):
        if not isinstance(backend_variable, backend.Variable):
            raise TypeError(
                "`backend_variable` must be a `backend.Variable`. "
                f"Recevied: backend_variable={backend_variable} of type "
                f"({type(backend_variable)})"
            )
        return tf.Variable(
            backend_variable.value,
            dtype=backend_variable.dtype,
            trainable=backend_variable.trainable,
            name=backend_variable.name,
        )

    def _get_concrete_fn(self, endpoint):
        """Workaround for some SavedModel quirks."""
        if endpoint in self._endpoint_signatures:
            return getattr(self._tf_trackable, endpoint)
        else:
            traces = getattr(self._tf_trackable, endpoint)._trackable_children(
                "saved_model"
            )
            return list(traces.values())[0]

    def _get_variables_used_by_endpoints(self):
        fns = [self._get_concrete_fn(name) for name in self._endpoint_names]
        return _list_variables_used_by_fns(fns)

    def _filter_and_track_resources(self):
        """Track resources used by endpoints / referenced in `track()` calls."""
        # Start by extracting variables from endpoints.
        fns = [self._get_concrete_fn(name) for name in self._endpoint_names]
        tvs, ntvs = _list_variables_used_by_fns(fns)
        self._tf_trackable._all_variables = list(tvs + ntvs)

        # Next, track lookup tables.
        # Hopefully, one day this will be automated at the tf.function level.
        self._tf_trackable._misc_assets = []
        from tensorflow.saved_model.experimental import TrackableResource

        if hasattr(self, "_tracked"):
            for root in self._tracked:
                descendants = tf.train.TrackableView(root).descendants()
                for trackable in descendants:
                    if isinstance(trackable, TrackableResource):
                        self._tf_trackable._misc_assets.append(trackable)


def export_saved_model(
    model, filepath, verbose=None, input_signature=None, **kwargs
):
    """Export the model as a TensorFlow SavedModel artifact for inference.

    This method lets you export a model to a lightweight SavedModel artifact
    that contains the model's forward pass only (its `call()` method)
    and can be served via e.g. TensorFlow Serving. The forward pass is
    registered under the name `serve()` (see example below).

    The original code of the model (including any custom layers you may
    have used) is *no longer* necessary to reload the artifact -- it is
    entirely standalone.

    Args:
        filepath: `str` or `pathlib.Path` object. The path to save the artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which uses the default value set by different backends and
            formats.
        input_signature: Optional. Specifies the shape and dtype of the model
            inputs. Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
            `backend.KerasTensor`, or backend tensor. If not provided, it will
            be automatically computed. Defaults to `None`.
        **kwargs: Additional keyword arguments:
            - Specific to the JAX backend:
                - `is_static`: Optional `bool`. Indicates whether `fn` is
                    static. Set to `False` if `fn` involves state updates
                    (e.g., RNG seeds).
                - `jax2tf_kwargs`: Optional `dict`. Arguments for
                    `jax2tf.convert`. See [`jax2tf.convert`](
                        https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                    If `native_serialization` and `polymorphic_shapes` are not
                    provided, they are automatically computed.

    **Note:** This feature is currently supported only with TensorFlow, JAX and
    Torch backends. Support for the Torch backend is experimental.

    **Note:** The dynamic shape feature is not yet supported with Torch
    backend. As a result, you must fully define the shapes of the inputs using
    `input_signature`. If `input_signature` is not provided, all instances of
    `None` (such as the batch size) will be replaced with `1`.

    Example:

    ```python
    # Export the model as a TensorFlow SavedModel artifact
    model.export("path/to/location", format="tf_saved_model")

    # Load the artifact in a different process/environment
    reloaded_artifact = tf.saved_model.load("path/to/location")
    predictions = reloaded_artifact.serve(input_data)
    ```

    If you would like to customize your serving endpoints, you can
    use the lower-level `keras.export.ExportArchive` class. The
    `export()` method relies on `ExportArchive` internally.
    """
    if verbose is None:
        verbose = True  # Defaults to `True` for all backends.
    export_archive = ExportArchive()
    if input_signature is None:
        input_signature = get_input_signature(model)

    export_archive.track_and_add_endpoint(
        DEFAULT_ENDPOINT_NAME, model, input_signature, **kwargs
    )
    export_archive.write_out(filepath, verbose=verbose)


def _print_signature(fn, name, verbose=True):
    concrete_fn = fn._list_all_concrete_functions()[0]
    pprinted_signature = concrete_fn.pretty_printed_signature(verbose=verbose)
    lines = pprinted_signature.split("\n")
    lines = [f"* Endpoint '{name}'"] + lines[1:]
    endpoint = "\n".join(lines)
    return endpoint


def _list_variables_used_by_fns(fns):
    trainable_variables = []
    non_trainable_variables = []
    trainable_variables_ids = set()
    non_trainable_variables_ids = set()
    for fn in fns:
        if hasattr(fn, "concrete_functions"):
            concrete_functions = fn.concrete_functions
        elif hasattr(fn, "get_concrete_function"):
            concrete_functions = [fn.get_concrete_function()]
        else:
            concrete_functions = [fn]
        for concrete_fn in concrete_functions:
            for v in concrete_fn.trainable_variables:
                if id(v) not in trainable_variables_ids:
                    trainable_variables.append(v)
                    trainable_variables_ids.add(id(v))

            for v in concrete_fn.variables:
                if (
                    id(v) not in trainable_variables_ids
                    and id(v) not in non_trainable_variables_ids
                ):
                    non_trainable_variables.append(v)
                    non_trainable_variables_ids.add(id(v))
    return trainable_variables, non_trainable_variables

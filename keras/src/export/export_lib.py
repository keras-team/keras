"""Library for exporting inference-only Keras models/layers."""

import inspect
import itertools
import string

from absl import logging

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.layers import Layer
from keras.src.models import Functional
from keras.src.models import Sequential
from keras.src.utils import io_utils
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.export.ExportArchive")
class ExportArchive:
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
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
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
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
    )
    export_archive.add_endpoint(
        name="call_training",
        fn=lambda x: model.call(x, training=True),
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
    )
    export_archive.write_out("path/to/location")
    ```

    **Note on resource tracking:**

    `ExportArchive` is able to automatically track all `tf.Variables` used
    by its endpoints, so most of the time calling `.track(model)`
    is not strictly required. However, if your model uses lookup layers such
    as `IntegerLookup`, `StringLookup`, or `TextVectorization`,
    it will need to be tracked explicitly via `.track(model)`.

    Explicit tracking is also required if you need to be able to access
    the properties `variables`, `trainable_variables`, or
    `non_trainable_variables` on the revived archive.
    """

    def __init__(self):
        self._endpoint_names = []
        self._endpoint_signatures = {}
        self.tensorflow_version = tf.__version__

        self._tf_trackable = tf.__internal__.tracking.AutoTrackable()
        self._tf_trackable.variables = []
        self._tf_trackable.trainable_variables = []
        self._tf_trackable.non_trainable_variables = []

        if backend.backend() == "jax":
            self._backend_variables = []
            self._backend_trainable_variables = []
            self._backend_non_trainable_variables = []

        if backend.backend() not in ("tensorflow", "jax"):
            raise NotImplementedError(
                "The export API is only compatible with JAX and TF backends."
            )

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
        """Track the variables (and other assets) of a layer or model.

        By default, all variables used by an endpoint function
        are automatically tracked when you call `add_endpoint()`.
        However, non-variables assets such as lookup tables
        need to be tracked manually. Note that lookup tables
        used by built-in Keras layers
        (`TextVectorization`, `IntegerLookup`, `StringLookup`)
        are automatically tracked in `add_endpoint()`.

        Arguments:
            resource: A trackable TensorFlow resource.
        """
        if backend.backend() == "tensorflow" and not isinstance(
            resource, tf.__internal__.tracking.Trackable
        ):
            raise ValueError(
                "Invalid resource type. Expected an instance of a "
                "TensorFlow `Trackable` (such as a Keras `Layer` or `Model`). "
                f"Received instead an object of type '{type(resource)}'. "
                f"Object received: {resource}"
            )
        if backend.backend() == "jax" and not isinstance(
            resource, backend.jax.layer.JaxLayer
        ):
            raise ValueError(
                "Invalid resource type. Expected an instance of a "
                "JAX-based Keras `Layer` or `Model`. "
                f"Received instead an object of type '{type(resource)}'. "
                f"Object received: {resource}"
            )
        if isinstance(resource, Layer):
            if not resource.built:
                raise ValueError(
                    "The layer provided has not yet been built. "
                    "It must be built before export."
                )

        # Layers in `_tracked` are not part of the trackables that get saved,
        # because we're creating the attribute in a
        # no_automatic_dependency_tracking scope.
        if not hasattr(self, "_tracked"):
            self._tracked = []
        self._tracked.append(resource)

        if isinstance(resource, Layer):
            # Variables in the lists below are actually part of the trackables
            # that get saved, because the lists are created in __init__.
            if backend.backend() == "jax":

                trainable_variables = tree.flatten(resource.trainable_variables)
                non_trainable_variables = tree.flatten(
                    resource.non_trainable_variables
                )
                self._backend_trainable_variables += trainable_variables
                self._backend_non_trainable_variables += non_trainable_variables
                self._backend_variables = (
                    self._backend_trainable_variables
                    + self._backend_non_trainable_variables
                )

                self._tf_trackable.trainable_variables += [
                    tf.Variable(v) for v in trainable_variables
                ]
                self._tf_trackable.non_trainable_variables += [
                    tf.Variable(v) for v in non_trainable_variables
                ]
                self._tf_trackable.variables = (
                    self._tf_trackable.trainable_variables
                    + self._tf_trackable.non_trainable_variables
                )
            else:
                self._tf_trackable.variables += resource.variables
                self._tf_trackable.trainable_variables += (
                    resource.trainable_variables
                )
                self._tf_trackable.non_trainable_variables += (
                    resource.non_trainable_variables
                )

    def add_endpoint(self, name, fn, input_signature=None, jax2tf_kwargs=None):
        """Register a new serving endpoint.

        Arguments:
            name: Str, name of the endpoint.
            fn: A function. It should only leverage resources
                (e.g. `tf.Variable` objects or `tf.lookup.StaticHashTable`
                objects) that are available on the models/layers
                tracked by the `ExportArchive` (you can call `.track(model)`
                to track a new model).
                The shape and dtype of the inputs to the function must be
                known. For that purpose, you can either 1) make sure that
                `fn` is a `tf.function` that has been called at least once, or
                2) provide an `input_signature` argument that specifies the
                shape and dtype of the inputs (see below).
            input_signature: Used to specify the shape and dtype of the
                inputs to `fn`. List of `tf.TensorSpec` objects (one
                per positional input argument of `fn`). Nested arguments are
                allowed (see below for an example showing a Functional model
                with 2 input arguments).
            jax2tf_kwargs: Optional. A dict for arguments to pass to `jax2tf`.
                Supported only when the backend is JAX. See documentation for
                [`jax2tf.convert`](
                    https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                The values for `native_serialization` and `polymorphic_shapes`,
                if not provided, are automatically computed.

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
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
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
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
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
                    tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
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
                    "x1": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                    "x2": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
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
        """
        if name in self._endpoint_names:
            raise ValueError(f"Endpoint name '{name}' is already taken.")

        if jax2tf_kwargs and backend.backend() != "jax":
            raise ValueError(
                "'jax2tf_kwargs' is only supported with the jax backend. "
                f"Current backend: {backend.backend()}"
            )

        if input_signature:
            if backend.backend() == "tensorflow":
                decorated_fn = tf.function(
                    fn, input_signature=input_signature, autograph=False
                )
            else:  # JAX backend

                # 1. Create a stateless wrapper for `fn`
                # 2. jax2tf the stateless wrapper
                # 3. Create a stateful function that binds the variables with
                #    the jax2tf converted stateless wrapper
                # 4. Make the signature of the stateful function the same as the
                #    original function
                # 5. Wrap in a `tf.function`
                def stateless_fn(variables, *args, **kwargs):
                    state_mapping = zip(self._backend_variables, variables)
                    with StatelessScope(state_mapping=state_mapping) as scope:
                        output = fn(*args, **kwargs)

                    # Gather updated non-trainable variables
                    non_trainable_variables = []
                    for var in self._backend_non_trainable_variables:
                        new_value = scope.get_current_value(var)
                        non_trainable_variables.append(new_value)
                    return output, non_trainable_variables

                jax2tf_stateless_fn = self._convert_jax2tf_function(
                    stateless_fn,
                    input_signature,
                    jax2tf_kwargs=jax2tf_kwargs,
                )

                def stateful_fn(*args, **kwargs):
                    output, non_trainable_variables = jax2tf_stateless_fn(
                        # Change the trackable `ListWrapper` to a plain `list`
                        list(self._tf_trackable.variables),
                        *args,
                        **kwargs,
                    )
                    for var, new_value in zip(
                        self._tf_trackable.non_trainable_variables,
                        non_trainable_variables,
                    ):
                        var.assign(new_value)
                    return output

                # Note: we truncate the number of parameters to what is
                # specified by `input_signature`.
                fn_signature = inspect.signature(fn)
                fn_parameters = list(fn_signature.parameters.values())
                stateful_fn.__signature__ = inspect.Signature(
                    parameters=fn_parameters[0 : len(input_signature)],
                    return_annotation=fn_signature.return_annotation,
                )

                decorated_fn = tf.function(
                    stateful_fn,
                    input_signature=input_signature,
                    autograph=False,
                )
            self._endpoint_signatures[name] = input_signature
        else:
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
                    "        tf.TensorSpec(\n"
                    "            shape=(None, 224, 224, 3),\n"
                    "            dtype=tf.float32,\n"
                    "        )\n"
                    "    ],\n"
                    ")"
                )
        setattr(self._tf_trackable, name, decorated_fn)
        self._endpoint_names.append(name)
        return decorated_fn

    def add_variable_collection(self, name, variables):
        """Register a set of variables to be retrieved after reloading.

        Arguments:
            name: The string name for the collection.
            variables: A tuple/list/set of `tf.Variable` instances.

        Example:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        # Register an endpoint
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
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
            variables = tree.flatten(tree.map_structure(tf.Variable, variables))
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
        if backend.backend() == "tensorflow":
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
        from keras.src.layers import IntegerLookup
        from keras.src.layers import StringLookup
        from keras.src.layers import TextVectorization

        if hasattr(self, "_tracked"):
            for root in self._tracked:
                descendants = tf.train.TrackableView(root).descendants()
                for trackable in descendants:
                    if isinstance(
                        trackable,
                        (IntegerLookup, StringLookup, TextVectorization),
                    ):
                        self._tf_trackable._misc_assets.append(trackable)

    def _convert_jax2tf_function(self, fn, input_signature, jax2tf_kwargs=None):
        from jax.experimental import jax2tf

        if jax2tf_kwargs is None:
            jax2tf_kwargs = {}

        if "native_serialization" not in jax2tf_kwargs:
            jax2tf_kwargs["native_serialization"] = (
                self._check_device_compatible()
            )

        variables_shapes = self._to_polymorphic_shape(
            self._backend_variables, allow_none=False
        )
        if "polymorphic_shapes" in jax2tf_kwargs:
            input_shapes = jax2tf_kwargs["polymorphic_shapes"]
        else:
            input_shapes = self._to_polymorphic_shape(input_signature)
        jax2tf_kwargs["polymorphic_shapes"] = [variables_shapes] + input_shapes

        return jax2tf.convert(fn, **jax2tf_kwargs)

    def _to_polymorphic_shape(self, struct, allow_none=True):
        if allow_none:
            # Generates unique names: a, b, ... z, aa, ab, ... az, ba, ... zz
            # for unknown non-batch dims. Defined here to be scope per endpoint.
            dim_names = itertools.chain(
                string.ascii_lowercase,
                itertools.starmap(
                    lambda a, b: a + b,
                    itertools.product(string.ascii_lowercase, repeat=2),
                ),
            )

        def convert_shape(x):
            poly_shape = []
            for index, dim in enumerate(list(x.shape)):
                if dim is not None:
                    poly_shape.append(str(dim))
                elif not allow_none:
                    raise ValueError(
                        f"Illegal None dimension in {x} with shape {x.shape}"
                    )
                elif index == 0:
                    poly_shape.append("batch")
                else:
                    poly_shape.append(next(dim_names))
            return "(" + ", ".join(poly_shape) + ")"

        return tree.map_structure(convert_shape, struct)

    def _check_device_compatible(self):
        from jax import default_backend as jax_device

        if (
            jax_device() == "gpu"
            and len(tf.config.list_physical_devices("GPU")) == 0
        ):
            logging.warning(
                "JAX backend is using GPU for export, but installed "
                "TF package cannot access GPU, so reloading the model with "
                "the TF runtime in the same environment will not work. "
                "To use JAX-native serialization for high-performance export "
                "and serving, please install `tensorflow-gpu` and ensure "
                "CUDA version compatibility between your JAX and TF "
                "installations."
            )
            return False
        else:
            return True


def export_model(model, filepath, verbose=True):
    export_archive = ExportArchive()
    export_archive.track(model)
    if isinstance(model, (Functional, Sequential)):
        input_signature = tree.map_structure(_make_tensor_spec, model.inputs)
        if isinstance(input_signature, list) and len(input_signature) > 1:
            input_signature = [input_signature]
        export_archive.add_endpoint("serve", model.__call__, input_signature)
    else:
        input_signature = _get_input_signature(model)
        if not input_signature or not model._called:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
            )
        export_archive.add_endpoint("serve", model.__call__, input_signature)
    export_archive.write_out(filepath, verbose=verbose)


def _get_input_signature(model):
    shapes_dict = getattr(model, "_build_shapes_dict", None)
    if not shapes_dict:
        return None

    def make_tensor_spec(structure):
        # We need to turn wrapper structures like TrackingDict or _DictWrapper
        # into plain Python structures because they don't work with jax2tf/JAX.
        if isinstance(structure, dict):
            return {k: make_tensor_spec(v) for k, v in structure.items()}
        elif isinstance(structure, tuple):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return tf.TensorSpec(
                    shape=(None,) + structure[1:], dtype=model.input_dtype
                )
            return tuple(make_tensor_spec(v) for v in structure)
        elif isinstance(structure, list):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return tf.TensorSpec(
                    shape=[None] + structure[1:], dtype=model.input_dtype
                )
            return [make_tensor_spec(v) for v in structure]
        else:
            raise ValueError(
                f"Unsupported type {type(structure)} for {structure}"
            )

    return [make_tensor_spec(value) for value in shapes_dict.values()]


@keras_export("keras.layers.TFSMLayer")
class TFSMLayer(Layer):
    """Reload a Keras model/layer that was saved via SavedModel / ExportArchive.

    Arguments:
        filepath: `str` or `pathlib.Path` object. The path to the SavedModel.
        call_endpoint: Name of the endpoint to use as the `call()` method
            of the reloaded layer. If the SavedModel was created
            via `model.export()`,
            then the default endpoint name is `'serve'`. In other cases
            it may be named `'serving_default'`.

    Example:

    ```python
    model.export("path/to/artifact")
    reloaded_layer = TFSMLayer("path/to/artifact")
    outputs = reloaded_layer(inputs)
    ```

    The reloaded object can be used like a regular Keras layer, and supports
    training/fine-tuning of its trainable weights. Note that the reloaded
    object retains none of the internal structure or custom methods of the
    original object -- it's a brand new layer created around the saved
    function.

    **Limitations:**

    * Only call endpoints with a single `inputs` tensor argument
    (which may optionally be a dict/tuple/list of tensors) are supported.
    For endpoints with multiple separate input tensor arguments, consider
    subclassing `TFSMLayer` and implementing a `call()` method with a
    custom signature.
    * If you need training-time behavior to differ from inference-time behavior
    (i.e. if you need the reloaded object to support a `training=True` argument
    in `__call__()`), make sure that the training-time call function is
    saved as a standalone endpoint in the artifact, and provide its name
    to the `TFSMLayer` via the `call_training_endpoint` argument.
    """

    def __init__(
        self,
        filepath,
        call_endpoint="serve",
        call_training_endpoint=None,
        trainable=True,
        name=None,
        dtype=None,
    ):
        if backend.backend() != "tensorflow":
            raise NotImplementedError(
                "The TFSMLayer is only currently supported with the "
                "TensorFlow backend."
            )

        # Initialize an empty layer, then add_weight() etc. as needed.
        super().__init__(trainable=trainable, name=name, dtype=dtype)

        self._reloaded_obj = tf.saved_model.load(filepath)

        self.filepath = filepath
        self.call_endpoint = call_endpoint
        self.call_training_endpoint = call_training_endpoint

        # Resolve the call function.
        if hasattr(self._reloaded_obj, call_endpoint):
            # Case 1: it's set as an attribute.
            self.call_endpoint_fn = getattr(self._reloaded_obj, call_endpoint)
        elif call_endpoint in self._reloaded_obj.signatures:
            # Case 2: it's listed in the `signatures` field.
            self.call_endpoint_fn = self._reloaded_obj.signatures[call_endpoint]
        else:
            raise ValueError(
                f"The endpoint '{call_endpoint}' "
                "is neither an attribute of the reloaded SavedModel, "
                "nor an entry in the `signatures` field of "
                "the reloaded SavedModel. Select another endpoint via "
                "the `call_endpoint` argument. Available endpoints for "
                "this SavedModel: "
                f"{list(self._reloaded_obj.signatures.keys())}"
            )

        # Resolving the training function.
        if call_training_endpoint:
            if hasattr(self._reloaded_obj, call_training_endpoint):
                self.call_training_endpoint_fn = getattr(
                    self._reloaded_obj, call_training_endpoint
                )
            elif call_training_endpoint in self._reloaded_obj.signatures:
                self.call_training_endpoint_fn = self._reloaded_obj.signatures[
                    call_training_endpoint
                ]
            else:
                raise ValueError(
                    f"The endpoint '{call_training_endpoint}' "
                    "is neither an attribute of the reloaded SavedModel, "
                    "nor an entry in the `signatures` field of "
                    "the reloaded SavedModel. Available endpoints for "
                    "this SavedModel: "
                    f"{list(self._reloaded_obj.signatures.keys())}"
                )

        # Add trainable and non-trainable weights from the call_endpoint_fn.
        all_fns = [self.call_endpoint_fn]
        if call_training_endpoint:
            all_fns.append(self.call_training_endpoint_fn)
        tvs, ntvs = _list_variables_used_by_fns(all_fns)
        for v in tvs:
            self._add_existing_weight(v)
        for v in ntvs:
            self._add_existing_weight(v)
        self.built = True

    def _add_existing_weight(self, weight):
        """Tracks an existing weight."""
        self._track_variable(weight)

    def call(self, inputs, training=False, **kwargs):
        if training:
            if self.call_training_endpoint:
                return self.call_training_endpoint_fn(inputs, **kwargs)
        return self.call_endpoint_fn(inputs, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            # Note: this is not intended to be portable.
            "filepath": self.filepath,
            "call_endpoint": self.call_endpoint,
            "call_training_endpoint": self.call_training_endpoint,
        }
        return {**base_config, **config}


def _make_tensor_spec(x):
    shape = (None,) + x.shape[1:]
    return tf.TensorSpec(shape, dtype=x.dtype, name=x.name)


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

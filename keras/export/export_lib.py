# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Library for exporting inference-only Keras models/layers."""

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

from keras.engine import base_layer
from keras.engine import functional
from keras.engine import sequential
from keras.utils import io_utils


@keras_export("keras.export.ExportArchive")
class ExportArchive(tf.__internal__.tracking.AutoTrackable):
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
    """

    def __init__(self):
        self._endpoint_names = []
        self._endpoint_signatures = {}
        self._trackables = []
        self.tensorflow_version = tf.__version__

    def track(self, layer):
        """Track the variables (and other resources) of a layer or model."""
        if not isinstance(layer, base_layer.Layer):
            raise ValueError(
                "Invalid layer type. Expected an instance of "
                "`keras.layers.Layer` or `keras.Model`. "
                f"Received instead an object of type '{type(layer)}'. "
                f"Object received: {layer}"
            )

        if not layer.built:
            raise ValueError(
                "The layer provided has not yet been built. "
                "It must be built before export."
            )

        self._trackables = list(layer._trackable_children().values())
        self.variables = list(layer.variables)
        self.trainable_variables = list(layer.trainable_variables)
        self.non_trainable_variables = list(layer.non_trainable_variables)

    def add_endpoint(self, name, fn, input_signature=None):
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

        if input_signature:
            decorated_fn = tf.function(fn, input_signature=input_signature)
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
        setattr(self, name, decorated_fn)
        self._endpoint_names.append(name)

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
        if not all(isinstance(v, tf.Variable) for v in variables):
            raise ValueError(
                "Expected all elements in `variables` to be "
                "`tf.Variable` instances. Found instead the following types: "
                f"{list(set(type(v) for v in variables))}"
            )
        setattr(self, name, list(variables))

    def write_out(self, filepath, options=None):
        """Write the corresponding SavedModel to disk.

        Arguments:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the artifact.
            options: `tf.saved_model.SaveOptions` object that specifies
                SavedModel saving options.

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
        if not self._trackables:
            raise ValueError("No assets are being tracked. Call track().")
        signatures = {}
        for name in self._endpoint_names:
            signatures[name] = self._get_concrete_fn(name)
        # Add "serving_default" signature key for TFServing
        if "serving_default" not in self._endpoint_names:
            signatures["serving_default"] = self._get_concrete_fn(
                self._endpoint_names[0]
            )
        tf.saved_model.save(
            self, filepath, options=options, signatures=signatures
        )

        # Print out available endpoints
        endpoints = "\n\n".join(
            _print_signature(getattr(self, name), name)
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
            return getattr(self, endpoint)
        else:
            traces = getattr(self, endpoint)._trackable_children("saved_model")
            return list(traces.values())[0]


def export_model(model, filepath):
    export_archive = ExportArchive()
    export_archive.track(model)
    if isinstance(model, (functional.Functional, sequential.Sequential)):
        input_signature = tf.nest.map_structure(_make_tensor_spec, model.inputs)
        export_archive.add_endpoint("serve", model.__call__, input_signature)
    else:
        save_spec = model._get_save_spec()
        if not save_spec:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
            )
        input_signature = [save_spec]
        export_archive.add_endpoint("serve", model.__call__, input_signature)
    export_archive.write_out(filepath)


class ReloadedLayer(base_layer.Layer):
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
    reloaded_layer = ReloadedLayer("path/to/artifact")
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
    subclassing `ReloadedLayer` and implementing a `call()` method with a
    custom signature.
    * If you need training-time behavior to differ from inference-time behavior
    (i.e. if you need the reloaded object to support a `training=True` argument
    in `__call__()`), make sure that the training-time call function is
    saved as a standalone endpoint in the artifact, and provide its name
    to the `ReloadedLayer` via the `call_training_endpoint` argument.
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
                f"The endpoint '{call_endpoint}' is neither an "
                "attribute of the reloaded SavedModel, nor an entry "
                "in the `signatures` field of the reloaded SavedModel. "
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
                    f"The endpoint '{call_training_endpoint}' is "
                    "neither an attribute of the reloaded SavedModel, "
                    "nor an entry in the `signatures` field of "
                    "the reloaded SavedModel. "
                )

        # Add trainable and non-trainable weights from the call_endpoint_fn.
        all_fns = [self.call_endpoint_fn]
        if call_training_endpoint:
            all_fns.append(self.call_training_endpoint_fn)
        trainable_variables_ids = set()
        non_trainable_variables_ids = set()
        for fn in all_fns:
            # The function may or may not be already a concrete function
            if hasattr(fn, "concrete_functions"):
                concrete_functions = fn.concrete_functions
            else:
                concrete_functions = [fn]
            for concrete_fn in concrete_functions:
                for v in concrete_fn.trainable_variables:
                    if id(v) not in trainable_variables_ids:
                        self._add_existing_weight(v, trainable=True)
                        trainable_variables_ids.add(id(v))

                for v in concrete_fn.variables:
                    if (
                        id(v) not in trainable_variables_ids
                        and id(v) not in non_trainable_variables_ids
                    ):
                        self._add_existing_weight(v, trainable=False)
                        non_trainable_variables_ids.add(id(v))
        self.built = True

    def _add_existing_weight(self, weight, trainable):
        """Calls add_weight() to register but not create an existing weight."""
        self.add_weight(
            name=weight.name,
            shape=weight.shape,
            dtype=weight.dtype,
            trainable=trainable,
            getter=lambda *_, **__: weight,
        )

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
    return tf.TensorSpec(x.shape, dtype=x.dtype)


def _print_signature(fn, name):
    concrete_fn = fn._list_all_concrete_functions()[0]
    pprinted_signature = concrete_fn.pretty_printed_signature(verbose=True)
    lines = pprinted_signature.split("\n")
    lines = [f"* Endpoint '{name}'"] + lines[1:]
    endpoint = "\n".join(lines)
    return endpoint

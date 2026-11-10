"""Tests for the XLA SavedModel export archive."""

import importlib.util
import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.backend import config
from keras.src.export import saved_model
from keras.src.export import xla_saved_model_archive
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.module_utils import xla_saved_model

# The C++ runtime is an optional binary package (wheels for Python >= 3.12
# only); the serving round-trip tests are skipped without it.
MODEL_RUNNER_AVAILABLE = (
    importlib.util.find_spec("xla_saved_model_runner") is not None
)


class CustomModel(models.Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def call(self, input):
        output = input
        for layer in self.layer_list:
            output = layer(output)
        return output


def get_model(type="sequential", input_shape=(10,), layer_list=None):
    layer_list = layer_list or [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    if type == "sequential":
        model = models.Sequential(layer_list)
        model.build((None,) + input_shape)
        return model
    elif type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    elif type == "subclass":
        model = CustomModel(layer_list)
        model(np.zeros((1,) + input_shape, dtype="float32"))
        return model


def _load_manifest(export_dir):
    from xla_saved_model._src import core as xsm

    return xsm.load(export_dir)


def exported_function_names(export_dir):
    """Names of the exported (StableHLO) functions in the manifest."""
    manifest = _load_manifest(export_dir)
    return [
        name
        for name, obj in manifest.objects.items()
        if obj.HasField("function")
    ]


def _find_function(manifest, signature):
    # Without orchestration (single endpoint) the function is named after the
    # signature; with orchestration (multiple endpoints) it is named
    # `{signature}_{model_name}_{method_key}`.
    prefix = f"{signature}_{xla_saved_model_archive.DEFAULT_MODEL_NAME}_"
    names = [
        name
        for name, obj in manifest.objects.items()
        if obj.HasField("function")
        and (name == signature or name.startswith(prefix))
    ]
    if len(names) != 1:
        raise ValueError(
            f"Expected exactly one function for signature '{signature}', "
            f"found: {names}"
        )
    return manifest.objects[names[0]].function


def _type_to_specs(type_proto):
    """Converts a manifest `Type` proto to a structure of `(shape, dtype)`."""
    from xla_saved_model.protos import type_pb2

    kind = type_proto.WhichOneof("type")
    if kind == "leaf":
        tensor_type = type_proto.leaf.tensor_type
        shape = tuple(
            dim.size if dim.HasField("size") else None
            for dim in tensor_type.shape.shape_with_known_rank.dimension_sizes
        )
        return (shape, type_pb2.DType.Name(tensor_type.dtype))
    if kind == "dict":
        return {
            key: _type_to_specs(value)
            for key, value in type_proto.dict.string_to_type.items()
        }
    if kind == "list":
        return [_type_to_specs(e) for e in type_proto.list.elements]
    if kind == "tuple":
        return tuple(_type_to_specs(e) for e in type_proto.tuple.elements)
    if kind == "none":
        return None
    raise ValueError(f"Unsupported type: {kind}")


def exported_signature(export_dir, signature):
    """Returns the `(inputs, outputs)` specs of an exported signature.

    Both are structures matching the serving pytrees whose leaves are
    `(shape, dtype)` tuples; dynamic dimensions are `None`. This is what the
    XLA Serving Engine reads to match request/response tensors by name.
    """
    function = _find_function(_load_manifest(export_dir), signature)
    # The StableHLO function takes `((params, inputs), kwargs)`.
    args = function.signature.input.tuple.elements[0]
    inputs = _type_to_specs(args.tuple.elements[1])
    outputs = _type_to_specs(function.signature.output)
    return inputs, outputs


def load_checkpoint(export_dir):
    """Restores the model weights checkpoint referenced by the manifest."""
    import orbax.checkpoint as ocp

    manifest = _load_manifest(export_dir)
    checkpoint_path = manifest.objects[
        "weights"
    ].value.external.data.file_system_location.string_path
    return ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        os.path.join(export_dir, checkpoint_path)
    )


def model_runner(export_dir):
    """Loads an exported model with the XLA Serving Engine C++ runtime."""
    from xla_saved_model_runner import model_runner as runner_lib

    return runner_lib.ModelRunner(model_path=export_dir)


def assert_signature(test_case, export_dir, signature, inputs, outputs):
    """Checks an exported signature against reference inputs and outputs.

    Always verifies, from the manifest, that the signature exists and that
    its input and output names, dtypes and ranks match `inputs`/`outputs`
    (dicts of name to array). When the XLA Serving Engine runtime
    (`xla_saved_model_runner`) is installed, also runs the signature with
    `inputs` and checks that the results match `outputs` numerically.
    """
    input_specs, output_specs = exported_signature(export_dir, signature)
    for specs, arrays in ((input_specs, inputs), (output_specs, outputs)):
        test_case.assertEqual(set(specs.keys()), set(arrays.keys()))
        for name, array in arrays.items():
            shape, dtype = specs[name]
            array = np.asarray(array)
            test_case.assertEqual(len(shape), array.ndim, msg=name)
            for dim, size in zip(shape, array.shape):
                if dim is not None:
                    test_case.assertEqual(dim, size, msg=name)
            test_case.assertEqual(dtype, _DTYPE_NAMES[array.dtype.name])
    if MODEL_RUNNER_AVAILABLE:
        results = model_runner(export_dir).signature(signature)
        results = results.run_with_dict_return(**inputs)
        test_case.assertEqual(set(results.keys()), set(outputs.keys()))
        for name, expected in outputs.items():
            test_case.assertAllClose(results[name], expected, msg=name)


# NumPy dtype names to manifest `DType` enum names.
_DTYPE_NAMES = {"float32": "f32", "int32": "i32", "bool": "i1"}


@pytest.mark.skipif(
    config.backend() != "jax",
    reason="XLA SavedModel export only supports the JAX backend.",
)
@pytest.mark.skipif(
    not xla_saved_model.available,
    reason="Requires the `xla_saved_model` package.",
)
class XLASavedModelArchiveTest(testing.TestCase):
    def _export_dir(self, name="exported_model"):
        return os.path.join(self.get_temp_dir(), name)

    def test_export_archive_factory(self):
        archive = saved_model.ExportArchive(format="xla_saved_model")
        self.assertIsInstance(
            archive, xla_saved_model_archive.XLASavedModelArchive
        )

    @parameterized.named_parameters(
        ("sequential", "sequential"),
        ("functional", "functional"),
        ("subclass", "subclass"),
    )
    def test_standard_model_export(self, model_type):
        export_dir = self._export_dir()
        model = get_model(model_type)
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        endpoint_fn = archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        # The returned callable runs the stateless function with the
        # current variable values.
        self.assertAllClose(endpoint_fn(ref_input), ref_output)

        archive.write_out(export_dir)

        self.assertTrue(os.path.exists(os.path.join(export_dir, "manifest.pb")))
        self.assertTrue(os.path.isdir(os.path.join(export_dir, "checkpoint")))

        # The checkpoint holds exactly the tracked variables (trainable first,
        # then non-trainable), in the order the endpoint receives them.
        for restored, variable in zip(
            load_checkpoint(export_dir), archive.variables, strict=True
        ):
            self.assertAllClose(restored, variable.value)
        assert_signature(
            self,
            export_dir,
            "serve",
            {"inputs": ref_input},
            {"output_0": ref_output},
        )
        # Symbolic batch dimension: a different batch size also works.
        other_input = np.random.normal(size=(5, 10)).astype("float32")
        self.assertAllClose(endpoint_fn(other_input), model(other_input))
        assert_signature(
            self,
            export_dir,
            "serve",
            {"inputs": other_input},
            {"output_0": model(other_input)},
        )

    def test_track_and_add_endpoint(self):
        export_dir = self._export_dir()
        model = get_model("sequential")
        ref_input = np.random.normal(size=(2, 10)).astype("float32")
        ref_output = model(ref_input)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track_and_add_endpoint(
            "call",
            model,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        self.assertLen(archive.variables, len(model.variables))
        archive.write_out(export_dir, verbose=False)
        assert_signature(
            self,
            export_dir,
            "call",
            {"inputs": ref_input},
            {"output_0": ref_output},
        )

    def test_multi_input_model_export(self):
        export_dir = self._export_dir()
        x1 = layers.Input(shape=(3,), name="x1")
        x2 = layers.Input(shape=(4,), name="x2")
        output = layers.Dense(2)(layers.Concatenate()([x1, x2]))
        model = models.Model(inputs=[x1, x2], outputs=output)

        ref_inputs = [
            np.random.normal(size=(3, 3)).astype("float32"),
            np.random.normal(size=(3, 4)).astype("float32"),
        ]
        ref_output = model(ref_inputs)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        endpoint_fn = archive.add_endpoint(
            "serve",
            model.__call__,
            # One positional argument that is a list of two tensors.
            input_signature=[
                [
                    layers.InputSpec(
                        shape=(None, 3), dtype="float32", name="x1"
                    ),
                    layers.InputSpec(
                        shape=(None, 4), dtype="float32", name="x2"
                    ),
                ]
            ],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        self.assertAllClose(endpoint_fn(ref_inputs), ref_output)
        assert_signature(
            self,
            export_dir,
            "serve",
            {"x1": ref_inputs[0], "x2": ref_inputs[1]},
            {"output_0": ref_output},
        )

    def test_dict_input_model_export(self):
        export_dir = self._export_dir()
        x1 = layers.Input(shape=(3,), name="x1")
        x2 = layers.Input(shape=(4,), name="x2")
        output = layers.Dense(2)(layers.Concatenate()([x1, x2]))
        model = models.Model(inputs={"x1": x1, "x2": x2}, outputs=output)

        ref_inputs = {
            "x1": np.random.normal(size=(3, 3)).astype("float32"),
            "x2": np.random.normal(size=(3, 4)).astype("float32"),
        }
        ref_output = model(ref_inputs)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        endpoint_fn = archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                {
                    "x1": layers.InputSpec(shape=(None, 3), dtype="float32"),
                    "x2": layers.InputSpec(shape=(None, 4), dtype="float32"),
                }
            ],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        self.assertAllClose(endpoint_fn(ref_inputs), ref_output)
        assert_signature(
            self, export_dir, "serve", ref_inputs, {"output_0": ref_output}
        )

    def test_multiple_endpoints(self):
        export_dir = self._export_dir()
        model = get_model("functional")
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        # Compute the training reference first: the stateful `training=True`
        # call updates the BatchNormalization moving statistics, which the
        # inference references below must include.
        ref_training = model(ref_input, training=True)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]
        inference_fn = archive.add_endpoint(
            "call_inference",
            lambda x: model(x, training=False),
            input_signature=input_signature,
            platforms=["cpu"],
        )
        training_fn = archive.add_endpoint(
            "call_training",
            lambda x: model(x, training=True),
            input_signature=input_signature,
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)

        ref_inference = model(ref_input, training=False)
        self.assertAllClose(inference_fn(ref_input), ref_inference)
        # The exported training function is stateless: it does not update the
        # moving statistics, so it keeps returning the same values.
        self.assertAllClose(training_fn(ref_input), ref_training)
        self.assertAllClose(training_fn(ref_input), ref_training)
        assert_signature(
            self,
            export_dir,
            "call_inference",
            {"inputs": ref_input},
            {"output_0": ref_inference},
        )
        assert_signature(
            self,
            export_dir,
            "call_training",
            {"inputs": ref_input},
            {"output_0": ref_training},
        )
        # Only the registered endpoints are exported.
        self.assertEqual(
            sorted(exported_function_names(export_dir)),
            [
                "call_inference_keras_model_call_inference",
                "call_training_keras_model_call_training",
            ],
        )

    def test_warmup_examples_and_options(self):
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
            warmup_examples=[{"inputs": np.zeros((1, 10), dtype="float32")}],
        )
        archive.write_out(
            export_dir, options={"save_shlo_to_file": True}, verbose=False
        )
        self.assertTrue(os.path.exists(os.path.join(export_dir, "manifest.pb")))
        # The warmup requests are written under `assets.extra`.
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    export_dir, "assets.extra", "xla_serving_warmup_requests"
                )
            )
        )

    def test_polymorphic_shapes_override(self):
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            polymorphic_shapes=["b, 10"],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        ref_input = np.random.normal(size=(4, 10)).astype("float32")
        self.assertEqual(
            exported_signature(export_dir, "serve")[0],
            {"inputs": ((None, 10), "f32")},
        )
        assert_signature(
            self,
            export_dir,
            "serve",
            {"inputs": ref_input},
            {"output_0": model(ref_input)},
        )

    def test_errors(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]

        with self.assertRaisesRegex(ValueError, "has not yet been built"):
            archive.track(models.Sequential([layers.Dense(2)]))
        with self.assertRaisesRegex(ValueError, "Invalid resource type"):
            archive.track("not a layer")
        with self.assertRaisesRegex(ValueError, "No endpoints have been set"):
            archive.write_out(self._export_dir())
        with self.assertRaisesRegex(ValueError, "must provide an `input_sig"):
            archive.add_endpoint("serve", model.__call__)
        with self.assertRaisesRegex(ValueError, "Unsupported keyword arg"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=input_signature,
                jax2tf_kwargs={},
            )
        archive.track(model)
        archive.add_endpoint(
            "serve", model.__call__, input_signature=input_signature
        )
        with self.assertRaisesRegex(ValueError, "already taken"):
            archive.add_endpoint(
                "serve", model.__call__, input_signature=input_signature
            )
        with self.assertRaisesRegex(ValueError, "to be a list/tuple/set"):
            archive.add_variable_collection("vars", model.variables[0])
        with self.assertRaisesRegex(ValueError, "`keras.Variable` instances"):
            archive.add_variable_collection("vars", [np.zeros((2,))])
        archive.add_variable_collection("vars", model.variables)
        with self.assertRaisesRegex(ValueError, "already taken"):
            archive.add_variable_collection("vars", model.variables)
        with self.assertRaisesRegex(ValueError, "`options` must be a dict"):
            archive.write_out(self._export_dir(), options="bad")

    def test_variable_collections(self):
        import orbax.checkpoint as ocp

        export_dir = self._export_dir()
        model = get_model("sequential")
        extra = backend.Variable(np.arange(4, dtype="float32"), name="extra")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        archive.add_variable_collection("my_vars", model.trainable_variables)
        archive.add_variable_collection("extra", [extra])
        archive.write_out(export_dir, verbose=False)

        collections = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
            os.path.join(
                export_dir, xla_saved_model_archive.VARIABLE_COLLECTIONS_SUBDIR
            )
        )
        self.assertEqual(set(collections.keys()), {"my_vars", "extra"})
        self.assertLen(collections["my_vars"], len(model.trainable_variables))
        for restored, variable in zip(
            collections["my_vars"], model.trainable_variables
        ):
            self.assertAllClose(restored, variable.value)
        self.assertAllClose(collections["extra"][0], np.arange(4))

        # The collections are not part of the model weights loaded by the
        # server: the model checkpoint only contains the tracked variables.
        self.assertLen(load_checkpoint(export_dir), len(model.variables))
        ref_input = np.random.normal(size=(2, 10)).astype("float32")
        assert_signature(
            self,
            export_dir,
            "serve",
            {"inputs": ref_input},
            {"output_0": model(ref_input)},
        )

    def test_untracked_variables_error(self):
        model = get_model("sequential")
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]

        # A bound method of an untracked model is detected explicitly.
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.add_endpoint("serve", model.__call__, input_signature)
        with self.assertRaisesRegex(ValueError, "have not been tracked"):
            archive.write_out(self._export_dir())

        # Otherwise, exporting without any tracked variables is rejected: the
        # format requires a weights checkpoint.
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.add_endpoint("serve", lambda x: model(x), input_signature)
        with self.assertRaisesRegex(ValueError, "No variables have been"):
            archive.write_out(self._export_dir())

    def test_track_deduplicates_variables(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.track(model)  # Tracking twice must not duplicate variables.
        archive.track(model.layers[0])
        self.assertLen(archive.variables, len(model.variables))
        self.assertLen(
            archive.trainable_variables, len(model.trainable_variables)
        )
        self.assertLen(
            archive.non_trainable_variables,
            len(model.non_trainable_variables),
        )
        self.assertEqual(
            archive.variables,
            archive.trainable_variables + archive.non_trainable_variables,
        )

    def test_single_spec_and_dynamic_non_batch_dims(self):
        # A 3D input with a dynamic sequence length: the non-batch `None` dim
        # gets its own symbol, and the exported function accepts any length.
        export_dir = self._export_dir()
        model = models.Sequential([layers.Dense(4)])
        model.build((None, None, 8))
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        # A single spec (not wrapped in a list) is accepted.
        endpoint_fn = archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=layers.InputSpec(
                shape=(None, None, 8), dtype="float32"
            ),
            jax_export_kwargs={"platforms": ["cpu"]},
        )
        archive.write_out(export_dir, verbose=False)
        self.assertEqual(
            exported_signature(export_dir, "serve")[0],
            {"inputs": ((None, None, 8), "f32")},
        )
        for seq_len in (3, 7):
            ref_input = np.random.normal(size=(2, seq_len, 8)).astype("float32")
            self.assertAllClose(endpoint_fn(ref_input), model(ref_input))
            assert_signature(
                self,
                export_dir,
                "serve",
                {"inputs": ref_input},
                {"output_0": model(ref_input)},
            )

    def test_polymorphic_shapes_single_string_and_concrete_dim(self):
        export_dir = self._export_dir()
        model = models.Sequential([layers.Dense(4)])
        model.build((None, None, 8))
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        # A single polymorphic shape string (not wrapped in a list) is
        # accepted, and can pin a `None` dimension to a concrete size.
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=layers.InputSpec(
                shape=(None, None, 8), dtype="float32"
            ),
            polymorphic_shapes="batch, 5, 8",
            platforms=["cpu"],
        )
        self.assertEqual(
            archive._endpoints["serve"]["input_signature"]["inputs"].shape,
            ("batch", 5, 8),
        )
        archive.write_out(export_dir, verbose=False)
        self.assertEqual(
            exported_signature(export_dir, "serve")[0],
            {"inputs": ((None, 5, 8), "f32")},
        )
        ref_input = np.random.normal(size=(2, 5, 8)).astype("float32")
        assert_signature(
            self,
            export_dir,
            "serve",
            {"inputs": ref_input},
            {"output_0": model(ref_input)},
        )

    def test_polymorphic_shapes_none_and_short_entries(self):
        # `None` means "use the concrete spec shape" (as in
        # `jax.export.symbolic_args_specs`); rank-1 and rank-0 strings have
        # no comma.
        export_dir = self._export_dir()
        inputs = layers.Input(shape=(10,), name="x")
        scale = layers.Input(shape=(), name="s")
        bias = layers.Input(batch_shape=(4,), name="b")
        outputs = layers.Dense(1)(inputs) * scale[:, None] + ops.mean(bias)
        model = models.Model(inputs=[inputs, scale, bias], outputs=outputs)
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                [
                    layers.InputSpec(
                        shape=(None, 10), dtype="float32", name="x"
                    ),
                    layers.InputSpec(shape=(None,), dtype="float32", name="s"),
                    layers.InputSpec(shape=(4,), dtype="float32", name="b"),
                ]
            ],
            polymorphic_shapes=[["batch, 10", "batch", None]],
            platforms=["cpu"],
        )
        serving = archive._endpoints["serve"]["input_signature"]
        self.assertEqual(serving["x"].shape, ("batch", 10))
        self.assertEqual(serving["s"].shape, ("batch",))
        self.assertEqual(serving["b"].shape, (4,))
        archive.write_out(export_dir, verbose=False)
        self.assertEqual(
            exported_signature(export_dir, "serve")[0],
            {
                "x": ((None, 10), "f32"),
                "s": ((None,), "f32"),
                "b": ((4,), "f32"),
            },
        )
        ref_inputs = {
            "x": np.random.normal(size=(3, 10)).astype("float32"),
            "s": np.random.normal(size=(3,)).astype("float32"),
            "b": np.random.normal(size=(4,)).astype("float32"),
        }
        ref_output = model([ref_inputs["x"], ref_inputs["s"], ref_inputs["b"]])
        assert_signature(
            self, export_dir, "serve", ref_inputs, {"output_0": ref_output}
        )

    def test_polymorphic_shapes_none_requires_concrete_shape(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        with self.assertRaisesRegex(ValueError, "Illegal None dimension"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=[
                    layers.InputSpec(shape=(None, 10), dtype="float32")
                ],
                polymorphic_shapes=[None],
            )
        with self.assertRaisesRegex(ValueError, "must be a `jax.export`"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=[
                    layers.InputSpec(shape=(None, 10), dtype="float32")
                ],
                polymorphic_shapes=[123],
            )

    def test_remote_path_error(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track_and_add_endpoint(
            "serve",
            model,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
        )
        with self.assertRaisesRegex(ValueError, "remote path is not"):
            archive.write_out("gs://bucket/model")

    def test_polymorphic_shapes_rank_mismatch_error(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        with self.assertRaisesRegex(ValueError, "does not match the rank"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=[
                    layers.InputSpec(shape=(None, 10), dtype="float32")
                ],
                polymorphic_shapes=["batch"],
            )

    def test_backend_tensor_input_signature(self):
        # Backend tensors / KerasTensors can be used as the input signature;
        # their batch dimension is made dynamic.
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[ops.ones((2, 10), dtype="float32")],
            platforms=["cpu"],
        )
        self.assertEqual(
            archive._endpoints["serve"]["input_signature"]["inputs"].shape,
            ("batch", 10),
        )
        archive.write_out(export_dir, verbose=False)
        ref_input = np.random.normal(size=(4, 10)).astype("float32")
        assert_signature(
            self,
            export_dir,
            "serve",
            {"inputs": ref_input},
            {"output_0": model(ref_input)},
        )

    def test_endpoint_names_are_exported_verbatim(self):
        # Exactly one serving signature per registered endpoint, no extras.
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serving_default",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        self.assertEqual(
            exported_function_names(export_dir), ["serving_default"]
        )

    def test_untracked_owner_error(self):
        # Tracking a different model is not enough: the endpoint's own model
        # must be tracked, otherwise its weights would be baked in as
        # constants.
        model = get_model("sequential")
        other = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(other)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
        )
        with self.assertRaisesRegex(ValueError, "have not been tracked"):
            archive.write_out(self._export_dir())

    @pytest.mark.skipif(not tf.available, reason="Requires TensorFlow.")
    def test_unknown_rank_error(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        with self.assertRaisesRegex(ValueError, "known rank"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=[tf.TensorSpec(shape=None, dtype="float32")],
            )
        # Scalar and partially-known `tf.TensorSpec`s are accepted.
        archive.add_endpoint(
            "serve",
            lambda x, s: model(x) * s,
            input_signature=[
                tf.TensorSpec(shape=(None, 10), dtype="float32"),
                tf.TensorSpec(shape=(), dtype="float32"),
            ],
        )

    @pytest.mark.skipif(
        not MODEL_RUNNER_AVAILABLE,
        reason="Requires the `xla_saved_model_runner` package.",
    )
    def test_load_and_run_with_model_runner(self):
        """Loads the artifact with the XLA Serving Engine runtime and runs it.

        This exercises the actual serving path (manifest, orchestration,
        checkpoint loading, named inputs/outputs, dynamic batch dimension)
        rather than the Python reload used by the other tests.
        """
        export_dir = self._export_dir()
        model = get_model("functional")
        model.export(export_dir, format="xla_saved_model", platforms=["cpu"])

        runner = model_runner(export_dir)
        self.assertEqual(runner.ifrt_platform_name(), "cpu")
        # `model.export()` registers the forward pass as `serving_default`,
        # the signature the XLA Serving Engine uses by default.
        self.assertEqual(runner.signature_names(), ["serving_default"])

        for batch_size in (3, 5):
            ref_input = np.random.normal(size=(batch_size, 10)).astype(
                "float32"
            )
            ref_output = np.asarray(model(ref_input))
            out = runner.signature("serving_default").run_with_dict_return(
                inputs=ref_input
            )
            self.assertEqual(list(out.keys()), ["output_0"])
            self.assertEqual(out["output_0"].dtype, np.float32)
            self.assertEqual(out["output_0"].shape, (batch_size, 1))
            self.assertAllClose(out["output_0"], ref_output)

    @pytest.mark.skipif(
        not MODEL_RUNNER_AVAILABLE,
        reason="Requires the `xla_saved_model_runner` package.",
    )
    def test_load_and_run_multiple_endpoints_with_model_runner(self):
        export_dir = self._export_dir()
        x1 = layers.Input(shape=(3,), name="x1")
        x2 = layers.Input(shape=(4,), name="x2")
        output = layers.Dense(2)(layers.Concatenate()([x1, x2]))
        model = models.Model(inputs={"x1": x1, "x2": x2}, outputs=output)
        ref_inputs = {
            "x1": np.random.normal(size=(3, 3)).astype("float32"),
            "x2": np.random.normal(size=(3, 4)).astype("float32"),
        }
        ref_output = np.asarray(model(ref_inputs))

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        input_signature = [
            {
                "x1": layers.InputSpec(shape=(None, 3), dtype="float32"),
                "x2": layers.InputSpec(shape=(None, 4), dtype="float32"),
            }
        ]
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=input_signature,
            platforms=["cpu"],
        )
        # Dict outputs keep their keys as output names.
        archive.add_endpoint(
            "serve_named",
            lambda inputs: {"logits": model(inputs)},
            input_signature=input_signature,
            platforms=["cpu"],
            warmup_examples=[
                {
                    "x1": np.zeros((1, 3), dtype="float32"),
                    "x2": np.zeros((1, 4), dtype="float32"),
                }
            ],
        )
        archive.write_out(export_dir, verbose=False)

        runner = model_runner(export_dir)
        self.assertEqual(
            sorted(runner.signature_names()), ["serve", "serve_named"]
        )
        out = runner.signature("serve").run_with_dict_return(**ref_inputs)
        self.assertEqual(list(out.keys()), ["output_0"])
        self.assertAllClose(out["output_0"], ref_output)

        out = runner.signature("serve_named").run_with_dict_return(**ref_inputs)
        self.assertEqual(list(out.keys()), ["logits"])
        self.assertAllClose(out["logits"], ref_output)

        # `run()` returns the outputs as a list, in output order.
        (out,) = runner.signature("serve").run(**ref_inputs)
        self.assertAllClose(out, ref_output)

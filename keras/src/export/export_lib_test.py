"""Tests for inference-only model/layer exporting utilities."""

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src import tree
from keras.src import utils
from keras.src.export import export_lib
from keras.src.saving import saving_lib
from keras.src.testing.test_utils import named_product


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
        return models.Sequential(layer_list)
    elif type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    elif type == "subclass":
        return CustomModel(layer_list)


@pytest.mark.skipif(
    backend.backend() not in ("tensorflow", "jax"),
    reason="Export only currently supports the TF and JAX backends.",
)
class ExportArchiveTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_standard_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.serve(ref_input))
        # Test with a different batch size
        revived_model.serve(tf.random.normal((6, 10)))

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_with_rng_export(self, model_type):

        class RandomLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.seed_generator = backend.random.SeedGenerator()

            def call(self, inputs):
                return inputs + random.uniform(
                    ops.shape(inputs), seed=self.seed_generator
                )

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type, layer_list=[RandomLayer()])
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertEqual(ref_output.shape, revived_model.serve(ref_input).shape)
        # Test with a different batch size
        input = tf.random.normal((6, 10))
        output1 = revived_model.serve(input)
        output2 = revived_model.serve(input)
        # Verify RNG seeding works and produces random outputs
        self.assertNotAllClose(output1, output2)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_with_non_trainable_state_export(self, model_type):

        class StateLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.counter = self.add_variable(
                    (), "zeros", "int32", trainable=False
                )

            def call(self, inputs):
                self.counter.assign_add(1)
                return ops.array(inputs), ops.array(self.counter.value)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type, layer_list=[StateLayer()])
        model(tf.random.normal((3, 10)))

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)

        # The non-trainable counter is expected to increment
        input = tf.random.normal((6, 10))
        output1, counter1 = revived_model.serve(input)
        self.assertAllClose(output1, input)
        self.assertAllClose(counter1, 2)
        output2, counter2 = revived_model.serve(input)
        self.assertAllClose(output2, input)
        self.assertAllClose(counter2, 3)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_with_tf_data_layer(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type, layer_list=[layers.Rescaling(scale=2.0)])
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.serve(ref_input))
        # Test with a different batch size
        revived_model.serve(tf.random.normal((6, 10)))

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):

        class TupleModel(models.Model):

            def call(self, inputs):
                x, y = inputs
                return ops.add(x, y)

        class ArrayModel(models.Model):

            def call(self, inputs):
                x = inputs[0]
                y = inputs[1]
                return ops.add(x, y)

        class DictModel(models.Model):

            def call(self, inputs):
                x = inputs["x"]
                y = inputs["y"]
                return ops.add(x, y)

        if struct_type == "tuple":
            model = TupleModel()
            ref_input = (tf.random.normal((3, 10)), tf.random.normal((3, 10)))
        elif struct_type == "array":
            model = ArrayModel()
            ref_input = [tf.random.normal((3, 10)), tf.random.normal((3, 10))]
        elif struct_type == "dict":
            model = DictModel()
            ref_input = {
                "x": tf.random.normal((3, 10)),
                "y": tf.random.normal((3, 10)),
            }

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        ref_output = model(tree.map_structure(ops.convert_to_tensor, ref_input))

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.serve(ref_input))
        # Test with a different batch size
        bigger_input = tree.map_structure(
            lambda x: tf.concat([x, x], axis=0), ref_input
        )
        revived_model.serve(bigger_input)

        # Test with keras.saving_lib
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.keras"
        )
        saving_lib.save_model(model, temp_filepath)
        revived_model = saving_lib.load_model(
            temp_filepath,
            {
                "TupleModel": TupleModel,
                "ArrayModel": ArrayModel,
                "DictModel": DictModel,
            },
        )
        self.assertAllClose(ref_output, revived_model(ref_input))
        export_lib.export_model(revived_model, self.get_temp_dir())

    def test_model_with_multiple_inputs(self):

        class TwoInputsModel(models.Model):
            def call(self, x, y):
                return x + y

            def build(self, y_shape, x_shape):
                self.built = True

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = TwoInputsModel()
        ref_input_x = tf.random.normal((3, 10))
        ref_input_y = tf.random.normal((3, 10))
        ref_output = model(ref_input_x, ref_input_y)

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.serve(ref_input_x, ref_input_y)
        )
        # Test with a different batch size
        revived_model.serve(
            tf.random.normal((6, 10)), tf.random.normal((6, 10))
        )

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_low_level_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model(model_type)
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        # Test variable tracking
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        self.assertLen(export_archive.variables, 8)
        self.assertLen(export_archive.trainable_variables, 6)
        self.assertLen(export_archive.non_trainable_variables, 2)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.call(ref_input))
        # Test with a different batch size
        revived_model.call(tf.random.normal((6, 10)))

    def test_low_level_model_export_with_alias(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        fn = export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(
            temp_filepath,
            tf.saved_model.SaveOptions(function_aliases={"call_alias": fn}),
        )
        revived_model = tf.saved_model.load(
            temp_filepath,
            options=tf.saved_model.LoadOptions(
                experimental_load_function_aliases=True
            ),
        )
        self.assertAllClose(
            ref_output, revived_model.function_aliases["call_alias"](ref_input)
        )
        # Test with a different batch size
        revived_model.function_aliases["call_alias"](tf.random.normal((6, 10)))

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_low_level_model_export_with_dynamic_dims(self, model_type):

        class ReductionLayer(layers.Layer):
            def call(self, inputs):
                return ops.max(inputs, axis=1)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model(
            model_type,
            input_shape=[(None,), (None,)],
            layer_list=[layers.Concatenate(), ReductionLayer()],
        )
        ref_input = [tf.random.normal((3, 8)), tf.random.normal((3, 6))]
        ref_output = model(ref_input)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[
                [
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                ]
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.call(ref_input))
        # Test with a different batch size
        revived_model.call([tf.random.normal((6, 8)), tf.random.normal((6, 6))])
        # Test with a different batch size and different dynamic sizes
        revived_model.call([tf.random.normal((6, 3)), tf.random.normal((6, 5))])

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is only for the JAX backend.",
    )
    def test_low_level_model_export_with_jax2tf_kwargs(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            jax2tf_kwargs={
                "native_serialization": True,
                "native_serialization_platforms": ("cpu", "tpu"),
            },
        )
        with self.assertRaisesRegex(
            ValueError, "native_serialization_platforms.*bogus"
        ):
            export_archive.add_endpoint(
                "call2",
                model.__call__,
                input_signature=[
                    tf.TensorSpec(shape=(None, 10), dtype=tf.float32)
                ],
                jax2tf_kwargs={
                    "native_serialization": True,
                    "native_serialization_platforms": ("cpu", "bogus"),
                },
            )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.call(ref_input))

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is only for the JAX backend.",
    )
    def test_low_level_model_export_with_jax2tf_polymorphic_shapes(self):

        class SquareLayer(layers.Layer):
            def call(self, inputs):
                return ops.matmul(inputs, inputs)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = CustomModel([SquareLayer()])
        ref_input = tf.random.normal((3, 10, 10))
        ref_output = model(ref_input)
        signature = [tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)]

        with self.assertRaises(TypeError):
            # This will fail because the polymorphic_shapes that is
            # automatically generated will not account for the fact that
            # dynamic dimensions 1 and 2 must have the same value.
            export_archive = export_lib.ExportArchive()
            export_archive.track(model)
            export_archive.add_endpoint(
                "call",
                model.__call__,
                input_signature=signature,
                jax2tf_kwargs={},
            )
            export_archive.write_out(temp_filepath)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=signature,
            jax2tf_kwargs={"polymorphic_shapes": ["(batch, a, a)"]},
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.call(ref_input))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="This test is native to the TF backend.",
    )
    def test_endpoint_registration_tf_function(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        # Test variable tracking
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        self.assertLen(export_archive.variables, 8)
        self.assertLen(export_archive.trainable_variables, 6)
        self.assertLen(export_archive.non_trainable_variables, 2)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        # Test registering an endpoint that is a tf.function (called)
        my_endpoint(ref_input)  # Trace fn

        export_archive.add_endpoint(
            "call",
            my_endpoint,
        )
        export_archive.write_out(temp_filepath)

        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(ref_output, revived_model.call(ref_input))
        self.assertLen(revived_model.variables, 8)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 2)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is native to the JAX backend.",
    )
    def test_jax_endpoint_registration_tf_function(self):
        model = get_model()
        ref_input = np.random.normal(size=(3, 10))
        model(ref_input)

        # build a JAX function
        def model_call(x):
            return model(x)

        from jax import default_backend as jax_device
        from jax.experimental import jax2tf

        native_jax_compatible = not (
            jax_device() == "gpu"
            and len(tf.config.list_physical_devices("GPU")) == 0
        )
        # now, convert JAX function
        converted_model_call = jax2tf.convert(
            model_call,
            native_serialization=native_jax_compatible,
            polymorphic_shapes=["(b, 10)"],
        )

        # you can now build a TF inference function
        @tf.function(
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            autograph=False,
        )
        def infer_fn(x):
            return converted_model_call(x)

        ref_output = infer_fn(ref_input)

        # Export with TF inference function as endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model")
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint("serve", infer_fn)
        export_archive.write_out(temp_filepath)

        # Reload and verify outputs
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(ref_output, revived_model.serve(ref_input))
        self.assertLen(revived_model.variables, 8)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 2)

        # Assert all variables wrapped as `tf.Variable`
        assert isinstance(export_archive.variables[0], tf.Variable)
        assert isinstance(export_archive.trainable_variables[0], tf.Variable)
        assert isinstance(
            export_archive.non_trainable_variables[0], tf.Variable
        )

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is native to the JAX backend.",
    )
    def test_jax_multi_unknown_endpoint_registration(self):
        window_size = 100

        X = np.random.random((1024, window_size, 1))
        Y = np.random.random((1024, window_size, 1))

        model = models.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="relu"),
            ]
        )

        model.compile(optimizer="adam", loss="mse")

        model.fit(X, Y, batch_size=32)

        # build a JAX function
        def model_call(x):
            return model(x)

        from jax import default_backend as jax_device
        from jax.experimental import jax2tf

        native_jax_compatible = not (
            jax_device() == "gpu"
            and len(tf.config.list_physical_devices("GPU")) == 0
        )
        # now, convert JAX function
        converted_model_call = jax2tf.convert(
            model_call,
            native_serialization=native_jax_compatible,
            polymorphic_shapes=["(b, t, 1)"],
        )

        # you can now build a TF inference function
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
            ],
            autograph=False,
        )
        def infer_fn(x):
            return converted_model_call(x)

        ref_input = np.random.random((1024, window_size, 1))
        ref_output = infer_fn(ref_input)

        # Export with TF inference function as endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model")
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint("serve", infer_fn)
        export_archive.write_out(temp_filepath)

        # Reload and verify outputs
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(ref_output, revived_model.serve(ref_input))
        self.assertLen(revived_model.variables, 6)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 0)

        # Assert all variables wrapped as `tf.Variable`
        assert isinstance(export_archive.variables[0], tf.Variable)
        assert isinstance(export_archive.trainable_variables[0], tf.Variable)

    def test_layer_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.BatchNormalization()
        ref_input = tf.random.normal((3, 10))
        ref_output = layer(ref_input)  # Build layer (important)

        export_archive = export_lib.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_layer.call(ref_input))

    def test_multi_input_output_functional_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        x1 = layers.Input((2,))
        x2 = layers.Input((2,))
        y1 = layers.Dense(3)(x1)
        y2 = layers.Dense(3)(x2)
        model = models.Model([x1, x2], [y1, y2])

        ref_inputs = [tf.random.normal((3, 2)), tf.random.normal((3, 2))]
        ref_outputs = model(ref_inputs)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                [
                    tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                ]
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_outputs[0], revived_model.serve(ref_inputs)[0])
        self.assertAllClose(ref_outputs[1], revived_model.serve(ref_inputs)[1])
        # Test with a different batch size
        revived_model.serve(
            [tf.random.normal((6, 2)), tf.random.normal((6, 2))]
        )

        # Now test dict inputs
        model = models.Model({"x1": x1, "x2": x2}, [y1, y2])

        ref_inputs = {
            "x1": tf.random.normal((3, 2)),
            "x2": tf.random.normal((3, 2)),
        }
        ref_outputs = model(ref_inputs)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                {
                    "x1": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                    "x2": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                }
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_outputs[0], revived_model.serve(ref_inputs)[0])
        self.assertAllClose(ref_outputs[1], revived_model.serve(ref_inputs)[1])
        # Test with a different batch size
        revived_model.serve(
            {
                "x1": tf.random.normal((6, 2)),
                "x2": tf.random.normal((6, 2)),
            }
        )

    # def test_model_with_lookup_table(self):
    #     tf.debugging.disable_traceback_filtering()
    #     temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
    #     text_vectorization = layers.TextVectorization()
    #     text_vectorization.adapt(["one two", "three four", "five six"])
    #     model = models.Sequential(
    #         [
    #             layers.Input(shape=(), dtype="string"),
    #             text_vectorization,
    #             layers.Embedding(10, 32),
    #             layers.Dense(1),
    #         ]
    #     )
    #     ref_input = tf.convert_to_tensor(["one two three four"])
    #     ref_output = model(ref_input)

    #     export_lib.export_model(model, temp_filepath)
    #     revived_model = tf.saved_model.load(temp_filepath)
    #     self.assertAllClose(ref_output, revived_model.serve(ref_input))

    def test_track_multiple_layers(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        layer_1 = layers.Dense(2)
        ref_input_1 = tf.random.normal((3, 4))
        ref_output_1 = layer_1(ref_input_1)
        layer_2 = layers.Dense(3)
        ref_input_2 = tf.random.normal((3, 5))
        ref_output_2 = layer_2(ref_input_2)

        export_archive = export_lib.ExportArchive()
        export_archive.add_endpoint(
            "call_1",
            layer_1.call,
            input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32)],
        )
        export_archive.add_endpoint(
            "call_2",
            layer_2.call,
            input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output_1, revived_layer.call_1(ref_input_1))
        self.assertAllClose(ref_output_2, revived_layer.call_2(ref_input_2))

    def test_non_standard_layer_signature(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.MultiHeadAttention(2, 2)
        x1 = tf.random.normal((3, 2, 2))
        x2 = tf.random.normal((3, 2, 2))
        ref_output = layer(x1, x2)  # Build layer (important)
        export_archive = export_lib.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_layer.call(x1, x2))

    def test_non_standard_layer_signature_with_kwargs(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.MultiHeadAttention(2, 2)
        x1 = tf.random.normal((3, 2, 2))
        x2 = tf.random.normal((3, 2, 2))
        ref_output = layer(x1, x2)  # Build layer (important)
        export_archive = export_lib.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_layer.call(query=x1, value=x2))
        # Test with a different batch size
        revived_layer.call(
            query=tf.random.normal((6, 2, 2)), value=tf.random.normal((6, 2, 2))
        )

    def test_variable_collection(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(2),
                layers.Dense(2),
            ]
        )

        # Test variable tracking
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.add_variable_collection(
            "my_vars", model.layers[1].weights
        )

        self.assertLen(export_archive._tf_trackable.my_vars, 2)
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertLen(revived_model.my_vars, 2)

    def test_export_model_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Model has not been built
        model = models.Sequential([layers.Dense(2)])
        with self.assertRaisesRegex(ValueError, "It must be built"):
            export_lib.export_model(model, temp_filepath)

        # Subclassed model has not been called
        model = get_model("subclass")
        model.build((2, 10))
        with self.assertRaisesRegex(ValueError, "It must be called"):
            export_lib.export_model(model, temp_filepath)

    def test_export_archive_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Dense(2)])
        model(tf.random.normal((2, 3)))

        # Endpoint name reuse
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        with self.assertRaisesRegex(ValueError, "already taken"):
            export_archive.add_endpoint(
                "call",
                model.__call__,
                input_signature=[
                    tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
                ],
            )

        # Write out with no endpoints
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(ValueError, "No endpoints have been set"):
            export_archive.write_out(temp_filepath)

        # Invalid object type
        with self.assertRaisesRegex(ValueError, "Invalid resource type"):
            export_archive = export_lib.ExportArchive()
            export_archive.track("model")

        # Set endpoint with no input signature
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(
            ValueError, "you must provide an `input_signature`"
        ):
            export_archive.add_endpoint("call", model.__call__)

        # Set endpoint that has never been called
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(
            ValueError, "you must either provide a function"
        ):
            export_archive.add_endpoint("call", my_endpoint)

    def test_export_no_assets(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Case where there are legitimately no assets.
        model = models.Sequential([layers.Flatten()])
        model(tf.random.normal((2, 3)))
        export_archive = export_lib.ExportArchive()
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_export_method(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        model.export(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(ref_output, revived_model.serve(ref_input))
        # Test with a different batch size
        revived_model.serve(tf.random.normal((6, 10)))


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="TFSM Layer reloading is only for the TF backend.",
)
class TestTFSMLayer(testing.TestCase):
    def test_reloading_export_archive(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_lib.export_model(model, temp_filepath)
        reloaded_layer = export_lib.TFSMLayer(temp_filepath)
        self.assertAllClose(reloaded_layer(ref_input), ref_output, atol=1e-7)
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )

        # TODO(nkovela): Expand test coverage/debug fine-tuning and
        # non-trainable use cases here.

    def test_reloading_default_saved_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        tf.saved_model.save(model, temp_filepath)
        reloaded_layer = export_lib.TFSMLayer(
            temp_filepath, call_endpoint="serving_default"
        )
        # The output is a dict, due to the nature of SavedModel saving.
        new_output = reloaded_layer(ref_input)
        self.assertAllClose(
            new_output[list(new_output.keys())[0]],
            ref_output,
            atol=1e-7,
        )
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )

    def test_call_training(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        utils.set_random_seed(1337)
        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(10),
                layers.Dropout(0.99999),
            ]
        )
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="call_inference",
            fn=lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.add_endpoint(
            name="call_training",
            fn=lambda x: model(x, training=True),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        reloaded_layer = export_lib.TFSMLayer(
            temp_filepath,
            call_endpoint="call_inference",
            call_training_endpoint="call_training",
        )
        inference_output = reloaded_layer(
            tf.random.normal((1, 10)), training=False
        )
        training_output = reloaded_layer(
            tf.random.normal((1, 10)), training=True
        )
        self.assertAllClose(np.mean(training_output), 0.0, atol=1e-7)
        self.assertNotAllClose(np.mean(inference_output), 0.0, atol=1e-7)

    def test_serialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_lib.export_model(model, temp_filepath)
        reloaded_layer = export_lib.TFSMLayer(temp_filepath)

        # Test reinstantiation from config
        config = reloaded_layer.get_config()
        rereloaded_layer = export_lib.TFSMLayer.from_config(config)
        self.assertAllClose(rereloaded_layer(ref_input), ref_output, atol=1e-7)

        # Test whole model saving with reloaded layer inside
        model = models.Sequential([reloaded_layer])
        temp_model_filepath = os.path.join(self.get_temp_dir(), "m.keras")
        model.save(temp_model_filepath, save_format="keras_v3")
        reloaded_model = saving_lib.load_model(
            temp_model_filepath,
            custom_objects={"TFSMLayer": export_lib.TFSMLayer},
        )
        self.assertAllClose(reloaded_model(ref_input), ref_output, atol=1e-7)

    def test_errors(self):
        # Test missing call endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Input((2,)), layers.Dense(3)])
        export_lib.export_model(model, temp_filepath)
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            export_lib.TFSMLayer(temp_filepath, call_endpoint="wrong")

        # Test missing call training endpoint
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            export_lib.TFSMLayer(
                temp_filepath,
                call_endpoint="serve",
                call_training_endpoint="wrong",
            )

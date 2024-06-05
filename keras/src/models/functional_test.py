import os
import warnings

import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import saving
from keras.src import testing
from keras.src.layers.core.input_layer import Input
from keras.src.layers.input_spec import InputSpec
from keras.src.models import Functional
from keras.src.models import Model


class FunctionalTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basic_flow_multi_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs, name="basic")
        model.summary()

        self.assertEqual(model.name, "basic")
        self.assertIsInstance(model, Functional)
        self.assertIsInstance(model, Model)

        # Eager call
        in_val = [np.random.random((2, 3)), np.random.random((2, 3))]
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2, name="input_a_2")
        input_b_2 = Input(shape=(3,), batch_size=2, name="input_b_2")
        in_val = [input_a_2, input_b_2]
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    @pytest.mark.requires_trainable_backend
    def test_scalar_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(), batch_size=2, name="input_b")
        outputs = input_a + input_b[:, None]
        model = Functional([input_a, input_b], outputs)
        model.summary()

        in_val = [np.zeros((2, 3)), np.ones((2,))]
        out_val = model(in_val)
        self.assertAllClose(out_val, np.ones((2, 3)))

    @pytest.mark.requires_trainable_backend
    def test_mutable_state(self):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        x = layers.Dense(5)(inputs)
        outputs = layers.Dense(5)(x)
        model = Functional(inputs, outputs)
        # Allow attaching state to a model that isn't directly part of the DAG.
        # Most useful for functional subclasses.
        model.extra_layer = layers.Dense(5)

    @pytest.mark.requires_trainable_backend
    def test_basic_flow_multi_output(self):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        x = layers.Dense(5)(inputs)
        output_a = layers.Dense(4)(x)
        output_b = layers.Dense(5)(x)
        model = Functional(inputs, [output_a, output_b])

        # Eager call
        in_val = np.random.random((2, 3))
        out_val = model(in_val)
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

        # Symbolic call
        out_val = model(Input(shape=(3,), batch_size=2))
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

    @pytest.mark.requires_trainable_backend
    def test_basic_flow_dict_io(self):
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)

        with self.assertRaisesRegex(
            ValueError, "All `inputs` values must be KerasTensors"
        ):
            model = Functional({"a": "input_a", "b": input_b}, outputs)

        with self.assertRaisesRegex(
            ValueError, "All `outputs` values must be KerasTensors"
        ):
            model = Functional({"a": input_a, "b": input_b}, "outputs")

        model = Functional({"a": input_a, "b": input_b}, outputs)

        # Eager call
        in_val = {"a": np.random.random((2, 3)), "b": np.random.random((2, 3))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        input_b_2 = Input(shape=(3,), batch_size=2)
        in_val = {"a": input_a_2, "b": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    def test_basic_flow_as_a_submodel(self):
        # Build submodel
        submodel_inputs = Input([4])
        submodel_outputs = layers.Flatten()(submodel_inputs)
        submodel = Model(submodel_inputs, submodel_outputs)

        inputs = Input((None, 4))
        outputs = layers.TimeDistributed(submodel)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        x = np.random.random((2, 3, 4))
        y = model(x)
        self.assertEqual(y.shape, (2, 3, 4))

    @pytest.mark.requires_trainable_backend
    def test_named_input_dict_io(self):
        input_a = Input(shape=(3,), batch_size=2, name="a")
        x = layers.Dense(5)(input_a)
        outputs = layers.Dense(4)(x)

        model = Functional(input_a, outputs)

        # Eager call
        in_val = {"a": np.random.random((2, 3))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        in_val = {"a": input_a_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    @pytest.mark.requires_trainable_backend
    def test_input_dict_with_extra_field(self):
        input_a = Input(shape=(3,), batch_size=2, name="a")
        x = input_a * 5
        outputs = x + 2

        model = Functional({"a": input_a}, outputs)

        # Eager call
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            in_val = {
                "a": np.random.random((2, 3)),
                "b": np.random.random((2, 1)),
            }
            out_val = model(in_val)
            self.assertEqual(out_val.shape, (2, 3))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Symbolic call
            input_a_2 = Input(shape=(3,), batch_size=2)
            input_b_2 = Input(shape=(1,), batch_size=2)
            in_val = {"a": input_a_2, "b": input_b_2}
            out_val = model(in_val)
            self.assertEqual(out_val.shape, (2, 3))

    @pytest.mark.requires_trainable_backend
    def test_layer_getters(self):
        # Test mixing ops and layers
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5, name="dense_1")(x)
        outputs = layers.Dense(4, name="dense_2")(x)
        model = Functional([input_a, input_b], outputs)

        self.assertEqual(len(model.layers), 4)
        self.assertEqual(len(model._operations), 5)
        self.assertEqual(model.get_layer(index=0).name, "input_a")
        self.assertEqual(model.get_layer(index=1).name, "input_b")
        self.assertEqual(model.get_layer(index=2).name, "dense_1")
        self.assertEqual(model.get_layer(index=3).name, "dense_2")
        self.assertEqual(model.get_layer(name="dense_1").name, "dense_1")

    @pytest.mark.requires_trainable_backend
    def test_training_arg(self):
        class Canary(layers.Layer):
            def call(self, x, training=False):
                assert training
                return x

            def compute_output_spec(self, x, training=False):
                return backend.KerasTensor(x.shape, dtype=x.dtype)

        inputs = Input(shape=(3,), batch_size=2)
        outputs = Canary()(inputs)
        model = Functional(inputs, outputs)
        model(np.random.random((2, 3)), training=True)

    def test_mask_arg(self):
        # TODO
        pass

    @pytest.mark.requires_trainable_backend
    def test_passing_inputs_by_name(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs)

        # Eager call
        in_val = {
            "input_a": np.random.random((2, 3)),
            "input_b": np.random.random((2, 3)),
        }
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2, name="input_a_2")
        input_b_2 = Input(shape=(3,), batch_size=2, name="input_b_2")
        in_val = {"input_a": input_a_2, "input_b": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    @pytest.mark.requires_trainable_backend
    def test_rank_standardization(self):
        # Downranking
        inputs = Input(shape=(3,), batch_size=2)
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs)
        out_val = model(np.random.random((2, 3, 1)))
        self.assertEqual(out_val.shape, (2, 3))

        # Upranking
        inputs = Input(shape=(3, 1), batch_size=2)
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs)
        out_val = model(np.random.random((2, 3)))
        self.assertEqual(out_val.shape, (2, 3, 3))

    @pytest.mark.requires_trainable_backend
    def test_dtype_standardization(self):
        float_input = Input(shape=(2,), dtype="float16")
        int_input = Input(shape=(2,), dtype="int32")
        float_output = float_input + 2
        int_output = int_input + 2
        model = Functional((float_input, int_input), (float_output, int_output))
        float_data, int_data = model((np.ones((2, 2)), np.ones((2, 2))))

        self.assertEqual(backend.standardize_dtype(float_data.dtype), "float16")
        self.assertEqual(backend.standardize_dtype(int_data.dtype), "int32")

    @pytest.mark.requires_trainable_backend
    def test_serialization(self):
        # Test basic model
        inputs = Input(shape=(3,), batch_size=2)
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs, trainable=False)
        self.run_class_serialization_test(model)

        # Test multi-io model
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        xa = layers.Dense(5, name="middle_a")(input_a)
        xb = layers.Dense(5, name="middle_b")(input_b)
        output_a = layers.Dense(4, name="output_a")(xa)
        output_b = layers.Dense(4, name="output_b")(xb)
        model = Functional(
            [input_a, input_b], [output_a, output_b], name="func"
        )
        self.run_class_serialization_test(model)

        # Test model that includes floating ops
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5, name="middle")(x)
        output_a = layers.Dense(4, name="output_a")(x)
        output_b = layers.Dense(4, name="output_b")(x)
        model = Functional(
            [input_a, input_b], [output_a, output_b], name="func"
        )
        self.run_class_serialization_test(model)

        # Test model with dict i/o
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional({"a": input_a, "b": input_b}, outputs)
        self.run_class_serialization_test(model)

    @pytest.mark.requires_trainable_backend
    def test_bad_input_spec(self):
        # Single input
        inputs = Input(shape=(4,))
        outputs = layers.Dense(2)(inputs)
        model = Functional(inputs, outputs)
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "expects 1 input"):
            model([np.zeros((2, 4)), np.zeros((2, 4))])

        # List input
        input_a = Input(shape=(4,), name="a")
        input_b = Input(shape=(4,), name="b")
        x = input_a + input_b
        outputs = layers.Dense(2)(x)
        model = Functional([input_a, input_b], outputs)
        with self.assertRaisesRegex(ValueError, "expects 2 input"):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model([np.zeros((2, 3)), np.zeros((2, 4))])

        # Dict input
        model = Functional({"a": input_a, "b": input_b}, outputs)
        with self.assertRaisesRegex(ValueError, "expects 2 input"):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model({"a": np.zeros((2, 3)), "b": np.zeros((2, 4))})

    @pytest.mark.requires_trainable_backend
    def test_manual_input_spec(self):
        inputs = Input(shape=(None, 3))
        outputs = layers.Dense(2)(inputs)
        model = Functional(inputs, outputs)
        model.input_spec = InputSpec(shape=(None, 4, 3))
        with self.assertRaisesRegex(
            ValueError,
            r"expected shape=\(None, 4, 3\), found shape=\(2, 3, 3\)",
        ):
            model(np.zeros((2, 3, 3)))
        model(np.zeros((2, 4, 3)))

    def test_functional_slicing(self):
        inputs = Input(shape=(None, 2), name="input")
        x1 = layers.Dense(3, name="dense1")(inputs)
        x2 = layers.Dense(4, name="dense2")(x1)
        outputs = layers.Dense(5, name="dense3")(x2)

        full_model = Functional(inputs, outputs, name="full_model")
        self.assertLen(full_model.layers, 4)

        partial_model_1 = Functional(x2, outputs, name="partial1")
        self.assertLen(partial_model_1.layers, 2)  # input_layer, dense3
        self.assertIsInstance(partial_model_1.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_1.layers[1].name, "dense3")

        partial_model_2 = Functional(x1, x2, name="partial2")
        self.assertLen(partial_model_2.layers, 2)  # input_layer, dense2
        self.assertIsInstance(partial_model_2.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_2.layers[1].name, "dense2")

        partial_model_3 = Functional(
            full_model.get_layer("dense2").input, outputs, name="partial3"
        )
        self.assertLen(partial_model_3.layers, 3)  # input_layer, dense2, dense3
        self.assertIsInstance(partial_model_3.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_3.layers[1].name, "dense2")
        self.assertEqual(partial_model_3.layers[2].name, "dense3")

        partial_model_4 = Functional(
            full_model.get_layer("dense1").input,
            full_model.get_layer("dense2").output,
            name="partial4",
        )
        self.assertLen(partial_model_4.layers, 3)  # input_layer, dense1, dense2
        self.assertIsInstance(partial_model_4.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_4.layers[1].name, "dense1")
        self.assertEqual(partial_model_4.layers[2].name, "dense2")

    def test_deeply_nested_model(self):
        i1, i2, i3 = Input((1,)), Input((2,)), Input((3,))
        o1, o2, o3 = (
            layers.Dense(1)(i1),
            layers.Dense(2)(i2),
            layers.Dense(3)(i3),
        )
        model = Model(
            {"1": i1, "others": {"2": i2, "3": i3}},
            {"1": o1, "others": {"2": o2, "3": o3}},
        )
        out_eager = model(
            {
                "1": np.ones((2, 1)),
                "others": {"2": np.ones((2, 2)), "3": np.ones((2, 3))},
            }
        )
        out_symbolic = model(
            {
                "1": Input((1,), batch_size=2),
                "others": {
                    "2": Input((2,), batch_size=2),
                    "3": Input((3,), batch_size=2),
                },
            }
        )
        for out in [out_eager, out_symbolic]:
            self.assertIsInstance(out, dict)
            self.assertEqual(set(out.keys()), {"1", "others"})
            self.assertEqual(out["1"].shape, (2, 1))
            self.assertIsInstance(out["others"], dict)
            self.assertEqual(set(out["others"].keys()), {"2", "3"})
            self.assertEqual(out["others"]["2"].shape, (2, 2))
            self.assertEqual(out["others"]["3"].shape, (2, 3))

        # Test serialization boundaries
        temp_filepath = os.path.join(self.get_temp_dir(), "deeply_nested.keras")
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)
        new_out_eager = loaded_model(
            {
                "1": np.ones((2, 1)),
                "others": {"2": np.ones((2, 2)), "3": np.ones((2, 3))},
            }
        )
        self.assertAllClose(out_eager["1"], new_out_eager["1"])
        self.assertAllClose(
            out_eager["others"]["2"], new_out_eager["others"]["2"]
        )
        self.assertAllClose(
            out_eager["others"]["3"], new_out_eager["others"]["3"]
        )

    def test_optional_inputs(self):
        class OptionalInputLayer(layers.Layer):
            def call(self, x, y=None):
                if y is not None:
                    return x + y
                return x

            def compute_output_shape(self, x_shape):
                return x_shape

        i1 = Input((2,))
        i2 = Input((2,), optional=True)
        outputs = OptionalInputLayer()(i1, i2)
        model = Model([i1, i2], outputs)

        # Eager test
        out = model([np.ones((2, 2)), None])
        self.assertAllClose(out, np.ones((2, 2)))
        # Note: it's not intended to work in symbolic mode (yet).

    def test_add_loss(self):
        # TODO
        pass

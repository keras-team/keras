import warnings

import numpy as np
import pytest

from keras import backend
from keras import layers
from keras import testing
from keras.layers.core.input_layer import Input
from keras.layers.input_spec import InputSpec
from keras.models import Functional
from keras.models import Model


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
            ValueError, "all values in the dict must be KerasTensors"
        ):
            model = Functional({"aa": [input_a], "bb": input_b}, outputs)

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
        with self.assertRaisesRegex(ValueError, "expected 1 input"):
            model([np.zeros((2, 4)), np.zeros((2, 4))])

        # List input
        input_a = Input(shape=(4,), name="a")
        input_b = Input(shape=(4,), name="b")
        x = input_a + input_b
        outputs = layers.Dense(2)(x)
        model = Functional([input_a, input_b], outputs)
        with self.assertRaisesRegex(ValueError, "expected 2 input"):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model([np.zeros((2, 3)), np.zeros((2, 4))])

        # Dict input
        model = Functional({"a": input_a, "b": input_b}, outputs)
        with self.assertRaisesRegex(ValueError, "expected 2 input"):
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

    def test_add_loss(self):
        # TODO
        pass

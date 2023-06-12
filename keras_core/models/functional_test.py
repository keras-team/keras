import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import testing
from keras_core.layers.core.input_layer import Input
from keras_core.models import Functional
from keras_core.models import Model


class FunctionalTest(testing.TestCase):
    def test_basic_flow_multi_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs, name="basic")
        model.summary()

        self.assertEqual(model.name, "basic")
        self.assertTrue(isinstance(model, Functional))
        self.assertTrue(isinstance(model, Model))

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

    def test_scalar_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(), batch_size=2, name="input_b")
        outputs = input_a + input_b[:, None]
        model = Functional([input_a, input_b], outputs)
        model.summary()

        in_val = [np.zeros((2, 3)), np.ones((2,))]
        out_val = model(in_val)
        self.assertAllClose(out_val, np.ones((2, 3)))

    def test_basic_flow_multi_output(self):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        x = layers.Dense(5)(inputs)
        output_a = layers.Dense(4)(x)
        output_b = layers.Dense(5)(x)
        model = Functional(inputs, [output_a, output_b])

        # Eager call
        in_val = np.random.random((2, 3))
        out_val = model(in_val)
        self.assertTrue(isinstance(out_val, list))
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

        # Symbolic call
        out_val = model(Input(shape=(3,), batch_size=2))
        self.assertTrue(isinstance(out_val, list))
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

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

        with self.assertRaisesRegex(
            ValueError, "all keys in the dict must match the names"
        ):
            model = Functional({"aa": input_a, "bb": input_b}, outputs)

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

    def test_dtype_standardization(self):
        float_input = Input(shape=(2,), dtype="float16")
        int_input = Input(shape=(2,), dtype="int32")
        float_output = float_input + 2
        int_output = int_input + 2
        model = Functional((float_input, int_input), (float_output, int_output))
        float_data, int_data = model((np.ones((2, 2)), np.ones((2, 2))))
        self.assertEqual(backend.standardize_dtype(float_data.dtype), "float16")
        self.assertEqual(backend.standardize_dtype(int_data.dtype), "int32")

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

    def test_add_loss(self):
        # TODO
        pass

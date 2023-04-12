import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import operations as ops
from keras_core import testing
from keras_core.layers.core.input_layer import Input
from keras_core.models.functional import Functional


class FunctionalTest(testing.TestCase):
    def test_basic_flow_multi_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs)

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
        pass

    def test_rank_standardization(self):
        pass

    def test_serialization(self):
        # TODO
        pass

    def test_add_loss(self):
        # TODO
        pass

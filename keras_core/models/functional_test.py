import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import operations as ops
from keras_core import testing
from keras_core.layers.core.input_layer import Input
from keras_core.models.functional import Functional


class FunctionalTest(testing.TestCase):
    def test_basic_flow(self):
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

    def test_layer_getters(self):
        # Test mixing ops and layers
        pass

    def test_training_arg(self):
        pass

    def test_mask_arg(self):
        pass

    def test_shape_inference(self):
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

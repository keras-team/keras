from keras_core import layers
from keras_core import testing
from keras_core.layers.core.input_layer import Input
from keras_core.models.functional import Functional
from keras_core.models.model import Model


class ModelTest(testing.TestCase):
    def test_functional_rerouting(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Model([input_a, input_b], outputs)
        self.assertTrue(isinstance(model, Functional))

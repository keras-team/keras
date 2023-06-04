from keras_core import layers
from keras_core import testing
from keras_core.layers.core.input_layer import Input
from keras_core.models.functional import Functional
from keras_core.models.model import Model
from keras_core.models.model import model_from_json


class ModelTest(testing.TestCase):
    def _get_model(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Model([input_a, input_b], outputs)
        return model

    def test_functional_rerouting(self):
        model = self._get_model()
        self.assertTrue(isinstance(model, Functional))

    def test_json_serialization(self):
        model = self._get_model()
        json_string = model.to_json()
        new_model = model_from_json(json_string)
        self.assertEqual(json_string, new_model.to_json())

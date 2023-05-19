import numpy as np

from keras_core import testing
from keras_core.layers.activations import elu


class ELUTest(testing.TestCase):
    def test_config(self):
        elu_layer = elu.ELU()
        self.run_class_serialization_test(elu_layer)

    def test_elu(self):
        self.run_layer_test(
            elu.ELU,
            init_kwargs={},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

        x = np.random.random((2, 5))
        elu_layer = elu.ELU()
        result = elu_layer(x[np.newaxis, :])[0]
        self.assertAllClose(result, x, rtol=1e-05)

import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import testing


class TextVectorizationTest(testing.TestCase):
    def test_config(self):
        layer = layers.Hashing(
            num_bins=8,
            output_mode="int",
        )
        self.run_class_serialization_test(layer)

    def test_correctness(self):
        layer = layers.Hashing(num_bins=3)
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        output = layer(inp)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[1], [0], [1], [1], [2]]))

        layer = layers.Hashing(num_bins=3, mask_value="")
        inp = [["A"], ["B"], [""], ["C"], ["D"]]
        output = layer(inp)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[1], [1], [0], [2], [2]]))

        layer = layers.Hashing(num_bins=3, salt=[133, 137])
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        output = layer(inp)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[1], [2], [1], [0], [2]]))

        layer = layers.Hashing(num_bins=3, salt=133)
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        output = layer(inp)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[0], [0], [2], [1], [0]]))

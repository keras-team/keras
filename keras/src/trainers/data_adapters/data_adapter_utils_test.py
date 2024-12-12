import numpy as np
from absl.testing import parameterized

from keras.src import testing
from keras.src.trainers.data_adapters.data_adapter_utils import (
    class_weight_to_sample_weights,
)


class TestClassWeightToSampleWeights(testing.TestCase):
    @parameterized.named_parameters(
        [
            # Simple case, where y is flat
            (
                "simple_class_labels",
                np.array([0, 1, 0, 2]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 2.0, 1.0, 3.0]),
            ),
            # Testing with one-hot encoded labels,
            # so basically the argmax statement
            (
                "one_hot_encoded_labels",
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 2.0, 1.0, 3.0]),
            ),
            # 3 is not mapped, so it's assigned the default weight (1)
            (
                "unmapped_class",
                np.array([0, 3, 0, 2]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 1.0, 1.0, 3.0]),
            ),
            (
                "multi_dimensional_input",
                np.array([[0], [1], [0], [2]]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 2.0, 1.0, 3.0]),
            ),
            (
                "all_unmapped",
                np.array([0, 1, 0, 2]),
                {},
                np.array([1.0, 1.0, 1.0, 1.0]),
            ),
        ]
    )
    def test_class_weight_to_sample_weights(self, y, class_weight, expected):
        self.assertAllClose(
            class_weight_to_sample_weights(y, class_weight), expected
        )

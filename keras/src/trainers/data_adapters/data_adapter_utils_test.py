import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
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

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_class_weight_to_sample_weights_torch_specific(self):
        import torch

        y = torch.from_numpy(np.array([0, 1, 0, 2]))
        self.assertAllClose(
            class_weight_to_sample_weights(y, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )
        y_one_hot = torch.from_numpy(
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.assertAllClose(
            class_weight_to_sample_weights(y_one_hot, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_class_weight_to_sample_weights_jax_specific(self):
        import jax

        y = jax.numpy.asarray(np.array([0, 1, 0, 2]))
        self.assertAllClose(
            class_weight_to_sample_weights(y, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )
        y_one_hot = jax.numpy.asarray(
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.assertAllClose(
            class_weight_to_sample_weights(y_one_hot, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="tensorflow only"
    )
    def test_class_weight_to_sample_weights_tf_specific(self):
        import tensorflow as tf

        y = tf.convert_to_tensor(np.array([0, 1, 0, 2]))
        self.assertAllClose(
            class_weight_to_sample_weights(y, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )
        y_one_hot = tf.convert_to_tensor(
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.assertAllClose(
            class_weight_to_sample_weights(y_one_hot, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )

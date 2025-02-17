import numpy as np
import pytest

from keras.src import layers
from keras.src.testing import test_case


class ActivityRegularizationTest(test_case.TestCase):
    def test_correctness(self):
        layer = layers.ActivityRegularization(l1=0.2, l2=0.3)
        layer(2 * np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 4 * 0.3 + 2 * 0.2)

    @pytest.mark.requires_trainable_backend
    def test_activity_regularization_basics(self):
        self.run_layer_test(
            layers.ActivityRegularization,
            {"l1": 0.1, "l2": 0.2},
            input_shape=(2, 3),
            input_dtype="float32",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=1,
            supports_masking=True,
            assert_built_after_instantiation=True,
        )

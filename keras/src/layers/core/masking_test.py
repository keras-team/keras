import numpy as np
import pytest

from keras.src import layers
from keras.src import models
from keras.src import testing


class MaskingTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_masking_basics(self):
        self.run_layer_test(
            layers.Masking,
            init_kwargs={"mask_value": 0.0},
            input_shape=(2, 3, 2),
            expected_output_shape=(2, 3, 2),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @pytest.mark.requires_trainable_backend
    def test_masking_correctness(self):
        x = np.array(
            [
                [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]],
                [[2.0, 2.0], [0.0, 0.0], [2.0, 1.0]],
            ]
        )
        expected_mask = [[False, True, False], [True, False, True]]

        layer = layers.Masking(mask_value=0.0)
        self.assertAllClose(layer.compute_mask(x), expected_mask)

        test_obj = self

        class TestLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.supports_masking = True

            def compute_output_shape(self, input_shape):
                return input_shape

            def call(self, inputs, mask=None):
                assert mask is not None
                test_obj.assertAllClose(mask, expected_mask)
                return inputs

        model = models.Sequential(
            [
                layers.Masking(mask_value=0.0),
                TestLayer(),
            ]
        )
        model(x)

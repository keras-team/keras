import numpy as np
import pytest

from keras import layers
from keras import testing
from keras.backend.common.keras_tensor import KerasTensor


class UpSamplingTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_upsampling_1d(self):
        self.run_layer_test(
            layers.UpSampling1D,
            init_kwargs={"size": 2},
            input_shape=(3, 5, 4),
            expected_output_shape=(3, 10, 4),
            expected_output_dtype="float32",
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_upsampling_1d_correctness(self):
        self.assertAllClose(
            layers.UpSampling1D(size=2)(np.arange(12).reshape((2, 2, 3))),
            np.array(
                [
                    [
                        [0.0, 1.0, 2.0],
                        [0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0],
                    ],
                    [
                        [6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0],
                        [9.0, 10.0, 11.0],
                    ],
                ]
            ),
        )

    def test_upsampling_1d_correctness_with_ones(self):
        self.assertAllClose(
            layers.UpSampling1D(size=3)(np.ones((2, 1, 5))), np.ones((2, 3, 5))
        )

    def test_upsampling_1d_with_dynamic_batch_size(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(layers.UpSampling1D(size=2)(x).shape, (None, 4, 3))
        self.assertEqual(layers.UpSampling1D(size=4)(x).shape, (None, 8, 3))

    def test_upsampling_1d_with_dynamic_shape(self):
        y = KerasTensor([2, None, 3])
        self.assertEqual(layers.UpSampling1D(size=2)(y).shape, (2, None, 3))
        self.assertEqual(layers.UpSampling1D(size=4)(y).shape, (2, None, 3))

        z = KerasTensor([2, 3, None])
        self.assertEqual(layers.UpSampling1D(size=2)(z).shape, (2, 6, None))
        self.assertEqual(layers.UpSampling1D(size=4)(z).shape, (2, 12, None))

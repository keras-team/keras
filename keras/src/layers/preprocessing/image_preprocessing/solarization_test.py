import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import random
from keras.src import testing


class SolarizationTest(testing.TestCase):
    def _test_input_output(self, layer, input_value, expected_value, dtype):
        input = np.ones(shape=(2, 224, 224, 3), dtype=dtype) * input_value
        expected_output = ops.clip(
            (
                np.ones(shape=(2, 224, 224, 3), dtype=layer.compute_dtype)
                * expected_value
            ),
            0,
            255,
        )
        output = layer(input)
        self.assertAllClose(output, expected_output)

    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.Solarization,
            init_kwargs={
                "addition_factor": 0.75,
                "value_range": (20, 200),
                "threshold_factor": (0, 1),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    @parameterized.named_parameters(
        ("0_255", 0, 255),
        ("64_191", 64, 191),
        ("127_128", 127, 128),
        ("191_64", 191, 64),
        ("255_0", 255, 0),
    )
    def test_output_values(self, input_value, expected_value):
        solarization = layers.Solarization(value_range=(0, 255))

        self._test_input_output(
            layer=solarization,
            input_value=input_value,
            expected_value=expected_value,
            dtype="uint8",
        )

    @parameterized.named_parameters(
        ("0_0", 0, 0),
        ("191_64", 191, 64),
        ("255_0", 255, 0),
    )
    def test_only_values_above_threshold_are_solarized(
        self, input_value, output_value
    ):
        solarization = layers.Solarization(
            threshold_factor=(128.0 / 255.0, 128.0 / 255.0),
            value_range=(0, 255),
        )

        self._test_input_output(
            layer=solarization,
            input_value=input_value,
            expected_value=output_value,
            dtype="uint8",
        )

    def test_random_augmentation_applied_per_sample(self):
        image = random.uniform((16, 16, 3), minval=0, maxval=255)
        images = ops.stack([image, image])
        layer = layers.Solarization(
            value_range=(0, 255), threshold_factor=0.5, addition_factor=0.5
        )
        outputs = layer(images)
        self.assertNotAllClose(outputs[0], outputs[1])

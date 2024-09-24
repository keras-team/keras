import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import testing


def squared_l2_norm(x):
    x = backend.convert_to_numpy(x)
    return np.sum(x**2)


class UnitNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_un_basics(self):
        self.run_layer_test(
            layers.UnitNormalization,
            init_kwargs={"axis": -1},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            supports_masking=True,
            assert_built_after_instantiation=True,
        )
        self.run_layer_test(
            layers.UnitNormalization,
            init_kwargs={"axis": (1, 2)},
            input_shape=(1, 3, 3),
            expected_output_shape=(1, 3, 3),
            supports_masking=True,
            assert_built_after_instantiation=True,
        )

    def test_invalid_axis(self):
        with self.assertRaisesRegex(
            TypeError,
            (
                "Invalid value for `axis` argument: expected an int or a "
                "list/tuple of ints."
            ),
        ):
            layers.UnitNormalization(axis={"axis": -1})

    def test_correctness(self):
        layer = layers.UnitNormalization(axis=-1)
        inputs = np.random.normal(size=(2, 3))
        outputs = layer(inputs)
        self.assertAllClose(squared_l2_norm(outputs[0, :]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :]), 1.0)

        layer = layers.UnitNormalization(axis=(1, 2))
        inputs = np.random.normal(size=(2, 3, 3))
        outputs = layer(inputs)
        self.assertAllClose(squared_l2_norm(outputs[0, :, :]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :, :]), 1.0)

        layer = layers.UnitNormalization(axis=1)
        inputs = np.random.normal(size=(2, 3, 2))
        outputs = layer(inputs)
        self.assertAllClose(squared_l2_norm(outputs[0, :, 0]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :, 0]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[0, :, 1]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :, 1]), 1.0)

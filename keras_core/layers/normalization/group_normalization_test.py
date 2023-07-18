import numpy as np
import pytest

from keras_core import constraints
from keras_core import layers
from keras_core import regularizers
from keras_core import testing


class GroupNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_groupnorm(self):
        self.run_layer_test(
            layers.GroupNormalization,
            init_kwargs={
                "gamma_regularizer": regularizers.L2(0.01),
                "beta_regularizer": regularizers.L2(0.01),
            },
            input_shape=(3, 4, 32),
            expected_output_shape=(3, 4, 32),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,
            supports_masking=True,
        )

        self.run_layer_test(
            layers.GroupNormalization,
            init_kwargs={
                "groups": 4,
                "gamma_constraint": constraints.UnitNorm(),
                "beta_constraint": constraints.UnitNorm(),
            },
            input_shape=(3, 4, 4),
            expected_output_shape=(3, 4, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_correctness_instance_norm(self):
        instance_norm_layer = layers.GroupNormalization(
            groups=4, axis=-1, scale=False, center=False
        )

        inputs = np.array([[[-1.0, 1.0, 0, 2.0], [1.0, 3.0, -4, -2.0]]])

        expected_instance_norm_output = np.array(
            [[[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]]
        )

        self.assertAllClose(
            instance_norm_layer(inputs),
            expected_instance_norm_output,
            atol=1e-3,
        )

    def test_correctness_1d(self):
        layer_with_1_group = layers.GroupNormalization(
            groups=1, axis=-1, scale=False, center=False
        )
        layer_with_2_groups = layers.GroupNormalization(
            groups=2, axis=1, scale=False, center=False
        )

        inputs = np.array([[-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0, -2.0]])

        expected_output_1_group = np.array(
            [[-0.898, -0.898, 0.539, 0.539, 1.257, 1.257, -0.180, -1.616]],
        )
        self.assertAllClose(
            layer_with_1_group(inputs),
            expected_output_1_group,
            atol=1e-3,
        )

        expected_output_2_groups = np.array(
            [[-1.0, -1.0, 1.0, 1.0, 0.904, 0.904, -0.301, -1.507]]
        )
        self.assertAllClose(
            layer_with_2_groups(inputs),
            expected_output_2_groups,
            atol=1e-3,
        )

    def test_correctness_2d(self):
        layer_with_1_group = layers.GroupNormalization(
            groups=1, axis=-1, scale=False, center=False
        )
        layer_with_2_groups = layers.GroupNormalization(
            groups=2, axis=2, scale=False, center=False
        )

        inputs = np.array([[[-1.0, -1.0, 2.0, 2.0], [1.0, 1.0, 0, -2.0]]])

        expected_output_1_group = np.array(
            [[[-0.898, -0.898, 1.257, 1.257], [0.539, 0.539, -0.180, -1.616]]]
        )

        self.assertAllClose(
            layer_with_1_group(inputs),
            expected_output_1_group,
            atol=1e-3,
        )

        expected_output_2_groups = np.array(
            [[[-1.0, -1.0, 0.904, 0.904], [1.0, 1.0, -0.301, -1.507]]]
        )
        self.assertAllClose(
            layer_with_2_groups(inputs),
            expected_output_2_groups,
            atol=1e-3,
        )

# Copyright 2022 The Keras Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import tensorflow.compat.v2 as tf

import keras
from keras.initializers import Constant
from keras.layers import GroupNormalization
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


def _build_group_normalization_model(norm):
    model = keras.models.Sequential()
    model.add(norm)
    model.compile(
        loss="mse",
        optimizer="rmsprop",
        run_eagerly=test_utils.should_run_eagerly(),
    )

    return model


@test_utils.run_v2_only
class GroupNormalizationTest(test_combinations.TestCase):
    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_trainable_weights(self):
        # Check if weights get initialized correctly
        layer = GroupNormalization(groups=1, scale=False, center=False)
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

        # Check if weights get initialized correctly
        layer = GroupNormalization(groups=1, scale=True, center=True)
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.weights), 2)

    @test_combinations.run_all_keras_modes
    def test_groupnorm(self):
        test_utils.layer_test(
            GroupNormalization,
            kwargs={
                "gamma_regularizer": keras.regularizers.l2(0.01),
                "beta_regularizer": keras.regularizers.l2(0.01),
            },
            input_shape=(3, 4, 32),
        )

        test_utils.layer_test(
            GroupNormalization,
            kwargs={
                "groups": 4,
                "gamma_constraint": keras.constraints.UnitNorm(),
                "beta_constraint": keras.constraints.UnitNorm(),
            },
            input_shape=(3, 4, 4),
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_1d(self):
        layer_with_1_group = GroupNormalization(
            groups=1, axis=-1, input_shape=(8,), scale=False, center=False
        )
        layer_with_2_groups = GroupNormalization(
            groups=2, axis=1, input_shape=(8,), scale=False, center=False
        )

        inputs = tf.constant(
            [-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0, -2.0], shape=(1, 8)
        )

        expected_output_1_group = tf.constant(
            [-0.898, -0.898, 0.539, 0.539, 1.257, 1.257, -0.180, -1.616],
            shape=(1, 8),
        )
        self.assertAllClose(
            _build_group_normalization_model(layer_with_1_group)(inputs),
            expected_output_1_group,
            atol=1e-3,
        )

        expected_output_2_groups = tf.constant(
            [-1.0, -1.0, 1.0, 1.0, 0.904, 0.904, -0.301, -1.507], shape=(1, 8)
        )
        self.assertAllClose(
            _build_group_normalization_model(layer_with_2_groups)(inputs),
            expected_output_2_groups,
            atol=1e-3,
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_2d(self):
        layer_with_1_group = GroupNormalization(
            groups=1, axis=-1, input_shape=(2, 4), scale=False, center=False
        )
        layer_with_2_groups = GroupNormalization(
            groups=2, axis=2, input_shape=(2, 4), scale=False, center=False
        )

        inputs = tf.constant(
            [[-1.0, -1.0, 2.0, 2.0], [1.0, 1.0, 0, -2.0]], shape=(1, 2, 4)
        )

        expected_output_1_group = tf.constant(
            [[-0.898, -0.898, 1.257, 1.257], [0.539, 0.539, -0.180, -1.616]],
            shape=(1, 2, 4),
        )
        self.assertAllClose(
            _build_group_normalization_model(layer_with_1_group)(inputs),
            expected_output_1_group,
            atol=1e-3,
        )

        expected_output_2_groups = tf.constant(
            [[-1.0, -1.0, 0.904, 0.904], [1.0, 1.0, -0.301, -1.507]],
            shape=(1, 2, 4),
        )
        self.assertAllClose(
            _build_group_normalization_model(layer_with_2_groups)(inputs),
            expected_output_2_groups,
            atol=1e-3,
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_instance_norm(self):
        instance_norm_layer = GroupNormalization(
            groups=4, axis=-1, input_shape=(2, 4), scale=False, center=False
        )

        inputs = tf.constant(
            [[-1.0, 1.0, 0, 2.0], [1.0, 3.0, -4, -2.0]], shape=(1, 2, 4)
        )

        expected_instance_norm_output = tf.constant(
            [[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0]], shape=(1, 2, 4)
        )
        self.assertAllClose(
            _build_group_normalization_model(instance_norm_layer)(inputs),
            expected_instance_norm_output,
            atol=1e-3,
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_with_centering(self):
        normalization_layer = GroupNormalization(
            groups=2,
            axis=-1,
            input_shape=(8,),
            scale=False,
            center=True,
            beta_initializer=Constant(10),
        )

        inputs = tf.constant(
            [-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0, -2.0], shape=(1, 8)
        )

        expected_output = tf.constant(
            [9.0, 9.0, 11.0, 11.0, 10.904, 10.904, 9.699, 8.493], shape=(1, 8)
        )
        self.assertAllClose(
            _build_group_normalization_model(normalization_layer)(inputs),
            expected_output,
            atol=1e-3,
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_with_scaling(self):
        normalization_layer = GroupNormalization(
            groups=2,
            axis=-1,
            input_shape=(8,),
            scale=True,
            center=False,
            gamma_initializer=Constant(2),
        )

        inputs = tf.constant(
            [-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0, -2.0], shape=(1, 8)
        )

        expected_output = tf.constant(
            [-2.0, -2.0, 2.0, 2.0, 1.809, 1.808, -0.602, -3.014], shape=(1, 8)
        )
        self.assertAllClose(
            _build_group_normalization_model(normalization_layer)(inputs),
            expected_output,
            atol=1e-3,
        )

    def test_validates_groups_against_channels(self):
        with self.assertRaisesRegex(
            ValueError, r"must be a multiple of the number of channels"
        ):
            norm = GroupNormalization(groups=3, axis=-1)
            norm.build(input_shape=(2, 10))

        with self.assertRaisesRegex(
            ValueError, r"cannot be more than the number of channels"
        ):
            norm = GroupNormalization(groups=32, axis=-1)
            norm.build(input_shape=(2, 8))

    def test_validates_known_number_of_channels(self):
        with self.assertRaisesRegex(
            ValueError, r"tensor should have a defined dimension"
        ):
            norm = GroupNormalization(axis=-1)
            norm.build(input_shape=(1, 32, None))

    def test_rejects_invalid_axis(self):
        with self.assertRaisesRegex(
            ValueError, r"Invalid value for `axis` argument"
        ):
            norm = GroupNormalization(axis=-4)
            norm.build(input_shape=(64, 32, 32))
        with self.assertRaisesRegex(
            ValueError, r"Invalid value for `axis` argument"
        ):
            norm = GroupNormalization(axis=3)
            norm.build(input_shape=(64, 32, 32))


if __name__ == "__main__":
    tf.test.main()

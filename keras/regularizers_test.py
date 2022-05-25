# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for Keras regularizers."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import regularizers
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import np_utils

DATA_DIM = 5
NUM_CLASSES = 2


class KerasRegularizersTest(test_combinations.TestCase, parameterized.TestCase):
    def create_model(
        self,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
    ):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                NUM_CLASSES,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                input_shape=(DATA_DIM,),
            )
        )
        return model

    def regularizer_fn_tensor(x):
        return tf.constant(0.0)

    def regularizer_fn_scalar(x):
        return 0.0

    class RegularizerTensor(regularizers.Regularizer):
        def __call__(self, x):
            return tf.constant(0.0)

    class RegularizerScalar(regularizers.Regularizer):
        def __call__(self, x):
            return 0.0

    def get_data(self):
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=10,
            test_samples=10,
            input_shape=(DATA_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
        y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
        return (x_train, y_train), (x_test, y_test)

    def create_multi_input_model_from(self, layer1, layer2):
        input_1 = keras.layers.Input(shape=(DATA_DIM,))
        input_2 = keras.layers.Input(shape=(DATA_DIM,))
        out1 = layer1(input_1)
        out2 = layer2(input_2)
        out = keras.layers.Average()([out1, out2])
        model = keras.models.Model([input_1, input_2], out)
        model.add_loss(keras.backend.mean(out2))
        model.add_loss(tf.reduce_sum(input_1))
        return model

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1()),
            ("l2", regularizers.l2()),
            ("l1_l2", regularizers.l1_l2()),
            ("l2_zero", keras.regularizers.l2(0.0)),
            ("function_tensor", regularizer_fn_tensor),
            ("function_scalar", regularizer_fn_scalar),
            ("lambda_tensor", lambda x: tf.constant(0.0)),
            ("lambda_scalar", lambda x: 0.0),
            ("regularizer_base_class", regularizers.Regularizer()),
            ("regularizer_custom_class_tensor", RegularizerTensor()),
            ("regularizer_custom_class_scalar", RegularizerScalar()),
        ]
    )
    def test_kernel_regularization(self, regularizer):
        (x_train, y_train), _ = self.get_data()
        model = self.create_model(kernel_regularizer=regularizer)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        self.assertEqual(len(model.losses), 1)
        model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=0)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1()),
            ("l2", regularizers.l2()),
            ("l1_l2", regularizers.l1_l2()),
            ("l2_zero", keras.regularizers.l2(0.0)),
            ("function_tensor", regularizer_fn_tensor),
            ("function_scalar", regularizer_fn_scalar),
            ("lambda_tensor", lambda x: tf.constant(0.0)),
            ("lambda_scalar", lambda x: 0.0),
            ("regularizer_base_class", regularizers.Regularizer()),
            ("regularizer_custom_class_tensor", RegularizerTensor()),
            ("regularizer_custom_class_scalar", RegularizerScalar()),
        ]
    )
    def test_bias_regularization(self, regularizer):
        (x_train, y_train), _ = self.get_data()
        model = self.create_model(bias_regularizer=regularizer)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        self.assertEqual(len(model.losses), 1)
        model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=0)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1()),
            ("l2", regularizers.l2()),
            ("l1_l2", regularizers.l1_l2()),
            ("l2_zero", keras.regularizers.l2(0.0)),
            ("function_tensor", regularizer_fn_tensor),
            ("function_scalar", regularizer_fn_scalar),
            ("lambda_tensor", lambda x: tf.constant(0.0)),
            ("lambda_scalar", lambda x: 0.0),
            ("regularizer_base_class", regularizers.Regularizer()),
            ("regularizer_custom_class_tensor", RegularizerTensor()),
            ("regularizer_custom_class_scalar", RegularizerScalar()),
        ]
    )
    def test_activity_regularization(self, regularizer):
        (x_train, y_train), _ = self.get_data()
        model = self.create_model(activity_regularizer=regularizer)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        self.assertEqual(len(model.losses), 1 if tf.executing_eagerly() else 1)
        model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=0)

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_zero_regularization(self):
        # Verifies that training with zero regularization works.
        x, y = np.ones((10, 10)), np.ones((10, 3))
        model = test_utils.get_model_from_layers(
            [
                keras.layers.Dense(
                    3, kernel_regularizer=keras.regularizers.l2(0)
                )
            ],
            input_shape=(10,),
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        model.fit(x, y, batch_size=5, epochs=1)

    def test_custom_regularizer_saving(self):
        def my_regularizer(weights):
            return tf.reduce_sum(tf.abs(weights))

        inputs = keras.Input((10,))
        outputs = keras.layers.Dense(1, kernel_regularizer=my_regularizer)(
            inputs
        )
        model = keras.Model(inputs, outputs)
        model2 = model.from_config(
            model.get_config(),
            custom_objects={"my_regularizer": my_regularizer},
        )
        self.assertEqual(model2.layers[1].kernel_regularizer, my_regularizer)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1()),
            ("l2", regularizers.l2()),
            ("l1_l2", regularizers.l1_l2()),
        ]
    )
    def test_regularization_shared_layer(self, regularizer):
        dense_layer = keras.layers.Dense(
            NUM_CLASSES,
            kernel_regularizer=regularizer,
            activity_regularizer=regularizer,
        )
        model = self.create_multi_input_model_from(dense_layer, dense_layer)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        self.assertLen(model.losses, 5)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1()),
            ("l2", regularizers.l2()),
            ("l1_l2", regularizers.l1_l2()),
        ]
    )
    def test_regularization_shared_model(self, regularizer):
        dense_layer = keras.layers.Dense(
            NUM_CLASSES,
            kernel_regularizer=regularizer,
            activity_regularizer=regularizer,
        )

        input_tensor = keras.layers.Input(shape=(DATA_DIM,))
        dummy_model = keras.models.Model(
            input_tensor, dense_layer(input_tensor)
        )

        model = self.create_multi_input_model_from(dummy_model, dummy_model)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        self.assertLen(model.losses, 6)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1()),
            ("l2", regularizers.l2()),
            ("l1_l2", regularizers.l1_l2()),
        ]
    )
    def test_regularization_shared_layer_in_different_models(self, regularizer):
        shared_dense = keras.layers.Dense(
            NUM_CLASSES,
            kernel_regularizer=regularizer,
            activity_regularizer=regularizer,
        )
        models = []
        for _ in range(2):
            input_tensor = keras.layers.Input(shape=(DATA_DIM,))
            unshared_dense = keras.layers.Dense(
                NUM_CLASSES, kernel_regularizer=regularizer
            )
            out = unshared_dense(shared_dense(input_tensor))
            models.append(keras.models.Model(input_tensor, out))

        model = self.create_multi_input_model_from(
            layer1=models[0], layer2=models[1]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )

        # We expect to see 9 losses on the model:
        # - 2 from the 2 add_loss calls on the outer model.
        # - 3 from the weight regularizers on the shared_dense layer,
        # unshared_dense in inner model 1, unshared_dense in inner model 2.
        # - 4 from activity regularizers on the shared_dense layer.
        self.assertLen(model.losses, 9)

    def test_deserialization_error(self):
        with self.assertRaisesRegex(
            ValueError, "Could not interpret regularizer"
        ):
            keras.regularizers.get(0)

    @parameterized.named_parameters(
        [
            ("l1", regularizers.l1(l1=None), 0.01),
            ("l2", regularizers.l2(l2=None), 0.01),
            ("l1_l2", regularizers.l1_l2(l1=None, l2=None), 0.0),
        ]
    )
    def test_default_value_when_init_with_none(
        self, regularizer, expected_value
    ):
        expected_value = np.asarray(expected_value)
        if hasattr(regularizer, "l1"):
            self.assertAllClose(regularizer.l1, expected_value)
        if hasattr(regularizer, "l2"):
            self.assertAllClose(regularizer.l2, expected_value)

    @test_utils.run_v2_only
    def test_orthogonal_regularizer(self):
        # Test correctness.
        factor = 0.1
        reg_rows = regularizers.OrthogonalRegularizer(
            factor=factor, mode="rows"
        )
        reg_cols = regularizers.OrthogonalRegularizer(
            factor=factor, mode="columns"
        )

        # Test with square matrix
        inputs = tf.constant(
            [[1, 1, 1, 1], [2, 0, 0, 0], [0, 0, 3, 1]], dtype="float32"
        )
        normalized_rows = tf.math.l2_normalize(inputs, axis=1)
        normalized_cols = tf.math.l2_normalize(inputs, axis=0)
        rows_pairs = [
            tf.reduce_sum(normalized_rows[0] * normalized_rows[1]),
            tf.reduce_sum(normalized_rows[0] * normalized_rows[2]),
            tf.reduce_sum(normalized_rows[1] * normalized_rows[2]),
        ]
        col_pairs = [
            tf.reduce_sum(normalized_cols[:, 0] * normalized_cols[:, 1]),
            tf.reduce_sum(normalized_cols[:, 0] * normalized_cols[:, 2]),
            tf.reduce_sum(normalized_cols[:, 0] * normalized_cols[:, 3]),
            tf.reduce_sum(normalized_cols[:, 1] * normalized_cols[:, 2]),
            tf.reduce_sum(normalized_cols[:, 1] * normalized_cols[:, 3]),
            tf.reduce_sum(normalized_cols[:, 2] * normalized_cols[:, 3]),
        ]
        num_row_pairs = 3
        num_col_pairs = 6
        # Expected: factor * sum(pairwise_dot_products_of_rows) / num_row_pairs
        self.assertAllClose(
            reg_rows(inputs), factor * sum(rows_pairs) / num_row_pairs
        )
        # Expected: factor * sum(pairwise_dot_products_of_columns) /
        # num_col_pairs
        self.assertAllClose(
            reg_cols(inputs), factor * sum(col_pairs) / num_col_pairs
        )

        # Test incorrect usage.
        with self.assertRaisesRegex(ValueError, "must have rank 2"):
            reg_rows(tf.constant([1, 1], dtype="float32"))

        # Test serialization
        self.assertDictEqual(
            reg_cols.get_config(), {"factor": factor, "mode": "columns"}
        )

        # Test usage in model.
        model_inputs = keras.Input((3,))
        model_outputs = keras.layers.Dense(4, kernel_regularizer=reg_rows)(
            model_inputs
        )
        model = keras.Model(model_inputs, model_outputs)
        model.compile(optimizer="rmsprop", loss="mse")
        model.fit(
            np.random.random((16, 3)), np.random.random((16, 4)), epochs=1
        )

        # Test serialization and deserialiation as part of model.
        inputs = tf.constant([[1, 1, 1], [2, 0, 0], [0, 0, 3]], dtype="float32")
        outputs = model(inputs)
        config = model.get_config()
        weights = model.get_weights()
        model = keras.Model.from_config(config)
        model.set_weights(weights)
        self.assertAllClose(model(inputs), outputs, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()

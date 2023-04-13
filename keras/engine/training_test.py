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
"""Tests for training routines."""


import collections
import io
import sys
import tempfile

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import backend
from keras import layers as layers_module
from keras import losses
from keras import metrics as metrics_module
from keras.callbacks import Callback
from keras.engine import input_layer
from keras.engine import sequential
from keras.engine import training as training_module
from keras.engine import training_utils_v1
from keras.layers.preprocessing import string_lookup
from keras.mixed_precision import policy
from keras.optimizers import legacy as optimizer_legacy
from keras.optimizers import rmsprop
from keras.optimizers import sgd as sgd_experimental
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import data_utils
from keras.utils import io_utils
from keras.utils import np_utils

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.rmsprop import (
    RMSPropOptimizer,
)

try:
    import scipy.sparse as scipy_sparse
except ImportError:
    scipy_sparse = None


class TrainingTest(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_model_instrumentation(self):
        layers = [
            layers_module.Dense(10, dtype=np.float64),
            layers_module.Dense(10, dtype=np.float64),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(1,))

        self.assertTrue(model._instrumented_keras_api)
        self.assertTrue(model._instrumented_keras_model_class)
        self.assertFalse(model._instrumented_keras_layer_class)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_fit_training_arg(self):
        class ReturnTraining(layers_module.Layer):
            def call(self, inputs, training):
                if training:
                    return inputs + tf.constant([100], "float32")
                else:
                    return inputs + tf.constant([0], "float32")

        model = sequential.Sequential([ReturnTraining()])
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        hist = model.fit(x=np.array([0.0]), y=np.array([0.0]))
        self.assertAllClose(hist.history["loss"][0], 10000)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_fit_on_empty(self):
        model = sequential.Sequential([layers_module.Dense(1)])
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        with self.assertRaisesRegex(
            ValueError, "Unexpected result of `train_function`.*"
        ):
            model.fit(x=np.array([]), y=np.array([]))

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_compile_fit_with_jit_compile(self):
        # Test with jit_compile = True
        model = sequential.Sequential([layers_module.Dense(1)])
        model.compile("sgd", loss="mse", run_eagerly=False, jit_compile=True)
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model.fit(x, y, epochs=2)
        # Test fcompile fit for a RNN model
        model = sequential.Sequential()
        model.add(
            layers_module.TimeDistributed(
                layers_module.Embedding(5, 6, mask_zero=True),
                input_shape=(None, None),
            )
        )  # N by t_1 by t_2 by 6
        model.add(
            layers_module.TimeDistributed(
                layers_module.SimpleRNN(7, return_sequences=True)
            )
        )
        model.add(
            layers_module.TimeDistributed(
                layers_module.SimpleRNN(8, return_sequences=False)
            )
        )
        model.add(layers_module.SimpleRNN(1, return_sequences=False))
        model.compile(optimizer="sgd", loss="mse", jit_compile=True)
        model_input = np.random.randint(
            low=1, high=5, size=(10, 3, 4), dtype="int32"
        )
        for i in range(4):
            model_input[i, i:, i:] = 0
        model.fit(
            model_input, np.random.random((10, 1)), epochs=1, batch_size=10
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_compile_fit_evaluate_predict_with_mirrored_strategy(self):
        # Test with jit_compile = True
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = sequential.Sequential([layers_module.Dense(1)])
        model.compile("sgd", loss="mse", run_eagerly=False, jit_compile=True)
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model.fit(x, y, epochs=2)
        model.evaluate(x, y)
        model.predict(x)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_distribution_reduction_method_sum_default_train_step(self):

        strategy = tf.distribute.MirroredStrategy(
            ["/cpu:1", "/cpu:2", "/cpu:3", "/cpu:4"]
        )
        BATCH_SIZE = 10

        # A model that always outputs `1`:
        with strategy.scope():
            inputs = layers_module.Input(shape=(1,), name="my_input")
            outputs = layers_module.Dense(
                units=1, kernel_initializer="zeros", bias_initializer="ones"
            )(inputs)
            model = training_module.Model(inputs, outputs)

        model.trainable = False
        model.compile(optimizer="sgd", loss="mean_absolute_error")

        # Data points are always equal to `2`:
        x, y = 2 * np.ones((40, 1)), 2 * np.ones((40, 1))

        # For every output x_i = 1, every target y_i = 2,
        #   loss_i     = |1-2| = 1; and
        #   loss_total = sum([1, 1, ..., 1]) / BATCH_SIZE = 1.0
        history = model.fit(x, y, epochs=1, batch_size=BATCH_SIZE)
        self.assertAllClose(history.history["loss"][-1], 1.0)

        eval_output = model.evaluate(x, y, batch_size=BATCH_SIZE)
        self.assertAllClose(eval_output, 1.0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_distribution_reduction_method_sum_custom_train_step(self):

        strategy = tf.distribute.MirroredStrategy(
            ["/cpu:1", "/cpu:2", "/cpu:3", "/cpu:4"]
        )
        BATCH_SIZE = 10

        class MyModel(training_module.Model):
            @staticmethod
            def reduce_loss(loss_value, global_batch_size):
                REDUCTION_AXES = range(1, backend.ndim(loss_value))
                loss_value = tf.reduce_mean(loss_value, axis=REDUCTION_AXES)
                return tf.nn.compute_average_loss(
                    loss_value, global_batch_size=global_batch_size
                )

            def train_step(self, data):
                loss_value = tf.ones_like(data[0])
                return {
                    "loss": MyModel.reduce_loss(
                        loss_value, global_batch_size=BATCH_SIZE
                    )
                }

            def test_step(self, data):
                loss_value = tf.ones_like(data[0])
                return {
                    "metric": MyModel.reduce_loss(
                        loss_value, global_batch_size=BATCH_SIZE
                    )
                }

        with strategy.scope():
            inputs = layers_module.Input(shape=(1,), name="my_input")
            outputs = layers_module.Dense(1)(inputs)
            model = MyModel(inputs, outputs)

        model.compile()

        x, y = np.ones((40, 1)), np.ones((40, 1))
        history = model.fit(x, y, epochs=2, batch_size=BATCH_SIZE)
        self.assertAllClose(history.history["loss"][-1], 1.0)

        eval_output = model.evaluate(x, y, batch_size=BATCH_SIZE)
        self.assertAllClose(eval_output, 1.0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_verify_xla_compile_with_jit_compile(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = np.array([[1, 2, 3, 4], [4, 3, 1, 0]])
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_data = keras.Input(shape=(None,), dtype=tf.string)
            layer = string_lookup.StringLookup(vocabulary=vocab_data)
            int_data = layer(input_data)
            model = keras.Model(inputs=input_data, outputs=int_data)
            model.compile(
                "sgd", loss="mse", run_eagerly=False, jit_compile=True
            )
            # Added a string op unsupported by XLA compiler to make sure that an
            # error is thrown, This ensures that the graph is indeed being
            # compiled using XLA
            with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "Graph execution error"
            ):
                model.fit(input_array, expected_output, epochs=1)
                model.predict(input_array)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_jit_compile_for_compile_evaluate_predict(self):
        # Test with jit_compile = True for model.compile(), model.evaluate(),
        # model.predict()
        model = sequential.Sequential([layers_module.Dense(1)])
        self.assertIsNone(model._jit_compile)
        model.compile("sgd", loss="mse", run_eagerly=False, jit_compile=True)
        self.assertTrue(model._jit_compile)
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model.fit(x, y, epochs=2)
        model.evaluate(x, y)
        model.predict(x)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_jit_compile_true_for_evaluate_predict_but_false_for_compile(self):
        # Test with jit_compile = True for model.compile(), model.evaluate(),
        # model.predict()
        model = sequential.Sequential([layers_module.Dense(1)])
        self.assertIsNone(model._jit_compile)
        self.assertIsNone(model.jit_compile)
        model.compile("sgd", loss="mse")
        model.jit_compile = True
        self.assertTrue(model._jit_compile)
        self.assertTrue(model.jit_compile)
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model.fit(x, y, epochs=2)
        model.evaluate(x, y)
        model.predict(x)
        self.assertTrue(model._jit_compile)
        self.assertTrue(model.jit_compile)
        model.compile("sgd", loss="mse", jit_compile=False)
        self.assertFalse(model._jit_compile)
        self.assertFalse(model.jit_compile)
        model.compile("sgd", loss="mse", jit_compile=True)
        self.assertTrue(model._jit_compile)
        self.assertTrue(model.jit_compile)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_predict_xla_compile_with_jit_compile_setter_false_then_true(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_data = keras.Input(shape=(None,), dtype=tf.string)
            # Added a string op unsupported by XLA compiler to make sure that an
            # error is thrown, This ensures that the graph is indeed being
            # compiled using XLA
            layer = string_lookup.StringLookup(vocabulary=vocab_data)
            int_data = layer(input_data)
            model = keras.Model(inputs=input_data, outputs=int_data)
            # Compiled without jit_compile
            model.predict(input_array)
            model.jit_compile = True
            with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "Graph execution error"
            ):
                model.predict(input_array)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_fit_without_loss_at_compile(self):
        model = sequential.Sequential([layers_module.Dense(1)])
        model.compile("sgd", run_eagerly=test_utils.should_run_eagerly())
        x, y = np.ones((10, 1)), np.ones((10, 1))
        with self.assertRaisesRegex(ValueError, "No loss found..*"):
            model.fit(x, y, epochs=2)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_fit_without_loss_at_compile_but_with_add_loss(self):
        class MyModel(sequential.Sequential):
            def call(self, x):
                self.add_loss(tf.reduce_sum(x))
                return x

        model = MyModel([layers_module.Dense(1)])
        model.compile("sgd", run_eagerly=test_utils.should_run_eagerly())
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model.fit(x, y, epochs=2)

    @test_combinations.run_all_keras_modes
    def test_run_eagerly_setting(self):
        model = sequential.Sequential([layers_module.Dense(1)])
        run_eagerly = test_utils.should_run_eagerly()
        model.compile("sgd", "mse", run_eagerly=run_eagerly)
        self.assertEqual(model.run_eagerly, run_eagerly)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        ("train_on_batch", "train_on_batch"),
        ("test_on_batch", "test_on_batch"),
        ("predict_on_batch", "predict_on_batch"),
        ("fit", "fit"),
        ("evaluate", "evaluate"),
        ("predict", "predict"),
    )
    def test_disallow_methods_inside_tf_function(self, method_name):
        model = sequential.Sequential([layers_module.Dense(1)])
        run_eagerly = test_utils.should_run_eagerly()
        model.compile("sgd", "mse", run_eagerly=run_eagerly)

        @tf.function
        def my_fn():
            getattr(model, method_name)(1)

        error_msg = "inside a `tf.function`"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            my_fn()

    @test_combinations.run_all_keras_modes
    def test_fit_and_validate_learning_phase(self):
        class ReturnTraining(layers_module.Layer):
            def call(self, inputs):
                return backend.in_train_phase(
                    lambda: tf.ones_like(inputs), lambda: tf.zeros_like(inputs)
                )

        model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
        model.compile(
            "sgd", loss="mae", run_eagerly=test_utils.should_run_eagerly()
        )

        inputs = np.ones((40, 2), dtype=np.float32)
        targets = np.ones((40, 1), dtype=np.float32)

        # Test correctness with `steps_per_epoch`.
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(10)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(10)
        history = model.fit(
            train_dataset, epochs=2, verbose=1, validation_data=val_dataset
        )

        # The training loss should be 0.0
        self.assertAllClose(history.history["loss"][0], 0.0)
        # The validation loss should be 1.0.
        self.assertAllClose(history.history["val_loss"][0], 1.0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_warn_on_evaluate(self):
        i = layers_module.Input((1,))
        x = np.ones((100, 1))
        y = np.ones((100, 1))
        sample_weight = np.ones((100,))
        model = training_module.Model(i, i)
        model.compile(loss="mse", metrics=["mse"])

        logging.set_verbosity(2)
        with self.assertLogs(level=2) as logs:
            model.evaluate(x, y, sample_weight=sample_weight)
        self.assertTrue(
            any(
                "`evaluate()` received a value for `sample_weight`" in log
                for log in logs.output
            )
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_sample_weight_warning_disable(self):
        i = layers_module.Input((1,))
        x = np.ones((100, 1))
        y = np.ones((100, 1))
        sample_weight = np.ones((100,))
        model = training_module.Model(i, i)
        model.compile(loss="mse", metrics=["mse"], weighted_metrics=[])

        logging.set_verbosity(2)
        with self.assertLogs(level=2) as logs:
            model.evaluate(x, y, sample_weight=sample_weight)
        self.assertFalse(
            any(
                "`evaluate()` received a value for `sample_weight`" in log
                for log in logs.output
            )
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_warn_on_evaluate_with_tf_dataset(self):
        i = layers_module.Input((1,))

        x = tf.ones((100, 1), tf.float32)
        y = tf.ones((100, 1), tf.float32)
        sample_weight = tf.ones((100,), dtype=tf.float32)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (x, y, sample_weight)
        ).batch(10)
        model = training_module.Model(i, i)
        model.compile(loss="mse", metrics=["mse"])

        logging.set_verbosity(2)
        with self.assertLogs(level=2) as logs:
            model.evaluate(val_dataset)
        self.assertTrue(
            any(
                "`evaluate()` received a value for `sample_weight`" in log
                for log in logs.output
            )
        )

    @test_combinations.run_all_keras_modes
    def test_fit_and_validate_training_arg(self):
        class ReturnTraining(layers_module.Layer):
            def call(self, inputs, training=None):
                return backend.in_train_phase(
                    lambda: tf.ones_like(inputs),
                    lambda: tf.zeros_like(inputs),
                    training=training,
                )

        model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
        model.compile(
            "sgd", loss="mae", run_eagerly=test_utils.should_run_eagerly()
        )

        inputs = np.ones((40, 2), dtype=np.float32)
        targets = np.ones((40, 1), dtype=np.float32)

        # Test correctness with `steps_per_epoch`.
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(10)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(10)
        history = model.fit(
            train_dataset, epochs=2, verbose=1, validation_data=val_dataset
        )

        # The training loss should be 0.0
        self.assertAllClose(history.history["loss"][0], 0.0)
        # The validation loss should be 1.0.
        self.assertAllClose(history.history["val_loss"][0], 1.0)

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_target_dtype_matches_output(self):
        def loss_fn(labels, preds):
            self.assertEqual(labels.dtype, preds.dtype)
            return labels - preds

        layers = [
            layers_module.Dense(10, dtype=np.float64),
            layers_module.Dense(10, dtype=np.float64),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(1,))
        inputs = np.ones(shape=(10, 1), dtype=np.float64)
        targets = np.ones(shape=(10, 1), dtype=np.float64)
        model.compile(
            "sgd", loss=loss_fn, run_eagerly=test_utils.should_run_eagerly()
        )
        model.train_on_batch(inputs, targets)
        model.test_on_batch(inputs, targets)
        self.assertEqual(model.predict(inputs).dtype, np.float64)

    @test_combinations.run_all_keras_modes
    def test_fit_and_validate_nested_training_arg(self):
        class NestedReturnTraining(layers_module.Layer):
            def call(self, inputs, training=None):
                return backend.in_train_phase(
                    lambda: tf.ones_like(inputs),
                    lambda: tf.zeros_like(inputs),
                    training=training,
                )

        class ReturnTraining(layers_module.Layer):
            def __init__(self, input_shape=None, **kwargs):
                super().__init__(input_shape=input_shape, **kwargs)
                self._nested_layer = None

            def build(self, input_shape):
                self._nested_layer = NestedReturnTraining()
                self.built = True

            def call(self, inputs):
                return self._nested_layer(inputs)

        model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
        model.compile(
            "sgd", loss="mae", run_eagerly=test_utils.should_run_eagerly()
        )

        inputs = np.ones((40, 2), dtype=np.float32)
        targets = np.ones((40, 1), dtype=np.float32)

        # Test correctness with `steps_per_epoch`.
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(10)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(10)
        history = model.fit(
            train_dataset, epochs=2, verbose=1, validation_data=val_dataset
        )

        # The training loss should be 0.0
        self.assertAllClose(history.history["loss"][0], 0.0)
        # The validation loss should be 1.0.
        self.assertAllClose(history.history["val_loss"][0], 1.0)

    @test_combinations.run_with_all_model_types(exclude_models="sequential")
    @test_combinations.run_all_keras_modes
    def test_fit_on_arrays(self):
        input_a = layers_module.Input(shape=(3,), name="input_a")
        input_b = layers_module.Input(shape=(3,), name="input_b")

        dense = layers_module.Dense(4, name="dense")
        dropout = layers_module.Dropout(0.5, name="dropout")
        branch_a = [input_a, dense]
        branch_b = [input_b, dense, dropout]

        model = test_utils.get_multi_io_model(branch_a, branch_b)

        optimizer = RMSPropOptimizer(learning_rate=0.001)
        loss = "mse"
        loss_weights = [1.0, 0.5]
        model.compile(
            optimizer,
            loss,
            metrics=[metrics_module.CategoricalAccuracy(), "mae"],
            loss_weights=loss_weights,
            run_eagerly=test_utils.should_run_eagerly(),
        )

        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))

        output_d_np = np.random.random((10, 4))
        output_e_np = np.random.random((10, 4))

        # Test fit at different verbosity
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=1,
            batch_size=5,
            verbose=0,
        )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=1,
            batch_size=5,
            verbose=1,
        )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=2,
            batch_size=5,
            verbose=2,
        )
        model.train_on_batch(
            [input_a_np, input_b_np], [output_d_np, output_e_np]
        )

        # Test with validation data
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            validation_data=(
                [input_a_np, input_b_np],
                [output_d_np, output_e_np],
            ),
            epochs=1,
            batch_size=5,
            verbose=0,
        )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            validation_data=(
                [input_a_np, input_b_np],
                [output_d_np, output_e_np],
            ),
            epochs=2,
            batch_size=5,
            verbose=1,
        )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            validation_data=(
                [input_a_np, input_b_np],
                [output_d_np, output_e_np],
            ),
            epochs=2,
            batch_size=5,
            verbose=2,
        )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            validation_data=[
                [input_a_np, input_b_np],
                [output_d_np, output_e_np],
            ],
            epochs=2,
            batch_size=5,
            verbose=2,
        )
        # Test with validation split
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=2,
            batch_size=5,
            verbose=0,
            validation_split=0.2,
        )

        if test_utils.get_model_type() == "functional":
            # Test with dictionary inputs
            model.fit(
                {"input_a": input_a_np, "input_b": input_b_np},
                {"dense": output_d_np, "dropout": output_e_np},
                epochs=1,
                batch_size=5,
                verbose=0,
            )
            model.fit(
                {"input_a": input_a_np, "input_b": input_b_np},
                {"dense": output_d_np, "dropout": output_e_np},
                epochs=1,
                batch_size=5,
                verbose=1,
            )
            model.fit(
                {"input_a": input_a_np, "input_b": input_b_np},
                {"dense": output_d_np, "dropout": output_e_np},
                validation_data=(
                    {"input_a": input_a_np, "input_b": input_b_np},
                    {"dense": output_d_np, "dropout": output_e_np},
                ),
                epochs=1,
                batch_size=5,
                verbose=0,
            )
            model.train_on_batch(
                {"input_a": input_a_np, "input_b": input_b_np},
                {"dense": output_d_np, "dropout": output_e_np},
            )

        # Test with lists for loss, metrics
        loss = ["mae", "mse"]
        model.compile(
            optimizer,
            loss,
            metrics=[metrics_module.CategoricalAccuracy(), "mae"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=1,
            batch_size=5,
            verbose=0,
        )

        # Test with dictionaries for loss, metrics, loss weights
        if test_utils.get_model_type() == "functional":
            loss = {"dense": "mse", "dropout": "mae"}
            loss_weights = {"dense": 1.0, "dropout": 0.5}
            metrics = {
                "dense": "mse",
                "dropout": metrics_module.CategoricalAccuracy(),
            }
            model.compile(
                optimizer,
                loss,
                metrics=metrics,
                loss_weights=loss_weights,
                run_eagerly=test_utils.should_run_eagerly(),
            )
        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=1,
            batch_size=5,
            verbose=0,
        )

        # Build single-input model
        x = layers_module.Input(shape=(3,), name="input_a")
        y = layers_module.Dense(4)(x)
        model = training_module.Model(x, y)
        model.compile(
            optimizer, loss="mse", run_eagerly=test_utils.should_run_eagerly()
        )
        # This will work
        model.fit([input_a_np], output_d_np, epochs=1)

        # Test model on a list of floats
        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 4))

        # Test execution on inputs that are lists of scalars.
        # TF2 and TF1 have slightly different semantics:
        if tf.executing_eagerly():
            # In TF2 to avoid any ambiguity when there are nested lists
            # the entire input gets converted to a
            # single numpy array (& it only works in the case of a single io
            # model)
            model.fit(
                np.ndarray.tolist(input_a_np),
                np.ndarray.tolist(input_b_np),
                epochs=2,
                batch_size=5,
                verbose=2,
            )
        else:
            # In TF1 there was logic to try disambiguating between the
            # individual inputs when lists are nested. This allowed multi-io
            # functional models to support lists of scalars as input, but it
            # caused ambiguity issues for subclass models & made it trickier to
            # pass multi-dimensional inputs as lists of scalars to single io
            # models. This was an excessive amount of complexity for what boiled
            # down to a convenience method we were mainly just using for writing
            # tests.
            model.fit(
                [np.ndarray.tolist(input_a_np)],
                [np.ndarray.tolist(input_b_np)],
                epochs=2,
                batch_size=5,
                verbose=2,
            )

    @test_combinations.run_all_keras_modes
    def test_evaluate_predict_on_arrays(self):
        a = layers_module.Input(shape=(3,), name="input_a")
        b = layers_module.Input(shape=(3,), name="input_b")

        dense = layers_module.Dense(4, name="dense")
        c = dense(a)
        d = dense(b)
        e = layers_module.Dropout(0.5, name="dropout")(c)

        model = training_module.Model([a, b], [d, e])

        optimizer = RMSPropOptimizer(learning_rate=0.001)
        loss = "mse"
        loss_weights = [1.0, 0.5]
        model.compile(
            optimizer,
            loss,
            metrics=["mae", metrics_module.CategoricalAccuracy()],
            loss_weights=loss_weights,
            sample_weight_mode=None,
            run_eagerly=test_utils.should_run_eagerly(),
        )

        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))

        output_d_np = np.random.random((10, 4))
        output_e_np = np.random.random((10, 4))

        # Test evaluate at different verbosity
        out = model.evaluate(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            batch_size=5,
            verbose=0,
        )
        self.assertEqual(len(out), 7)
        out = model.evaluate(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            batch_size=5,
            verbose=1,
        )
        self.assertEqual(len(out), 7)
        out = model.evaluate(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            batch_size=5,
            verbose=2,
        )
        self.assertEqual(len(out), 7)
        out = model.test_on_batch(
            [input_a_np, input_b_np], [output_d_np, output_e_np]
        )
        self.assertEqual(len(out), 7)

        # Test evaluate with dictionary inputs
        model.evaluate(
            {"input_a": input_a_np, "input_b": input_b_np},
            {"dense": output_d_np, "dropout": output_e_np},
            batch_size=5,
            verbose=0,
        )
        model.evaluate(
            {"input_a": input_a_np, "input_b": input_b_np},
            {"dense": output_d_np, "dropout": output_e_np},
            batch_size=5,
            verbose=1,
        )

        # Test predict
        out = model.predict([input_a_np, input_b_np], batch_size=5)
        self.assertEqual(len(out), 2)
        out = model.predict({"input_a": input_a_np, "input_b": input_b_np})
        self.assertEqual(len(out), 2)
        out = model.predict_on_batch(
            {"input_a": input_a_np, "input_b": input_b_np}
        )
        self.assertEqual(len(out), 2)

    def _make_sequence_input_functions(self, input_type):
        # train and test
        xy_namedtuple = collections.namedtuple("xy_namedtuple", ["x", "y"])

        # predict
        x_namedtuple = collections.namedtuple("x_namedtuple", ["x"])

        if input_type == "dataset":
            dataset = tf.data.Dataset.range(16).map(
                lambda _: tf.ones(shape=(1,))
            )

            xy_dataset = tf.data.Dataset.zip((dataset, dataset)).batch(4)
            x_dataset = dataset.batch(4)

            def xy_function(use_namedtuple):
                return (
                    xy_dataset.map(xy_namedtuple)
                    if use_namedtuple
                    else xy_dataset
                )

            def x_function(use_namedtuple):
                return (
                    x_dataset.map(x_namedtuple) if use_namedtuple else x_dataset
                )

            return xy_function, x_function

        elif input_type == "generator":

            def xy_generator(use_namedtuple):
                x, y = np.ones((4, 1)), np.ones((4, 1))
                for _ in range(4):
                    if use_namedtuple:
                        yield xy_namedtuple(x, y)
                    else:
                        yield x, y

            def x_generator(use_namedtuple):
                x = np.ones((4, 1))
                for _ in range(4):
                    if use_namedtuple:
                        yield x_namedtuple(x)
                    else:
                        yield x

            return xy_generator, x_generator

        elif input_type == "sequence":

            class XYSequence(data_utils.Sequence):
                def __init__(self, use_namedtuple):
                    self._use_namedtuple = use_namedtuple
                    super().__init__()

                def __getitem__(self, idx):
                    x, y = np.ones((4, 1)), np.ones((4, 1))
                    if self._use_namedtuple:
                        return xy_namedtuple(x, y)
                    return x, y

                def __len__(self):
                    return 4

            class XSequence(data_utils.Sequence):
                def __init__(self, use_namedtuple):
                    self._use_namedtuple = use_namedtuple
                    super().__init__()

                def __getitem__(self, idx):
                    x = np.ones((4, 1))
                    if self._use_namedtuple:
                        return x_namedtuple(x)
                    return x

                def __len__(self):
                    return 4

            return XYSequence, XSequence

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @test_combinations.run_with_all_model_types
    @parameterized.named_parameters(
        ("dataset", "dataset"),
        ("generator", "generator"),
        ("sequence", "sequence"),
    )
    def test_sequence_input_types(self, input_type):
        """Ensure that namedtuples and tuples are plumbed identically."""
        if not tf.executing_eagerly():
            self.skipTest("Improved checking is only present in data_adapter.")

        xy_function, x_function = self._make_sequence_input_functions(
            input_type
        )
        fit_kwargs, evaluate_kwargs, predict_kwargs = {}, {}, {}
        if input_type == "generator":
            fit_kwargs["steps_per_epoch"] = 4
            evaluate_kwargs["steps"] = 4
            predict_kwargs["steps"] = 4

        model = test_utils.get_small_mlp(1, 1, 1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )

        model.fit(xy_function(use_namedtuple=False), **fit_kwargs)
        model.evaluate(xy_function(use_namedtuple=False), **evaluate_kwargs)
        model.predict(x_function(use_namedtuple=False), **predict_kwargs)

    @test_combinations.run_all_keras_modes
    def test_custom_mapping_in_config(self):
        class MyModel(training_module.Model):
            def call(self, inputs):
                return inputs

            def get_config(self):
                self.a = {}
                return {"a": self.a}

        model = MyModel()
        self.assertIn('{"a": {}}', model.to_json())

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_get_config_default(self):
        class MyModel(training_module.Model):
            def __init__(self, units):
                super().__init__()
                self.units = units

            def call(self, inputs):
                return inputs

        # Test default config with named args
        model = MyModel(units=10)
        config = model.get_config()
        self.assertLen(config, 1)
        self.assertEqual(config["units"], 10)
        model = model.from_config(config)
        self.assertDictEqual(model.get_config(), config)

        # Test default config with positinal args
        model = MyModel(10)
        config = model.get_config()
        self.assertLen(config, 1)
        self.assertEqual(config["units"], 10)
        model = model.from_config(config)
        self.assertDictEqual(model.get_config(), config)

        # Test non-serializable
        model = MyModel(units=np.int32(10))
        config = model.get_config()
        self.assertNotIn("units", config)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_get_config_kwargs(self):
        class MyModel(training_module.Model):
            def __init__(self, units, **kwargs):
                super().__init__()
                self.units = units

            def call(self, inputs):
                return inputs

        model = MyModel(10, extra=1)
        config = model.get_config()
        # config = {'name': 'my_model', 'trainable': True, 'dtype': 'float32',
        # 'extra': 1, 'units': 10}
        self.assertLen(config, 5)
        self.assertEqual(config["units"], 10)
        self.assertEqual(config["extra"], 1)
        model = model.from_config(config)
        self.assertDictEqual(model.get_config(), config)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_get_config_override(self):
        class MyModel(training_module.Model):
            def __init__(self, units):
                super().__init__()
                self.units = units

            def call(self, inputs):
                return inputs

            def get_config(self):
                config = {"units": int(self.units)}
                config.update(super().get_config())
                return config

        model = MyModel(units=np.int32(10))
        config = model.get_config()
        self.assertLen(config, 1)
        self.assertEqual(config["units"], 10)
        model = model.from_config(config)
        self.assertDictEqual(model.get_config(), config)

    def test_training_on_sparse_data_with_dense_placeholders_v1(self):
        with tf.Graph().as_default():
            if scipy_sparse is None:
                return

            test_inputs = [
                scipy_sparse.random(6, 3, density=0.25).tocsr()
                for _ in range(2)
            ]
            test_outputs = [
                scipy_sparse.random(6, i, density=0.25).tocsr()
                for i in range(3, 5)
            ]
            in1 = layers_module.Input(shape=(3,))
            in2 = layers_module.Input(shape=(3,))
            out1 = layers_module.Dropout(0.5, name="dropout")(in1)
            out2 = layers_module.Dense(4, name="dense_1")(in2)
            model = training_module.Model([in1, in2], [out1, out2])
            model.predict(test_inputs, batch_size=2)
            optimizer = "rmsprop"
            model.compile(
                optimizer,
                "mse",
                metrics=["mae", metrics_module.CategoricalAccuracy()],
            )
            model.fit(
                test_inputs,
                test_outputs,
                epochs=1,
                batch_size=2,
                validation_split=0.5,
            )
            model.evaluate(test_inputs, test_outputs, batch_size=2)

    @test_combinations.run_all_keras_modes
    def test_compile_with_sparse_placeholders(self):
        inputs = layers_module.Input(shape=(10,), sparse=True)
        weights = tf.Variable(
            np.ones((10, 1)).astype(np.float32), name="weights"
        )
        weights_mult = lambda x: tf.sparse.sparse_dense_matmul(x, weights)
        output_layer = layers_module.Lambda(weights_mult)(inputs)
        model = training_module.Model([inputs], output_layer)
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

    @test_combinations.run_all_keras_modes
    def test_that_trainable_disables_updates(self):
        val_a = np.random.random((10, 4))
        val_out = np.random.random((10, 4))

        a = layers_module.Input(shape=(4,))
        layer = layers_module.BatchNormalization(input_shape=(4,))
        b = layer(a)
        model = training_module.Model(a, b)

        model.trainable = False
        if not tf.compat.v1.executing_eagerly_outside_functions():
            self.assertEmpty(model.updates)

        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        if not tf.compat.v1.executing_eagerly_outside_functions():
            self.assertEmpty(model.updates)

        x1 = model.predict(val_a)
        model.train_on_batch(val_a, val_out)
        x2 = model.predict(val_a)
        self.assertAllClose(x1, x2, atol=1e-7)

        model.trainable = True
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        if not tf.compat.v1.executing_eagerly_outside_functions():
            self.assertAllGreater(len(model.updates), 0)

        model.train_on_batch(val_a, val_out)
        x2 = model.predict(val_a)
        assert np.abs(np.sum(x1 - x2)) > 1e-5

        layer.trainable = False
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        if not tf.compat.v1.executing_eagerly_outside_functions():
            self.assertEmpty(model.updates)

        x1 = model.predict(val_a)
        model.train_on_batch(val_a, val_out)
        x2 = model.predict(val_a)
        self.assertAllClose(x1, x2, atol=1e-7)

    def test_weight_deduplication_in_methods(self):
        inp = layers_module.Input(shape=(1,))
        bn = layers_module.BatchNormalization()
        d = layers_module.Dense(1)

        m0 = training_module.Model(inp, d(bn(inp)))
        m1 = training_module.Model(inp, d(bn(inp)))

        x0 = m0(inp)
        x1 = m1(inp)
        x = layers_module.Add()([x0, x1])

        model = training_module.Model(inp, x)
        self.assertLen(model.trainable_weights, 4)
        self.assertLen(model.non_trainable_weights, 2)
        self.assertLen(model.weights, 6)

    @test_combinations.run_all_keras_modes
    def test_weight_deduplication(self):
        class WatchingLayer(layers_module.Layer):
            def __init__(self, dense_to_track):
                # This will cause the kernel and bias to be double counted,
                # effectively doubling the learning rate if weights are not
                # deduped.
                self._kernel = dense_to_track.kernel
                self._bias = dense_to_track.bias
                super().__init__()

        inp = layers_module.Input(shape=(1,))
        dense_layer = layers_module.Dense(1)
        dense_output = dense_layer(inp)  # This will build the dense kernel

        # Deterministically set weights to make the test repeatable.
        dense_layer.set_weights([np.ones((1, 1)), np.zeros((1,))])
        output = WatchingLayer(dense_layer)(dense_output)

        model = training_module.Model(inp, output)

        # 0.25 is the edge of the radius of convergence for the double apply
        # case. At lr=0.24, the double apply case will very slowly descend
        # while the correct case will drop very quickly.
        model.compile(
            loss="mse",
            optimizer=optimizer_legacy.gradient_descent.SGD(0.24),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.ones((64 * 2,))
        y = 4.5 * x - 3.0

        history = model.fit(x, y, batch_size=64, epochs=2, verbose=2)

        # If the gradient apply is duplicated then the loss after 2 epochs will
        # be ~0.15, compared to the correct answer of O(1e-7).
        self.assertLess(history.history["loss"][-1], 1e-6)

    @test_combinations.run_all_keras_modes
    def test_weight_shared_across_layers(self):
        class AddWeightLayer(layers_module.Layer):
            def __init__(self, trainable_var, non_trainable_var):
                self.trainable_var = trainable_var
                self.non_trainable_var = non_trainable_var
                super().__init__()

            def call(self, inputs):
                return inputs + self.trainable_var

        class LayerWithWeightSharedLayers(layers_module.Layer):
            def __init__(self):
                super().__init__()
                shared_trainable_var = tf.Variable(1.0)
                shared_non_trainable_var = tf.Variable(1.0, trainable=False)
                self.layer1 = AddWeightLayer(
                    shared_trainable_var, shared_non_trainable_var
                )
                self.layer2 = AddWeightLayer(
                    shared_trainable_var, shared_non_trainable_var
                )

            def call(self, inputs):
                return self.layer2(self.layer1(inputs))

        l = LayerWithWeightSharedLayers()
        layers = list(l._flatten_layers(include_self=False, recursive=False))
        self.assertEqual(layers, [l.layer1, l.layer2])
        self.assertEqual(
            l.variables, [l.layer1.trainable_var, l.layer1.non_trainable_var]
        )
        self.assertEqual(l.trainable_variables, [l.layer1.trainable_var])
        self.assertEqual(
            l.non_trainable_variables, [l.layer1.non_trainable_var]
        )
        self.assertLen(l.get_weights(), 2)

    @test_combinations.run_all_keras_modes
    def test_weight_tracking_for_template(self):
        def variable_scoped_function(trainable=True):
            return tf.compat.v1.get_variable(
                "dummy",
                shape=[1],
                trainable=trainable,
                initializer=tf.compat.v1.zeros_initializer(),
            )

        def nested_template():
            nested1 = tf.compat.v1.make_template(
                "nested", variable_scoped_function
            )
            nested2 = tf.compat.v1.make_template(
                "nested", variable_scoped_function
            )
            v1 = nested1()
            v2 = nested2()

            # nested1 and nested2 should not share variables
            self.assertIsNot(v1, v2)

            # Variables created by nested1 should be isolated from variables
            # created by nested2.
            self.assertEqual(1, len(nested1.variables))
            self.assertEqual(1, len(nested2.variables))
            self.assertIs(nested1.variables[0], v1)
            self.assertIs(nested2.variables[0], v2)
            self.assertEqual(1, len(nested1.trainable_variables))
            self.assertEqual(1, len(nested2.trainable_variables))
            self.assertIs(nested1.trainable_variables[0], v1)
            self.assertIs(nested2.trainable_variables[0], v2)
            self.assertEqual(len(nested1.non_trainable_variables), 0)
            self.assertEqual(len(nested2.non_trainable_variables), 0)
            return v1, v2

        tmpl1 = tf.compat.v1.make_template("s1", nested_template)
        tmpl2 = tf.compat.v1.make_template("s1", nested_template)

        v1, v2 = tmpl1()
        v5, v6 = tmpl2()

        model = training_module.Model()
        model.template = tmpl1
        self.assertEqual(2, len(model.variables))
        self.assertIs(model.variables[0], v1)
        self.assertIs(model.variables[1], v2)
        self.assertEqual(2, len(model.variables))
        self.assertIs(model.trainable_variables[0], v1)
        self.assertIs(model.trainable_variables[1], v2)
        self.assertEqual(len(model.non_trainable_variables), 0)
        model.templates = [tmpl2]
        for v, w in zip(model.variables, [v1, v2, v5, v6]):
            self.assertIs(v, w)
        for v, w in zip(model.trainable_variables, [v1, v2, v5, v6]):
            self.assertIs(v, w)
        self.assertEqual(len(model.non_trainable_variables), 0)
        # Make sure losses, layers, and updates aren't broken by having a
        # Template in the mix, which does not expose any updates or losses.
        self.assertEqual([], model.layers)
        self.assertEqual([], model.updates)
        self.assertEqual([], model.losses)
        self.assertEqual([], model.templates.layers)
        self.assertEqual([], model.templates.updates)
        self.assertEqual([], model.templates.losses)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_logs_passed_to_callbacks(self):
        input_dim = 5
        num_classes = 1

        class TestCallback(Callback):
            def __init__(self):
                super().__init__()
                self.epoch_end_logs = None
                self.batch_end_logs = None
                self.epoch_end_call_count = 0
                self.batch_end_call_count = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_end_logs = logs
                self.epoch_end_call_count += 1

            def on_batch_end(self, batch, logs=None):
                self.batch_end_logs = logs
                self.batch_end_call_count += 1

        model = test_utils.get_small_sequential_mlp(
            num_hidden=10, num_classes=num_classes, input_dim=input_dim
        )
        model.compile(
            loss="binary_crossentropy",
            metrics=["acc"],
            weighted_metrics=["mae"],
            optimizer=RMSPropOptimizer(learning_rate=0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        np.random.seed(1337)
        (x_train, y_train), (_, _) = test_utils.get_test_data(
            train_samples=10,
            test_samples=10,
            input_shape=(input_dim,),
            num_classes=num_classes,
        )

        test_callback = TestCallback()
        model.fit(
            x_train,
            y_train,
            batch_size=2,
            epochs=2,
            verbose=0,
            callbacks=[test_callback],
            validation_data=(x_train, y_train),
        )
        self.assertEqual(test_callback.batch_end_call_count, 10)
        self.assertEqual(test_callback.epoch_end_call_count, 2)

        self.assertSetEqual(
            set(test_callback.batch_end_logs.keys()),
            set(["acc", "loss", "mae"]),
        )
        self.assertSetEqual(
            set(test_callback.epoch_end_logs.keys()),
            set(["acc", "loss", "mae", "val_acc", "val_loss", "val_mae"]),
        )

    @test_combinations.run_all_keras_modes
    def test_mismatched_output_shape_and_target_shape(self):
        model = sequential.Sequential(
            [
                layers_module.Dense(2, input_shape=(3, 4)),
                layers_module.Dense(5),
            ]
        )
        model.compile(
            RMSPropOptimizer(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        # Test with Numpy data
        x_train = np.random.random((10, 3, 4)).astype(np.float32)
        y_train = np.random.randint(0, 5, size=(10, 3)).astype(np.float32)
        model.fit(x_train, y_train, batch_size=5, epochs=1)

        # Test with iterator
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat(10)
        dataset = dataset.batch(10)
        model.fit(dataset, epochs=1, steps_per_epoch=2)

        if tf.executing_eagerly():
            # Test with eager execution
            model.compile(
                RMSPropOptimizer(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                run_eagerly=True,
            )
            model.fit(x_train, y_train, batch_size=5, epochs=1)

            # Test with eager execution and iterator
            model.fit(dataset, epochs=1, steps_per_epoch=2)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_losses_in_defun(self):
        layer = layers_module.Dense(1, kernel_regularizer="l1")
        layer(tf.ones([1, 10]))

        @tf.function
        def get_losses():
            return layer.losses

        self.assertAllEqual(
            self.evaluate(layer.losses), self.evaluate(get_losses())
        )

    @test_combinations.run_all_keras_modes
    def test_logging(self):
        mock_stdout = io.StringIO()
        model = sequential.Sequential()
        model.add(layers_module.Dense(10, activation="relu"))
        model.add(layers_module.Dense(1, activation="sigmoid"))
        model.compile(
            RMSPropOptimizer(learning_rate=0.001),
            loss="binary_crossentropy",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        io_utils.enable_interactive_logging()
        with tf.compat.v1.test.mock.patch.object(sys, "stdout", mock_stdout):
            model.fit(
                np.ones((10, 10), "float32"),
                np.ones((10, 1), "float32"),
                epochs=10,
            )
        self.assertTrue("Epoch 5/10" in mock_stdout.getvalue())

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_training_with_loss_instance(self):
        a = layers_module.Input(shape=(3,), name="input_a")
        b = layers_module.Input(shape=(3,), name="input_b")

        dense = layers_module.Dense(4, name="dense")
        c = dense(a)
        d = dense(b)
        e = layers_module.Dropout(0.5, name="dropout")(c)

        model = training_module.Model([a, b], [d, e])
        loss_weights = [1.0, 0.5]
        model.compile(
            RMSPropOptimizer(learning_rate=0.001),
            loss=losses.MeanSquaredError(),
            metrics=[metrics_module.CategoricalAccuracy(), "mae"],
            loss_weights=loss_weights,
        )

        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))

        output_d_np = np.random.random((10, 4))
        output_e_np = np.random.random((10, 4))

        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=1,
            batch_size=5,
        )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_static_batch_in_input_layer(self):
        if tf.executing_eagerly():
            self.skipTest("Not inferred in eager.")

        class Counter(Callback):
            def __init__(self):
                self.batches = 0

            def on_batch_end(self, batch, logs=None):
                self.batches += 1

        x, y = np.ones((64, 10), "float32"), np.ones((64, 1), "float32")

        for batch_size, expected_batches in [(None, 2), (4, 16)]:
            inputs = input_layer.Input(batch_size=batch_size, shape=(10,))
            outputs = layers_module.Dense(1, activation="sigmoid")(inputs)
            model = training_module.Model(inputs, outputs)

            model.compile(
                optimizer_legacy.adam.Adam(0.001), "binary_crossentropy"
            )
            counter = Counter()
            model.fit(x, y, callbacks=[counter])
            self.assertEqual(counter.batches, expected_batches)

            model = sequential.Sequential(
                [layers_module.Dense(1, batch_input_shape=(batch_size, 10))]
            )
            model.compile(
                optimizer_legacy.adam.Adam(0.001), "binary_crossentropy"
            )
            counter = Counter()
            model.fit(x, y, callbacks=[counter])
            self.assertEqual(counter.batches, expected_batches)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_static_batch_in_input_layer_consistency_checks(self):
        if tf.executing_eagerly():
            self.skipTest("Not inferred in eager.")
        x, y = np.ones((64, 10), "float32"), np.ones((64, 1), "float32")

        inputs = input_layer.Input(batch_size=2, shape=(10,))
        outputs = layers_module.Dense(1, activation="sigmoid")(inputs)
        model = training_module.Model(inputs, outputs)
        model.compile(optimizer_legacy.adam.Adam(0.001), "binary_crossentropy")
        with self.assertRaisesRegex(
            ValueError, "incompatible with the specified batch size"
        ):
            model.fit(x, y, batch_size=4)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_compatible_batch_size_functional_model(self):
        class MyLayer(layers_module.Layer):
            def call(self, inputs):
                return tf.concat(inputs, axis=0)

        input1 = input_layer.Input(batch_size=2, shape=(10,))
        input2 = input_layer.Input(batch_size=3, shape=(10,))
        outputs = MyLayer()([input1, input2])
        with tf.compat.v1.test.mock.patch.object(
            logging, "warning"
        ) as mock_warn:
            training_module.Model([input1, input2], outputs)
            self.assertEqual(
                mock_warn.call_args_list[0][0][0],
                "Found incompatible static batch sizes among the inputs. "
                "Batch sizes: [2, 3]",
            )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_calling_subclass_model_on_different_datasets(self):
        class SubclassedModel(training_module.Model):
            def call(self, inputs):
                return inputs * 2

        model = SubclassedModel()
        dataset_one = tf.data.Dataset.from_tensor_slices([[0], [1]]).batch(2)
        dataset_two = tf.data.Dataset.from_tensor_slices(
            [[3], [4], [5], [6], [7], [8]]
        ).batch(2)
        self.assertAllEqual([[0], [2]], model.predict(dataset_one, steps=1))
        self.assertAllEqual(
            [[6], [8], [10], [12]], model.predict(dataset_two, steps=2)
        )

    @test_combinations.generate(test_combinations.combine(mode=["eager"]))
    def test_training_on_sparse_categorical_crossentropy_loss_with_softmax(
        self,
    ):
        np.random.seed(1337)
        train_x = np.ones((100, 4))
        train_y = np.random.randint(0, 1, size=(100, 1))

        reference_model = test_utils.get_small_sequential_mlp(
            16, 2, input_dim=4
        )
        reference_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=True,
        )
        fixed_weights = reference_model.get_weights()
        reference_model_loss = reference_model.train_on_batch(train_x, train_y)

        test_model = test_utils.get_small_sequential_mlp(16, 2, input_dim=4)
        test_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=False,
        )
        test_model.set_weights(fixed_weights)
        test_model_loss = test_model.train_on_batch(train_x, train_y)
        self.assertAlmostEqual(test_model_loss, reference_model_loss, places=4)

    @test_combinations.generate(test_combinations.combine(mode=["eager"]))
    def test_training_on_categorical_crossentropy_loss_with_softmax(self):
        np.random.seed(1337)
        train_x = np.ones((100, 4))
        train_y = np_utils.to_categorical(
            np.random.randint(0, 1, size=(100, 1)), 2
        )

        reference_model = test_utils.get_small_sequential_mlp(
            16, 2, input_dim=4
        )
        reference_model.compile(
            loss="categorical_crossentropy",
            optimizer=rmsprop.RMSprop(learning_rate=0.001),
            run_eagerly=True,
        )
        fixed_weights = reference_model.get_weights()
        reference_model_loss = reference_model.train_on_batch(train_x, train_y)

        test_model = test_utils.get_small_sequential_mlp(16, 2, input_dim=4)
        test_model.compile(
            loss="categorical_crossentropy",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=False,
        )
        test_model.set_weights(fixed_weights)
        test_model_loss = test_model.train_on_batch(train_x, train_y)
        self.assertAlmostEqual(test_model_loss, reference_model_loss, places=4)

    @test_combinations.generate(test_combinations.combine(mode=["eager"]))
    def test_training_on_binary_crossentropy_loss(self):
        train_x = np.ones((100, 4), dtype=np.float32)
        train_y = np.ones((100, 1), dtype=np.float32)
        reference_model = test_utils.get_small_sequential_mlp(
            16, 1, input_dim=4
        )
        reference_model.compile(
            loss="binary_crossentropy",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=True,
        )
        fixed_weights = reference_model.get_weights()
        reference_model_loss = reference_model.train_on_batch(train_x, train_y)

        test_model = test_utils.get_small_sequential_mlp(16, 1, input_dim=4)
        test_model.compile(
            loss="binary_crossentropy",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=False,
        )
        test_model.set_weights(fixed_weights)
        test_model_loss = test_model.train_on_batch(train_x, train_y)
        self.assertAlmostEqual(test_model_loss, reference_model_loss, places=4)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        ("default", 1, 4),
        ("integer_two", 2, 2),
        ("integer_four", 4, 1),
        ("simple_list", [1, 3, 4], 3),
        ("duplicated_list", [4, 2, 2], 2),
    )
    def test_validation_freq(self, validation_freq, expected_runs):
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model = test_utils.get_small_mlp(2, 1, 10)
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())

        class ValCounter(Callback):
            def __init__(self):
                self.val_runs = 0

            def on_test_begin(self, logs=None):
                self.val_runs += 1

        val_counter = ValCounter()
        model.fit(
            x,
            y,
            epochs=4,
            validation_data=(x, y),
            validation_freq=validation_freq,
            callbacks=[val_counter],
        )
        self.assertEqual(val_counter.val_runs, expected_runs)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_validation_steps_without_data(self):
        if tf.executing_eagerly():
            self.skipTest("Check removed in new `fit`")
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model = test_utils.get_small_mlp(2, 1, 10)
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())

        with self.assertRaisesRegex(
            ValueError,
            "`validation_steps` should not be specified if "
            "`validation_data` is None.",
        ):
            model.fit(x, y, epochs=4, validation_data=None, validation_steps=3)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_layer_with_variable_output(self):
        class VariableOutputLayer(layers_module.Layer):
            def build(self, input_shape):
                self.v = self.add_weight(
                    "output_var", shape=(2, 5), initializer="ones"
                )

            def call(self, inputs):
                return self.v

        model = test_utils.get_model_from_layers(
            [VariableOutputLayer(), layers_module.Dense(1)], input_shape=(10,)
        )
        # TODO(omalleyt): Make this work with `run_eagerly=True`.
        model.compile("sgd", "mse", run_eagerly=False)
        model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2, epochs=5)

        self.assertLen(model.trainable_variables, 3)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    @test_utils.enable_v2_dtype_behavior
    def test_model_dtype(self):
        class AssertTypeLayer(layers_module.Layer):
            def call(self, inputs):
                assert inputs.dtype.name == self.dtype, (
                    "Input tensor has type %s which does not match assert "
                    "type %s" % (inputs.dtype.name, self.assert_type)
                )
                return inputs + 1.0

        for dtype in ("float16", "float32", "float64"):
            model = test_utils.get_model_from_layers(
                [AssertTypeLayer(dtype=dtype)], input_shape=(10,)
            )
            model.compile(
                "sgd", "mse", run_eagerly=test_utils.should_run_eagerly()
            )

            x = np.ones((10, 10))
            y = np.ones((10, 10))
            model.fit(x, y)
            model.test_on_batch(x, y)
            model(x)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    @test_utils.enable_v2_dtype_behavior
    def test_model_input_dtype(self):
        model = test_utils.get_small_mlp(1, 10, 10)
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        x = np.ones((10, 10)).astype(np.float64)
        y = np.ones((10, 10)).astype(np.float64)
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        model.fit(dataset)
        self.assertEqual(model._compute_dtype, "float32")

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_subclassed_model_with_training_arg(self):
        class LayerWithTrainingArg(layers_module.Layer):
            def call(self, inputs, training=None):
                self.training = training
                return inputs

        class ModelWithTrainingArg(training_module.Model):
            def __init__(self):
                super().__init__()
                self.l1 = LayerWithTrainingArg()

            def call(self, inputs, training=None):
                self.training = training
                inputs = self.l1(inputs, training=training)
                return inputs

        x = np.zeros((1, 2))
        model = ModelWithTrainingArg()
        model.compile(
            loss="mse",
            optimizer="sgd",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(x, x, epochs=1)

        if tf.executing_eagerly():
            expected_training_arg = True
        else:
            expected_training_arg = backend.symbolic_learning_phase()

        self.assertIs(model.training, expected_training_arg)
        self.assertIs(model.l1.training, expected_training_arg)

    @test_combinations.run_all_keras_modes
    def test_error_when_model_is_not_compiled(self):
        inputs = input_layer.Input(shape=(1,))
        outputs = layers_module.Dense(1)(inputs)
        model = training_module.Model(inputs, outputs)
        with self.assertRaisesRegex(RuntimeError, "must compile your model"):
            model.fit(np.ones((1, 1)), np.ones((1, 1)))

        class MyModel(training_module.Model):
            def call(self, x):
                self.add_loss(tf.reduce_sum(x))
                return x

        model = MyModel()
        with self.assertRaisesRegex(RuntimeError, "must compile your model"):
            model.fit(np.random.random((32, 1)), epochs=2)

    @test_combinations.run_all_keras_modes
    @test_utils.enable_v2_dtype_behavior
    def test_losses_of_different_dtypes(self):
        inp = input_layer.Input(shape=(2,))
        out_1 = layers_module.Dense(
            2, dtype="float32", kernel_regularizer="l2"
        )(inp)
        out_2 = layers_module.Dense(
            2, dtype="float16", kernel_regularizer="l2"
        )(inp)
        model = training_module.Model(inp, [out_1, out_2])
        extra_loss = tf.reduce_sum(tf.cast(out_2, "float64"))
        model.add_loss(extra_loss)
        model.compile(
            "sgd", ["mse", "mse"], run_eagerly=test_utils.should_run_eagerly()
        )
        x, y = np.ones((10, 2)), np.ones((10, 2))
        model.fit(x, [y, y])

    @test_combinations.run_all_keras_modes
    @test_utils.enable_v2_dtype_behavior
    def test_losses_of_different_dtypes_with_subclassed_model(self):
        class MyModel(training_module.Model):
            def build(self, _):
                self.dense = layers_module.Dense(2)

            def call(self, inputs):
                self.add_loss(tf.cast(tf.nn.l2_loss(inputs), "float64"))
                return self.dense(inputs)

        model = MyModel(dtype="float32")
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        x, y = np.ones((10, 2)), np.ones((10, 2))
        model.fit(x, y)

    @test_combinations.run_all_keras_modes
    @test_utils.enable_v2_dtype_behavior
    def test_regularizer_of_different_dtype(self):
        inp = input_layer.Input(shape=(2,))

        def regularizer(weight):
            return tf.cast(tf.nn.l2_loss(weight), "float64")

        out = layers_module.Dense(
            2, dtype="float32", kernel_regularizer=regularizer
        )(inp)
        model = training_module.Model(inp, out)
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        x, y = np.ones((10, 2)), np.ones((10, 2))
        model.fit(x, y)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_outputs_are_floats(self):
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model = sequential.Sequential([layers_module.Dense(1)])
        model.compile(
            "sgd",
            "mse",
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        history = model.fit(x, y, epochs=2)
        self.assertIsInstance(history.history["loss"][0], float)
        self.assertIsInstance(history.history["accuracy"][0], float)

        loss, accuracy = model.train_on_batch(x, y)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)

        loss, accuracy = model.evaluate(x, y)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)

        loss, accuracy = model.test_on_batch(x, y)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_int_output(self):
        x, y = np.ones((10, 1)), np.ones((10, 1))
        model = sequential.Sequential([layers_module.Dense(1)])

        class MyMetric(metrics_module.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                del y_true, y_pred, sample_weight

            def result(self):
                return tf.constant(1, dtype="int64")

        model.compile(
            "sgd",
            "mse",
            metrics=[MyMetric()],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        history = model.fit(x, y, epochs=2)
        self.assertIsInstance(history.history["my_metric"][0], int)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @test_utils.enable_v2_dtype_behavior
    def test_mixed_precision(self):
        x, y = np.ones((10, 1)), np.ones((10, 1))
        policy.set_global_policy("mixed_float16")
        model = sequential.Sequential([layers_module.Dense(1)])
        optimizer = sgd_experimental.SGD()
        model.compile(
            optimizer,
            "mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(x, y, epochs=2)
        policy.set_global_policy("float32")

    @test_combinations.run_all_keras_modes
    def test_calling_aggregate_gradient(self):
        class _Optimizer(optimizer_legacy.gradient_descent.SGD):
            """Mock optimizer to check if _aggregate_gradient is called."""

            _HAS_AGGREGATE_GRAD = True

            def __init__(self):
                self.aggregate_gradients_called = False
                super().__init__(name="MyOptimizer")

            def _aggregate_gradients(self, grads):
                self.aggregate_gradients_called = True
                return super()._aggregate_gradients(grads)

        mock_optimizer = _Optimizer()

        model = sequential.Sequential()
        model.add(layers_module.Dense(10, activation="relu"))

        model.compile(
            mock_optimizer, "mse", run_eagerly=test_utils.should_run_eagerly()
        )
        x, y = np.ones((10, 10)), np.ones((10, 10))
        model.fit(x, y)
        self.assertEqual(model.optimizer.aggregate_gradients_called, True)

        class _OptimizerOverrideApplyGradients(_Optimizer):
            """Override apply_gradients.

            To test the case where the optimizer does not define the
            experimental_aggregate_gradients parameter.
            """

            _HAS_AGGREGATE_GRAD = False

            def apply_gradients(self, grads_and_vars, name=None):
                return super().apply_gradients(grads_and_vars, name)

        mock_optimizer = _OptimizerOverrideApplyGradients()
        model.compile(
            mock_optimizer, "mse", run_eagerly=test_utils.should_run_eagerly()
        )
        x, y = np.ones((10, 10)), np.ones((10, 10))
        model.fit(x, y)
        self.assertEqual(model.optimizer.aggregate_gradients_called, True)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_gradients_are_none(self):
        class DenseWithExtraWeight(layers_module.Dense):
            def build(self, input_shape):
                # Gradients w.r.t. extra_weights are None
                self.extra_weight_1 = self.add_weight(
                    "extra_weight_1", shape=(), initializer="ones"
                )
                super().build(input_shape)
                self.extra_weight_2 = self.add_weight(
                    "extra_weight_2", shape=(), initializer="ones"
                )

        model = sequential.Sequential(
            [DenseWithExtraWeight(4, input_shape=(4,))]
        )
        # Test clipping can handle None gradients
        opt = optimizer_legacy.adam.Adam(clipnorm=1.0, clipvalue=1.0)
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())
        inputs = np.random.normal(size=(64, 4))
        targets = np.random.normal(size=(64, 4))
        old_kernel = model.get_weights()[1]
        model.fit(inputs, targets)
        new_kernel = model.get_weights()[1]
        self.assertNotAllEqual(old_kernel, new_kernel)

    @test_combinations.run_all_keras_modes
    def test_layer_ordering(self):
        class MyLayer(layers_module.Layer):
            pass

        class MyModel(training_module.Model):
            def __init__(self, name):
                super().__init__(name=name)

                self.weight = tf.Variable(0, name=name)

                self.direct_sublayer = MyLayer(name="direct")
                self.direct_sublayer.d = {"d": MyLayer(name="direct/dict")}

                self.dict_sublayer = {"d": MyLayer(name="dict")}
                self.dict_sublayer["d"].direct = MyLayer(name="dict/direct")

        model = MyModel("model")
        # All sublayers, including self and recursive sublayers.
        self.assertEqual(
            ["model", "direct", "direct/dict", "dict", "dict/direct"],
            [l.name for l in model._flatten_layers()],
        )
        # Only direct sublayers, including those in data structures.
        self.assertEqual(["direct", "dict"], [l.name for l in model.layers])

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_trainable_state_setting(self):
        class UpdateLayer(layers_module.Layer):
            def __init__(self):
                super().__init__()
                self.v = tf.Variable(0.0, trainable=False)

            def call(self, x):
                self.add_update(lambda: self.v.assign_add(1.0))
                return x * self.v

        layer = UpdateLayer()
        model_with_updates = sequential.Sequential([layer])
        model_with_updates.compile(
            "sgd", "mse", run_eagerly=test_utils.should_run_eagerly()
        )

        layer.trainable = False
        model_without_updates = sequential.Sequential([layer])
        model_without_updates.compile(
            "sgd", "mse", run_eagerly=test_utils.should_run_eagerly()
        )

        x, y = np.ones((10, 1)), np.ones((10, 1))

        self.assertEqual(self.evaluate(layer.v), 0.0)
        model_with_updates.fit(x, y, batch_size=10)
        # assign_add called.
        self.assertEqual(self.evaluate(layer.v), 1.0)
        model_without_updates.fit(x, y, batch_size=10)
        # assign_add not called.
        self.assertEqual(self.evaluate(layer.v), 1.0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        ("numpy_array", "numpy_array"),
        ("dataset_array", "dataset_array"),
        ("dataset_dict", "dataset_dict"),
    )
    def test_single_input_no_tuple_wrapping(self, input_type):
        x = np.ones((10, 1))

        if input_type == "numpy_array":
            batch_size = 3
            expected_data_type = tf.Tensor
        elif input_type == "dataset_array":
            x = tf.data.Dataset.from_tensor_slices(x).batch(3)
            batch_size = None
            expected_data_type = tf.Tensor
        else:
            x = {"my_input": x}
            x = tf.data.Dataset.from_tensor_slices(x).batch(3)
            batch_size = None
            expected_data_type = dict

        test_case = self

        class MyModel(training_module.Model):
            def train_step(self, data):
                # No tuple wrapping for single x input and no targets.
                test_case.assertIsInstance(data, expected_data_type)
                return super().train_step(data)

            def test_step(self, data):
                test_case.assertIsInstance(data, expected_data_type)
                return super().test_step(data)

            def predict_step(self, data):
                test_case.assertIsInstance(data, expected_data_type)
                return super().predict_step(data)

        inputs = layers_module.Input(shape=(1,), name="my_input")
        outputs = layers_module.Dense(1)(inputs)
        model = MyModel(inputs, outputs)
        model.add_loss(tf.reduce_sum(outputs))
        model.compile("sgd")
        model.fit(x, batch_size=batch_size)
        model.evaluate(x, batch_size=batch_size)
        model.predict(x, batch_size=batch_size)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        ("custom_metrics", False, True),
        ("compiled_metrics", True, False),
        ("both_compiled_and_custom_metrics", True, True),
    )
    def test_evaluate_with_custom_test_step(
        self, use_compiled_metrics, use_custom_metrics
    ):
        class MyModel(training_module.Model):
            def test_step(self, data):
                x, y = data
                pred = self(x)
                metrics = {}
                if use_compiled_metrics:
                    self.compiled_metrics.update_state(y, pred)
                    self.compiled_loss(y, pred)
                    for metric in self.metrics:
                        metrics[metric.name] = metric.result()
                if use_custom_metrics:
                    custom_metrics = {
                        "mean": tf.reduce_mean(pred),
                        "sum": tf.reduce_sum(pred),
                    }
                    metrics.update(custom_metrics)
                return metrics

        inputs = layers_module.Input((2,))
        outputs = layers_module.Dense(3)(inputs)
        model = MyModel(inputs, outputs)
        if use_compiled_metrics:
            model.compile(
                "adam",
                "mse",
                metrics=["mae", "mape"],
                run_eagerly=test_utils.should_run_eagerly(),
            )
        else:
            model.compile(
                "adam", "mse", run_eagerly=test_utils.should_run_eagerly()
            )
        x = np.random.random((4, 2))
        y = np.random.random((4, 3))
        results_list = model.evaluate(x, y)
        results_dict = model.evaluate(x, y, return_dict=True)
        self.assertLen(results_list, len(results_dict))
        if use_compiled_metrics and use_custom_metrics:
            self.assertLen(results_list, 5)
            self.assertEqual(
                results_list,
                [
                    results_dict["loss"],
                    results_dict["mae"],
                    results_dict["mape"],
                    results_dict["mean"],
                    results_dict["sum"],
                ],
            )
        if use_compiled_metrics and not use_custom_metrics:
            self.assertLen(results_list, 3)
            self.assertEqual(
                results_list,
                [
                    results_dict["loss"],
                    results_dict["mae"],
                    results_dict["mape"],
                ],
            )
        if not use_compiled_metrics and use_custom_metrics:
            self.assertLen(results_list, 2)
            self.assertEqual(
                results_list, [results_dict["mean"], results_dict["sum"]]
            )

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_model_make_function(self):
        layers = [
            layers_module.Dense(10, dtype=np.float64),
            layers_module.Dense(10, dtype=np.float64),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(1,))
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())

        original_train_function = model.make_train_function()
        self.assertIsNotNone(original_train_function)
        self.assertEqual(model.make_train_function(), original_train_function)
        # Check that we regenerate it without reusing the cached version.
        self.assertNotEqual(
            model.make_train_function(force=True), original_train_function
        )

        original_test_function = model.make_test_function()
        self.assertIsNotNone(original_test_function)
        self.assertEqual(model.make_test_function(), original_test_function)
        # Check that we regenerate it without reusing the cached version.
        self.assertNotEqual(
            model.make_test_function(force=True), original_test_function
        )

        original_predict_function = model.make_predict_function()
        self.assertIsNotNone(original_predict_function)
        self.assertEqual(
            model.make_predict_function(), original_predict_function
        )
        # Check that we regenerate it without reusing the cached version.
        self.assertNotEqual(
            model.make_predict_function(force=True), original_predict_function
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_custom_compute_metrics(self):
        class CustomMetric(metrics_module.Mean):
            def sq_diff_plus_x(self, x, y_true, y_pred):
                y_pred = tf.convert_to_tensor(y_pred)
                y_true = tf.cast(y_true, y_pred.dtype)
                sq_diff_plus_x = tf.add(
                    x, tf.math.squared_difference(y_pred, y_true)
                )
                return backend.mean(sq_diff_plus_x, axis=-1)

            def update_state(self, x, y_true, y_pred, sample_weight=None):
                matches = self.sq_diff_plus_x(x, y_true, y_pred)
                return super().update_state(matches)

        class MyModel(sequential.Sequential):
            def compute_metrics(self, x, y, y_pred, sample_weight):
                metric_results = super().compute_metrics(
                    x, y, y_pred, sample_weight
                )
                self.custom_metric.update_state(x, y, y_pred, sample_weight)
                metric_results[
                    "custom_metric_name"
                ] = self.custom_metric.result()
                return metric_results

        tensors = tf.random.uniform((10, 10)), tf.random.uniform((10,))
        dataset = tf.data.Dataset.from_tensor_slices(tensors).repeat().batch(1)
        model = MyModel([layers_module.Dense(10)])
        model.custom_metric = CustomMetric("my_metric")
        initial_result = model.custom_metric.result()
        optimizer = optimizer_legacy.gradient_descent.SGD()
        model.compile(optimizer, loss="mse", steps_per_execution=10)
        model.fit(dataset, epochs=2, steps_per_epoch=10, verbose=2)
        after_fit_result = model.custom_metric.result()

        self.assertEqual(self.evaluate(initial_result), 0.0)
        self.assertNotEqual(
            self.evaluate(initial_result), self.evaluate(after_fit_result)
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_custom_compute_loss(self):
        class MyModel(training_module.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_metric = metrics_module.Mean(name="loss")

            def compute_loss(self, x, y, y_pred, sample_weight):
                loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
                loss += tf.add_n(self.losses)
                self.loss_metric.update_state(loss)
                return loss

            def reset_metrics(self):
                self.loss_metric.reset_states()

            @property
            def metrics(self):
                return [self.loss_metric]

        tensors = tf.random.uniform((10, 10)), tf.random.uniform((10,))
        dataset = tf.data.Dataset.from_tensor_slices(tensors).repeat().batch(1)

        inputs = layers_module.Input(shape=(10,), name="my_input")
        outputs = layers_module.Dense(10)(inputs)
        model = MyModel(inputs, outputs)
        model.add_loss(tf.reduce_sum(outputs))

        optimizer = optimizer_legacy.gradient_descent.SGD()
        model.compile(optimizer, loss="mse", steps_per_execution=10)
        history = model.fit(dataset, epochs=2, steps_per_epoch=10)
        self.assertLen(history.history["loss"], 2)
        self.assertAllClose(
            history.history["loss"][1], model.loss_metric.result()
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        ("mixed_float16", "mixed_float16"), ("float32", "float32")
    )
    def test_ema_overwrite(self, test_policy):
        if not tf.__internal__.tf2.enabled():
            self.skipTest("EMA optimizer is only available in TF2.")
        policy.set_global_policy(test_policy)
        model = sequential.Sequential()
        model.add(input_layer.Input(shape=(4,)))
        model.add(layers_module.Dense(1, activation="relu"))

        tensors = tf.random.uniform((4, 4)), tf.random.uniform((4,))
        dataset = tf.data.Dataset.from_tensor_slices(tensors).repeat().batch(1)

        optimizer = sgd_experimental.SGD(use_ema=True, ema_momentum=1)
        model.compile(optimizer, loss="mse", steps_per_execution=10)
        initial_value = tf.Variable(model.trainable_variables[0])
        history = model.fit(dataset, epochs=2, steps_per_epoch=10)
        self.assertLen(history.history["loss"], 2)
        self.assertAllClose(initial_value, model.trainable_variables[0])
        policy.set_global_policy("float32")

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_get_verbosity(self):
        class MyStrategy(tf.distribute.Strategy):
            def __init__(self):
                self._should_use_with_coordinator = True

        with self.assertRaisesRegex(ValueError, "`verbose=1` is not allowed"):
            training_module._get_verbosity(1, MyStrategy())

        io_utils.enable_interactive_logging()
        self.assertEqual(
            training_module._get_verbosity("auto", MyStrategy()), 2
        )
        self.assertEqual(
            training_module._get_verbosity(
                "auto", tf.distribute.MirroredStrategy()
            ),
            1,
        )
        self.assertEqual(
            training_module._get_verbosity(2, tf.distribute.MirroredStrategy()),
            2,
        )

        io_utils.disable_interactive_logging()
        self.assertEqual(
            training_module._get_verbosity(
                "auto", tf.distribute.MirroredStrategy()
            ),
            2,
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_save_spec(self):
        class Model(training_module.Model):
            def call(
                self, arg_input_1, arg_input_2, keyword_input, training=None
            ):
                return 0

        # Test subclassed model save specs.
        model = Model()
        model(
            tf.ones([1, 1]),
            tf.ones([2, 2]),
            keyword_input=tf.ones([3, 3]),
            training=False,
        )
        spec = model.save_spec(dynamic_batch=False)
        self.assertEqual(spec[0][0].shape.as_list(), [1, 1])
        self.assertEqual(spec[0][1].shape.as_list(), [2, 2])
        self.assertEqual(spec[1]["keyword_input"].shape.as_list(), [3, 3])
        spec = model.save_spec(dynamic_batch=True)
        self.assertEqual(spec[0][0].shape.as_list(), [None, 1])

        # Test functional model save specs.
        input_1 = layers_module.Input((1,), batch_size=1)
        input_2 = layers_module.Input((2,), batch_size=2)
        input_3 = layers_module.Input((3,), batch_size=3)
        output = model(input_1, input_2, keyword_input=input_3, training=True)
        functional = training_module.Model([input_1, input_2, input_3], output)
        # Functional models should ignore dynamic_batch if the input layers have
        # a known batch size.
        spec = functional.save_spec(dynamic_batch=True)
        input_specs = spec[0][0]
        self.assertEqual(input_specs[0].shape.as_list(), [1, 1])
        self.assertEqual(input_specs[1].shape.as_list(), [2, 2])
        self.assertEqual(input_specs[2].shape.as_list(), [3, 3])


class TestExceptionsAndWarnings(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @test_combinations.run_with_all_model_types
    def test_fit_on_no_output(self):
        inputs = layers_module.Input((3,))
        outputs = layers_module.Dense(2)(inputs)
        model = training_module.Model(inputs, outputs)
        model.compile("rmsprop", "mse")
        x = np.zeros((32, 3))
        with self.assertRaisesRegex(ValueError, "Target data is missing..*"):
            model.fit(x)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @test_combinations.run_with_all_model_types
    def test_fit_on_wrong_output_type(self):
        inputs1 = layers_module.Input((3,), name="a")
        inputs2 = layers_module.Input((3,), name="b")
        x = layers_module.Concatenate()([inputs1, inputs2])
        outputs = layers_module.Dense(2, name="c")(x)
        model = training_module.Model([inputs1, inputs2], outputs)
        model.compile("rmsprop", "mse")
        x = np.zeros((32, 3))
        y = np.zeros((32, 2))
        with self.assertRaisesRegex(ValueError, "Target data is missing..*"):
            model.fit({"a": x, "b": x, "c": y})

    @test_combinations.run_all_keras_modes
    def test_compile_warning_for_loss_missing_output(self):
        with self.cached_session():
            inp = layers_module.Input(shape=(16,), name="input_a")
            out_1 = layers_module.Dense(8, name="dense_1")(inp)
            out_2 = layers_module.Dense(
                3, activation="softmax", name="dense_2"
            )(out_1)
            model = training_module.Model(inputs=[inp], outputs=[out_1, out_2])
            optimizer = RMSPropOptimizer(learning_rate=0.001)

            model.compile(
                optimizer,
                loss={
                    "dense_2": "categorical_crossentropy",
                },
                metrics={
                    "dense_2": "categorical_accuracy",
                    "dense_1": metrics_module.CategoricalAccuracy(),
                },
                run_eagerly=test_utils.should_run_eagerly(),
            )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_predict_error_with_empty_x(self):
        inputs = layers_module.Input(shape=(2,))
        outputs = layers_module.Dense(4)(inputs)
        model = training_module.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse")

        with self.assertRaisesRegex(
            ValueError, "Unexpected result of `predict_function`.*"
        ):
            model.predict(np.array([]))

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        ("dynamic", 0, False),
        ("dynamic_multistep", 10, False),
        ("static", 0, True),
        ("static_multistep", 10, True),
    )
    def test_predict_structured(self, spe, static_batch):
        inputs = layers_module.Input(shape=(2,))
        outputs = layers_module.Dense(2)(inputs)
        model = training_module.Model(
            inputs=inputs,
            outputs={"out": outputs},
        )
        model.compile(
            loss="mse",
            steps_per_execution=spe,
            run_eagerly=test_utils.should_run_eagerly(),
        )
        xdata = np.random.uniform(size=(8, 2)).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((xdata, xdata))
        dataset = dataset.batch(8, drop_remainder=static_batch)
        ret = model.predict(dataset, steps=1)
        tf.nest.assert_same_structure(ret, {"out": ""})

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_on_batch_error_inconsistent_batch_size(self):
        input_node1 = layers_module.Input(shape=(5,))
        input_node2 = layers_module.Input(shape=(5,))
        output_node = layers_module.Concatenate()([input_node1, input_node2])
        output_node = layers_module.Dense(4)(output_node)
        model = training_module.Model([input_node1, input_node2], output_node)
        model.compile(loss="mse")

        with self.assertRaisesRegex(
            ValueError, "Data cardinality is ambiguous"
        ):
            model.train_on_batch(
                [np.ones((10, 5)), np.ones((10, 5))], np.ones((11, 4))
            )

        with self.assertRaisesRegex(
            ValueError, "Data cardinality is ambiguous"
        ):
            model.test_on_batch(
                [np.ones((10, 5)), np.ones((10, 5))], np.ones((11, 4))
            )

        with self.assertRaisesRegex(
            ValueError, "Data cardinality is ambiguous"
        ):
            model.predict_on_batch([np.ones((10, 5)), np.ones((11, 5))])


class LossWeightingTest(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes
    def test_class_weights(self):
        num_classes = 5
        batch_size = 5
        epochs = 10
        weighted_class = 3
        weight = 0.5
        train_samples = 1000
        test_samples = 1000
        input_dim = 5
        learning_rate = 0.001

        model = test_utils.get_small_sequential_mlp(
            num_hidden=10, num_classes=num_classes, input_dim=input_dim
        )
        model.compile(
            loss="categorical_crossentropy",
            metrics=["acc", metrics_module.CategoricalAccuracy()],
            weighted_metrics=["mae", metrics_module.CategoricalAccuracy()],
            optimizer=RMSPropOptimizer(learning_rate=learning_rate),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=train_samples,
            test_samples=test_samples,
            input_shape=(input_dim,),
            num_classes=num_classes,
        )
        int_y_test = y_test.copy()
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        test_ids = np.where(int_y_test == np.array(weighted_class))[0]

        class_weight = dict([(i, 1.0) for i in range(num_classes)])
        class_weight[weighted_class] = weight

        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 3,
            verbose=0,
            class_weight=class_weight,
            validation_data=(x_train, y_train),
        )
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 2,
            verbose=0,
            class_weight=class_weight,
        )
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 2,
            verbose=0,
            class_weight=class_weight,
            validation_split=0.1,
        )

        model.train_on_batch(
            x_train[:batch_size],
            y_train[:batch_size],
            class_weight=class_weight,
        )
        ref_score = model.evaluate(x_test, y_test, verbose=0)  # noqa: F841
        score = model.evaluate(  # noqa: F841
            x_test[test_ids, :], y_test[test_ids, :], verbose=0
        )
        # TODO(b/152990697): Fix the class weights test here.
        # self.assertLess(score[0], ref_score[0])

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_segmentation_class_weights(self):
        num_channels = 3
        num_classes = 5
        batch_size = 2
        image_width = 8

        input_shape = (batch_size, image_width, image_width, num_channels)
        output_shape = (batch_size, image_width, image_width, num_classes)

        model = sequential.Sequential([layers_module.Conv2D(num_classes, 1)])

        model.compile(
            loss="categorical_crossentropy",
            metrics=["acc", metrics_module.CategoricalAccuracy()],
            weighted_metrics=["mae", metrics_module.CategoricalAccuracy()],
            optimizer="adam",
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = tf.random.uniform(input_shape)
        y = tf.random.uniform(output_shape, dtype=tf.int32, maxval=num_classes)

        # Class weights are just the class value + 1
        class_weight = dict([(i, i + 1) for i in range(num_classes)])

        # This test simply asserts that the model can be compiled and fit
        # can run without error. Verification that the class weights are
        # applied correctly is performed in data_adapter_test.
        model.fit(x, y, class_weight=class_weight, steps_per_epoch=1)

        sample_weight = np.array([x + 1 for x in range(batch_size)])
        model.fit(
            x,
            y,
            class_weight=class_weight,
            sample_weight=sample_weight,
            steps_per_epoch=1,
        )

    @test_combinations.run_all_keras_modes
    def test_temporal_sample_weights(self):
        num_classes = 5
        batch_size = 5
        epochs = 10
        weighted_class = 3
        weight = 10.0
        train_samples = 1000
        test_samples = 1000
        input_dim = 5
        timesteps = 3
        learning_rate = 0.001

        with self.cached_session():
            model = sequential.Sequential()
            model.add(
                layers_module.TimeDistributed(
                    layers_module.Dense(num_classes),
                    input_shape=(timesteps, input_dim),
                )
            )
            model.add(layers_module.Activation("softmax"))

            np.random.seed(1337)
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=train_samples,
                test_samples=test_samples,
                input_shape=(input_dim,),
                num_classes=num_classes,
            )
            int_y_test = y_test.copy()
            int_y_train = y_train.copy()
            # convert class vectors to binary class matrices
            y_train = np_utils.to_categorical(y_train, num_classes)
            y_test = np_utils.to_categorical(y_test, num_classes)
            test_ids = np.where(int_y_test == np.array(weighted_class))[0]

            sample_weight = np.ones((y_train.shape[0]))
            sample_weight[int_y_train == weighted_class] = weight

            temporal_x_train = np.reshape(
                x_train, (len(x_train), 1, x_train.shape[1])
            )
            temporal_x_train = np.repeat(temporal_x_train, timesteps, axis=1)
            temporal_x_test = np.reshape(
                x_test, (len(x_test), 1, x_test.shape[1])
            )
            temporal_x_test = np.repeat(temporal_x_test, timesteps, axis=1)

            temporal_y_train = np.reshape(
                y_train, (len(y_train), 1, y_train.shape[1])
            )
            temporal_y_train = np.repeat(temporal_y_train, timesteps, axis=1)
            temporal_y_test = np.reshape(
                y_test, (len(y_test), 1, y_test.shape[1])
            )
            temporal_y_test = np.repeat(temporal_y_test, timesteps, axis=1)

            temporal_sample_weight = np.reshape(
                sample_weight, (len(sample_weight), 1)
            )
            temporal_sample_weight = np.repeat(
                temporal_sample_weight, timesteps, axis=1
            )

            model.compile(
                RMSPropOptimizer(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=["acc", metrics_module.CategoricalAccuracy()],
                weighted_metrics=["mae", metrics_module.CategoricalAccuracy()],
                sample_weight_mode="temporal",
                run_eagerly=test_utils.should_run_eagerly(),
            )

            model.fit(
                temporal_x_train,
                temporal_y_train,
                batch_size=batch_size,
                epochs=epochs // 3,
                verbose=0,
                sample_weight=temporal_sample_weight,
            )
            model.fit(
                temporal_x_train,
                temporal_y_train,
                batch_size=batch_size,
                epochs=epochs // 3,
                verbose=0,
                sample_weight=temporal_sample_weight,
                validation_split=0.1,
            )

            model.train_on_batch(
                temporal_x_train[:batch_size],
                temporal_y_train[:batch_size],
                sample_weight=temporal_sample_weight[:batch_size],
            )
            model.test_on_batch(
                temporal_x_train[:batch_size],
                temporal_y_train[:batch_size],
                sample_weight=temporal_sample_weight[:batch_size],
            )
            ref_score = model.evaluate(
                temporal_x_test, temporal_y_test, verbose=0
            )
            if not tf.executing_eagerly():
                score = model.evaluate(
                    temporal_x_test[test_ids],
                    temporal_y_test[test_ids],
                    verbose=0,
                )
                self.assertLess(score[0], ref_score[0])

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types(exclude_models="sequential")
    def test_fit_with_incorrect_weights(self):
        input_a = layers_module.Input(shape=(3,), name="input_a")
        input_b = layers_module.Input(shape=(3,), name="input_b")

        dense = layers_module.Dense(2, name="output_1")
        dropout = layers_module.Dropout(0.5, name="output_2")
        branch_a = [input_a, dense]
        branch_b = [input_b, dense, dropout]

        model = test_utils.get_multi_io_model(branch_a, branch_b)
        model.compile(
            optimizer="adam",
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x = np.random.random((10, 3))
        y = np.random.random((10, 2))

        with self.assertRaises(ValueError):
            model.fit([x, x], [y, y], epochs=1, sample_weight={"unknown": x})

        with self.assertRaises(ValueError):
            model.fit([x, x], [y, y], epochs=1, class_weight={"unknown": 1})

    @test_combinations.run_all_keras_modes
    def test_default_sample_weight(self):
        """Verifies that fit works without having to set sample_weight."""
        num_classes = 5
        input_dim = 5
        timesteps = 3
        learning_rate = 0.001

        with self.cached_session():
            model = sequential.Sequential()
            model.add(
                layers_module.TimeDistributed(
                    layers_module.Dense(num_classes),
                    input_shape=(timesteps, input_dim),
                )
            )

            x = np.random.random((10, timesteps, input_dim))
            y = np.random.random((10, timesteps, num_classes))
            optimizer = RMSPropOptimizer(learning_rate=learning_rate)

            # sample_weight_mode is a list and mode value is None
            model.compile(
                optimizer,
                loss="mse",
                sample_weight_mode=[None],
                run_eagerly=test_utils.should_run_eagerly(),
            )
            model.fit(x, y, epochs=1, batch_size=10)

            # sample_weight_mode is a list and mode value is `temporal`
            model.compile(
                optimizer,
                loss="mse",
                sample_weight_mode=["temporal"],
                run_eagerly=test_utils.should_run_eagerly(),
            )
            model.fit(x, y, epochs=1, batch_size=10)

            # sample_weight_mode is a dict and mode value is None
            model.compile(
                optimizer,
                loss="mse",
                sample_weight_mode={"time_distributed": None},
                run_eagerly=test_utils.should_run_eagerly(),
            )
            model.fit(x, y, epochs=1, batch_size=10)

            # sample_weight_mode is a dict and mode value is `temporal`
            model.compile(
                optimizer,
                loss="mse",
                sample_weight_mode={"time_distributed": "temporal"},
                run_eagerly=test_utils.should_run_eagerly(),
            )
            model.fit(x, y, epochs=1, batch_size=10)

            # sample_weight_mode is a not a list/dict and mode value is None
            model.compile(
                optimizer,
                loss="mse",
                sample_weight_mode=None,
                run_eagerly=test_utils.should_run_eagerly(),
            )
            model.fit(x, y, epochs=1, batch_size=10)

            # sample_weight_mode is a not a list/dict and mode value is
            # `temporal`
            model.compile(
                optimizer,
                loss="mse",
                sample_weight_mode="temporal",
                run_eagerly=test_utils.should_run_eagerly(),
            )
            model.fit(x, y, epochs=1, batch_size=10)

    def test_sample_weight_tensor(self):
        """Tests that sample weight may be defined as a tensor in the graph."""
        with tf.compat.v1.get_default_graph().as_default():
            # Create a simple pass-through model
            inputs = layers_module.Input(shape=1, name="input_layer")
            model = training_module.Model(inputs=inputs, outputs=inputs)
            model.compile(loss="mean_absolute_error", optimizer="adam")

            # Prepare sample weights iterator tensor
            sample_weights = tf.constant([[0, 0.4, 1, 1], [2, 0.4, 0.3, 1]])
            dataset = tf.data.Dataset.from_tensor_slices(sample_weights)
            sample_weights = tf.compat.v1.data.make_one_shot_iterator(
                dataset
            ).get_next()
            sample_weights = training_utils_v1.standardize_sample_weights(
                sample_weights, model.output_names
            )

            # Update model loss with sample weight tensor.
            model._compile_weights_loss_and_weighted_metrics(sample_weights)

            feeds = {
                "input_layer:0": [[0], [0], [0], [0]],
                "input_layer_target:0": [[1], [1], [1], [1]],
            }
            with self.cached_session() as sess:
                self.assertAllClose(
                    (0.4 + 1 + 1) / 4,
                    sess.run(model.total_loss, feed_dict=feeds),
                )
                self.assertAllClose(
                    (2 + 0.4 + 0.3 + 1) / 4,
                    sess.run(model.total_loss, feed_dict=feeds),
                )


@test_combinations.run_all_keras_modes
class MaskingTest(test_combinations.TestCase):
    def _get_model(self, input_shape=None):
        layers = [
            layers_module.Masking(mask_value=0),
            layers_module.TimeDistributed(
                layers_module.Dense(1, kernel_initializer="one")
            ),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape)
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=test_utils.should_run_eagerly(),
        )
        return model

    @test_combinations.run_with_all_model_types
    def test_masking(self):
        model = self._get_model(input_shape=(2, 1))
        x = np.array([[[1], [1]], [[0], [0]]])
        y = np.array([[[1], [1]], [[1], [1]]])
        loss = model.train_on_batch(x, y)
        self.assertEqual(loss, 0)

    @test_combinations.run_with_all_model_types(exclude_models="functional")
    def test_masking_deferred(self):
        model = self._get_model()
        x = np.array([[[1], [1]], [[0], [0]]])
        y = np.array([[[1], [1]], [[1], [1]]])
        loss = model.train_on_batch(x, y)
        self.assertEqual(loss, 0)

    def test_mask_argument_in_layer(self):
        # Test that the mask argument gets correctly passed to a layer in the
        # functional API.

        class CustomMaskedLayer(layers_module.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, inputs, mask=None):
                assert mask is not None
                return inputs

            def compute_output_shape(self, input_shape):
                return input_shape

        x = np.random.random((5, 3))
        inputs = layers_module.Input((3,))
        masked = layers_module.Masking(mask_value=0)(inputs)
        outputs = CustomMaskedLayer()(masked)

        model = training_module.Model(inputs, outputs)
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=test_utils.should_run_eagerly(),
        )
        y = np.random.random((5, 3))
        model.train_on_batch(x, y)


@test_combinations.run_all_keras_modes
class TestDynamicTrainability(test_combinations.TestCase):
    def test_trainable_warning(self):
        x = np.random.random((5, 3))
        y = np.random.random((5, 2))

        model = sequential.Sequential()
        model.add(layers_module.Dense(2, input_dim=3))
        model.trainable = False
        model.compile(
            "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
        )
        model.trainable = True
        model.train_on_batch(x, y)
        self.assertRaises(Warning)

    def test_trainable_argument(self):
        with self.cached_session():
            x = np.random.random((5, 3))
            y = np.random.random((5, 2))

            model = sequential.Sequential()
            model.add(layers_module.Dense(2, input_dim=3, trainable=False))
            model.compile(
                "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
            )
            out = model.predict(x)
            model.train_on_batch(x, y)
            out_2 = model.predict(x)
            self.assertAllClose(out, out_2)

            # test with nesting
            inputs = layers_module.Input(shape=(3,))
            output = model(inputs)
            model = training_module.Model(inputs, output)
            model.compile(
                "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
            )
            out = model.predict(x)
            model.train_on_batch(x, y)
            out_2 = model.predict(x)
            self.assertAllClose(out, out_2)

    def test_layer_trainability_switch(self):
        # with constructor argument, in Sequential
        model = sequential.Sequential()
        model.add(layers_module.Dense(2, trainable=False, input_dim=1))
        self.assertListEqual(model.trainable_weights, [])

        # by setting the `trainable` argument, in Sequential
        model = sequential.Sequential()
        layer = layers_module.Dense(2, input_dim=1)
        model.add(layer)
        self.assertListEqual(model.trainable_weights, layer.trainable_weights)
        layer.trainable = False
        self.assertListEqual(model.trainable_weights, [])

        # with constructor argument, in Model
        x = layers_module.Input(shape=(1,))
        y = layers_module.Dense(2, trainable=False)(x)
        model = training_module.Model(x, y)
        self.assertListEqual(model.trainable_weights, [])

        # by setting the `trainable` argument, in Model
        x = layers_module.Input(shape=(1,))
        layer = layers_module.Dense(2)
        y = layer(x)
        model = training_module.Model(x, y)
        self.assertListEqual(model.trainable_weights, layer.trainable_weights)
        layer.trainable = False
        self.assertListEqual(model.trainable_weights, [])

    def test_model_trainability_switch(self):
        # a non-trainable model has no trainable weights
        x = layers_module.Input(shape=(1,))
        y = layers_module.Dense(2)(x)
        model = training_module.Model(x, y)
        model.trainable = False
        self.assertListEqual(model.trainable_weights, [])

        # same for Sequential
        model = sequential.Sequential()
        model.add(layers_module.Dense(2, input_dim=1))
        model.trainable = False
        self.assertListEqual(model.trainable_weights, [])

    def test_nested_model_trainability(self):
        # a Sequential inside a Model
        inner_model = sequential.Sequential()
        inner_model.add(layers_module.Dense(2, input_dim=1))

        x = layers_module.Input(shape=(1,))
        y = inner_model(x)
        outer_model = training_module.Model(x, y)
        self.assertListEqual(
            outer_model.trainable_weights, inner_model.trainable_weights
        )
        inner_model.trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])
        inner_model.trainable = True
        inner_model.layers[-1].trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])

        # a Sequential inside a Sequential
        inner_model = sequential.Sequential()
        inner_model.add(layers_module.Dense(2, input_dim=1))
        outer_model = sequential.Sequential()
        outer_model.add(inner_model)
        self.assertListEqual(
            outer_model.trainable_weights, inner_model.trainable_weights
        )
        inner_model.trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])
        inner_model.trainable = True
        inner_model.layers[-1].trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])

        # a Model inside a Model
        x = layers_module.Input(shape=(1,))
        y = layers_module.Dense(2)(x)
        inner_model = training_module.Model(x, y)
        x = layers_module.Input(shape=(1,))
        y = inner_model(x)
        outer_model = training_module.Model(x, y)
        self.assertListEqual(
            outer_model.trainable_weights, inner_model.trainable_weights
        )
        inner_model.trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])
        inner_model.trainable = True
        inner_model.layers[-1].trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])

        # a Model inside a Sequential
        x = layers_module.Input(shape=(1,))
        y = layers_module.Dense(2)(x)
        inner_model = training_module.Model(x, y)
        outer_model = sequential.Sequential()
        outer_model.add(inner_model)
        self.assertListEqual(
            outer_model.trainable_weights, inner_model.trainable_weights
        )
        inner_model.trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])
        inner_model.trainable = True
        inner_model.layers[-1].trainable = False
        self.assertListEqual(outer_model.trainable_weights, [])

    def test_gan_workflow(self):
        shared_layer = layers_module.BatchNormalization()

        inputs1 = input_layer.Input(10)
        outputs1 = shared_layer(inputs1)
        model1 = training_module.Model(inputs1, outputs1)
        shared_layer.trainable = False
        model1.compile(
            "sgd", "mse", run_eagerly=test_utils.should_run_eagerly()
        )

        inputs2 = input_layer.Input(10)
        outputs2 = shared_layer(inputs2)
        model2 = training_module.Model(inputs2, outputs2)
        shared_layer.trainable = True
        model2.compile(
            "sgd", "mse", run_eagerly=test_utils.should_run_eagerly()
        )

        x, y = np.ones((10, 10)), np.ones((10, 10))

        out1_0 = model1.predict_on_batch(x)
        model1.train_on_batch(x, y)
        out1_1 = model1.predict_on_batch(x)
        self.assertAllClose(out1_0, out1_1)

        out2_0 = model2.predict_on_batch(x)
        model2.train_on_batch(x, y)
        out2_1 = model2.predict_on_batch(x)
        self.assertNotAllClose(out2_0, out2_1)

    def test_toggle_value(self):
        input_0 = layers_module.Input(shape=(1,))
        dense_0 = layers_module.Dense(
            1, kernel_initializer="ones", bias_initializer="ones"
        )
        dense_1 = layers_module.Dense(
            1, kernel_initializer="ones", bias_initializer="ones"
        )
        result = layers_module.Add()([dense_0(input_0), dense_1(input_0)])
        model = training_module.Model(input_0, result)
        dense_0.trainable = False
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())

        x = np.ones((10, 1))
        y = 5 * x + 2
        model.train_on_batch(x, y)
        dense_0.trainable = True
        model.train_on_batch(x, y)
        kernel, bias = dense_0.get_weights()
        self.assertAllEqual([kernel[0, 0], bias[0]], [1.0, 1.0])

        kernel, bias = dense_1.get_weights()
        self.assertAllClose([kernel[0, 0], bias[0]], [1.1176, 1.1176])


class TestTrainingWithDataTensors(test_combinations.TestCase):
    def test_training_and_eval_methods_on_symbolic_tensors_single_io(self):
        with tf.Graph().as_default():
            x = layers_module.Input(shape=(3,), name="input")
            y = layers_module.Dense(4, name="dense")(x)
            model = training_module.Model(x, y)

            optimizer = RMSPropOptimizer(learning_rate=0.001)
            loss = "mse"
            model.compile(
                optimizer,
                loss,
                metrics=["mae", metrics_module.CategoricalAccuracy()],
            )

            inputs = backend.zeros(shape=(10, 3))
            targets = backend.zeros(shape=(10, 4))

            model.fit(inputs, targets, epochs=1, steps_per_epoch=2, verbose=0)
            model.evaluate(inputs, targets, steps=2, verbose=0)
            model.predict(inputs, steps=2)
            model.train_on_batch(inputs, targets)
            model.test_on_batch(inputs, targets)
            model.fit(
                inputs,
                targets,
                epochs=1,
                steps_per_epoch=2,
                verbose=0,
                validation_data=(inputs, targets),
                validation_steps=2,
            )

            # Test with dynamic shape
            inputs = tf.compat.v1.placeholder_with_default(
                np.zeros((2, 3)), shape=tf.TensorShape([None, 3])
            )
            targets = tf.compat.v1.placeholder_with_default(
                np.zeros((2, 4)), shape=tf.TensorShape([None, 4])
            )
            self.assertEqual(inputs.shape.dims[0].value, None)
            model.fit(inputs, targets, epochs=1, steps_per_epoch=2, verbose=0)
            model.evaluate(inputs, targets, steps=2, verbose=0)
            model.predict(inputs, steps=2)
            model.train_on_batch(inputs, targets)
            model.test_on_batch(inputs, targets)
            model.fit(
                inputs,
                targets,
                epochs=1,
                steps_per_epoch=2,
                verbose=0,
                validation_data=(inputs, targets),
                validation_steps=2,
            )

    def test_training_and_eval_methods_on_symbolic_tensors_multi_io(self):
        a = layers_module.Input(shape=(3,), name="input_a")
        b = layers_module.Input(shape=(3,), name="input_b")

        dense = layers_module.Dense(4, name="dense")
        c = dense(a)
        d = dense(b)
        e = layers_module.Dropout(0.5, name="dropout")(c)

        model = training_module.Model([a, b], [d, e])

        optimizer = "rmsprop"
        loss = "mse"
        loss_weights = [1.0, 0.5]
        model.compile(
            optimizer,
            loss,
            metrics=["mae", metrics_module.CategoricalAccuracy()],
            loss_weights=loss_weights,
        )

        input_a_tf = tf.zeros(shape=(10, 3))
        input_b_tf = tf.zeros(shape=(10, 3))

        output_d_tf = tf.zeros(shape=(10, 4))
        output_e_tf = tf.zeros(shape=(10, 4))

        model.fit(
            [input_a_tf, input_b_tf],
            [output_d_tf, output_e_tf],
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
        )
        model.train_on_batch(
            [input_a_tf, input_b_tf], [output_d_tf, output_e_tf]
        )

        # Test with dictionary inputs
        model.fit(
            {"input_a": input_a_tf, "input_b": input_b_tf},
            {"dense": output_d_tf, "dropout": output_e_tf},
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
        )
        model.fit(
            {"input_a": input_a_tf, "input_b": input_b_tf},
            {"dense": output_d_tf, "dropout": output_e_tf},
            validation_data=(
                {"input_a": input_a_tf, "input_b": input_b_tf},
                {"dense": output_d_tf, "dropout": output_e_tf},
            ),
            epochs=1,
            steps_per_epoch=2,
            validation_steps=2,
            verbose=0,
        )
        model.train_on_batch(
            {"input_a": input_a_tf, "input_b": input_b_tf},
            {"dense": output_d_tf, "dropout": output_e_tf},
        )

        # Test with validation data
        model.fit(
            [input_a_tf, input_b_tf],
            [output_d_tf, output_e_tf],
            validation_data=(
                [input_a_tf, input_b_tf],
                [output_d_tf, output_e_tf],
            ),
            epochs=1,
            steps_per_epoch=2,
            validation_steps=2,
            verbose=0,
        )
        # Test evaluation / prediction methods
        model.evaluate(
            [input_a_tf, input_b_tf],
            [output_d_tf, output_e_tf],
            steps=2,
            verbose=0,
        )
        model.predict([input_a_tf, input_b_tf], steps=2)
        model.test_on_batch(
            [input_a_tf, input_b_tf], [output_d_tf, output_e_tf]
        )

    @tf_test_utils.run_deprecated_v1
    def test_model_with_input_feed_tensor(self):
        """We test building a model with a TF variable as input.

        We should be able to call fit, evaluate, predict,
        by only passing them data for the placeholder inputs
        in the model.
        """
        with tf.Graph().as_default(), self.cached_session():
            input_a_np = np.random.random((10, 3))
            input_b_np = np.random.random((10, 3))

            output_a_np = np.random.random((10, 4))
            output_b_np = np.random.random((10, 3))

            input_v = tf.Variable(input_a_np, dtype="float32")
            self.evaluate(tf.compat.v1.variables_initializer([input_v]))
            a = input_layer.Input(tensor=input_v)
            b = input_layer.Input(shape=(3,), name="input_b")

            a_2 = layers_module.Dense(4, name="dense_1")(a)
            dp = layers_module.Dropout(0.5, name="dropout")
            b_2 = dp(b)

            model = training_module.Model([a, b], [a_2, b_2])
            model.summary()

            optimizer = "rmsprop"
            loss = "mse"
            loss_weights = [1.0, 0.5]
            model.compile(
                optimizer,
                loss,
                metrics=["mean_squared_error"],
                loss_weights=loss_weights,
                sample_weight_mode=None,
            )

            # test train_on_batch
            out = model.train_on_batch(input_b_np, [output_a_np, output_b_np])
            out = model.train_on_batch(
                {"input_b": input_b_np}, [output_a_np, output_b_np]
            )
            out = model.test_on_batch(
                {"input_b": input_b_np}, [output_a_np, output_b_np]
            )
            out = model.predict_on_batch({"input_b": input_b_np})

            # test fit
            out = model.fit(
                {"input_b": input_b_np},
                [output_a_np, output_b_np],
                epochs=1,
                batch_size=10,
            )
            out = model.fit(
                input_b_np, [output_a_np, output_b_np], epochs=1, batch_size=10
            )

            # test evaluate
            out = model.evaluate(
                {"input_b": input_b_np},
                [output_a_np, output_b_np],
                batch_size=10,
            )
            out = model.evaluate(
                input_b_np, [output_a_np, output_b_np], batch_size=10
            )

            # test predict
            out = model.predict({"input_b": input_b_np}, batch_size=10)
            out = model.predict(input_b_np, batch_size=10)
            self.assertEqual(len(out), 2)

            # Now test a model with a single input
            # i.e. we don't pass any data to fit the model.
            self.evaluate(tf.compat.v1.variables_initializer([input_v]))
            a = input_layer.Input(tensor=input_v)
            a_2 = layers_module.Dense(4, name="dense_1")(a)
            a_2 = layers_module.Dropout(0.5, name="dropout")(a_2)
            model = training_module.Model(a, a_2)
            model.summary()

            optimizer = "rmsprop"
            loss = "mse"
            model.compile(optimizer, loss, metrics=["mean_squared_error"])

            # test train_on_batch
            out = model.train_on_batch(None, output_a_np)
            out = model.train_on_batch(None, output_a_np)
            out = model.test_on_batch(None, output_a_np)
            out = model.predict_on_batch(None)
            out = model.train_on_batch([], output_a_np)
            out = model.train_on_batch({}, output_a_np)

            # test fit
            _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=3)
            _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=3)

            # test evaluate
            _ = model.evaluate(None, output_a_np, steps=3)
            _ = model.evaluate(None, output_a_np, steps=3)

            # test predict
            out = model.predict(None, steps=3)
            out = model.predict(None, steps=3)
            self.assertEqual(out.shape, (10 * 3, 4))

            # Same, without learning phase
            # i.e. we don't pass any data to fit the model.
            self.evaluate(tf.compat.v1.variables_initializer([input_v]))
            a = input_layer.Input(tensor=input_v)
            a_2 = layers_module.Dense(4, name="dense_1")(a)
            model = training_module.Model(a, a_2)
            model.summary()

            optimizer = "rmsprop"
            loss = "mse"
            model.compile(optimizer, loss, metrics=["mean_squared_error"])

            # test train_on_batch
            out = model.train_on_batch(None, output_a_np)
            out = model.train_on_batch(None, output_a_np)
            out = model.test_on_batch(None, output_a_np)
            out = model.predict_on_batch(None)
            out = model.train_on_batch([], output_a_np)
            out = model.train_on_batch({}, output_a_np)

            # test fit
            _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=10)
            _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=10)

            # test evaluate
            _ = model.evaluate(None, output_a_np, steps=10)
            _ = model.evaluate(None, output_a_np, steps=10)

            # test predict
            out = model.predict(None, steps=3)
            out = model.predict(None, steps=3)
            self.assertEqual(out.shape, (10 * 3, 4))

    @test_combinations.run_all_keras_modes
    def test_model_with_partial_loss(self):
        with self.cached_session():
            a = input_layer.Input(shape=(3,), name="input_a")
            a_2 = layers_module.Dense(4, name="dense_1")(a)
            dp = layers_module.Dropout(0.5, name="dropout")
            a_3 = dp(a_2)
            model = training_module.Model(a, [a_2, a_3])

            optimizer = "rmsprop"
            loss = {"dropout": "mse"}
            model.compile(optimizer, loss, metrics=["mae"])

            input_a_np = np.random.random((10, 3))
            output_a_np = np.random.random((10, 4))

            # test train_on_batch
            _ = model.train_on_batch(input_a_np, output_a_np)
            _ = model.test_on_batch(input_a_np, output_a_np)
            # fit
            _ = model.fit(input_a_np, output_a_np)
            # evaluate
            _ = model.evaluate(input_a_np, output_a_np)

            # Same without dropout.
            a = input_layer.Input(shape=(3,), name="input_a")
            a_2 = layers_module.Dense(4, name="dense_1")(a)
            a_3 = layers_module.Dense(4, name="dense_2")(a_2)
            model = training_module.Model(a, [a_2, a_3])

            optimizer = "rmsprop"
            loss = {"dense_2": "mse"}
            model.compile(optimizer, loss, metrics={"dense_1": "mae"})

            # test train_on_batch
            _ = model.train_on_batch(input_a_np, output_a_np)
            _ = model.test_on_batch(input_a_np, output_a_np)
            # fit
            _ = model.fit(input_a_np, output_a_np)
            # evaluate
            _ = model.evaluate(input_a_np, output_a_np)

    def test_model_with_external_loss(self):
        with tf.Graph().as_default(), self.cached_session():
            # None loss, only regularization loss.
            a = input_layer.Input(shape=(3,), name="input_a")
            a_2 = layers_module.Dense(
                4,
                name="dense_1",
                kernel_regularizer="l1",
                bias_regularizer="l2",
            )(a)
            dp = layers_module.Dropout(0.5, name="dropout")
            a_3 = dp(a_2)

            model = training_module.Model(a, [a_2, a_3])

            optimizer = "rmsprop"
            loss = None
            model.compile(optimizer, loss, metrics=["mae"])

            input_a_np = np.random.random((10, 3))

            # test train_on_batch
            out = model.train_on_batch(input_a_np, None)
            out = model.test_on_batch(input_a_np, None)
            # fit
            out = model.fit(input_a_np, None)
            # evaluate
            out = model.evaluate(input_a_np, None)

            # No dropout, external loss.
            a = input_layer.Input(shape=(3,), name="input_a")
            a_2 = layers_module.Dense(4, name="dense_1")(a)
            a_3 = layers_module.Dense(4, name="dense_2")(a)

            model = training_module.Model(a, [a_2, a_3])
            model.add_loss(backend.mean(a_3 + a_2))

            optimizer = "rmsprop"
            loss = None
            model.compile(optimizer, loss, metrics=["mae"])

            # test train_on_batch
            out = model.train_on_batch(input_a_np, None)
            out = model.test_on_batch(input_a_np, None)
            # fit
            out = model.fit(input_a_np, None)
            # evaluate
            out = model.evaluate(input_a_np, None)

            # Test model with no external data at all.
            input_v = tf.Variable(input_a_np, dtype="float32")
            self.evaluate(tf.compat.v1.variables_initializer([input_v]))
            a = input_layer.Input(tensor=input_v)
            a_2 = layers_module.Dense(4, name="dense_1")(a)
            a_2 = layers_module.Dropout(0.5, name="dropout")(a_2)
            model = training_module.Model(a, a_2)
            model.add_loss(backend.mean(a_2))

            model.compile(
                optimizer="rmsprop", loss=None, metrics=["mean_squared_error"]
            )

            # test train_on_batch
            out = model.train_on_batch(None, None)
            out = model.test_on_batch(None, None)
            out = model.predict_on_batch(None)

            # Test multi-output model with no external data at all.
            self.evaluate(tf.compat.v1.variables_initializer([input_v]))
            a = input_layer.Input(tensor=input_v)
            a_1 = layers_module.Dense(4, name="dense_1")(a)
            a_2 = layers_module.Dropout(0.5, name="dropout")(a_1)
            model = training_module.Model(a, [a_1, a_2])
            model.add_loss(backend.mean(a_2))

            model.compile(
                optimizer="rmsprop", loss=None, metrics=["mean_squared_error"]
            )

            # test train_on_batch
            out = model.train_on_batch(None, None)
            out = model.test_on_batch(None, None)
            out = model.predict_on_batch(None)

            out = model.predict(None, steps=3)
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0].shape, (10 * 3, 4))
            self.assertEqual(out[1].shape, (10 * 3, 4))

    def test_target_tensors(self):
        with tf.Graph().as_default(), self.cached_session():
            # single-output, as list
            model = sequential.Sequential()
            model.add(layers_module.Dense(4, input_shape=(4,), name="dense"))
            input_val = np.random.random((10, 4))
            target_val = np.random.random((10, 4))
            target = backend.variable(target_val)
            model.compile(
                optimizer="rmsprop", loss="mse", target_tensors=[target]
            )
            model.train_on_batch(input_val, None)

            # single-output, as single tensor
            model.compile(
                optimizer="rmsprop", loss="mse", target_tensors=target
            )
            model.train_on_batch(input_val, None)

            # single-output, as dict
            model.compile(
                optimizer="rmsprop",
                loss="mse",
                target_tensors={"dense": target},
            )
            model.train_on_batch(input_val, None)

            # test invalid arguments
            with self.assertRaises(TypeError):
                model.compile(
                    optimizer="rmsprop", loss="mse", target_tensors=set()
                )
            with self.assertRaises(ValueError):
                model.compile(
                    optimizer="rmsprop",
                    loss="mse",
                    target_tensors=[target, target],
                )
            with self.assertRaises(ValueError):
                model.compile(
                    optimizer="rmsprop",
                    loss="mse",
                    target_tensors={"dense2": None},
                )
            with self.assertRaises(ValueError):
                model.compile(
                    optimizer="rmsprop", loss="mse", target_tensors=[target]
                )
                model.train_on_batch(input_val, target_val)

            # multi-output, as list
            input_val = np.random.random((10, 4))
            target_val_a = np.random.random((10, 4))
            target_val_b = np.random.random((10, 4))
            target_a = backend.variable(target_val_a)
            target_b = backend.variable(target_val_b)

            inputs = layers_module.Input(shape=(4,))
            output_a = layers_module.Dense(4, name="dense_a")(inputs)
            output_b = layers_module.Dense(4, name="dense_b")(inputs)
            model = training_module.Model(inputs, [output_a, output_b])
            model.compile(
                optimizer="rmsprop",
                loss="mse",
                target_tensors=[target_a, target_b],
            )
            model.train_on_batch(input_val, None)

            # multi-output, as dict
            model.compile(
                optimizer="rmsprop",
                loss="mse",
                target_tensors={"dense_a": target_a, "dense_b": target_b},
            )
            model.train_on_batch(input_val, None)

            # test with sample weights
            model.compile(
                optimizer="rmsprop",
                loss="mse",
                metrics=["mae", metrics_module.CategoricalAccuracy()],
                target_tensors=[target_a, target_b],
            )
            model.train_on_batch(
                input_val,
                None,
                sample_weight={"dense_a": np.random.random((10,))},
            )

    def test_model_custom_target_tensors(self):
        with tf.Graph().as_default(), self.cached_session():
            a = input_layer.Input(shape=(3,), name="input_a")
            b = input_layer.Input(shape=(3,), name="input_b")

            a_2 = layers_module.Dense(4, name="dense_1")(a)
            dp = layers_module.Dropout(0.5, name="dropout")
            b_2 = dp(b)

            y = backend.placeholder([10, 4], name="y")
            y1 = backend.placeholder([10, 3], name="y1")
            y2 = backend.placeholder([7, 5], name="y2")
            model = training_module.Model([a, b], [a_2, b_2])

            optimizer = "rmsprop"
            loss = "mse"
            loss_weights = [1.0, 0.5]

            # test list of target tensors
            with self.assertRaises(ValueError):
                model.compile(
                    optimizer,
                    loss,
                    metrics=[],
                    loss_weights=loss_weights,
                    sample_weight_mode=None,
                    target_tensors=[y, y1, y2],
                )
            model.compile(
                optimizer,
                loss,
                metrics=[],
                loss_weights=loss_weights,
                sample_weight_mode=None,
                target_tensors=[y, y1],
            )
            input_a_np = np.random.random((10, 3))
            input_b_np = np.random.random((10, 3))

            output_a_np = np.random.random((10, 4))
            output_b_np = np.random.random((10, 3))

            _ = model.train_on_batch(
                [input_a_np, input_b_np],
                [output_a_np, output_b_np],
                {
                    "dense_1": np.random.random((10,)),
                    "dropout": np.random.random((10,)),
                },
            )
            # test dictionary of target_tensors
            with self.assertRaises(ValueError):
                model.compile(
                    optimizer,
                    loss,
                    metrics=[],
                    loss_weights=loss_weights,
                    sample_weight_mode=None,
                    target_tensors={"does_not_exist": y2},
                )
            # test dictionary of target_tensors
            model.compile(
                optimizer,
                loss,
                metrics=[],
                loss_weights=loss_weights,
                sample_weight_mode=None,
                target_tensors={"dense_1": y, "dropout": y1},
            )
            _ = model.train_on_batch(
                [input_a_np, input_b_np],
                [output_a_np, output_b_np],
                {
                    "dense_1": np.random.random((10,)),
                    "dropout": np.random.random((10,)),
                },
            )

            # test with custom TF placeholder as target
            pl_target_a = tf.compat.v1.placeholder("float32", shape=(None, 4))
            model.compile(
                optimizer="rmsprop",
                loss="mse",
                target_tensors={"dense_1": pl_target_a},
            )
            model.train_on_batch(
                [input_a_np, input_b_np], [output_a_np, output_b_np]
            )


class TestTrainingWithMetrics(test_combinations.TestCase):
    """Training tests related to metrics."""

    @test_combinations.run_all_keras_modes
    def test_metrics_names(self):
        a = layers_module.Input(shape=(3,), name="input_a")
        b = layers_module.Input(shape=(3,), name="input_b")

        dense = layers_module.Dense(4, name="dense")
        c = dense(a)
        d = dense(b)
        e = layers_module.Dropout(0.5, name="dropout")(c)

        model = training_module.Model([a, b], [d, e])

        optimizer = RMSPropOptimizer(learning_rate=0.001)
        metrics = ["mse", metrics_module.BinaryAccuracy()]
        model.compile(
            optimizer,
            loss="mae",
            metrics=metrics,
            run_eagerly=test_utils.should_run_eagerly(),
        )

        mse_metric = "mse" if tf.executing_eagerly() else "mean_squared_error"
        reference_metric_names = [
            "loss",
            "dense_loss",
            "dropout_loss",
            "dense_" + mse_metric,
            "dense_binary_accuracy",
            "dropout_" + mse_metric,
            "dropout_binary_accuracy",
        ]

        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))

        output_d_np = np.random.random((10, 4))
        output_e_np = np.random.random((10, 4))

        model.fit(
            [input_a_np, input_b_np],
            [output_d_np, output_e_np],
            epochs=1,
            batch_size=5,
        )
        self.assertEqual(reference_metric_names, model.metrics_names)

    @test_combinations.run_all_keras_modes
    def test_metric_state_reset_between_fit_and_evaluate(self):
        model = sequential.Sequential()
        model.add(layers_module.Dense(3, activation="relu", input_dim=4))
        model.add(layers_module.Dense(1, activation="sigmoid"))
        acc_obj = metrics_module.BinaryAccuracy()
        model.compile(
            loss="mae",
            metrics=[acc_obj],
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x_train = np.random.random((100, 4))
        y_train = np.random.random((100, 1))
        model.fit(x_train, y_train, batch_size=5, epochs=2)
        self.assertEqual(self.evaluate(acc_obj.count), 100)

        x_test = np.random.random((10, 4))
        y_test = np.random.random((10, 1))
        model.evaluate(x_test, y_test, batch_size=5)
        self.assertEqual(self.evaluate(acc_obj.count), 10)

    @test_combinations.run_all_keras_modes
    def test_metric_state_reset_between_test_on_batch_and_evaluate(self):
        model = sequential.Sequential()
        model.add(layers_module.Dense(3, activation="relu", input_dim=4))
        model.add(layers_module.Dense(1, activation="sigmoid"))
        acc_obj = metrics_module.BinaryAccuracy()
        model.compile(
            loss="mae",
            metrics=[acc_obj],
            optimizer=RMSPropOptimizer(learning_rate=0.001),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x_test = np.random.random((10, 4))
        y_test = np.random.random((10, 1))
        loss, acc = model.test_on_batch(x_test[:2], y_test[:2])
        loss_eval, acc_eval = model.evaluate(x_test, y_test)
        loss_1, acc_1 = model.test_on_batch(x_test[:2], y_test[:2])
        loss_eval_1, acc_eval_1 = model.evaluate(x_test, y_test)
        self.assertEqual(loss, loss_1)
        self.assertEqual(acc, acc_1)
        self.assertEqual(loss_eval, loss_eval_1)
        self.assertEqual(acc_eval, acc_eval_1)

    @test_combinations.run_with_all_model_types(exclude_models=["sequential"])
    @test_combinations.run_all_keras_modes
    def test_metrics_valid_compile_input_formats(self):
        inp_1 = layers_module.Input(shape=(1,), name="input_1")
        inp_2 = layers_module.Input(shape=(1,), name="input_2")
        x = layers_module.Dense(3, kernel_initializer="ones", trainable=False)
        out_1 = layers_module.Dense(
            1, kernel_initializer="ones", name="output_1", trainable=False
        )
        out_2 = layers_module.Dense(
            1, kernel_initializer="ones", name="output_2", trainable=False
        )

        branch_a = [inp_1, x, out_1]
        branch_b = [inp_2, x, out_2]
        model = test_utils.get_multi_io_model(branch_a, branch_b)

        # list of metrics.
        model.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=[metrics_module.MeanSquaredError()],
            weighted_metrics=[metrics_module.MeanSquaredError()],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        # list of list of metrics.
        model.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=[
                metrics_module.MeanSquaredError(),
                [metrics_module.MeanSquaredError(), metrics_module.Accuracy()],
            ],
            weighted_metrics=[
                metrics_module.MeanSquaredError(),
                [metrics_module.MeanSquaredError(), metrics_module.Accuracy()],
            ],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        # dict of metrics.
        model.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics={
                "output_1": metrics_module.MeanSquaredError(),
                "output_2": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.Accuracy(),
                ],
            },
            weighted_metrics={
                "output_1": metrics_module.MeanSquaredError(),
                "output_2": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.Accuracy(),
                ],
            },
            run_eagerly=test_utils.should_run_eagerly(),
        )

    @test_combinations.run_all_keras_modes
    def test_metrics_masking(self):
        np.random.seed(1337)
        model = sequential.Sequential()
        model.add(layers_module.Masking(mask_value=0, input_shape=(2, 1)))
        model.add(
            layers_module.TimeDistributed(
                layers_module.Dense(1, kernel_initializer="ones")
            )
        )
        model.compile(
            RMSPropOptimizer(learning_rate=0.001),
            loss="mse",
            weighted_metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        # verify that masking is applied.
        x = np.array(
            # third row is masked
            [[[1], [1]], [[1], [1]], [[0], [0]]]
        )
        y = np.array([[[1], [1]], [[0], [1]], [[1], [1]]])

        scores = model.test_on_batch(x, y)
        self.assertArrayNear(scores, [0.25, 0.75], 0.0001)

        # verify that masking is combined with sample weights.
        w = np.array([3, 2, 4])
        scores = model.test_on_batch(x, y, sample_weight=w)
        self.assertArrayNear(scores, [0.5, 0.8], 0.0001)

        scores = model.train_on_batch(x, y)
        self.assertArrayNear(scores, [0.25, 0.75], 0.0001)

        scores = model.train_on_batch(x, y, sample_weight=w)
        self.assertArrayNear(scores, [0.5 - 0.001037, 0.8], 0.0001)

    @test_combinations.run_all_keras_modes
    def test_add_metric_with_tensor_on_model(self):
        x = layers_module.Input(shape=(1,))
        y = layers_module.Dense(1, kernel_initializer="ones")(x)
        model = training_module.Model(x, y)
        model.add_metric(tf.reduce_sum(y), name="metric_1", aggregation="mean")

        if tf.executing_eagerly():
            # This is not a use case in v1 graph mode.
            mean_result = metrics_module.Mean()(y)
            with self.assertRaisesRegex(
                ValueError, "Expected a symbolic Tensor for the metric value"
            ):
                model.add_metric(mean_result, name="metric_2")
        else:
            with self.assertRaisesRegex(
                ValueError, "Using the result of calling a `Metric` object "
            ):
                with backend.get_graph().as_default():
                    model.add_metric(metrics_module.Mean(name="metric_2")(y))

        model.compile(
            "sgd", loss="mse", run_eagerly=test_utils.should_run_eagerly()
        )

        inputs = np.ones(shape=(10, 1))
        targets = np.ones(shape=(10, 1))
        history = model.fit(
            inputs,
            targets,
            epochs=2,
            batch_size=5,
            validation_data=(inputs, targets),
        )
        self.assertEqual(history.history["metric_1"][-1], 5)
        self.assertEqual(history.history["val_metric_1"][-1], 5)

        eval_results = model.evaluate(inputs, targets, batch_size=5)
        self.assertEqual(eval_results[-1], 5)

        model.predict(inputs, batch_size=5)
        model.train_on_batch(inputs, targets)
        model.test_on_batch(inputs, targets)

    @test_combinations.run_all_keras_modes
    def test_add_metric_in_model_call(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")
                self.mean = metrics_module.Mean(name="metric_1")

            def call(self, x):
                self.add_metric(
                    tf.reduce_sum(x), name="metric_2", aggregation="mean"
                )
                # Provide same name as in the instance created in __init__
                # for eager mode
                self.add_metric(self.mean(x), name="metric_1")
                return self.dense1(x)

        model = TestModel()
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))
        history = model.fit(
            x, y, epochs=2, batch_size=5, validation_data=(x, y)
        )
        self.assertAlmostEqual(history.history["metric_1"][-1], 1, 0)
        self.assertAlmostEqual(history.history["val_metric_1"][-1], 1, 0)
        self.assertAlmostEqual(history.history["metric_2"][-1], 5, 0)
        self.assertAlmostEqual(history.history["val_metric_2"][-1], 5, 0)

        eval_results = model.evaluate(x, y, batch_size=5)
        self.assertAlmostEqual(eval_results[1], 1, 0)
        self.assertAlmostEqual(eval_results[2], 5, 0)

        model.predict(x, batch_size=5)
        model.train_on_batch(x, y)
        model.test_on_batch(x, y)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_add_metric_in_layer_call(self):
        class TestLayer(layers_module.Layer):
            def build(self, input_shape):
                self.a = self.add_weight(
                    "a", (1, 1), initializer="ones", trainable=False
                )
                self.built = True

            def call(self, inputs):
                self.add_metric(
                    tf.reduce_sum(inputs), name="metric_1", aggregation="mean"
                )
                return inputs + 1

        layers = [
            TestLayer(input_shape=(1,)),
            layers_module.Dense(2, kernel_initializer="ones"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(1,))
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))
        history = model.fit(
            x, y, epochs=2, batch_size=5, validation_data=(x, y)
        )
        self.assertEqual(history.history["metric_1"][-1], 5)
        self.assertAlmostEqual(history.history["val_metric_1"][-1], 5, 0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_model_metrics_list(self):
        class LayerWithAddMetric(layers_module.Layer):
            def __init__(self):
                super().__init__()
                self.dense = layers_module.Dense(1, kernel_initializer="ones")

            def __call__(self, inputs):
                outputs = self.dense(inputs)
                self.add_metric(
                    tf.reduce_sum(outputs), name="metric_1", aggregation="mean"
                )
                return outputs

        class LayerWithNestedAddMetricLayer(layers_module.Layer):
            def __init__(self):
                super().__init__()
                self.layer = LayerWithAddMetric()

            def call(self, inputs):
                outputs = self.layer(inputs)
                self.add_metric(
                    tf.reduce_sum(outputs), name="metric_2", aggregation="mean"
                )
                return outputs

        x = layers_module.Input(shape=(1,))
        y = LayerWithNestedAddMetricLayer()(x)

        model = training_module.Model(x, y)
        model.add_metric(tf.reduce_sum(y), name="metric_3", aggregation="mean")

        if tf.executing_eagerly():
            # This is not a use case in v1 graph mode.
            mean_result = metrics_module.Mean()(y)
            with self.assertRaisesRegex(
                ValueError, "Expected a symbolic Tensor for the metric value"
            ):
                model.add_metric(mean_result, name="metric_4")

        else:
            with self.assertRaisesRegex(
                ValueError, "Using the result of calling a `Metric` object "
            ):
                with backend.get_graph().as_default():
                    model.add_metric(metrics_module.Mean(name="metric_4")(y))

        model.compile(
            "sgd",
            loss="mse",
            metrics=[metrics_module.Accuracy("metric_4")],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)

        # Verify that the metrics added using `compile` and `add_metric` API are
        # included
        self.assertEqual(
            [m.name for m in model.metrics],
            ["loss", "metric_4", "metric_2", "metric_1", "metric_3"],
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_model_metrics_list_in_call(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")

            def call(self, x):
                self.add_metric(
                    tf.reduce_sum(x), name="metric_1", aggregation="mean"
                )
                return self.dense1(x)

        model = TestModel()
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            metrics=[metrics_module.Accuracy("acc")],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))
        model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

        self.assertEqual(
            [m.name for m in model.metrics], ["loss", "acc", "metric_1"]
        )

    @test_combinations.run_all_keras_modes
    def test_multiple_add_metric_calls(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")
                self.mean1 = metrics_module.Mean(name="metric_1")
                self.mean2 = metrics_module.Mean(name="metric_2")

            def call(self, x):
                self.add_metric(self.mean2(x), name="metric_2")
                self.add_metric(self.mean1(x), name="metric_1")
                self.add_metric(
                    tf.reduce_sum(x), name="metric_3", aggregation="mean"
                )
                return self.dense1(x)

        model = TestModel()
        self.assertListEqual(
            [m.name for m in model.metrics], ["metric_1", "metric_2"]
        )
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))
        history = model.fit(
            x, y, epochs=2, batch_size=5, validation_data=(x, y)
        )
        self.assertAlmostEqual(history.history["metric_1"][-1], 1, 0)
        self.assertAlmostEqual(history.history["metric_2"][-1], 1, 0)
        self.assertAlmostEqual(history.history["metric_3"][-1], 5, 0)

        eval_results = model.evaluate(x, y, batch_size=5)
        self.assertArrayNear(eval_results[1:4], [1, 1, 5], 0.1)

        model.predict(x, batch_size=5)
        model.train_on_batch(x, y)
        model.test_on_batch(x, y)

    @test_combinations.run_all_keras_modes
    def test_multiple_add_metric_calls_layer(self):
        class TestLayer(layers_module.Layer):
            def __init__(self):
                super().__init__(name="test_layer")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")
                self.m1 = metrics_module.Mean(name="m_1")
                self.m2 = [
                    metrics_module.Mean(name="m_2"),
                    metrics_module.Mean(name="m_3"),
                ]
                self.m3 = {
                    "mean4": metrics_module.Mean(name="m_4"),
                    "mean5": metrics_module.Mean(name="m_5"),
                }

            def call(self, x):
                self.add_metric(self.m2[0](x))
                self.add_metric(self.m2[1](x))
                self.add_metric(self.m1(x))
                self.add_metric(self.m3["mean4"](x))
                self.add_metric(self.m3["mean5"](x))
                self.add_metric(
                    tf.reduce_sum(x), name="m_6", aggregation="mean"
                )
                return self.dense1(x)

        layer = TestLayer()
        self.assertListEqual(
            [m.name for m in layer.metrics], ["m_1", "m_2", "m_3", "m_4", "m_5"]
        )

        layer(np.ones((10, 10)))
        self.assertListEqual(
            [m.name for m in layer.metrics],
            ["m_1", "m_2", "m_3", "m_4", "m_5", "m_6"],
        )

    @test_combinations.run_all_keras_modes
    def test_duplicate_metric_name_in_add_metric(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")
                self.mean = metrics_module.Mean(name="metric_1")
                self.mean2 = metrics_module.Mean(name="metric_1")

            def call(self, x):
                self.add_metric(self.mean(x), name="metric_1")
                return self.dense1(x)

        model = TestModel()
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))
        with self.assertRaisesRegex(
            ValueError,
            "Please provide different names for the metrics you have added. "
            'We found 2 metrics with the name: "metric_1"',
        ):
            model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

    @test_combinations.run_all_keras_modes
    def test_add_metric_without_name(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")

            def call(self, x):
                self.add_metric(tf.reduce_sum(x), aggregation="mean")
                return self.dense1(x)

        model = TestModel()
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))

        with self.assertRaisesRegex(
            ValueError, "Please provide a name for your metric like"
        ):
            model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

    @test_combinations.run_all_keras_modes
    def test_add_metric_correctness(self):
        inputs = input_layer.Input(shape=(1,))
        targets = input_layer.Input(shape=(1,))

        class Bias(layers_module.Layer):
            def build(self, input_shape):
                self.bias = self.add_weight("bias", (1,), initializer="zeros")
                self.mae = metrics_module.MeanAbsoluteError(name="mae_1")

            def call(self, inputs):
                inputs, targets = inputs
                outputs = inputs + self.bias
                self.add_metric(self.mae(targets, outputs), name="mae_1")
                return outputs

        outputs = Bias()([inputs, targets])
        model = training_module.Model([inputs, targets], outputs)

        model.add_metric(
            metrics_module.mean_absolute_error(targets, outputs),
            name="mae_2",
            aggregation="mean",
        )

        model.compile(
            loss="mae",
            optimizer=optimizer_legacy.gradient_descent.SGD(0.1),
            metrics=[metrics_module.MeanAbsoluteError(name="mae_3")],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([[0.5], [2.0], [3.5]])
        history = model.fit([x, y], y, batch_size=3, epochs=5)

        expected_val = [1.0, 0.9, 0.8, 0.7, 0.6]
        for key in ["loss", "mae_1", "mae_2", "mae_3"]:
            self.assertAllClose(history.history[key], expected_val, 1e-3)

    @test_combinations.run_all_keras_modes
    def test_add_metric_order(self):
        class MyLayer(layers_module.Layer):
            def call(self, inputs, training=None, mask=None):
                self.add_metric(
                    tf.ones([32]) * 2.0, name="two", aggregation="mean"
                )
                return inputs

        class MyModel(training_module.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._sampler = MyLayer(name="sampler")

            def call(self, inputs, training=None, mask=None):
                z = self._sampler(inputs)
                self.add_metric(
                    tf.ones([32]) * 1.0, name="one", aggregation="mean"
                )
                self.add_metric(
                    tf.ones([32]) * 3.0, name="three", aggregation="mean"
                )
                return z

        xdata = np.random.uniform(size=[32, 16]).astype(np.float32)
        dataset_train = tf.data.Dataset.from_tensor_slices((xdata, xdata))
        dataset_train = dataset_train.batch(32, drop_remainder=True)

        model = MyModel()
        model.compile(
            optimizer="sgd",
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        history = model.fit(dataset_train, epochs=3)
        self.assertDictEqual(
            history.history,
            {
                "loss": [0.0, 0.0, 0.0],
                "three": [3.0, 3.0, 3.0],
                "two": [2.0, 2.0, 2.0],
                "one": [1.0, 1.0, 1.0],
            },
        )

    @test_combinations.run_all_keras_modes
    def test_add_metric_aggregation_mean(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")

            def call(self, x):
                self.add_metric(
                    tf.reduce_sum(x), name="metric_1", aggregation="mean"
                )
                return self.dense1(x)

        model = TestModel()
        model.compile(
            "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
        )
        model.fit(np.ones(shape=(10, 1)), np.ones(shape=(10, 2)), batch_size=5)

    @test_combinations.run_all_keras_modes
    def test_add_metric_aggregation_none(self):
        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")
                self.mean = metrics_module.Mean(name="metric_1")

            def call(self, x):
                self.add_metric(self.mean(x), name="metric_1", aggregation=None)
                return self.dense1(x)

        model = TestModel()
        model.compile(
            "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
        )
        model.fit(np.ones(shape=(10, 1)), np.ones(shape=(10, 2)), batch_size=5)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def DISABLED_test_add_metric_invalid_aggregation(self):
        # TODO(psv): Re-enable test once it is fixed.
        x = layers_module.Input(shape=(1,))
        y = layers_module.Dense(1, kernel_initializer="ones")(x)
        model = training_module.Model(x, y)
        with self.assertRaisesRegex(
            ValueError, "only `mean` sample-wise metric aggregation"
        ):
            model.add_metric(
                tf.reduce_sum(y), name="metric_1", aggregation="sum"
            )

        with self.assertRaisesRegex(
            ValueError, "only `mean` sample-wise metric aggregation"
        ):
            model.add_metric(
                tf.reduce_sum(y), name="metric_1", aggregation=None
            )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_calling_evaluate_in_callback_during_fit(self):
        # Check fix for a bug that caused `evaluate` to hit a cached dataset
        # when run from inside a fit callback.
        x = layers_module.Input(shape=(2,))
        y = layers_module.Dense(2, kernel_initializer="ones", use_bias=False)(x)
        model = training_module.Model(x, y)

        ones = np.ones((10, 2), dtype=np.float32)
        zeros = np.zeros((10, 2), dtype=np.float32)
        train_ds = tf.data.Dataset.from_tensor_slices((ones, ones)).batch(5)
        val_ds_1 = tf.data.Dataset.from_tensor_slices((ones, ones)).batch(5)
        val_ds_2 = tf.data.Dataset.from_tensor_slices((zeros, zeros)).batch(5)
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())

        class MyCallback(Callback):
            def on_epoch_end(self, *args, **kwargs):
                eval_result = self.model.evaluate(val_ds_2)
                if abs(eval_result) > 1e-7:
                    raise AssertionError(
                        "Expected to hit the zeros dataset but got high loss "
                        "value of %s" % eval_result
                    )

        history = model.fit(
            train_ds, validation_data=val_ds_1, callbacks=[MyCallback()]
        )
        # Evaluate at the end of fit should hit the ones dataset (cached)
        self.assertGreater(abs(history.history["val_loss"][-1]), 0.1)
        # Standalone call to evaluate should not hit the cached dataset
        eval_result = model.evaluate(val_ds_2)
        self.assertLess(abs(eval_result), 1e-7)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_model_with_nested_compiled_model(self):
        class LayerWithAddMetric(layers_module.Layer):
            def __init__(self):
                super().__init__()
                self.dense = layers_module.Dense(1, kernel_initializer="ones")

            def call(self, inputs):
                outputs = self.dense(inputs)
                self.add_metric(
                    tf.reduce_sum(outputs), name="mean", aggregation="mean"
                )
                return outputs

        x = layers_module.Input(shape=(1,))
        y = LayerWithAddMetric()(x)

        inner_model = training_module.Model(x, y)
        inner_model.add_metric(
            tf.reduce_sum(y), name="mean1", aggregation="mean"
        )

        inner_model.compile(
            "sgd",
            loss="mse",
            metrics=[metrics_module.Accuracy("acc")],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        inner_model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)

        self.assertEqual(
            [m.name for m in inner_model.metrics],
            ["loss", "acc", "mean", "mean1"],
        )

        x = layers_module.Input(shape=[1])
        y = inner_model(x)
        outer_model = training_module.Model(x, y)
        outer_model.add_metric(
            tf.reduce_sum(y), name="mean2", aggregation="mean"
        )

        outer_model.compile(
            "sgd",
            loss="mse",
            metrics=[metrics_module.Accuracy("acc2")],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        outer_model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)
        self.assertEqual(
            [m.name for m in outer_model.metrics],
            ["loss", "acc2", "mean", "mean1", "mean2"],
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_model_with_metric_class_that_returns_dict(self):
        x = layers_module.Input(shape=(2,))
        y = layers_module.Dense(3)(x)
        model = training_module.Model(x, y)

        class DictMetric(metrics_module.Metric):
            def __init__(self):
                super().__init__()
                self.sample_count = tf.Variable(0)
                self.l2_sum = tf.Variable(0.0)

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.l2_sum.assign_add(
                    tf.reduce_sum(tf.square(y_true - y_pred))
                )
                self.sample_count.assign_add(tf.shape(y_true)[0])

            def reset_state(self):
                self.sample_count.assign(0)
                self.l2_sum.assign(0.0)

            def result(self):
                mse = self.l2_sum / tf.cast(self.sample_count, "float32")
                rmse = tf.sqrt(mse)
                return {"my_mse": mse, "my_rmse": rmse}

        model.compile(
            "sgd",
            "mse",
            metrics=["mae", DictMetric()],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        history = model.fit(np.ones((10, 2)), np.ones((10, 3)))
        self.assertEqual(
            list(history.history.keys()), ["loss", "mae", "my_mse", "my_rmse"]
        )
        list_evaluate_res = model.evaluate(np.ones((10, 2)), np.ones((10, 3)))
        self.assertEqual(len(list_evaluate_res), 4)
        dict_evaluate_res = model.evaluate(
            np.ones((10, 2)), np.ones((10, 3)), return_dict=True
        )
        self.assertEqual(
            list(dict_evaluate_res.keys()), ["loss", "mae", "my_mse", "my_rmse"]
        )
        list_train_on_batch_res = model.train_on_batch(
            np.ones((10, 2)), np.ones((10, 3))
        )
        self.assertEqual(len(list_train_on_batch_res), 4)
        dict_train_on_batch_res = model.train_on_batch(
            np.ones((10, 2)), np.ones((10, 3)), return_dict=True
        )
        self.assertEqual(
            list(dict_train_on_batch_res.keys()),
            ["loss", "mae", "my_mse", "my_rmse"],
        )
        list_test_on_batch_res = model.test_on_batch(
            np.ones((10, 2)), np.ones((10, 3))
        )
        self.assertEqual(len(list_test_on_batch_res), 4)
        dict_test_on_batch_res = model.test_on_batch(
            np.ones((10, 2)), np.ones((10, 3)), return_dict=True
        )
        self.assertEqual(
            list(dict_test_on_batch_res.keys()),
            ["loss", "mae", "my_mse", "my_rmse"],
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_add_metric_in_model_call_that_returns_dict(self):
        class DictMetric(metrics_module.Metric):
            def __init__(self):
                super().__init__()
                self.sample_count = tf.Variable(0)
                self.l2_sum = tf.Variable(0.0)

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.l2_sum.assign_add(
                    tf.reduce_sum(tf.square(y_true - y_pred))
                )
                self.sample_count.assign_add(tf.shape(y_true)[0])

            def reset_state(self):
                self.sample_count.assign(0)
                self.l2_sum.assign(0.0)

            def result(self):
                mse = self.l2_sum / tf.cast(self.sample_count, "float32")
                rmse = tf.sqrt(mse)
                return {"my_mse": mse, "my_rmse": rmse}

        class TestModel(training_module.Model):
            def __init__(self):
                super().__init__(name="test_model")
                self.dense1 = layers_module.Dense(2, kernel_initializer="ones")
                self.dict_metric = DictMetric()

            def call(self, x):
                self.add_metric(
                    tf.reduce_sum(x), name="metric_2", aggregation="mean"
                )
                # Provide same name as in the instance created in __init__
                # for eager mode
                self.add_metric(self.dict_metric(x, 1 - x), name="metric_1")
                return self.dense1(x)

        model = TestModel()
        model.compile(
            loss="mse",
            optimizer=RMSPropOptimizer(0.01),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = np.ones(shape=(10, 1))
        y = np.ones(shape=(10, 2))
        history = model.fit(
            x, y, epochs=2, batch_size=5, validation_data=(x, y)
        )
        self.assertAlmostEqual(history.history["metric_2"][-1], 5, 0)
        self.assertAlmostEqual(history.history["val_metric_2"][-1], 5, 0)
        self.assertAlmostEqual(history.history["my_mse"][-1], 1, 0)
        self.assertAlmostEqual(history.history["val_my_mse"][-1], 1, 0)
        self.assertAlmostEqual(history.history["my_rmse"][-1], 1, 0)
        self.assertAlmostEqual(history.history["val_my_rmse"][-1], 1, 0)

        eval_results = model.evaluate(x, y, batch_size=5, return_dict=True)
        self.assertAlmostEqual(eval_results["metric_2"], 5, 0)
        self.assertAlmostEqual(eval_results["my_mse"], 1, 0)
        self.assertAlmostEqual(eval_results["my_rmse"], 1, 0)

        model.predict(x, batch_size=5)
        model.train_on_batch(x, y)
        model.test_on_batch(x, y)


class BareUpdateLayer(layers_module.Layer):
    def build(self, input_shape):
        self.counter = self.add_weight(
            "counter",
            dtype="int32",
            shape=(),
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs):
        tf.compat.v1.assign_add(self.counter, 1)
        return tf.cast(self.counter, inputs.dtype) * inputs


class LambdaUpdateLayer(layers_module.Layer):
    def build(self, input_shape):
        self.counter = self.add_weight(
            "counter",
            dtype="int32",
            shape=(),
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs):
        # Make sure update isn't run twice.
        self.add_update(lambda: tf.compat.v1.assign_add(self.counter, 1))
        return tf.cast(self.counter, inputs.dtype) * inputs


class NestedUpdateLayer(layers_module.Layer):
    def build(self, input_shape):
        self.layer = BareUpdateLayer()
        self.layer.build(input_shape)

    @property
    def counter(self):
        return self.layer.counter

    def call(self, inputs):
        return self.layer(inputs)


class SubgraphUpdateLayer(layers_module.Layer):
    def build(self, input_shape):
        self.counter = self.add_weight(
            "counter",
            dtype="int32",
            shape=(),
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        if training:
            self.counter.assign(self.counter + 1)
        return inputs


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class TestAutoUpdates(test_combinations.TestCase):
    @test_combinations.run_with_all_model_types
    @parameterized.named_parameters(
        ("bare_update", BareUpdateLayer),
        ("lambda_update", LambdaUpdateLayer),
        ("nested_update", NestedUpdateLayer),
    )
    def test_updates_in_model(self, layer_builder):
        layer = layer_builder()
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model = test_utils.get_model_from_layers(
            [layer, layers_module.Dense(1)], input_shape=(10,)
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        model.fit(x, y, batch_size=2, epochs=1)
        self.assertEqual(self.evaluate(layer.counter), 5)

    @test_combinations.run_with_all_model_types
    def test_lambda_updates_trainable_false(self):
        x, y = np.ones((10, 10)), np.ones((10, 1))
        layer = LambdaUpdateLayer()
        model = test_utils.get_model_from_layers(
            [layer, layers_module.Dense(1)], input_shape=(10,)
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        model.fit(x, y, batch_size=2, epochs=1)
        self.assertEqual(self.evaluate(layer.counter), 5)
        layer.trainable = False
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        model.fit(x, y, batch_size=2, epochs=1)
        self.assertEqual(self.evaluate(layer.counter), 5)

    @test_combinations.run_with_all_model_types
    def test_subgraph_updates_in_model(self):
        layer = SubgraphUpdateLayer()
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model = test_utils.get_model_from_layers(
            [layer, layers_module.Dense(1)], input_shape=(10,)
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        model.fit(x, y, batch_size=2, epochs=1)
        self.assertEqual(self.evaluate(layer.counter), 5)

    @parameterized.named_parameters(
        ("bare_update", BareUpdateLayer),
        ("lambda_update", LambdaUpdateLayer),
        ("nested_update", NestedUpdateLayer),
    )
    def test_updates_standalone_layer(self, layer_builder):
        layer = layer_builder()
        y = layer(np.ones((10, 10)))
        self.evaluate(layer.counter.initializer)
        self.evaluate(y)
        self.assertEqual(self.evaluate(layer.counter), 1)

    def test_trainable_false_standalone_layer(self):
        layer = LambdaUpdateLayer()
        y = layer(np.ones((10, 10)))
        self.evaluate(layer.counter.initializer)
        self.evaluate(y)
        self.assertEqual(self.evaluate(layer.counter), 1)
        layer.trainable = False
        y = layer(np.ones((10, 10)))
        self.evaluate(y)
        self.assertEqual(self.evaluate(layer.counter), 1)

    @test_combinations.run_with_all_model_types
    def test_batchnorm_trainable_false(self):
        bn = layers_module.BatchNormalization()
        model = test_utils.get_model_from_layers(
            [bn, layers_module.Dense(1)], input_shape=(10,)
        )
        bn.trainable = False
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model.fit(x, y, batch_size=2, epochs=1)
        self.assertAllEqual(self.evaluate(bn.moving_mean), np.zeros((10,)))
        self.assertAllEqual(self.evaluate(bn.moving_variance), np.ones((10,)))


class TestFunctionTracing(test_combinations.TestCase):
    def _seq_model_and_data(self):
        model = sequential.Sequential(
            [layers_module.Dense(4, activation="relu")]
        )
        model.compile(loss="mse", optimizer="rmsprop")
        x = np.random.random((10, 6))
        y = np.random.random((10, 4))
        return model, x, y

    @test_combinations.run_all_keras_modes(
        always_skip_v1=True, always_skip_eager=True
    )
    def test_no_tracing_between_epoch(self):
        if _is_oss():
            self.skipTest("b/198729465")

        model, x, y = self._seq_model_and_data()

        logging.set_verbosity(1)
        with self.assertLogs(level=1) as logs:
            model.fit(x, y, epochs=10, batch_size=5, validation_data=(x, y))

        new_func_graph = "INFO:absl:Creating new FuncGraph for Python function"
        self.assertEqual(sum(new_func_graph in log for log in logs.output), 9)

    @test_combinations.run_all_keras_modes(
        always_skip_v1=True, always_skip_eager=True
    )
    def test_evaluate_no_cached_data(self):
        if _is_oss():
            self.skipTest("b/198729465")

        model, x, y = self._seq_model_and_data()

        new_func_graph = "INFO:absl:Creating new FuncGraph for Python function"
        logging.set_verbosity(1)
        with self.assertLogs(level=1) as eval_logs:
            for _ in range(6):
                model.evaluate(x, y, batch_size=5)
        self.assertEqual(
            sum(new_func_graph in log for log in eval_logs.output), 20
        )


class TestBuildCustomModel(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes
    def test_build_list_of_inputs(self):
        class MyModel(training_module.Model):
            def __init__(self):
                super().__init__()
                self.l1 = layers_module.Dense(1)
                self.l2 = layers_module.Dense(2)

            def call(self, x):
                a, b = x
                return self.l1(a) + self.l2(b)

        # List of tuples
        model = MyModel()
        model.build([(None, 1), (None, 2)])
        self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])
        self.assertEqual(model.l2.kernel.shape.as_list(), [2, 2])
        # List of lists
        model = MyModel()
        model.build([[None, 1], [None, 2]])
        self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])
        self.assertEqual(model.l2.kernel.shape.as_list(), [2, 2])

    @test_combinations.run_all_keras_modes
    def test_build_single_inputs(self):
        class MyModel(training_module.Model):
            def __init__(self):
                super().__init__()
                self.l1 = layers_module.Dense(1)

            def call(self, x):
                return self.l1(x)

        model = MyModel()
        model.build((None, 1))
        self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])
        model = MyModel()
        model.build([None, 1])
        self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])

    @test_combinations.run_all_keras_modes
    def test_build_dict_inputs(self):
        class MyModel(training_module.Model):
            def __init__(self):
                super().__init__()
                self.l1 = layers_module.Dense(1)

            def call(self, inputs):
                return self.l1(inputs["x"])

        model = MyModel()
        model.build({"x": [None, 16]})
        self.assertEqual(model.l1.kernel.shape.as_list(), [16, 1])

    def test_save_top_level_model_weights_h5(self):
        class MyModel(training_module.Model):
            def __init__(self):
                super().__init__()
                self.class_token = self.add_weight(
                    shape=(1,), name="class_token"
                )
                self.inner_layer = layers_module.Dense(1)

            def call(self, inputs):
                return self.inner_layer(inputs) * self.class_token

        h5_file = tempfile.mktemp(".h5")
        m1 = MyModel()
        m1.build((1, 1))
        m1.save_weights(h5_file)

        m2 = MyModel()
        m2.build((1, 1))
        m2.load_weights(h5_file)
        self.assertAllEqual(m1.get_weights(), m2.get_weights())
        m2.load_weights(h5_file, by_name=True)
        self.assertAllEqual(m1.get_weights(), m2.get_weights())


class ScalarDataModelTest(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_scalar_loss_reduction(self):
        class MyModel(training_module.Model):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(initializer="ones", name="kernel")
                self.b = self.add_weight(initializer="zeros", name="bias")

            def call(self, inputs):
                return inputs * self.w + self.b

        model = MyModel()
        model.compile(
            optimizer_legacy.gradient_descent.SGD(1e-2),
            loss="mse",
            metrics=["binary_accuracy"],
        )
        # learn y = x * 2 + 0.5
        x = np.array([3, 5, 5, 3, 5], dtype="float32")
        y = x * 2 + 0.5
        x2d = np.expand_dims(x, axis=-1)
        y2d = np.expand_dims(y, axis=-1)
        loss, acc = model.evaluate(x, y)
        loss2d, acc2d = model.evaluate(x2d, y2d)
        self.assertAllClose([loss, acc], [loss2d, acc2d], atol=1e-6)
        model.fit(x, y, epochs=20)
        preds = model.predict(x)
        self.assertEqual(preds.shape, (5,))
        self.assertAllClose(preds, y, atol=2e-1)


# Class used for testing.
class SubclassModel(training_module.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.d1 = layers_module.Dense(1000)
        self.d2 = layers_module.Dense(1000)
        self.dropout = layers_module.Dropout(0.1)

    def call(self, inputs, training=None):
        x = self.d1(inputs)
        x = self.dropout(x, training=training)
        return self.d2(x)


class TestVariableObjectPathMapping(test_combinations.TestCase):
    def test_subclass_model_get_weight_paths(self):
        model = SubclassModel()
        # Make sure the object path produce nothing when weights are not
        # initialized
        self.assertEmpty(model.get_weight_paths())

        model(tf.zeros((10, 10)))
        mapping = model.get_weight_paths()
        self.assertEqual(
            mapping.keys(), {"d1.kernel", "d1.bias", "d2.kernel", "d2.bias"}
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_functional_model_get_weight_paths(self):
        inputs = input_layer.Input(shape=(10,))
        x = layers_module.Dense(100, name="d1")(inputs)
        output = layers_module.Dense(200, name="d2", activation="softmax")(x)
        model = training_module.Model(inputs, output)
        mapping = model.get_weight_paths()
        self.assertEqual(
            mapping.keys(), {"d1.kernel", "d1.bias", "d2.kernel", "d2.bias"}
        )

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_sequential_model_get_weight_paths(self):
        model = sequential.Sequential(
            [
                layers_module.Dense(100, name="d1", input_shape=(10,)),
                layers_module.Dense(200, name="d2", activation="softmax"),
            ]
        )
        mapping = model.get_weight_paths()
        self.assertEqual(
            mapping.keys(), {"d1.kernel", "d1.bias", "d2.kernel", "d2.bias"}
        )


def _is_oss():
    """Returns whether the test is run under OSS."""
    return len(sys.argv) >= 1 and "bazel" in sys.argv[0]


if __name__ == "__main__":
    tf.test.main()

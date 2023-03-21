# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for saving utility functions."""

import os

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import backend
from keras.engine import sequential
from keras.feature_column import dense_features
from keras.optimizers.legacy import gradient_descent
from keras.saving.legacy import saving_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


class TraceModelCallTest(test_combinations.TestCase):
    def _assert_all_close(self, expected, actual):
        if not tf.executing_eagerly():
            with self.cached_session() as sess:
                backend._initialize_variables(sess)
                self.assertAllClose(expected, actual)
        else:
            self.assertAllClose(expected, actual)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_trace_model_outputs(self):
        input_dim = 5 if test_utils.get_model_type() == "functional" else None
        model = test_utils.get_small_mlp(10, 3, input_dim)
        inputs = tf.ones((8, 5))

        if input_dim is None:
            with self.assertRaisesRegex(
                ValueError, ".*input shape is not availabl*"
            ):
                saving_utils.trace_model_call(model)
            model._set_inputs(inputs)

        fn = saving_utils.trace_model_call(model)
        signature_outputs = fn(inputs)
        if model.output_names:
            expected_outputs = {model.output_names[0]: model(inputs)}
        else:
            expected_outputs = {"output_1": model(inputs)}

        self._assert_all_close(expected_outputs, signature_outputs)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_trace_model_outputs_after_fitting(self):
        input_dim = 5 if test_utils.get_model_type() == "functional" else None
        model = test_utils.get_small_mlp(10, 3, input_dim)
        model.compile(
            optimizer="sgd",
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(
            x=np.random.random((8, 5)).astype(np.float32),
            y=np.random.random((8, 3)).astype(np.float32),
            epochs=2,
        )

        inputs = tf.ones((8, 5))

        fn = saving_utils.trace_model_call(model)
        signature_outputs = fn(inputs)
        if model.output_names:
            expected_outputs = {model.output_names[0]: model(inputs)}
        else:
            expected_outputs = {"output_1": model(inputs)}

        self._assert_all_close(expected_outputs, signature_outputs)

    @test_combinations.run_with_all_model_types(exclude_models="sequential")
    @test_combinations.run_all_keras_modes
    def test_trace_multi_io_model_outputs(self):
        input_dim = 5
        num_classes = 3
        num_classes_b = 4
        input_a = keras.layers.Input(shape=(input_dim,), name="input_a")
        input_b = keras.layers.Input(shape=(input_dim,), name="input_b")

        dense = keras.layers.Dense(num_classes, name="dense")
        dense2 = keras.layers.Dense(num_classes_b, name="dense2")
        dropout = keras.layers.Dropout(0.5, name="dropout")
        branch_a = [input_a, dense]
        branch_b = [input_b, dense, dense2, dropout]

        model = test_utils.get_multi_io_model(branch_a, branch_b)

        input_a_ts = tf.constant(
            np.random.random((10, input_dim)).astype(np.float32)
        )
        input_b_ts = tf.constant(
            np.random.random((10, input_dim)).astype(np.float32)
        )

        if test_utils.get_model_type() == "subclass":
            with self.assertRaisesRegex(
                ValueError, ".*input shape is not availabl*"
            ):
                saving_utils.trace_model_call(model)

        model.compile(
            optimizer="sgd",
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(
            x=[
                np.random.random((8, input_dim)).astype(np.float32),
                np.random.random((8, input_dim)).astype(np.float32),
            ],
            y=[
                np.random.random((8, num_classes)).astype(np.float32),
                np.random.random((8, num_classes_b)).astype(np.float32),
            ],
            epochs=2,
        )

        fn = saving_utils.trace_model_call(model)
        # tf.function requires that the input structures match when calling a
        # ConcreteFunction. For some reason V1 models defines the inputs as a
        # list, while V2 models sets the inputs as a tuple.
        if (
            not tf.executing_eagerly()
            and test_utils.get_model_type() != "functional"
        ):
            signature_outputs = fn([input_a_ts, input_b_ts])
        else:
            signature_outputs = fn((input_a_ts, input_b_ts))
        outputs = model([input_a_ts, input_b_ts])
        if model.output_names:
            expected_outputs = {
                model.output_names[0]: outputs[0],
                model.output_names[1]: outputs[1],
            }
        else:
            expected_outputs = {"output_1": outputs[0], "output_2": outputs[1]}
        self._assert_all_close(expected_outputs, signature_outputs)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_trace_features_layer(self):
        columns = [tf.feature_column.numeric_column("x")]
        model = sequential.Sequential([dense_features.DenseFeatures(columns)])
        model_input = {"x": tf.constant([[1.0]])}
        model.predict(model_input, steps=1)
        fn = saving_utils.trace_model_call(model)
        self.assertAllClose({"output_1": [[1.0]]}, fn(model_input))

        columns = [
            tf.feature_column.numeric_column("x"),
            tf.feature_column.numeric_column("y"),
        ]
        model = sequential.Sequential([dense_features.DenseFeatures(columns)])
        model_input = {"x": tf.constant([[1.0]]), "y": tf.constant([[2.0]])}
        model.predict(model_input, steps=1)
        fn = saving_utils.trace_model_call(model)
        self.assertAllClose({"output_1": [[1.0, 2.0]]}, fn(model_input))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_specify_input_signature(self):
        model = test_utils.get_small_sequential_mlp(10, 3, None)
        inputs = tf.ones((8, 5))

        with self.assertRaisesRegex(
            ValueError, ".*input shape is not availabl*"
        ):
            saving_utils.trace_model_call(model)

        fn = saving_utils.trace_model_call(
            model, [tf.TensorSpec(shape=[None, 5], dtype=tf.float32)]
        )
        signature_outputs = fn(inputs)
        if model.output_names:
            expected_outputs = {model.output_names[0]: model(inputs)}
        else:
            expected_outputs = {"output_1": model(inputs)}
        self._assert_all_close(expected_outputs, signature_outputs)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_subclassed_model_with_input_signature(self):
        class Model(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = keras.layers.Dense(3, name="dense")

            @tf.function(
                input_signature=[
                    [
                        tf.TensorSpec([None, 5], tf.float32),
                        tf.TensorSpec([None], tf.float32),
                    ]
                ],
            )
            def call(self, inputs, *args):
                x, y = inputs
                return self.dense(x) + y

        model = Model()
        fn = saving_utils.trace_model_call(model)
        x = tf.ones((8, 5), dtype=tf.float32)
        y = tf.ones((3,), dtype=tf.float32)
        expected_outputs = {"output_1": model([x, y])}
        signature_outputs = fn([x, y])
        self._assert_all_close(expected_outputs, signature_outputs)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_model_with_fixed_input_dim(self):
        """Ensure that the batch_dim is removed when saving.

        When serving or retraining, it is important to reset the batch dim.
        This can be an issue inside of tf.function. See b/132783590 for context.
        """
        model = test_utils.get_small_mlp(10, 3, 5)

        loss_object = keras.losses.MeanSquaredError()
        optimizer = gradient_descent.SGD()

        @tf.function
        def train_step(data, labels):
            with tf.GradientTape() as tape:
                predictions = model(data)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        x = np.random.random((8, 5))
        y = np.random.random((8, 3))

        train_step(x, y)

        fn = saving_utils.trace_model_call(model)
        self.assertEqual(
            fn.structured_input_signature[0][0].shape.as_list(),
            tf.TensorShape([None, 5]).as_list(),
        )


def _import_and_infer(save_dir, inputs):
    """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session() as session:
        model = tf.compat.v1.saved_model.load(
            session, [tf.saved_model.SERVING], save_dir
        )
        signature = model.signature_def[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]
        assert set(inputs.keys()) == set(
            signature.inputs.keys()
        ), f"expected {signature.inputs.keys()}, found {inputs.keys()}"
        feed_dict = {}
        for arg_name in inputs.keys():
            feed_dict[
                graph.get_tensor_by_name(signature.inputs[arg_name].name)
            ] = inputs[arg_name]
        output_dict = {}
        for output_name, output_tensor_info in signature.outputs.items():
            output_dict[output_name] = graph.get_tensor_by_name(
                output_tensor_info.name
            )
        return session.run(output_dict, feed_dict=feed_dict)


class AutographedMetric(keras.metrics.Metric):
    def build(self, input_shape):
        pass

    def update_state(self, values):
        if tf.constant(False):
            x = 1
        else:
            x = 2
        return x

    def reset_states(self):
        pass

    def result(self):
        return tf.constant(0)

    def GetMean(self):
        return tf.constant(0)

    def GetCount(self):
        return tf.constant(0)


class BasicAutographedMetricLayer(keras.layers.Layer):
    def build(self, input_shape):
        self._metric = AutographedMetric()

    def call(self, inp):
        self._metric.update_state(inp)
        # TODO(b/172853147): Test control flow here.
        return inp


class BasicAutographedMetricModel(keras.models.Model):
    def __init__(self):
        super().__init__(name="test_model")
        self._layer = BasicAutographedMetricLayer()

    def call(self, inputs, **kwargs):
        return self._layer(inputs)


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class ModelSaveTest(test_combinations.TestCase):
    def test_model_save_preserves_autograph(self):
        model = BasicAutographedMetricModel()
        inputs = tf.ones((8, 5))
        model._set_inputs(inputs)

        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        tf.saved_model.save(model, save_dir)

        if model.output_names:
            output_name = model.output_names[0]
            input_name = model.input_names[0]
        else:
            output_name = "output_1"
            input_name = "input_1"

        self.assertAllClose(
            {output_name: model.predict_on_batch(inputs)},
            _import_and_infer(save_dir, {input_name: np.ones((8, 5))}),
        )

        # Test v2 loading.
        # TODO(mdan): tests using _import_and_infer should uniformly do this.
        self.assertAllClose(
            model.predict_on_batch(inputs),
            tf.saved_model.load(save_dir)(inputs),
        )

    def test_model_save(self):
        input_dim = 5
        model = test_utils.get_small_mlp(10, 3, input_dim)
        inputs = tf.ones((8, 5))

        if test_utils.get_model_type() == "subclass":
            model._set_inputs(inputs)

        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        tf.saved_model.save(model, save_dir)

        if model.output_names:
            output_name = model.output_names[0]
            input_name = model.input_names[0]
        else:
            output_name = "output_1"
            input_name = "input_1"

        self.assertAllClose(
            {output_name: model.predict_on_batch(inputs)},
            _import_and_infer(save_dir, {input_name: np.ones((8, 5))}),
        )


class ExtractModelMetricsTest(test_combinations.TestCase):
    def test_extract_model_metrics(self):
        # saving_utils.extract_model_metrics is used in V1 only API
        # keras.experimental.export_saved_model.
        with tf.Graph().as_default():
            a = keras.layers.Input(shape=(3,), name="input_a")
            b = keras.layers.Input(shape=(3,), name="input_b")

            dense = keras.layers.Dense(4, name="dense")
            c = dense(a)
            d = dense(b)
            e = keras.layers.Dropout(0.5, name="dropout")(c)

            model = keras.models.Model([a, b], [d, e])
            extract_metrics = saving_utils.extract_model_metrics(model)
            self.assertEqual(None, extract_metrics)

            extract_metric_names = [
                "dense_binary_accuracy",
                "dropout_binary_accuracy",
                "dense_mean_squared_error",
                "dropout_mean_squared_error",
            ]
            if tf.__internal__.tf2.enabled():
                extract_metric_names.extend(["dense_mae", "dropout_mae"])
            else:
                extract_metric_names.extend(
                    ["dense_mean_absolute_error", "dropout_mean_absolute_error"]
                )

            model_metric_names = [
                "loss",
                "dense_loss",
                "dropout_loss",
            ] + extract_metric_names
            model.compile(
                loss="mae",
                metrics=[
                    keras.metrics.BinaryAccuracy(),
                    "mae",
                    keras.metrics.mean_squared_error,
                ],
                optimizer=tf.compat.v1.train.RMSPropOptimizer(
                    learning_rate=0.01
                ),
            )
            extract_metrics = saving_utils.extract_model_metrics(model)
            self.assertEqual(set(model_metric_names), set(model.metrics_names))
            self.assertEqual(
                set(extract_metric_names), set(extract_metrics.keys())
            )


class UnbuiltModelSavingErrorMessageTest(test_combinations.TestCase):
    def setUp(self):
        super().setUp()
        if not tf.__internal__.tf2.enabled():
            self.skipTest("The test does not intend to cover TF1.")

    def test_sequential(self):
        model = sequential.Sequential([keras.layers.Dense(10)])
        optimizer = gradient_descent.SGD()
        model.compile(optimizer, loss="mse", steps_per_execution=10)

        # Forward pass not called yet. Input shape not available and thus error.
        with self.assertRaisesRegex(
            ValueError,
            "Model.*cannot be saved."
            "*specify an input shape either by calling.*",
        ):
            model.save(os.path.join(self.get_temp_dir(), "my_saved_model"))

    def test_functional(self):
        inputs = keras.Input(shape=(32,))
        outputs = keras.layers.Dense(1)(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        x = np.random.random((1000, 32))
        y = np.random.random((1000, 1))
        model.fit(x, y, epochs=3)

        # Functional model always has an input shape, so should save just fine.
        model.save(os.path.join(self.get_temp_dir(), "my_saved_model"))

    def test_subclass_forward_pass_by_layer_underscore_call(self):
        class CustomModel(keras.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dense1 = keras.layers.Dense(1)

            def train_step(self, data):
                x, y = data
                with tf.GradientTape() as tape:
                    y_pred = self.dense1(x, training=True)
                    loss = self.compiled_loss(y, y_pred)

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )
                return {}

        subclassed_model = CustomModel()
        subclassed_model.compile(optimizer="adam", loss="mse")

        x = np.random.random((1000, 32))
        y = np.random.random((1000, 1))
        subclassed_model.fit(x, y, epochs=1)

        # Saving of this subclassed model is supposed to raise an error, even if
        # `fit` has been called. This is because the model does not have
        # `call()` overridden. Forward pass using `layer.__call__` works for
        # training, but saving requires that `call()` be used.
        with self.assertRaisesRegex(
            ValueError,
            r"Model.*cannot be saved.*as opposed to `model.call\(\).*",
        ):
            subclassed_model.save(
                os.path.join(self.get_temp_dir(), "my_saved_model")
            )

    def test_subclass_forward_pass_by_model_call(self):
        class CustomModel(keras.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dense1 = keras.layers.Dense(1)

            def call(self, inputs):
                return self.dense1(inputs)

            def train_step(self, data):
                x, y = data
                with tf.GradientTape() as tape:
                    y_pred = self.call(x)
                    loss = self.compiled_loss(y, y_pred)

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )
                return {}

        subclassed_model = CustomModel()
        subclassed_model.compile(optimizer="adam", loss="mse")

        x = np.random.random((1000, 32))
        y = np.random.random((1000, 1))
        subclassed_model.fit(x, y, epochs=1)

        # Saving of this subclassed model is supposed to raise an error, even if
        # `fit` has been called. This is because the model has `call()`
        # overridden, but the forward pass uses `Model.call` as opposed to
        # `Model.__call__`, and as a result the `Model` is not really built. The
        # error message hints the user to use `Model.__call__`, i.e.,
        # `Model(inputs)` instead.
        with self.assertRaisesRegex(
            ValueError,
            r"Model.*cannot be saved.*as opposed to `model.call\(\).*",
        ):
            subclassed_model.save(
                os.path.join(self.get_temp_dir(), "my_saved_model")
            )


if __name__ == "__main__":
    tf.test.main()

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
"""Tests for Keras core layers."""

import os
import textwrap

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import initializers
from keras.layers import core
from keras.mixed_precision import policy
from keras.saving.serialization_lib import SafeModeScope
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class DropoutLayersTest(test_combinations.TestCase):
    def test_dropout(self):
        test_utils.layer_test(
            keras.layers.Dropout, kwargs={"rate": 0.5}, input_shape=(3, 2)
        )

        test_utils.layer_test(
            keras.layers.Dropout,
            kwargs={"rate": 0.5, "noise_shape": [3, 1]},
            input_shape=(3, 2),
        )

    def test_dropout_supports_masking(self):
        dropout = keras.layers.Dropout(0.5)
        self.assertEqual(True, dropout.supports_masking)

    def test_spatial_dropout_1d(self):
        test_utils.layer_test(
            keras.layers.SpatialDropout1D,
            kwargs={"rate": 0.5},
            input_shape=(2, 3, 4),
        )

    def test_spatial_dropout_2d(self):
        test_utils.layer_test(
            keras.layers.SpatialDropout2D,
            kwargs={"rate": 0.5},
            input_shape=(2, 3, 4, 5),
        )

        test_utils.layer_test(
            keras.layers.SpatialDropout2D,
            kwargs={"rate": 0.5, "data_format": "channels_first"},
            input_shape=(2, 3, 4, 5),
        )

    def test_spatial_dropout_3d(self):
        test_utils.layer_test(
            keras.layers.SpatialDropout3D,
            kwargs={"rate": 0.5},
            input_shape=(2, 3, 4, 4, 5),
        )

        test_utils.layer_test(
            keras.layers.SpatialDropout3D,
            kwargs={"rate": 0.5, "data_format": "channels_first"},
            input_shape=(2, 3, 4, 4, 5),
        )

    def test_dropout_partial_noise_shape(self):
        inputs = keras.Input(shape=(5, 10))
        layer = keras.layers.Dropout(0.5, noise_shape=(None, 1, None))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        out = model(np.ones((20, 5, 10)), training=True)
        out_np = keras.backend.get_value(out)
        # Test that dropout mask is shared across second dim.
        self.assertAllClose(out_np[:, 0, :], out_np[:, 1, :])

    def test_dropout_with_savemodel(self):
        inputs = keras.Input(shape=(5, 10))
        layer = keras.layers.Dropout(0.5, force_generator=True)
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        train = model(np.ones((20, 5, 10)), training=True)
        predict = model(np.ones((20, 5, 10)))
        # Make sure the weights from tf.random.Generator is not present in the
        # model which will cause weight loading issue for existing application
        # models if it contains dropout layer.
        self.assertEmpty(layer.get_weights())
        self.assertEmpty(model.get_weights())

        # Make sure the layer does dropout value when training
        self.assertNotAllClose(train, predict)

        model.save(
            os.path.join(self.get_temp_dir(), "savedmodel"), save_format="tf"
        )
        loaded_model = keras.models.load_model(
            os.path.join(self.get_temp_dir(), "savedmodel")
        )
        predict2 = loaded_model(np.ones((20, 5, 10)))

        self.assertAllClose(predict, predict2)
        # Make sure the model dropout different value after loading
        train2 = loaded_model(np.ones((20, 5, 10)), training=True)
        self.assertNotAllClose(train, train2)
        self.assertIsNotNone(loaded_model.layers[1]._random_generator)

        # Also make sure the checkpoint doesn't contain any variable from the
        # dropout layer, to keep the backward compatibility.
        checkpoint = tf.train.Checkpoint(model)
        save_path = checkpoint.save(
            os.path.join(self.get_temp_dir(), "checkpoint")
        )
        checkpoint_var_names = [
            name_value_tuple[0]
            for name_value_tuple in tf.train.list_variables(save_path)
        ]
        for name in checkpoint_var_names:
            self.assertNotIn("dropout", name)


@test_combinations.run_all_keras_modes
class LambdaLayerTest(test_combinations.TestCase):
    def test_lambda(self):
        with SafeModeScope(safe_mode=False):
            test_utils.layer_test(
                keras.layers.Lambda,
                kwargs={"function": lambda x: x + 1},
                input_shape=(3, 2),
            )

            test_utils.layer_test(
                keras.layers.Lambda,
                kwargs={
                    "function": lambda x, a, b: x * a + b,
                    "arguments": {"a": 0.6, "b": 0.4},
                },
                input_shape=(3, 2),
            )

        # test serialization with function
        def f(x):
            return x + 1

        ld = keras.layers.Lambda(f)
        config = ld.get_config()
        with SafeModeScope(safe_mode=False):
            ld = keras.layers.deserialize(
                {"class_name": "Lambda", "config": config}
            )
        self.assertEqual(ld.function(3), 4)

        # test with lambda
        ld = keras.layers.Lambda(
            lambda x: keras.backend.concatenate([tf.square(x), x])
        )
        config = ld.get_config()
        ld = keras.layers.Lambda.from_config(config)
        self.assertAllEqual(self.evaluate(ld.function([3])), [9, 3])

    def test_lambda_multiple_inputs(self):
        ld = keras.layers.Lambda(lambda x: x[0], output_shape=lambda x: x[0])
        x1 = np.ones([3, 2], np.float32)
        x2 = np.ones([3, 5], np.float32)
        out = ld([x1, x2])
        self.assertAllEqual(out.shape, [3, 2])

    def test_lambda_output_shape(self):
        l = keras.layers.Lambda(lambda x: x + 1, output_shape=(1, 1))
        l(keras.backend.variable(np.ones((1, 1))))
        self.assertEqual((1, 1), l.get_config()["output_shape"])

    def test_lambda_output_shape_function(self):
        def get_output_shape(input_shape):
            return 1 * input_shape

        l = keras.layers.Lambda(lambda x: x + 1, output_shape=get_output_shape)
        l(keras.backend.variable(np.ones((1, 1))))
        self.assertEqual("lambda", l.get_config()["output_shape_type"])

    def test_lambda_output_shape_autocalculate_multiple_inputs(self):
        def lambda_fn(x):
            return tf.matmul(x[0], x[1])

        l = keras.layers.Lambda(lambda_fn, dtype=tf.float64)
        output_shape = l.compute_output_shape([(10, 10), (10, 20)])
        self.assertAllEqual((10, 20), output_shape)
        output_signature = l.compute_output_signature(
            [
                tf.TensorSpec(dtype=tf.float64, shape=(10, 10)),
                tf.TensorSpec(dtype=tf.float64, shape=(10, 20)),
            ]
        )
        self.assertAllEqual((10, 20), output_signature.shape)
        self.assertAllEqual(tf.float64, output_signature.dtype)

    def test_lambda_output_shape_list_multiple_outputs(self):
        def lambda_fn(x):
            return x

        l = keras.layers.Lambda(lambda_fn, output_shape=[(10,), (20,)])
        output_shape = l.compute_output_shape([(10, 10), (10, 20)])
        self.assertAllEqual([(10, 10), (10, 20)], output_shape)

    def test_lambda_output_shape_tuple_with_none(self):
        def lambda_fn(x):
            return x

        l = keras.layers.Lambda(lambda_fn, output_shape=(None, 10))
        output_shape = l.compute_output_shape((5, 10, 20))
        self.assertAllEqual([5, None, 10], output_shape.as_list())

    def test_lambda_output_shape_function_multiple_outputs(self):
        def lambda_fn(x):
            return x

        def output_shape_fn(input_shape):
            return input_shape

        l = keras.layers.Lambda(lambda_fn, output_shape=output_shape_fn)
        output_shape = l.compute_output_shape([(10, 10), (10, 20)])
        self.assertAllEqual([(10, 10), (10, 20)], output_shape)

    def test_lambda_output_shape_nested(self):
        def lambda_fn(inputs):
            return (inputs[1]["a"], {"b": inputs[0]})

        l = keras.layers.Lambda(lambda_fn)
        output_shape = l.compute_output_shape(((10, 20), {"a": (10, 5)}))
        self.assertAllEqual(((10, 5), {"b": (10, 20)}), output_shape)

    def test_lambda_config_serialization(self):
        # Test serialization with output_shape and output_shape_type
        layer = keras.layers.Lambda(
            lambda x: x + 1, output_shape=(1, 1), mask=lambda i, m: m
        )
        layer(keras.backend.variable(np.ones((1, 1))))
        config = layer.get_config()

        with SafeModeScope(safe_mode=False):
            layer = keras.layers.deserialize(
                {"class_name": "Lambda", "config": config}
            )
        self.assertAllEqual(layer.function(1), 2)
        self.assertAllEqual(layer._output_shape, (1, 1))
        self.assertAllEqual(layer.mask(1, True), True)

        layer = keras.layers.Lambda.from_config(config)
        self.assertAllEqual(layer.function(1), 2)
        self.assertAllEqual(layer._output_shape, (1, 1))
        self.assertAllEqual(layer.mask(1, True), True)

    def test_lambda_with_training_arg(self):
        def fn(x, training=True):
            return keras.backend.in_train_phase(x, 2 * x, training=training)

        layer = keras.layers.Lambda(fn)
        x = keras.backend.ones(())
        train_out = layer(x, training=True)
        eval_out = layer(x, training=False)

        self.assertEqual(keras.backend.get_value(train_out), 1.0)
        self.assertEqual(keras.backend.get_value(eval_out), 2.0)

    def test_lambda_with_mask(self):
        def add_one(inputs):
            return inputs + 1.0

        def mask(unused_inputs, previous_mask):
            return previous_mask

        layer = keras.layers.Lambda(add_one, mask=mask)
        x = np.ones([5, 4, 3])
        x[:, -1, :] = 0
        masking = keras.layers.Masking()
        out = layer(masking(x))

        expected_out = np.full([5, 4, 3], 2.0)
        expected_out[:, -1, :] = 1.0
        expected_mask = np.ones([5, 4])
        expected_mask[:, -1] = 0.0

        self.assertAllClose(self.evaluate(out), expected_out)
        self.assertIsNotNone(out._keras_mask)
        self.assertAllClose(self.evaluate(out._keras_mask), expected_mask)

    def test_lambda_with_ragged_input(self):
        def add_one(inputs):
            return inputs + 1.0

        layer = keras.layers.Lambda(add_one)

        ragged_input = tf.ragged.constant([[1.0], [2.0, 3.0]])
        out = layer(ragged_input)
        expected_out = tf.ragged.constant([[2.0], [3.0, 4.0]])
        self.assertAllClose(out, expected_out)

    def test_lambda_deserialization_does_not_pollute_core(self):
        layer = keras.layers.Lambda(lambda x: x + 1)
        config = layer.get_config()
        keras.layers.Lambda.from_config(config)
        self.assertNotIn(self.__class__.__name__, dir(core))


class TestStatefulLambda(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_lambda_with_variable_in_model(self):
        v = tf.Variable(1.0, trainable=True)

        def lambda_fn(x, v):
            return x * v

        # While it is generally not advised to mix Variables with Lambda layers,
        # if the variables are explicitly set as attributes then they are still
        # tracked. This is consistent with the base Layer behavior.
        layer = keras.layers.Lambda(lambda_fn, arguments={"v": v})
        self.assertLen(layer.trainable_weights, 0)
        layer.v = v
        self.assertLen(layer.trainable_weights, 1)

        model = test_utils.get_model_from_layers([layer], input_shape=(10,))
        model.compile(
            keras.optimizers.legacy.gradient_descent.SGD(0.1),
            "mae",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x, y = np.ones((10, 10), "float32"), 2 * np.ones((10, 10), "float32")
        model.fit(x, y, batch_size=2, epochs=2, validation_data=(x, y))
        self.assertLen(model.trainable_weights, 1)
        self.assertAllClose(
            keras.backend.get_value(model.trainable_weights[0]), 2.0
        )

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_creation_inside_lambda(self):
        def lambda_fn(x):
            scale = tf.Variable(1.0, trainable=True, name="scale")
            shift = tf.Variable(1.0, trainable=True, name="shift")
            return x * scale + shift

        expected_error = textwrap.dedent(
            r"""
(    )?The following Variables were created within a Lambda layer \(shift_and_scale\)"""  # noqa: E501
            r"""
(    )?but are not tracked by said layer:
(    )?  <tf.Variable \'.*shift_and_scale/scale:0\'.+
(    )?  <tf.Variable \'.*shift_and_scale/shift:0\'.+
(    )?The layer cannot safely ensure proper Variable reuse.+"""
        )

        with self.assertRaisesRegex(ValueError, expected_error):
            layer = keras.layers.Lambda(lambda_fn, name="shift_and_scale")
            model = test_utils.get_model_from_layers([layer], input_shape=(1,))
            model(tf.ones((4, 1)))

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_transitive_variable_creation(self):
        dense = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")

        def bad_lambda_fn(x):
            return dense(x + 1)  # Dense layer is built on first call

        expected_error = textwrap.dedent(
            r"""
(    )?The following Variables were created within a Lambda layer \(bias_dense\)
(    )?but are not tracked by said layer:
(    )?  <tf.Variable \'.*bias_dense/dense/kernel:0\'.+
(    )?The layer cannot safely ensure proper Variable reuse.+"""
        )

        with self.assertRaisesRegex(ValueError, expected_error):
            layer = keras.layers.Lambda(bad_lambda_fn, name="bias_dense")
            model = test_utils.get_model_from_layers([layer], input_shape=(1,))
            model(tf.ones((4, 1)))

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_warns_on_variable_capture(self):
        v = tf.Variable(1.0, trainable=True)

        def lambda_fn(x):
            return x * v

        expected_warning = textwrap.dedent(
            r"""
(    )?The following Variables were used a Lambda layer\'s call \(lambda\), but
(    )?are not present in its tracked objects:
(    )?  <tf.Variable \'.*Variable:0\'.+
(    )?It is possible that this is intended behavior.+"""
        )

        layer = keras.layers.Lambda(lambda_fn)

        def patched_warn(msg):
            raise ValueError(msg)

        layer._warn = patched_warn

        with self.assertRaisesRegex(ValueError, expected_warning):
            model = test_utils.get_model_from_layers([layer], input_shape=(1,))
            model(tf.ones((4, 1)))

    @test_combinations.run_all_keras_modes
    @test_combinations.run_with_all_model_types
    def test_lambda_skip_state_variable_from_initializer(self):
        # Force the initializers to use the tf.random.Generator, which will
        # contain the state variable.
        kernel_initializer = initializers.RandomNormalV2()
        kernel_initializer._random_generator._rng_type = (
            kernel_initializer._random_generator.RNG_STATEFUL
        )
        dense = keras.layers.Dense(
            1, use_bias=False, kernel_initializer=kernel_initializer
        )

        def lambda_fn(x):
            return dense(x + 1)  # Dense layer is built on first call

        # While it is generally not advised to mix Variables with Lambda layers,
        # if the variables are explicitly set as attributes then they are still
        # tracked. This is consistent with the base Layer behavior.
        layer = keras.layers.Lambda(lambda_fn)
        layer.dense = dense

        model = test_utils.get_model_from_layers([layer], input_shape=(10,))
        model.compile(
            keras.optimizers.legacy.gradient_descent.SGD(0.1),
            "mae",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x, y = np.ones((10, 10), "float32"), 2 * np.ones((10, 10), "float32")
        model.fit(x, y, batch_size=2, epochs=2, validation_data=(x, y))
        self.assertLen(model.trainable_weights, 1)


@test_combinations.run_all_keras_modes
class CoreLayersTest(test_combinations.TestCase):
    def test_masking(self):
        test_utils.layer_test(
            keras.layers.Masking, kwargs={}, input_shape=(3, 2, 3)
        )

    def test_keras_mask(self):
        x = np.ones((10, 10))
        y = keras.layers.Masking(1.0)(x)
        self.assertTrue(hasattr(y, "_keras_mask"))
        self.assertIsNotNone(y._keras_mask)
        self.assertAllClose(self.evaluate(y._keras_mask), np.zeros((10,)))

    def test_compute_mask_with_positional_mask_arg(self):
        class MyLayer(keras.layers.Layer):
            def call(self, inputs, mask=None):
                return inputs

            def compute_mask(self, inputs, mask=None):
                if mask is not None:
                    return tf.ones(())
                else:
                    return tf.zeros(())

        x, mask = tf.ones((1, 1)), tf.ones((1, 1))
        layer = MyLayer()
        y = layer(x, mask)
        # Check that `mask` was correctly sent to `compute_mask`.
        self.assertEqual(keras.backend.get_value(y._keras_mask), 1)

    def test_activation(self):
        # with string argument
        test_utils.layer_test(
            keras.layers.Activation,
            kwargs={"activation": "relu"},
            input_shape=(3, 2),
        )

        # with function argument
        test_utils.layer_test(
            keras.layers.Activation,
            kwargs={"activation": keras.backend.relu},
            input_shape=(3, 2),
        )

    def test_dense(self):
        test_utils.layer_test(
            keras.layers.Dense, kwargs={"units": 3}, input_shape=(3, 2)
        )

        test_utils.layer_test(
            keras.layers.Dense, kwargs={"units": 3}, input_shape=(3, 4, 2)
        )

        test_utils.layer_test(
            keras.layers.Dense, kwargs={"units": 3}, input_shape=(None, None, 2)
        )

        test_utils.layer_test(
            keras.layers.Dense, kwargs={"units": 3}, input_shape=(3, 4, 5, 2)
        )

    def test_dense_output(self):
        dense_inputs = tf.convert_to_tensor(
            np.random.uniform(size=(10, 10)).astype("f")
        )
        # Create some sparse data where multiple rows and columns are missing.
        sparse_inputs = tf.SparseTensor(
            indices=np.random.randint(low=0, high=10, size=(5, 2)),
            values=np.random.uniform(size=(5,)).astype("f"),
            dense_shape=[10, 10],
        )
        sparse_inputs = tf.sparse.reorder(sparse_inputs)
        # Create some ragged data.
        ragged_inputs = tf.RaggedTensor.from_row_splits(
            np.random.uniform(size=(10, 10)).astype("f"),
            row_splits=[0, 4, 6, 6, 9, 10],
        )

        layer = keras.layers.Dense(
            5,
            kernel_initializer=keras.initializers.RandomUniform(),
            bias_initializer=keras.initializers.RandomUniform(),
            dtype="float32",
        )
        dense_outputs = layer(dense_inputs)
        sparse_outpus = layer(sparse_inputs)
        ragged_outputs = layer(ragged_inputs)

        expected_dense = tf.add(
            tf.matmul(dense_inputs, keras.backend.get_value(layer.kernel)),
            keras.backend.get_value(layer.bias),
        )
        expected_sparse = tf.add(
            tf.matmul(
                tf.sparse.to_dense(sparse_inputs),
                keras.backend.get_value(layer.kernel),
            ),
            keras.backend.get_value(layer.bias),
        )
        expected_ragged_values = tf.add(
            tf.matmul(
                ragged_inputs.flat_values, keras.backend.get_value(layer.kernel)
            ),
            keras.backend.get_value(layer.bias),
        )
        expected_ragged = tf.RaggedTensor.from_row_splits(
            expected_ragged_values, row_splits=[0, 4, 6, 6, 9, 10]
        )

        self.assertAllClose(dense_outputs, expected_dense)
        self.assertAllClose(sparse_outpus, expected_sparse)
        self.assertAllClose(ragged_outputs, expected_ragged)

    def test_dense_dtype(self):
        inputs = tf.convert_to_tensor(
            np.random.randint(low=0, high=7, size=(2, 2))
        )
        layer = keras.layers.Dense(5, dtype="float32")
        outputs = layer(inputs)
        self.assertEqual(outputs.dtype, "float32")

    def test_dense_with_policy(self):
        inputs = tf.convert_to_tensor(
            np.random.randint(low=0, high=7, size=(2, 2))
        )
        layer = keras.layers.Dense(5, dtype=policy.Policy("mixed_float16"))
        outputs = layer(inputs)
        output_signature = layer.compute_output_signature(
            tf.TensorSpec(dtype="float16", shape=(2, 2))
        )
        self.assertEqual(output_signature.dtype, tf.float16)
        self.assertEqual(output_signature.shape, (2, 5))
        self.assertEqual(outputs.dtype, "float16")
        self.assertEqual(layer.kernel.dtype, "float32")

    def test_dense_regularization(self):
        layer = keras.layers.Dense(
            3,
            kernel_regularizer=keras.regularizers.l1(0.01),
            bias_regularizer="l1",
            activity_regularizer="l2",
            name="dense_reg",
        )
        layer(keras.backend.variable(np.ones((2, 4))))
        self.assertEqual(3, len(layer.losses))

    def test_dense_constraints(self):
        k_constraint = keras.constraints.max_norm(0.01)
        b_constraint = keras.constraints.max_norm(0.01)
        layer = keras.layers.Dense(
            3, kernel_constraint=k_constraint, bias_constraint=b_constraint
        )
        layer(keras.backend.variable(np.ones((2, 4))))
        self.assertEqual(layer.kernel.constraint, k_constraint)
        self.assertEqual(layer.bias.constraint, b_constraint)

    def test_dense_layer_ragged_tensor(self):
        layer = keras.layers.Dense(2, kernel_initializer="ones", use_bias=False)

        # a.shape = [2, None, 2]; a.ragged_rank=1
        a = tf.ragged.constant(
            [[[1.0, 2], [3, 4], [5, 6]], [[7, 8]]], ragged_rank=1
        )
        a_out = layer(a)
        keras.backend.get_value(layer.kernel)  # ensures var is built in TF 1.x.
        self.assertAllEqual(a_out, [[[3.0, 3], [7, 7], [11, 11]], [[15, 15]]])

        # b.shape = [4, 2]; b.ragged_rank=1
        b = tf.RaggedTensor.from_uniform_row_length(
            [1.0, 2, 3, 4, 5, 6, 7, 8], 2
        )
        self.assertAllEqual(layer(b), [[3.0, 3], [7, 7], [11, 11], [15, 15]])

        # c.shape = [2, 2, 2]; c.ragged_rank=2
        c = tf.RaggedTensor.from_uniform_row_length(b, 2)
        self.assertAllEqual(
            layer(c), [[[3.0, 3], [7, 7]], [[11, 11], [15, 15]]]
        )

    def test_dense_layer_ragged_tensor_savedmodel(self):
        # Check that we don't get a deadlock when saving a Keras model with
        # a dense layer that processes RaggedTensors.  (This happened because
        # Dense.call() had a recursive call, which is not currently supported
        # by the @tf.function decorator.)

        class TestModel(keras.Model):
            def __init__(self):
                super().__init__()
                self._layer = keras.layers.Dense(
                    1, kernel_initializer="ones", use_bias=False
                )

            def call(self, inputs):
                return self._layer(inputs)

        model = TestModel()
        result = model(
            tf.RaggedTensor.from_row_lengths([[1.0], [2], [3]], [1, 2])
        )
        keras.backend.get_value(model._layer.kernel)  # required in TF 1.x.
        self.assertAllClose(result, [[[1.0]], [[2.0], [3.0]]])
        model.save(
            os.path.join(self.get_temp_dir(), "savedmodel"), save_format="tf"
        )

    def test_dense_layer_unsupported_ragged_tensor_error(self):
        layer = keras.layers.Dense(2)
        with self.assertRaisesRegex(
            ValueError,
            "The last dimension of the inputs to a Dense layer should "
            r"be defined. Found None. Full input shape received: .*",
        ):
            layer(tf.ragged.constant([[1.0, 2], [3, 4, 5]]))
        with self.assertRaisesRegex(
            ValueError,
            "Dense layer only supports RaggedTensors when the "
            r"innermost dimension is non-ragged. Received: inputs.shape=.*",
        ):
            layer.call(tf.ragged.constant([[1.0, 2], [3, 4, 5]]))


@test_combinations.run_all_keras_modes
class TFOpLambdaTest(test_combinations.TestCase):
    def test_non_tf_symbol(self):
        def dummy_func(a, b):
            return a + b

        layer = core.TFOpLambda(dummy_func)
        self.assertIsNone(layer.symbol)
        self.assertEqual(layer.name, "dummy_func")

        with self.assertRaisesRegex(
            ValueError, "was generated from .*dummy_func"
        ):
            layer.get_config()


if __name__ == "__main__":
    tf.test.main()

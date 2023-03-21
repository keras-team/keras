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
"""Tests for LSTM layer."""


import copy
import os
import shutil

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.layers.rnn import gru_lstm_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import np_utils

# isort: off
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import (
    test_util as tf_test_util,
)

# Global config for grappler setting that is used for graph mode test.
_rewrites = rewriter_config_pb2.RewriterConfig()
_rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
_rewrites.min_graph_nodes = -1
_graph_options = tf.compat.v1.GraphOptions(rewrite_options=_rewrites)
_config = tf.compat.v1.ConfigProto(graph_options=_graph_options)


@test_combinations.run_all_keras_modes(config=_config)
class LSTMGraphRewriteTest(test_combinations.TestCase):

    input_shape = 10
    output_shape = 8
    rnn_state_size = 8
    timestep = 4
    batch = 100
    epoch = 1

    @parameterized.named_parameters(
        ("non_tan_activation", "relu", "sigmoid", 0, False, True),
        ("non_sigmoid_recur_activation", "tanh", "relu", 0, False, True),
        ("use_recurrent_dropout", "tanh", "sigmoid", 0.1, False, True),
        ("unroll", "tanh", "sigmoid", 0, True, True),
        ("not_use_bias", "tanh", "sigmoid", 0, False, False),
    )
    @test_utils.run_v2_only
    def test_could_use_defun_backend(
        self,
        activation,
        recurrent_activation,
        recurrent_dropout,
        unroll,
        use_bias,
    ):
        layer = keras.layers.LSTM(
            1,
            activation=activation,
            recurrent_activation=recurrent_activation,
            recurrent_dropout=recurrent_dropout,
            unroll=unroll,
            use_bias=use_bias,
        )
        self.assertFalse(layer._could_use_gpu_kernel)

    @test_utils.run_v2_only
    def test_use_on_default_activation_with_gpu_kernel(self):
        layer = keras.layers.LSTM(1, activation=tf.tanh)
        self.assertTrue(layer._could_use_gpu_kernel)

        layer = keras.layers.LSTM(1, recurrent_activation=tf.sigmoid)
        self.assertTrue(layer._could_use_gpu_kernel)

    def test_static_shape_inference_LSTM(self):
        # GitHub issue: 15165
        timesteps = 3
        embedding_dim = 4
        units = 2

        model = keras.models.Sequential()
        inputs = keras.layers.Dense(
            embedding_dim, input_shape=(timesteps, embedding_dim)
        )
        model.add(inputs)
        layer = keras.layers.LSTM(units, return_sequences=True)
        model.add(layer)
        outputs = model.layers[-1].output
        self.assertEqual(outputs.shape.as_list(), [None, timesteps, units])

    def test_dynamic_behavior_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer = keras.layers.LSTM(units, input_shape=(None, embedding_dim))
        model = keras.models.Sequential()
        model.add(layer)
        model.compile(tf.compat.v1.train.GradientDescentOptimizer(0.001), "mse")
        x = np.random.random((num_samples, timesteps, embedding_dim))
        y = np.random.random((num_samples, units))
        model.train_on_batch(x, y)

    def test_stacking_LSTM(self):
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(10, return_sequences=True, unroll=False))
        model.add(keras.layers.LSTM(5, return_sequences=True, unroll=False))
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    def test_from_config_LSTM(self):
        layer_class = keras.layers.LSTM
        for stateful in (False, True):
            l1 = layer_class(units=1, stateful=stateful)
            l2 = layer_class.from_config(l1.get_config())
            assert l1.get_config() == l2.get_config()

    def test_specify_initial_state_keras_tensor(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        # Test with Keras tensor
        inputs = keras.Input((timesteps, embedding_dim))
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        layer = keras.layers.LSTM(units)
        if len(initial_state) == 1:
            output = layer(inputs, initial_state=initial_state[0])
        else:
            output = layer(inputs, initial_state=initial_state)
        self.assertTrue(
            any(
                initial_state[0] is t
                for t in layer._inbound_nodes[0].input_tensors
            )
        )

        model = keras.models.Model([inputs] + initial_state, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.train_on_batch([inputs] + initial_state, targets)

    def test_specify_initial_state_non_keras_tensor(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        # Test with non-Keras tensor
        inputs = keras.Input((timesteps, embedding_dim))
        initial_state = [
            keras.backend.random_normal_variable((num_samples, units), 0, 1)
            for _ in range(num_states)
        ]
        layer = keras.layers.LSTM(units)
        output = layer(inputs, initial_state=initial_state)

        model = keras.models.Model(inputs, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        targets = np.random.random((num_samples, units))
        model.train_on_batch(inputs, targets)

    def test_reset_states_with_values(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        layer = keras.layers.LSTM(units, stateful=True)
        layer.build((num_samples, timesteps, embedding_dim))
        initial_weight_count = len(layer.weights)
        layer.reset_states()
        assert len(layer.states) == num_states
        assert layer.states[0] is not None
        self.assertAllClose(
            keras.backend.eval(layer.states[0]),
            np.zeros(keras.backend.int_shape(layer.states[0])),
            atol=1e-4,
        )
        state_shapes = [
            keras.backend.int_shape(state) for state in layer.states
        ]
        values = [np.ones(shape) for shape in state_shapes]
        if len(values) == 1:
            values = values[0]
        layer.reset_states(values)
        self.assertAllClose(
            keras.backend.eval(layer.states[0]),
            np.ones(keras.backend.int_shape(layer.states[0])),
            atol=1e-4,
        )

        # Test with invalid data
        with self.assertRaises(ValueError):
            layer.reset_states([1] * (len(layer.states) + 1))

        self.assertEqual(initial_weight_count, len(layer.weights))
        # Variables in "states" shouldn't show up in .weights
        layer.states = tf.nest.map_structure(tf.Variable, values)
        layer.reset_states()
        self.assertEqual(initial_weight_count, len(layer.weights))

    def test_specify_state_with_masking(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        inputs = keras.Input((timesteps, embedding_dim))
        _ = keras.layers.Masking()(inputs)
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        output = keras.layers.LSTM(units)(inputs, initial_state=initial_state)

        model = keras.models.Model([inputs] + initial_state, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.train_on_batch([inputs] + initial_state, targets)

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message=(
            "Skipping as ROCm MIOpen does not support padded input yet."
        ),
    )
    def test_return_state(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        inputs = keras.Input(
            batch_shape=(num_samples, timesteps, embedding_dim)
        )
        masked = keras.layers.Masking()(inputs)
        layer = keras.layers.LSTM(units, return_state=True, stateful=True)
        outputs = layer(masked)
        state = outputs[1:]
        assert len(state) == num_states
        model = keras.models.Model(inputs, state[0])

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        state = model.predict(inputs)
        self.assertAllClose(
            keras.backend.eval(layer.states[0]), state, atol=1e-4
        )

    def test_state_reuse(self):
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        inputs = keras.Input(
            batch_shape=(num_samples, timesteps, embedding_dim)
        )
        layer = keras.layers.LSTM(
            units, return_state=True, return_sequences=True
        )
        outputs = layer(inputs)
        output, state = outputs[0], outputs[1:]
        output = keras.layers.LSTM(units)(output, initial_state=state)
        model = keras.models.Model(inputs, output)

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        model.predict(inputs)

    def test_initial_states_as_other_inputs(self):
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2
        num_states = 2
        layer_class = keras.layers.LSTM

        # Test with Keras tensor
        main_inputs = keras.Input((timesteps, embedding_dim))
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        inputs = [main_inputs] + initial_state

        layer = layer_class(units)
        output = layer(inputs)
        self.assertTrue(
            any(
                initial_state[0] is t
                for t in layer._inbound_nodes[0].input_tensors
            )
        )

        model = keras.models.Model(inputs, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )

        main_inputs = np.random.random((num_samples, timesteps, embedding_dim))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.train_on_batch([main_inputs] + initial_state, targets)

    @parameterized.named_parameters(("v0", 0), ("v1", 1), ("v2", 2))
    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message=(
            "Skipping as ROCm MIOpen does not support padded input yet."
        ),
    )
    def test_implementation_mode_LSTM(self, implementation_mode):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={"units": units, "implementation": implementation_mode},
            input_shape=(num_samples, timesteps, embedding_dim),
        )

        layer_class = keras.layers.LSTM
        k_constraint = keras.constraints.max_norm(0.01)
        r_constraint = keras.constraints.max_norm(0.01)
        b_constraint = keras.constraints.max_norm(0.01)
        layer = layer_class(
            5,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_constraint=k_constraint,
            recurrent_constraint=r_constraint,
            bias_constraint=b_constraint,
        )
        layer.build((None, None, embedding_dim))
        self.assertEqual(layer.cell.kernel.constraint, k_constraint)
        self.assertEqual(layer.cell.recurrent_kernel.constraint, r_constraint)
        self.assertEqual(layer.cell.bias.constraint, b_constraint)

        layer_class = keras.layers.LSTM
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(input_shape=(3, 4)))
        model.add(layer_class(units=5, return_sequences=True, unroll=False))
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message=(
            "Skipping as ROCm MIOpen does not support padded input yet."
        ),
    )
    def test_masking_with_stacking_LSTM(self):
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(input_shape=(3, 4)))
        model.add(keras.layers.LSTM(10, return_sequences=True, unroll=False))
        model.add(keras.layers.LSTM(5, return_sequences=True, unroll=False))
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
        )
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    @parameterized.named_parameters(
        # test_name, use_bias, bias_initializer, activation
        ("normal", True, "zeros"),
        ("no_bias", False, "zeros"),
        ("random_bias", True, "random_uniform"),
    )
    def test_lstm_model_save_load(self, use_bias, bias_initializer):
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir)
        h5_path = os.path.join(temp_dir, "test.h5")

        batch = 10
        timestep = 3
        input_dim = 5
        units = 2

        x = np.random.random((batch, timestep, input_dim))

        def build_model():
            inputs = keras.layers.Input(
                shape=[timestep, input_dim], dtype=tf.float32
            )
            layer = keras.layers.LSTM(
                units, use_bias=use_bias, bias_initializer=bias_initializer
            )
            output = layer(inputs)
            return keras.models.Model(inputs, output), layer

        model, layer = build_model()
        y_ref = model.predict(x)
        model.save_weights(h5_path)

        cloned_model, new_layer = build_model()
        cloned_model.load_weights(h5_path)
        y = cloned_model.predict(x)

        self.assertAllClose(y, y_ref)
        self.assertAllClose(layer.get_weights(), new_layer.get_weights())

    def test_lstm_output_on_multiple_kernel(self):
        x_train = np.random.random(
            (self.batch, self.timestep, self.input_shape)
        )

        inputs = keras.layers.Input(
            shape=[self.timestep, self.input_shape], dtype=tf.float32
        )
        with test_utils.device(should_use_gpu=False):
            layer = keras.layers.LSTM(self.rnn_state_size)
            output = layer(inputs)
            cpu_model = keras.models.Model(inputs, output)
            weights = cpu_model.get_weights()
        y_1 = cpu_model.predict(x_train)

        with test_utils.device(should_use_gpu=True):
            layer = keras.layers.LSTM(self.rnn_state_size)
            output = layer(inputs)
            gpu_model = keras.models.Model(inputs, output)
            gpu_model.set_weights(weights)
        y_2 = gpu_model.predict(x_train)

        self.assertAllClose(y_1, y_2)

    def test_return_sequences_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={"units": units, "return_sequences": True},
            input_shape=(num_samples, timesteps, embedding_dim),
        )

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message="Skipping as ROCm MIOpen does not support float64 yet.",
    )
    @test_utils.run_v2_only
    def test_float64_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={
                "units": units,
                "return_sequences": True,
                "dtype": "float64",
            },
            input_shape=(num_samples, timesteps, embedding_dim),
            input_dtype="float64",
        )

    def test_regularizers_LSTM(self):
        embedding_dim = 4
        layer_class = keras.layers.LSTM
        layer = layer_class(
            5,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_regularizer=keras.regularizers.l1(0.01),
            recurrent_regularizer=keras.regularizers.l1(0.01),
            bias_regularizer="l2",
            activity_regularizer="l1",
        )
        layer.build((None, None, 2))
        self.assertEqual(len(layer.losses), 3)
        x = keras.backend.variable(np.ones((2, 3, 2)))
        layer(x)
        if tf.executing_eagerly():
            self.assertEqual(len(layer.losses), 4)
        else:
            self.assertEqual(len(layer.get_losses_for(x)), 1)

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message=(
            "Skipping as ROCm MIOpen does not support padded input yet."
        ),
    )
    def test_statefulness_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer_class = keras.layers.LSTM
        model = keras.models.Sequential()
        model.add(
            keras.layers.Embedding(
                4,
                embedding_dim,
                mask_zero=True,
                input_length=timesteps,
                batch_input_shape=(num_samples, timesteps),
            )
        )
        layer = layer_class(
            units, return_sequences=False, stateful=True, weights=None
        )
        model.add(layer)
        model.compile(
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        out1 = model.predict(np.ones((num_samples, timesteps)))
        self.assertEqual(out1.shape, (num_samples, units))

        # train once so that the states change
        model.train_on_batch(
            np.ones((num_samples, timesteps)), np.ones((num_samples, units))
        )
        out2 = model.predict(np.ones((num_samples, timesteps)))

        # if the state is not reset, output should be different
        self.assertNotEqual(out1.max(), out2.max())

        # check that output changes after states are reset
        # (even though the model itself didn't change)
        layer.reset_states()
        out3 = model.predict(np.ones((num_samples, timesteps)))
        self.assertNotEqual(out2.max(), out3.max())

        # check that container-level reset_states() works
        model.reset_states()
        out4 = model.predict(np.ones((num_samples, timesteps)))
        self.assertAllClose(out3, out4, atol=1e-5)

        # check that the call to `predict` updated the states
        out5 = model.predict(np.ones((num_samples, timesteps)))
        self.assertNotEqual(out4.max(), out5.max())

        # Check masking
        layer.reset_states()

        left_padded_input = np.ones((num_samples, timesteps))
        left_padded_input[0, :1] = 0
        left_padded_input[1, :2] = 0
        out6 = model.predict(left_padded_input)

        layer.reset_states()

        right_padded_input = np.ones((num_samples, timesteps))
        right_padded_input[0, -1:] = 0
        right_padded_input[1, -2:] = 0
        out7 = model.predict(right_padded_input)

        layer.reset_states()

        mix_padded_input = np.ones((num_samples, timesteps))
        mix_padded_input[0, 1] = 0
        mix_padded_input[1, 0] = 0
        mix_padded_input[1, 2] = 0
        out8 = model.predict(mix_padded_input)

        self.assertAllClose(out7, out6, atol=1e-5)
        self.assertAllClose(out8, out7, atol=1e-5)

    def test_stateful_LSTM_training(self):
        # See b/123587692 for more context.
        vocab_size = 20
        embedding_dim = 10
        batch_size = 8
        timestep = 12
        units = 5
        x = np.random.randint(0, vocab_size, size=(batch_size, timestep))
        y = np.random.randint(0, vocab_size, size=(batch_size, timestep))

        model = keras.Sequential(
            [
                keras.layers.Embedding(
                    vocab_size,
                    embedding_dim,
                    batch_input_shape=[batch_size, timestep],
                ),
                keras.layers.LSTM(units, return_sequences=True, stateful=True),
                keras.layers.Dense(vocab_size),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(x, y, epochs=1, shuffle=False)

    def test_dropout_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={"units": units, "dropout": 0.1, "recurrent_dropout": 0.1},
            input_shape=(num_samples, timesteps, embedding_dim),
        )

    def test_bidirectional(self):
        batch = 128
        timestep = 20
        vocab_size = 1000
        model = keras.Sequential(
            [
                keras.layers.Embedding(vocab_size, 64),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(64, return_sequences=True)
                ),
                keras.layers.Bidirectional(keras.layers.LSTM(32)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        x = np.random.randint(0, vocab_size, size=(batch, timestep))
        y = np.random.randint(0, 1, size=(batch))
        model.fit(x, y, epochs=1, shuffle=False)
        model.evaluate(x, y)
        model.predict(x)

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message=(
            "Skipping as ROCm MIOpen does not support padded input yet."
        ),
    )
    @test_utils.run_v2_only
    def test_explicit_device_with_go_backward_and_mask(self):
        batch_size = 8
        timestep = 7
        masksteps = 5
        units = 4

        inputs = np.random.randn(batch_size, timestep, units).astype(np.float32)
        mask = np.ones((batch_size, timestep)).astype(bool)
        mask[:, masksteps:] = 0

        lstm_layer = keras.layers.LSTM(
            units, return_sequences=True, go_backwards=True
        )
        with test_utils.device(should_use_gpu=True):
            outputs_masked = lstm_layer(inputs, mask=tf.constant(mask))
            outputs_trimmed = lstm_layer(inputs[:, :masksteps])
        self.assertAllClose(outputs_masked[:, -masksteps:], outputs_trimmed)

    @tf_test_util.enable_output_all_intermediates
    def test_v1_session_behavior(self):
        with tf.compat.v1.get_default_graph().as_default():
            # See b/139132348 for more details.
            x = np.random.uniform(size=(100, 4, 8))
            y = np.random.uniform(size=(100, 1))
            dataset = (
                tf.data.Dataset.from_tensor_slices((x, y))
                .shuffle(100)
                .batch(32)
            )

            inp = keras.layers.Input(shape=(4, 8))
            layer = keras.layers.LSTM(1)(inp)
            layer = keras.layers.Dense(1)(layer)

            model = keras.models.Model(inp, layer)

            model.compile(loss="mse", optimizer="sgd")
            model.fit(dataset)

    def test_with_fully_masked_inputs(self):
        num_samples = 8
        timestep = 5
        embedding_dim = 4
        vocab_size = 20
        units = 2

        inputs = np.random.randint(0, vocab_size, size=(num_samples, timestep))
        # Set the first inputs to be fully zero.
        inputs[0, :] = 0.0

        model = keras.models.Sequential()
        model.add(
            keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                mask_zero=True,
                input_length=timestep,
                batch_input_shape=(num_samples, timestep),
            )
        )
        layer = keras.layers.LSTM(units)
        model.add(layer)
        model.compile(
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        # Make sure it doesn't crash with cudnn kernel.
        model.predict(inputs)

    # TODO (b/169895267): test with xla_gpu is disabled.
    def test_deepcopy(self):
        if not tf.executing_eagerly():
            self.skipTest("v2-only test")
        original_layer = keras.layers.LSTM(5)
        copied_layer = copy.deepcopy(original_layer)
        self.assertEqual(copied_layer.units, 5)
        self.assertEqual(
            original_layer.get_config(), original_layer.get_config()
        )

        # Copy layer before layer call on inputs without weight initialization.
        inputs = np.random.normal(size=[32, 10, 8]).astype(np.float32)
        original_layer = keras.layers.LSTM(4)
        copied_layer = copy.deepcopy(original_layer)
        outputs = original_layer(inputs)
        copied_outputs = copied_layer(inputs)
        self.assertNotAllClose(
            self.evaluate(outputs), self.evaluate(copied_outputs)
        )

        # Copy layer after layer call on inputs with weight initialization.
        original_layer = keras.layers.LSTM(4)
        outputs = original_layer(inputs)
        copied_layer = copy.deepcopy(original_layer)
        copied_outputs = copied_layer(inputs)
        self.assertAllClose(
            self.evaluate(outputs), self.evaluate(copied_outputs)
        )

    def _test_runtime_with_model(self, model):

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=self.batch,
            test_samples=0,
            input_shape=(self.timestep, self.input_shape),
            num_classes=self.output_shape,
        )
        y_train = np_utils.to_categorical(y_train, self.output_shape)

        model.compile(
            optimizer="sgd",
            loss=["categorical_crossentropy", None],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        existing_loss = 0
        for _ in range(self.epoch):
            history = model.fit(x_train, y_train)
            loss_value = history.history["loss"][0]

            self.assertNotEqual(existing_loss, loss_value)
            existing_loss = loss_value

        _, runtime_value = model.predict(x_train)
        if not tf.sysconfig.get_build_info()["is_rocm_build"]:
            if tf.test.is_gpu_available():
                self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_GPU)
            else:
                self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_CPU)

    @test_utils.run_v2_only
    def test_LSTM_runtime(self):
        layer = keras.layers.LSTM(self.rnn_state_size, return_runtime=True)

        inputs = keras.layers.Input(
            shape=[self.timestep, self.input_shape], dtype=tf.float32
        )

        outputs, runtime = layer(inputs)
        # Expand the runtime so that it is a 1D tensor instead of scalar.
        # TF model does not work with scalar model output, specially during
        # aggregation.
        runtime = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            runtime
        )
        model = keras.models.Model(inputs=inputs, outputs=[outputs, runtime])
        self._test_runtime_with_model(model)

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message=(
            "Skipping as ROCm MIOpen does not support padded input yet."
        ),
    )
    @test_utils.run_v2_only
    def test_LSTM_runtime_with_mask(self):
        # Masking will affect which backend is selected based on whether the
        # mask is strictly right padded.
        layer = keras.layers.LSTM(self.rnn_state_size, return_runtime=True)

        inputs = keras.layers.Input(
            shape=[self.timestep, self.input_shape], dtype=tf.float32
        )
        masked_inputs = keras.layers.Masking()(inputs)

        outputs, runtime = layer(masked_inputs)
        # Expand the runtime so that it is a 1D tensor instead of scalar.
        # TF model does not work with scalar model output, specially during
        # aggregation.
        runtime = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            runtime
        )
        model = keras.models.Model(inputs=inputs, outputs=[outputs, runtime])

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=self.batch,
            test_samples=0,
            input_shape=(self.timestep, self.input_shape),
            num_classes=self.output_shape,
        )
        y_train = np_utils.to_categorical(y_train, self.output_shape)

        model.compile(
            optimizer="sgd",
            loss=["categorical_crossentropy", None],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        model.fit(x_train, y_train)

        # Verify unpadded data.
        _, runtime_value = model.predict(x_train)
        if tf.test.is_gpu_available():
            self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_GPU)
        else:
            self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_CPU)

        # Update x/y to be right padded by setting the last timestep to 0
        x_train[:, -1, :] = 0
        y_train[:, -1] = 0
        _, runtime_value = model.predict(x_train)
        if tf.test.is_gpu_available():
            self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_GPU)
        else:
            self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_CPU)

        # Further update x/y to be mix padded (masks in the middle), and verify
        # only cpu kernel can be selected.
        x_train[:, -3, :] = 0
        y_train[:, -3] = 0
        _, runtime_value = model.predict(x_train)
        self.assertEqual(runtime_value[0], gru_lstm_utils.RUNTIME_CPU)

    @test_utils.run_v2_only
    def test_LSTM_runtime_with_cond(self):
        # This test is to demonstrate the graph rewrite of grappler plugin under
        # the condition that the function returns different number of internal
        # states.
        layer = keras.layers.LSTM(self.rnn_state_size, return_runtime=True)

        inputs = keras.layers.Input(
            shape=[self.timestep, self.input_shape], dtype=tf.float32
        )

        zeros = tf.zeros([self.batch, self.output_shape])
        dummy_runtime = gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_UNKNOWN)
        a = tf.constant(0)
        b = tf.constant(1)
        # Will always run the lstm layer.
        outputs, runtime = tf.cond(
            tf.less(a, b), lambda: layer(inputs), lambda: (zeros, dummy_runtime)
        )

        # Expand the runtime so that it is a 1D tensor instead of scalar.
        # TF model does not work with scalar model output, specially during
        # aggregation.
        runtime = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            runtime
        )
        model = keras.models.Model(inputs=inputs, outputs=[outputs, runtime])
        self._test_runtime_with_model(model)


@test_combinations.run_all_keras_modes
class LSTMLayerTest(test_combinations.TestCase):
    def test_return_sequences_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={"units": units, "return_sequences": True},
            input_shape=(num_samples, timesteps, embedding_dim),
        )

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message="Double type is yet not supported in ROCm",
    )
    @test_utils.run_v2_only
    def test_float64_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={
                "units": units,
                "return_sequences": True,
                "dtype": "float64",
            },
            input_shape=(num_samples, timesteps, embedding_dim),
            input_dtype="float64",
        )

    def test_static_shape_inference_LSTM(self):
        # GitHub issue: 15165
        timesteps = 3
        embedding_dim = 4
        units = 2

        model = keras.models.Sequential()
        inputs = keras.layers.Dense(
            embedding_dim, input_shape=(timesteps, embedding_dim)
        )
        model.add(inputs)
        layer = keras.layers.LSTM(units, return_sequences=True)
        model.add(layer)
        outputs = model.layers[-1].output
        self.assertEqual(outputs.shape.as_list(), [None, timesteps, units])

    def test_dynamic_behavior_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer = keras.layers.LSTM(units, input_shape=(None, embedding_dim))
        model = keras.models.Sequential()
        model.add(layer)
        model.compile(
            "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
        )

        x = np.random.random((num_samples, timesteps, embedding_dim))
        y = np.random.random((num_samples, units))
        model.train_on_batch(x, y)

    def test_dropout_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={"units": units, "dropout": 0.1, "recurrent_dropout": 0.1},
            input_shape=(num_samples, timesteps, embedding_dim),
        )

    def test_recurrent_dropout_with_implementation_restriction(self):
        layer = keras.layers.LSTM(2, recurrent_dropout=0.1, implementation=2)
        # The implementation is force to 1 due to the limit of
        # recurrent_dropout.
        self.assertEqual(layer.implementation, 1)

    @test_utils.run_v2_only
    def test_dropout_variable_name(self):
        layer = keras.layers.RNN(
            keras.layers.LSTMCell(2, dropout=0.1, force_generator=True)
        )
        layer(np.random.random((2, 3, 4)))
        self.assertEqual(
            layer.cell._random_generator._generator._state_var.name,
            "rnn/lstm_cell/StateVar:0",
        )

        layer = keras.layers.LSTM(2, dropout=0.1, force_generator=True)
        layer(np.random.random((2, 3, 4)))
        self.assertEqual(
            layer._random_generator._generator._state_var.name,
            "lstm/StateVar:0",
        )

    @parameterized.parameters([0, 1, 2])
    def test_implementation_mode_LSTM(self, implementation_mode):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            keras.layers.LSTM,
            kwargs={"units": units, "implementation": implementation_mode},
            input_shape=(num_samples, timesteps, embedding_dim),
        )

    def test_constraints_LSTM(self):
        embedding_dim = 4
        layer_class = keras.layers.LSTM
        k_constraint = keras.constraints.max_norm(0.01)
        r_constraint = keras.constraints.max_norm(0.01)
        b_constraint = keras.constraints.max_norm(0.01)
        layer = layer_class(
            5,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_constraint=k_constraint,
            recurrent_constraint=r_constraint,
            bias_constraint=b_constraint,
        )
        layer.build((None, None, embedding_dim))
        self.assertEqual(layer.cell.kernel.constraint, k_constraint)
        self.assertEqual(layer.cell.recurrent_kernel.constraint, r_constraint)
        self.assertEqual(layer.cell.bias.constraint, b_constraint)

    @parameterized.parameters([True, False])
    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message="Skipping as ROCm MIOpen does not support padded input.",
    )
    def test_with_masking_layer_LSTM(self, unroll):
        layer_class = keras.layers.LSTM
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(input_shape=(3, 4)))
        model.add(layer_class(units=5, return_sequences=True, unroll=unroll))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    @parameterized.parameters([True, False])
    def test_masking_with_stacking_LSTM(self, unroll):
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(input_shape=(3, 4)))
        lstm_cells = [keras.layers.LSTMCell(10), keras.layers.LSTMCell(5)]
        model.add(
            keras.layers.RNN(lstm_cells, return_sequences=True, unroll=unroll)
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    def test_from_config_LSTM(self):
        layer_class = keras.layers.LSTM
        for stateful in (False, True):
            l1 = layer_class(units=1, stateful=stateful)
            l2 = layer_class.from_config(l1.get_config())
            assert l1.get_config() == l2.get_config()

    def test_deep_copy_LSTM(self):
        cell = keras.layers.LSTMCell(5)
        copied_cell = copy.deepcopy(cell)
        self.assertEqual(copied_cell.units, 5)
        self.assertEqual(cell.get_config(), copied_cell.get_config())

    def test_specify_initial_state_keras_tensor(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        # Test with Keras tensor
        inputs = keras.Input((timesteps, embedding_dim))
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        layer = keras.layers.LSTM(units)
        if len(initial_state) == 1:
            output = layer(inputs, initial_state=initial_state[0])
        else:
            output = layer(inputs, initial_state=initial_state)
        self.assertTrue(
            any(
                initial_state[0] is t
                for t in layer._inbound_nodes[0].input_tensors
            )
        )

        model = keras.models.Model([inputs] + initial_state, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.AdamOptimizer(),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.train_on_batch([inputs] + initial_state, targets)

    def test_specify_initial_state_non_keras_tensor(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        # Test with non-Keras tensor
        inputs = keras.Input((timesteps, embedding_dim))
        initial_state = [
            keras.backend.random_normal_variable((num_samples, units), 0, 1)
            for _ in range(num_states)
        ]
        layer = keras.layers.LSTM(units)
        output = layer(inputs, initial_state=initial_state)

        model = keras.models.Model(inputs, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.AdamOptimizer(),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        targets = np.random.random((num_samples, units))
        model.train_on_batch(inputs, targets)

    def test_reset_states_with_values(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        layer = keras.layers.LSTM(units, stateful=True)
        layer.build((num_samples, timesteps, embedding_dim))
        layer.reset_states()
        assert len(layer.states) == num_states
        assert layer.states[0] is not None
        self.assertAllClose(
            keras.backend.eval(layer.states[0]),
            np.zeros(keras.backend.int_shape(layer.states[0])),
            atol=1e-4,
        )
        state_shapes = [
            keras.backend.int_shape(state) for state in layer.states
        ]
        values = [np.ones(shape) for shape in state_shapes]
        if len(values) == 1:
            values = values[0]
        layer.reset_states(values)
        self.assertAllClose(
            keras.backend.eval(layer.states[0]),
            np.ones(keras.backend.int_shape(layer.states[0])),
            atol=1e-4,
        )

        # Test with invalid data
        with self.assertRaises(ValueError):
            layer.reset_states([1] * (len(layer.states) + 1))

    def test_specify_state_with_masking(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        inputs = keras.Input((timesteps, embedding_dim))
        _ = keras.layers.Masking()(inputs)
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        output = keras.layers.LSTM(units)(inputs, initial_state=initial_state)

        model = keras.models.Model([inputs] + initial_state, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            run_eagerly=test_utils.should_run_eagerly(),
        )

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.train_on_batch([inputs] + initial_state, targets)

    def test_return_state(self):
        num_states = 2
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        inputs = keras.Input(
            batch_shape=(num_samples, timesteps, embedding_dim)
        )
        layer = keras.layers.LSTM(units, return_state=True, stateful=True)
        outputs = layer(inputs)
        state = outputs[1:]
        assert len(state) == num_states
        model = keras.models.Model(inputs, state[0])

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        state = model.predict(inputs)
        self.assertAllClose(
            keras.backend.eval(layer.states[0]), state, atol=1e-4
        )

    def test_state_reuse(self):
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2

        inputs = keras.Input(
            batch_shape=(num_samples, timesteps, embedding_dim)
        )
        layer = keras.layers.LSTM(
            units, return_state=True, return_sequences=True
        )
        outputs = layer(inputs)
        output, state = outputs[0], outputs[1:]
        output = keras.layers.LSTM(units)(output, initial_state=state)
        model = keras.models.Model(inputs, output)

        inputs = np.random.random((num_samples, timesteps, embedding_dim))
        outputs = model.predict(inputs)

    def test_initial_states_as_other_inputs(self):
        timesteps = 3
        embedding_dim = 4
        units = 3
        num_samples = 2
        num_states = 2
        layer_class = keras.layers.LSTM

        # Test with Keras tensor
        main_inputs = keras.Input((timesteps, embedding_dim))
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        inputs = [main_inputs] + initial_state

        layer = layer_class(units)
        output = layer(inputs)
        self.assertTrue(
            any(
                initial_state[0] is t
                for t in layer._inbound_nodes[0].input_tensors
            )
        )

        model = keras.models.Model(inputs, output)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.compat.v1.train.AdamOptimizer(),
            run_eagerly=test_utils.should_run_eagerly(),
        )

        main_inputs = np.random.random((num_samples, timesteps, embedding_dim))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.train_on_batch([main_inputs] + initial_state, targets)

    def test_regularizers_LSTM(self):
        embedding_dim = 4
        layer_class = keras.layers.LSTM
        layer = layer_class(
            5,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_regularizer=keras.regularizers.l1(0.01),
            recurrent_regularizer=keras.regularizers.l1(0.01),
            bias_regularizer="l2",
            activity_regularizer="l1",
        )
        layer.build((None, None, 2))
        self.assertEqual(len(layer.losses), 3)
        x = keras.backend.variable(np.ones((2, 3, 2)))
        layer(x)
        if tf.executing_eagerly():
            self.assertEqual(len(layer.losses), 4)
        else:
            self.assertEqual(len(layer.get_losses_for(x)), 1)

    @tf.test.disable_with_predicate(
        pred=tf.test.is_built_with_rocm,
        skip_message="Skipping as ROCm MIOpen does not support padded input.",
    )
    def test_statefulness_LSTM(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer_class = keras.layers.LSTM
        model = keras.models.Sequential()
        model.add(
            keras.layers.Embedding(
                4,
                embedding_dim,
                mask_zero=True,
                input_length=timesteps,
                batch_input_shape=(num_samples, timesteps),
            )
        )
        layer = layer_class(
            units, return_sequences=False, stateful=True, weights=None
        )
        model.add(layer)
        model.compile(
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        out1 = model.predict(np.ones((num_samples, timesteps)))
        self.assertEqual(out1.shape, (num_samples, units))

        # train once so that the states change
        model.train_on_batch(
            np.ones((num_samples, timesteps)), np.ones((num_samples, units))
        )
        out2 = model.predict(np.ones((num_samples, timesteps)))

        # if the state is not reset, output should be different
        self.assertNotEqual(out1.max(), out2.max())

        # check that output changes after states are reset
        # (even though the model itself didn't change)
        layer.reset_states()
        out3 = model.predict(np.ones((num_samples, timesteps)))
        self.assertNotEqual(out2.max(), out3.max())

        # check that container-level reset_states() works
        model.reset_states()
        out4 = model.predict(np.ones((num_samples, timesteps)))
        self.assertAllClose(out3, out4, atol=1e-5)

        # check that the call to `predict` updated the states
        out5 = model.predict(np.ones((num_samples, timesteps)))
        self.assertNotEqual(out4.max(), out5.max())

        # Check masking
        layer.reset_states()

        left_padded_input = np.ones((num_samples, timesteps))
        left_padded_input[0, :1] = 0
        left_padded_input[1, :2] = 0
        out6 = model.predict(left_padded_input)

        layer.reset_states()

        right_padded_input = np.ones((num_samples, timesteps))
        right_padded_input[0, -1:] = 0
        right_padded_input[1, -2:] = 0
        out7 = model.predict(right_padded_input)

        self.assertAllClose(out7, out6, atol=1e-5)

    @test_utils.run_v2_only
    def test_cloned_weight_names(self):
        inp = keras.Input([None, 3])
        rnn = keras.layers.LSTM(units=3)
        model = keras.Model(inp, rnn(inp))
        clone = keras.models.clone_model(model)

        model_names = [x.name for x in model.weights]
        clone_names = [x.name for x in clone.weights]
        self.assertEqual(model_names, clone_names)


if __name__ == "__main__":
    tf.test.main()

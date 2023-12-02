import numpy as np
import pytest

from keras import initializers
from keras import layers
from keras import testing


class SimpleRNNTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.Bidirectional,
            init_kwargs={"layer": layers.SimpleRNN(4)},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 8),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.Bidirectional,
            init_kwargs={
                "layer": layers.SimpleRNN(4),
                "backward_layer": layers.SimpleRNN(4, go_backwards=True),
                "merge_mode": "sum",
            },
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 4),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        forward_layer = layers.SimpleRNN(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer = layers.Bidirectional(
            layer=forward_layer,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.39687276, 0.39687276, 0.10004295, 0.10004295],
                    [0.7237238, 0.7237238, 0.53391594, 0.53391594],
                ]
            ),
            output,
        )

        layer = layers.Bidirectional(layer=forward_layer, merge_mode="ave")
        output = layer(sequence)
        self.assertAllClose(
            np.array([[0.24845785, 0.24845785], [0.6288199, 0.6288199]]),
            output,
        )

        layer = layers.Bidirectional(layer=forward_layer, merge_mode=None)
        output1, output2 = layer(sequence)
        self.assertAllClose(
            np.array([[0.39687276, 0.39687276], [0.7237238, 0.7237238]]),
            output1,
        )
        self.assertAllClose(
            np.array([[0.10004295, 0.10004295], [0.53391594, 0.53391594]]),
            output2,
        )

        backward_layer = layers.SimpleRNN(
            2,
            kernel_initializer=initializers.Constant(0.03),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.01),
            go_backwards=True,
        )
        layer = layers.Bidirectional(
            layer=forward_layer, backward_layer=backward_layer, merge_mode="mul"
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array([[0.08374989, 0.08374989], [0.6740834, 0.6740834]]),
            output,
        )

        forward_layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_sequences=True,
        )
        layer = layers.Bidirectional(layer=forward_layer, merge_mode="sum")
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [
                        [0.20937867, 0.20937867],
                        [0.34462988, 0.34462988],
                        [0.40290534, 0.40290534],
                    ],
                    [
                        [0.59829646, 0.59829646],
                        [0.6734641, 0.6734641],
                        [0.6479671, 0.6479671],
                    ],
                ]
            ),
            output,
        )

    def test_statefulness(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        forward_layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            stateful=True,
        )
        layer = layers.Bidirectional(layer=forward_layer)
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.26234663, 0.26234663, 0.16959146, 0.16959146],
                    [0.6137073, 0.6137073, 0.5381646, 0.5381646],
                ]
            ),
            output,
        )
        layer.reset_state()
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.26234663, 0.26234663, 0.16959146, 0.16959146],
                    [0.6137073, 0.6137073, 0.5381646, 0.5381646],
                ]
            ),
            output,
        )

    def test_pass_initial_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        initial_state = [
            np.arange(4).reshape((2, 2)).astype("float32") * 1,
            np.arange(4).reshape((2, 2)).astype("float32") * 2,
            np.arange(4).reshape((2, 2)).astype("float32") * 3,
            np.arange(4).reshape((2, 2)).astype("float32") * 4,
        ]
        forward_layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer = layers.Bidirectional(
            layer=forward_layer,
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array(
                [
                    [0.20794602, 0.4577124, 0.14046375, 0.48191673],
                    [0.6682636, 0.6711909, 0.60943645, 0.60950446],
                ]
            ),
            output,
        )

    def test_masking(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        forward_layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer = layers.Bidirectional(layer=forward_layer)
        mask = np.array([[True, True, False, True], [True, False, False, True]])
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.19393763, 0.19393763, 0.11669192, 0.11669192],
                    [0.30818558, 0.30818558, 0.28380975, 0.28380975],
                ]
            ),
            output,
        )

    def test_return_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        forward_layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_state=True,
        )
        layer = layers.Bidirectional(layer=forward_layer)
        output, h1, c1, h2, c2 = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.1990008, 0.1990008, 0.12659755, 0.12659755],
                    [0.52335435, 0.52335435, 0.44717982, 0.44717982],
                ]
            ),
            output,
        )
        self.assertAllClose(
            np.array([[0.1990008, 0.1990008], [0.52335435, 0.52335435]]),
            h1,
        )
        self.assertAllClose(
            np.array([[0.35567185, 0.35567185], [1.0492687, 1.0492687]]),
            c1,
        )
        self.assertAllClose(
            np.array([[0.12659755, 0.12659755], [0.44717982, 0.44717982]]),
            h2,
        )
        self.assertAllClose(
            np.array([[0.2501858, 0.2501858], [0.941473, 0.941473]]),
            c2,
        )

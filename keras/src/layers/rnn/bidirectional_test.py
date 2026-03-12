import numpy as np
import pytest

from keras.src import initializers
from keras.src import layers
from keras.src import testing


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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.Bidirectional(layer=forward_layer, merge_mode="ave")
        output = layer(sequence)
        self.assertAllClose(
            np.array([[0.24845785, 0.24845785], [0.6288199, 0.6288199]]),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.Bidirectional(layer=forward_layer, merge_mode=None)
        output1, output2 = layer(sequence)
        self.assertAllClose(
            np.array([[0.39687276, 0.39687276], [0.7237238, 0.7237238]]),
            output1,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array([[0.10004295, 0.10004295], [0.53391594, 0.53391594]]),
            output2,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array([[0.1990008, 0.1990008], [0.52335435, 0.52335435]]),
            h1,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array([[0.35567185, 0.35567185], [1.0492687, 1.0492687]]),
            c1,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array([[0.12659755, 0.12659755], [0.44717982, 0.44717982]]),
            h2,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array([[0.2501858, 0.2501858], [0.941473, 0.941473]]),
            c2,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    @pytest.mark.requires_trainable_backend
    def test_output_shape(self):
        x = np.array([[[101, 202], [303, 404]]])
        for merge_mode in ["ave", "concat", "mul", "sum", None]:
            sub_layer = layers.LSTM(2, return_state=True)
            layer = layers.Bidirectional(sub_layer, merge_mode=merge_mode)
            output = layer(x)
            output_shape = layer.compute_output_shape(x.shape)
            for out, shape in zip(output, output_shape):
                self.assertEqual(out.shape, shape)

        for merge_mode in ["concat", "ave", "mul", "sum"]:
            sub_layer = layers.LSTM(2, return_state=False)
            layer = layers.Bidirectional(sub_layer, merge_mode=merge_mode)
            output = layer(x)
            output_shape = layer.compute_output_shape(x.shape)
            self.assertEqual(output.shape, output_shape)

        # return_state=False & merge_mode=None
        sub_layer = layers.LSTM(2, return_state=False)
        layer = layers.Bidirectional(sub_layer, merge_mode=None)
        output = layer(x)
        output_shape = layer.compute_output_shape(x.shape)
        for out, shape in zip(output, output_shape):
            self.assertEqual(out.shape, shape)

    def test_keeps_use_cudnn(self):
        # keep use_cudnn if the layer has it
        for rnn_class in [layers.GRU, layers.LSTM]:
            for use_cudnn in [True, False, "auto"]:
                rnn = rnn_class(1, use_cudnn=use_cudnn)
                bidi = layers.Bidirectional(rnn)
                self.assertEqual(bidi.forward_layer.use_cudnn, use_cudnn)
                self.assertEqual(bidi.backward_layer.use_cudnn, use_cudnn)

        # otherwise ignore it
        rnn = layers.SimpleRNN(1)
        bidi = layers.Bidirectional(rnn)
        self.assertFalse(hasattr(bidi.forward_layer, "use_cudnn"))
        self.assertFalse(hasattr(bidi.backward_layer, "use_cudnn"))

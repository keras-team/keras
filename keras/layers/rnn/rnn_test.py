import numpy as np
import pytest

from keras import layers
from keras import ops
from keras import testing


class OneStateRNNCell(layers.Layer):
    def __init__(self, units, state_size=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = state_size if state_size else units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="ones",
            name="kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer="ones",
            name="recurrent_kernel",
        )
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = ops.matmul(inputs, self.kernel)
        output = h + ops.matmul(prev_output, self.recurrent_kernel)
        return output, [output]


class TwoStatesRNNCell(layers.Layer):
    def __init__(self, units, state_size=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = state_size if state_size else [units, units]
        self.output_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="ones",
            name="kernel",
        )
        self.recurrent_kernel_1 = self.add_weight(
            shape=(self.units, self.units),
            initializer="ones",
            name="recurrent_kernel_1",
        )
        self.recurrent_kernel_2 = self.add_weight(
            shape=(self.units, self.units),
            initializer="ones",
            name="recurrent_kernel_2",
        )
        self.built = True

    def call(self, inputs, states):
        prev_1 = states[0]
        prev_2 = states[0]
        h = ops.matmul(inputs, self.kernel)
        output_1 = h + ops.matmul(prev_1, self.recurrent_kernel_1)
        output_2 = h + ops.matmul(prev_2, self.recurrent_kernel_2)
        output = output_1 + output_2
        return output, [output_1, output_2]


class RNNTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": OneStateRNNCell(5, state_size=5)},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": OneStateRNNCell(5, state_size=[5])},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": OneStateRNNCell(5, state_size=(5,))},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": OneStateRNNCell(5), "return_sequences": True},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 2, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={
                "cell": OneStateRNNCell(5),
                "go_backwards": True,
                "unroll": True,
            },
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": TwoStatesRNNCell(5, state_size=[5, 5])},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": TwoStatesRNNCell(5, state_size=(5, 5))},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": TwoStatesRNNCell(5), "return_sequences": True},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 2, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            supports_masking=True,
        )

    def test_compute_output_shape_single_state(self):
        sequence = np.ones((3, 4, 5))
        layer = layers.RNN(OneStateRNNCell(8), return_sequences=False)
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape, (3, 8))

        layer = layers.RNN(OneStateRNNCell(8), return_sequences=True)
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape, (3, 4, 8))

        layer = layers.RNN(
            OneStateRNNCell(8), return_sequences=False, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape[0], (3, 8))
        self.assertEqual(output_shape[1], (3, 8))

        layer = layers.RNN(
            OneStateRNNCell(8), return_sequences=True, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape[0], (3, 4, 8))
        self.assertEqual(output_shape[1], (3, 8))

    def test_compute_output_shape_two_states(self):
        sequence = np.ones((3, 4, 5))
        layer = layers.RNN(TwoStatesRNNCell(8), return_sequences=False)
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape, (3, 8))

        layer = layers.RNN(TwoStatesRNNCell(8), return_sequences=True)
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape, (3, 4, 8))

        layer = layers.RNN(
            TwoStatesRNNCell(8), return_sequences=False, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape[0], (3, 8))
        self.assertEqual(output_shape[1], (3, 8))
        self.assertEqual(output_shape[2], (3, 8))

        layer = layers.RNN(
            TwoStatesRNNCell(8), return_sequences=True, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence.shape)
        self.assertEqual(output_shape[0], (3, 4, 8))
        self.assertEqual(output_shape[1], (3, 8))
        self.assertEqual(output_shape[2], (3, 8))

    def test_dynamic_shapes(self):
        sequence_shape = (None, None, 3)
        layer = layers.RNN(OneStateRNNCell(8), return_sequences=False)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, 8))

        layer = layers.RNN(OneStateRNNCell(8), return_sequences=True)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, None, 8))

        layer = layers.RNN(
            OneStateRNNCell(8), return_sequences=False, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, 8))
        self.assertEqual(output_shape[1], (None, 8))

        layer = layers.RNN(
            OneStateRNNCell(8), return_sequences=True, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, None, 8))
        self.assertEqual(output_shape[1], (None, 8))

        layer = layers.RNN(TwoStatesRNNCell(8), return_sequences=False)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, 8))

        layer = layers.RNN(TwoStatesRNNCell(8), return_sequences=True)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, None, 8))

        layer = layers.RNN(
            TwoStatesRNNCell(8), return_sequences=False, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, 8))
        self.assertEqual(output_shape[1], (None, 8))
        self.assertEqual(output_shape[2], (None, 8))

        layer = layers.RNN(
            TwoStatesRNNCell(8), return_sequences=True, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, None, 8))
        self.assertEqual(output_shape[1], (None, 8))
        self.assertEqual(output_shape[2], (None, 8))

    def test_forward_pass_single_state(self):
        sequence = np.ones((1, 2, 3))
        layer = layers.RNN(OneStateRNNCell(2), return_sequences=False)
        output = layer(sequence)
        self.assertAllClose(np.array([[9.0, 9.0]]), output)

        layer = layers.RNN(OneStateRNNCell(2), return_sequences=True)
        output = layer(sequence)
        self.assertAllClose(np.array([[[3.0, 3.0], [9.0, 9.0]]]), output)

        layer = layers.RNN(
            OneStateRNNCell(2), return_sequences=False, return_state=True
        )
        output, state = layer(sequence)
        self.assertAllClose(np.array([[9.0, 9.0]]), output)
        self.assertAllClose(np.array([[9.0, 9.0]]), state)

        layer = layers.RNN(
            OneStateRNNCell(2), return_sequences=True, return_state=True
        )
        output, state = layer(sequence)
        self.assertAllClose(np.array([[[3.0, 3.0], [9.0, 9.0]]]), output)
        self.assertAllClose(np.array([[9.0, 9.0]]), state)

    def test_forward_pass_two_states(self):
        sequence = np.ones((1, 2, 3))
        layer = layers.RNN(TwoStatesRNNCell(2), return_sequences=False)
        output = layer(sequence)
        self.assertAllClose(np.array([[18.0, 18.0]]), output)

        layer = layers.RNN(TwoStatesRNNCell(2), return_sequences=True)
        output = layer(sequence)
        self.assertAllClose(np.array([[[6.0, 6.0], [18.0, 18.0]]]), output)

        layer = layers.RNN(
            TwoStatesRNNCell(2), return_sequences=False, return_state=True
        )
        output, state1, state2 = layer(sequence)
        self.assertAllClose(np.array([[18.0, 18.0]]), output)
        self.assertAllClose(np.array([[9.0, 9.0]]), state1)
        self.assertAllClose(np.array([[9.0, 9.0]]), state2)

        layer = layers.RNN(
            TwoStatesRNNCell(2), return_sequences=True, return_state=True
        )
        output, state1, state2 = layer(sequence)
        self.assertAllClose(np.array([[[6.0, 6.0], [18.0, 18.0]]]), output)
        self.assertAllClose(np.array([[9.0, 9.0]]), state1)
        self.assertAllClose(np.array([[9.0, 9.0]]), state2)

    def test_passing_initial_state_single_state(self):
        sequence = np.ones((2, 3, 2))
        state = np.ones((2, 2))
        layer = layers.RNN(OneStateRNNCell(2), return_sequences=False)
        output = layer(sequence, initial_state=state)
        self.assertAllClose(np.array([[22.0, 22.0], [22.0, 22.0]]), output)

        layer = layers.RNN(
            OneStateRNNCell(2), return_sequences=False, return_state=True
        )
        output, state = layer(sequence, initial_state=state)
        self.assertAllClose(np.array([[22.0, 22.0], [22.0, 22.0]]), output)
        self.assertAllClose(np.array([[22.0, 22.0], [22.0, 22.0]]), state)

    def test_passing_initial_state_two_states(self):
        sequence = np.ones((2, 3, 2))
        state = [np.ones((2, 2)), np.ones((2, 2))]
        layer = layers.RNN(TwoStatesRNNCell(2), return_sequences=False)
        output = layer(sequence, initial_state=state)
        self.assertAllClose(np.array([[44.0, 44.0], [44.0, 44.0]]), output)

        layer = layers.RNN(
            TwoStatesRNNCell(2), return_sequences=False, return_state=True
        )
        output, state_1, state_2 = layer(sequence, initial_state=state)
        self.assertAllClose(np.array([[44.0, 44.0], [44.0, 44.0]]), output)
        self.assertAllClose(np.array([[22.0, 22.0], [22.0, 22.0]]), state_1)
        self.assertAllClose(np.array([[22.0, 22.0], [22.0, 22.0]]), state_2)

    def test_statefulness_single_state(self):
        sequence = np.ones((1, 2, 3))
        layer = layers.RNN(OneStateRNNCell(2), stateful=True)
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(np.array([[45.0, 45.0]]), output)

        layer = layers.RNN(OneStateRNNCell(2), stateful=True, return_state=True)
        layer(sequence)
        output, state = layer(sequence)
        self.assertAllClose(np.array([[45.0, 45.0]]), output)
        self.assertAllClose(np.array([[45.0, 45.0]]), state)

    def test_statefulness_two_states(self):
        sequence = np.ones((1, 2, 3))
        layer = layers.RNN(TwoStatesRNNCell(2), stateful=True)
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(np.array([[90.0, 90.0]]), output)

        layer = layers.RNN(
            TwoStatesRNNCell(2), stateful=True, return_state=True
        )
        layer(sequence)
        output, state_1, state_2 = layer(sequence)
        self.assertAllClose(np.array([[90.0, 90.0]]), output)
        self.assertAllClose(np.array([[45.0, 45.0]]), state_1)
        self.assertAllClose(np.array([[45.0, 45.0]]), state_2)

    def test_go_backwards(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.RNN(OneStateRNNCell(2), go_backwards=True)
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(np.array([[202.0, 202.0], [538.0, 538.0]]), output)

        layer = layers.RNN(OneStateRNNCell(2), stateful=True, return_state=True)
        layer(sequence)
        output, state = layer(sequence)
        self.assertAllClose(
            np.array([[954.0, 954.0], [3978.0, 3978.0]]), output
        )
        self.assertAllClose(np.array([[954.0, 954.0], [3978.0, 3978.0]]), state)

    def test_serialization(self):
        layer = layers.RNN(TwoStatesRNNCell(2), return_sequences=False)
        self.run_class_serialization_test(layer)

        layer = layers.RNN(OneStateRNNCell(2), return_sequences=False)
        self.run_class_serialization_test(layer)

    # TODO: test masking

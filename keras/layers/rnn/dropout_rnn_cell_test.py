import pytest

from keras import backend
from keras import layers
from keras import ops
from keras import testing
from keras.layers.rnn.dropout_rnn_cell import DropoutRNNCell


class RNNCellWithDropout(layers.Layer, DropoutRNNCell):
    def __init__(
        self, units, dropout=0.5, recurrent_dropout=0.5, seed=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed)
        self.units = units
        self.state_size = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

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

    def call(self, inputs, states, training=False):
        if training:
            dp_mask = self.get_dropout_mask(inputs)
            inputs *= dp_mask
        prev_output = states[0]
        h = ops.matmul(inputs, self.kernel)
        if training:
            rdp_mask = self.get_recurrent_dropout_mask(prev_output)
            prev_output *= rdp_mask
        output = h + ops.matmul(prev_output, self.recurrent_kernel)
        return output, [output]


class DropoutRNNCellTest(testing.TestCase):
    def test_seed_tracking(self):
        cell = RNNCellWithDropout(3, seed=1337)
        self.assertEqual(len(cell.non_trainable_variables), 1)
        layer = layers.RNN(cell)
        self.assertEqual(len(layer.non_trainable_variables), 1)

    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": RNNCellWithDropout(5, seed=1337)},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=1,
            supports_masking=True,
            run_mixed_precision_check=False,
        )

        # manually set dtype to mixed_float16 to run mixed precision check
        run_mixed_precision_check = True
        if backend.backend() == "torch":
            import torch

            run_mixed_precision_check = torch.cuda.is_available()
        if run_mixed_precision_check:
            self.run_layer_test(
                layers.RNN,
                init_kwargs={
                    "cell": RNNCellWithDropout(
                        5, seed=1337, dtype="mixed_float16"
                    ),
                    "dtype": "mixed_float16",
                },
                input_shape=(3, 2, 4),
                call_kwargs={"training": True},
                expected_output_shape=(3, 5),
                expected_num_trainable_weights=2,
                expected_num_non_trainable_weights=0,
                expected_num_non_trainable_variables=1,
                supports_masking=True,
                run_mixed_precision_check=False,
            )

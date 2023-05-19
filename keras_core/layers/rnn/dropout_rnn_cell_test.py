from keras_core import backend
from keras_core import layers
from keras_core import operations as ops
from keras_core import testing
from keras_core.layers.rnn.dropout_rnn_cell import DropoutRNNCell


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
    def test_basics(self):
        self.run_layer_test(
            layers.RNN,
            init_kwargs={"cell": RNNCellWithDropout(5, seed=1337)},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

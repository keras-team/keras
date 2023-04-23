from keras_core import activations
from keras_core import layers
from keras_core import testing


class ActivationTest(testing.TestCase):
    def test_dense_basics(self):
        self.run_layer_test(
            layers.Activation,
            init_kwargs={
                "activation": "relu",
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.Activation,
            init_kwargs={
                "activation": activations.gelu,
            },
            input_shape=(2, 2),
            expected_output_shape=(2, 2),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

import pytest

from keras.src import activations
from keras.src import layers
from keras.src import testing


class ActivationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_activation_basics(self):
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
            assert_built_after_instantiation=True,
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
            assert_built_after_instantiation=True,
        )

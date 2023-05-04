from keras_core import layers
from keras_core import testing


class IdentityTest(testing.TestCase):
    def test_identity_basics(self):
        self.run_layer_test(
            layers.Identity,
            init_kwargs={},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

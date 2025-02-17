import pytest

from keras.src import layers
from keras.src import ops
from keras.src import testing


class ExampleWrapper(layers.Wrapper):
    """Simple Wrapper subclass."""

    def call(self, inputs, **kwargs):
        return ops.cast(self.layer(inputs, **kwargs), self.compute_dtype)


class WrapperTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_wrapper_basics(self):
        self.run_layer_test(
            ExampleWrapper,
            init_kwargs={
                "layer": layers.Dense(2),
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            ExampleWrapper,
            init_kwargs={
                "layer": layers.Dense(2, activity_regularizer="l2"),
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=1,
            supports_masking=False,
        )
        self.run_layer_test(
            ExampleWrapper,
            init_kwargs={
                "layer": layers.Dense(2),
                "activity_regularizer": "l2",
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=1,
            supports_masking=False,
        )
        self.run_layer_test(
            ExampleWrapper,
            init_kwargs={
                "layer": layers.BatchNormalization(),
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_wrapper_invalid_layer(self):
        invalid_layer = "This is not a valid Keras layer."

        with self.assertRaisesRegex(
            ValueError,
            "Layer .* supplied to Wrapper isn't a supported layer type. "
            "Please ensure wrapped layer is a valid Keras layer.",
        ):
            layers.Wrapper(invalid_layer)

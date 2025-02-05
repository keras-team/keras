import numpy as np

from keras.src import layers
from keras.src import testing


class AdditiveAttentionTest(testing.TestCase):
    def test_attention_basics(self):
        # No scale
        self.run_layer_test(
            layers.AdditiveAttention,
            init_kwargs={
                "use_scale": True,
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )
        # With scale.
        self.run_layer_test(
            layers.AdditiveAttention,
            init_kwargs={
                "use_scale": False,
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    def test_attention_correctness(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        layer = layers.AdditiveAttention(use_scale=False)
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output, [[[1.727, 2.727], [2.272, 3.272]]], atol=1e-3
        )
        self.assertAllClose(
            scores, [[[0.636, 0.363], [0.363, 0.636]]], atol=1e-3
        )

    def test_attention_with_mask(self):
        layer = layers.AdditiveAttention(use_scale=False)
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        query_mask = np.array([[True, False]])
        value_mask = np.array([[True, False]])
        output, scores = layer(
            [query, value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        self.assertAllClose(output, [[[1.0, 1.0], [0.0, 0.0]]])
        self.assertAllClose(scores, [[[1.0, 0.0], [1.0, 0.0]]])

    def test_attention_errors(self):
        layer = layers.AdditiveAttention()
        tensor = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "must be called on a list"):
            layer(tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor, tensor, tensor])

        with self.assertRaisesRegex(ValueError, "layer mask must be a list"):
            layer([tensor, tensor], mask=tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor], mask=[tensor])

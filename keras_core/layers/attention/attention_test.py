import numpy as np

from keras_core import layers
from keras_core import testing


class DenseTest(testing.TestCase):
    def test_attention_basics(self):
        # No scale, no concat.
        self.run_layer_test(
            layers.Attention,
            init_kwargs={
                "score_mode": "dot",
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # Sale and concat.
        self.run_layer_test(
            layers.Attention,
            init_kwargs={
                "use_scale": True,
                "score_mode": "concat",
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_attention_correctness(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        # Dot.
        layer = layers.Attention(score_mode="dot")
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output, [[[2.462, 3.462], [1.538, 2.538]]], atol=1e-3
        )
        self.assertAllClose(
            scores, [[[0.269, 0.731], [0.731, 0.269]]], atol=1e-3
        )

        # Concat.
        layer = layers.Attention(score_mode="concat")
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
        layer = layers.Attention()
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
        layer = layers.Attention()
        tensor = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "must be called on a list"):
            layer(tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor, tensor, tensor])

        with self.assertRaisesRegex(ValueError, "layer mask must be a list"):
            layer([tensor, tensor], mask=tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor], mask=[tensor])

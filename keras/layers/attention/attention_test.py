import numpy as np

from keras import layers
from keras import testing


class AttentionTest(testing.TestCase):
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
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )
        # Scale and concat.
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
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
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

    def test_attention_with_dropout(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        layer_with_dropout = layers.Attention(dropout=0.2)
        layer_without_dropout = layers.Attention()

        output1, scores1 = layer_with_dropout(
            [query, value], return_attention_scores=True, training=True
        )
        output2, scores2 = layer_without_dropout(
            [query, value], return_attention_scores=True, training=True
        )
        self.assertNotAllClose(output1, output2)
        self.assertNotAllClose(scores1, scores2)

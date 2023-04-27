import numpy as np

from keras_core import layers
from keras_core.testing import test_case


class EmbeddingTest(test_case.TestCase):
    def test_embedding_basics(self):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 4, "output_dim": 3},
            input_shape=(2,),
            input_dtype="int32",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 5, "output_dim": 4, "mask_zero": True},
            input_shape=(2, 3),
            input_dtype="int64",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_correctness(self):
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]))

    def test_masking(self):
        layer = layers.Embedding(input_dim=3, output_dim=2, mask_zero=True)
        layer.build()
        out = layer.compute_mask(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([True, True, False]))

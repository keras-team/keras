import numpy as np

from keras_core import layers
from keras_core import testing


class HashedCrossingTest(testing.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.HashedCrossing,
            init_kwargs={
                "num_bins": 3,
                "output_mode": "int",
            },
            input_data=([1, 2], [4, 5]),
            expected_output_shape=(2,),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.HashedCrossing,
            init_kwargs={"num_bins": 4, "output_mode": "one_hot"},
            input_data=([1, 2], [4, 5]),
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_correctness(self):
        layer = layers.HashedCrossing(num_bins=5)
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        output = layer((feat1, feat2))
        self.assertAllClose(np.array([1, 4, 1, 1, 3]), output)

        layer = layers.HashedCrossing(num_bins=5, output_mode="one_hot")
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        output = layer((feat1, feat2))
        self.assertAllClose(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
            output,
        )

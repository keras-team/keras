"""Test DynamicEmbedding with Parameter server strategy."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.layers.experimental import dynamic_lookup

ds_combinations = tf.__internal__.distribute.combinations


class DistributedDynamiclookupTest(tf.test.TestCase, parameterized.TestCase):
    @ds_combinations.generate(
        tf.__internal__.test.combinations.combine(
            strategy=[
                ds_combinations.parameter_server_strategy_3worker_2ps_cpu
            ],
            mode="eager",
        )
    )
    def test_dynamic_lookup_with_pss(self, strategy):
        train_data = np.array(
            [
                ["a", "j", "c", "d", "e"],
                ["a", "h", "i", "j", "b"],
                ["i", "h", "c", "j", "e"],
            ]
        )
        train_labels = np.array([0, 1, 2])
        vocab = tf.constant(["a", "b", "c", "d", "e"])
        vocabulary_size = 5
        eviction_policy = "LFU"
        with strategy.scope():
            # Define the model
            model = keras.models.Sequential(
                [
                    dynamic_lookup.DynamicLookup(
                        vocabulary_size,
                        initial_vocabulary=vocab,
                        eviction_policy=eviction_policy,
                        name="dynamic_lookup",
                    ),
                    keras.layers.Flatten(),
                    keras.layers.Dense(3, activation="softmax"),
                ]
            )

            # Compile the model
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            result = model.fit(
                train_data,
                train_labels,
                epochs=10,
                batch_size=1,
                steps_per_epoch=1,
            )
            # Assert model trains
            self.assertEqual(result.history["loss"][0] > 0, True)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()

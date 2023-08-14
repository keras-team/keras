"""Test DynamicEmbedding with Parameter server strategy."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.callbacks import UpdateEmbeddingCallback
from keras.layers.experimental import dynamic_embedding

ds_combinations = tf.__internal__.distribute.combinations


class DistributedDynamicEmbeddingTest(tf.test.TestCase, parameterized.TestCase):
    @ds_combinations.generate(
        tf.__internal__.test.combinations.combine(
            strategy=[
                ds_combinations.parameter_server_strategy_3worker_2ps_cpu
            ],
            mode="eager",
        )
    )
    def test_dynamic_embedding_with_pss(self, strategy):
        # Generate dummy data
        train_data = np.array(
            [
                ["a", "j", "c", "d", "e"],
                ["a", "h", "i", "j", "b"],
                ["i", "h", "c", "j", "e"],
            ]
        )
        train_labels = np.array([0, 1, 2])
        vocab = tf.constant(["a", "b", "c", "d", "e"])
        eviction_policy = "LFU"
        with strategy.scope():
            # Define the model
            model = keras.models.Sequential(
                [
                    dynamic_embedding.DynamicEmbedding(
                        input_dim=5,
                        output_dim=2,
                        input_length=5,
                        eviction_policy=eviction_policy,
                        initial_vocabulary=vocab,
                    ),
                    keras.layers.Flatten(),
                    keras.layers.Dense(3, activation="softmax"),
                ]
            )
            update_embedding_callback = UpdateEmbeddingCallback(
                model.layers[0],
                interval=1,
            )
            with update_embedding_callback:
                # Compile the model
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                result = model.fit(
                    train_data,
                    train_labels,
                    epochs=100,
                    batch_size=1,
                    steps_per_epoch=2,
                    callbacks=[update_embedding_callback],
                )
                # Assert model trains
                self.assertEqual(result.history["loss"][0] > 0, True)
                self.assertTrue(
                    tf.reduce_all(
                        tf.not_equal(
                            model.layers[0].dynamic_lookup_layer.vocabulary,
                            vocab,
                        )
                    )
                )


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()

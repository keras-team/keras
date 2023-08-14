"""Test DynamicEmbeddingLayer."""

import numpy as np
import tensorflow as tf

from keras import layers
from keras import models
from keras.callbacks import UpdateEmbeddingCallback
from keras.layers.experimental import dynamic_embedding
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_utils.run_v2_only
class DynamicEmbeddingTest(test_combinations.TestCase):
    def test_dynamic_embedding_layer(self):
        input_ = np.array([["a", "j", "c", "d", "e"]])
        vocab = tf.constant(["a", "b", "c", "d", "e"])
        eviction_policy = "LFU"
        # Define the layer
        layer = dynamic_embedding.DynamicEmbedding(
            input_dim=5,
            output_dim=2,
            input_length=5,
            eviction_policy=eviction_policy,
            initial_vocabulary=vocab,
        )
        output = layer(input_)
        self.assertTrue(
            tf.reduce_all(tf.equal(tf.shape(output), tf.constant([1, 5, 2])))
        )
        self.assertTrue((layer.built))
        self.assertTrue((layer.dynamic_lookup_layer.built))
        self.assertTrue((layer.embedding_layer.built))

    def test_model_save_load(self):
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
        # Define the model
        model = models.Sequential(
            [
                dynamic_embedding.DynamicEmbedding(
                    input_dim=5,
                    output_dim=2,
                    input_length=5,
                    eviction_policy=eviction_policy,
                    initial_vocabulary=vocab,
                    name="dynamic_embedding",
                ),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_data,
            train_labels,
            epochs=10,
            batch_size=1,
        )
        # Save the model to a temporary file
        filepath = self.create_tempdir()
        model.save(filepath)
        # Load the model from the temporary file
        reloaded_model = models.load_model(filepath)
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    model.get_layer(
                        "dynamic_embedding"
                    ).dynamic_lookup_layer.vocabulary.numpy(),
                    reloaded_model.get_layer(
                        "dynamic_embedding"
                    ).dynamic_lookup_layer.vocabulary.numpy(),
                )
            )
        )

    def test_dynamic_embedding_layer_with_callback(self):
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
        # Define the model
        model = models.Sequential(
            [
                dynamic_embedding.DynamicEmbedding(
                    input_dim=5,
                    output_dim=2,
                    input_length=5,
                    eviction_policy=eviction_policy,
                    initial_vocabulary=vocab,
                ),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        update_embedding_callback = UpdateEmbeddingCallback(
            model.layers[0],
            interval=2,
        )
        with update_embedding_callback:
            result = model.fit(
                train_data,
                train_labels,
                epochs=100,
                batch_size=1,
                callbacks=[update_embedding_callback],
            )
        # Assert model trains
        self.assertEqual(result.history["loss"][0] > 0, True)
        # assert vocab is updated in DynamicLookup
        self.assertTrue(
            tf.reduce_all(
                tf.not_equal(
                    model.layers[0].dynamic_lookup_layer.vocabulary, vocab
                )
            )
        )
        # assert embedding matrix size
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    tf.shape(model.layers[0].embedding_layer.embeddings),
                    tf.constant([6, 2]),
                )
            )
        )

    def test_embedding_matrix_update(self):
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
        # Define the model
        model = models.Sequential(
            [
                dynamic_embedding.DynamicEmbedding(
                    input_dim=5,
                    output_dim=2,
                    input_length=5,
                    eviction_policy=eviction_policy,
                    initial_vocabulary=vocab,
                ),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # freeze training of all layers
        for layer in model.layers:
            layer.trainable = False
        # define update_embedding_callback to update embedding matrix and
        # vocabulary
        update_embedding_callback = UpdateEmbeddingCallback(
            model.layers[0],
            interval=5,
        )
        embedding_matrix_before = model.layers[0].embedding_layer.get_weights()
        with update_embedding_callback:
            model.fit(
                train_data,
                train_labels,
                epochs=100,
                batch_size=1,
                callbacks=[update_embedding_callback],
            )
        # assert the UpdateEmbeddingCallback did modify the embedding matrix
        self.assertNotEqual(
            model.layers[0].embedding_layer.get_weights(),
            embedding_matrix_before,
        )

    def test_get_vocabulary(self):
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
        # Define the model
        model = models.Sequential(
            [
                dynamic_embedding.DynamicEmbedding(
                    input_dim=5,
                    output_dim=2,
                    input_length=5,
                    eviction_policy=eviction_policy,
                    initial_vocabulary=vocab,
                ),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_data,
            train_labels,
            epochs=100,
            batch_size=1,
        )
        vocabulary_output = model.layers[0].get_vocabulary()
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    vocabulary_output,
                    vocab,
                )
            )
        )

    def test_default_initial_vocabulary(self):
        train_data = np.array(
            [
                ["a", "j", "c", "d", "e"],
                ["a", "h", "i", "j", "b"],
                ["i", "h", "c", "j", "e"],
            ]
        )
        train_labels = np.array([0, 1, 2])
        eviction_policy = "LFU"
        # Define the model
        model = models.Sequential(
            [
                dynamic_embedding.DynamicEmbedding(
                    input_dim=5,
                    output_dim=2,
                    input_length=5,
                    eviction_policy=eviction_policy,
                    initial_vocabulary=tf.string,
                    name="dynamic_embedding",
                ),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_data,
            train_labels,
            epochs=10,
            batch_size=1,
        )
        vocabulary_output = model.layers[0].get_vocabulary()
        self.assertEqual(vocabulary_output.dtype, tf.string)
        self.assertEqual(tf.shape(vocabulary_output)[0], 5)


if __name__ == "__main__":
    tf.test.main()

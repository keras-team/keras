"""Test for dynamic_lookup layer."""

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

import keras
from keras.layers.experimental import dynamic_lookup
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_utils.run_v2_only
class DynamicLookupTest(test_combinations.TestCase):
    def test_dynamic_lookup_layer(self):
        vocabulary_size = 5
        eviction_policy = "LFU"
        vocab = tf.constant(["apple", "banana", "cherry", "grape", "juice"])
        # vocab_frequency({apple:0, banana:0, cherry:0, grape:0, juice:0})
        # hash table size is 1.2Xvocab size. in this case 5x1.2 = 6
        layer = dynamic_lookup.DynamicLookup(
            vocabulary_size,
            initial_vocabulary=vocab,
            eviction_policy=eviction_policy,
        )

        input_1 = tf.constant(["apple", "banana", "cherry"])
        layer(input_1)
        # vocab_frequency({apple:1, banana:1, cherry:1, grape:0, juice:0})
        input_2 = tf.constant(["apple", "banana", "mango"])
        layer(input_2)
        # vocab_frequency({apple:2, banana:2, cherry:1, grape:0, juice:0, mango:
        # 1})
        input_3 = tf.constant(["fig", "date", "date"])
        layer(input_3)
        # vocab_frequency({apple:2, banana:2, cherry:1, fig:1, date:1, mango:1})
        input_4 = tf.constant(["banana", "jackfruit", "honeydew"])
        layer(input_4)
        # vocab_frequency({apple:2, banana:3, jackfruit:1, fig:1, date:1,
        # honeydew:1})
        input_5 = tf.constant(["banana", "apple", "jackfruit"])
        # vocab_frequency({apple:3, banana:4, jackfruit:2, fig:1, date:1,
        # honeydew:1})
        outputs = layer(input_5)
        expected_output = tf.constant([1, 0, 5], dtype=tf.int64)
        # verify if look up values are accurate
        self.assertTrue(tf.reduce_all(tf.equal(outputs, expected_output)))
        # Check the shape of the output
        self.assertEqual(outputs.shape, input_4.shape)

        # Check that the top-k vocab is correctly updated
        top_k_vocab = layer.get_top_vocabulary(3)
        expected_top_k_vocab = tf.constant(
            ["banana", "apple", "jackfruit"],
            dtype=tf.string,
        )
        self.assertTrue(
            tf.reduce_all(tf.equal(top_k_vocab, expected_top_k_vocab))
        )

    def test_layer_with_model(self):
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

        # Define the model
        model = keras.models.Sequential(
            [
                dynamic_lookup.DynamicLookup(
                    vocabulary_size,
                    initial_vocabulary=vocab,
                    eviction_policy=eviction_policy,
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
        )
        # Assert model trains
        self.assertEqual(result.history["loss"][0] > 0, True)

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
        vocabulary_size = 5
        eviction_policy = "LFU"

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
        model.fit(
            train_data,
            train_labels,
            epochs=10,
            batch_size=1,
        )
        # Save the model to a temporary file
        filepath = os.path.join(tempfile.gettempdir(), "tempdir")
        model.save(filepath)
        reloaded_model = keras.models.load_model(filepath)
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    model.get_layer("dynamic_lookup").vocabulary.numpy(),
                    reloaded_model.get_layer(
                        "dynamic_lookup"
                    ).vocabulary.numpy(),
                )
            )
        )
        shutil.rmtree(filepath)

    def test_dynamic_lookup_layer_learn_vocab_arg(self):
        vocabulary_size = 5
        eviction_policy = "LFU"
        vocab = tf.constant(["apple", "banana", "cherry", "grape", "juice"])
        # vocab_frequency({apple:0, banana:0, cherry:0, grape:0, juice:0})
        # hash table size is 1.2Xvocab size. in this case 5x1.2 = 6
        layer = dynamic_lookup.DynamicLookup(
            vocabulary_size,
            initial_vocabulary=vocab,
            eviction_policy=eviction_policy,
        )

        input_1 = tf.constant(["apple", "banana", "cherry"])
        layer(input_1, learn_vocab=False)
        input_2 = tf.constant(["apple", "banana", "mango"])
        layer(input_2, learn_vocab=False)
        input_3 = tf.constant(["fig", "date", "date"])
        layer(input_3, learn_vocab=False)
        input_4 = tf.constant(["banana", "jackfruit", "honeydew"])
        layer(input_4, learn_vocab=False)
        input_5 = tf.constant(["banana", "apple", "jackfruit"])
        layer(input_5, learn_vocab=False)
        # Check that the top-k vocab is not updated
        top_k_vocab = layer.get_top_vocabulary(5)
        expected_top_k_vocab = tf.constant(
            ["apple", "banana", "cherry", "grape", "juice"],
            dtype=tf.string,
        )
        self.assertTrue(
            tf.reduce_all(tf.equal(top_k_vocab, expected_top_k_vocab))
        )

    def test_get_vocabulary(self):
        vocabulary_size = 5
        eviction_policy = "LFU"
        vocab = tf.constant(["apple", "banana", "cherry", "grape", "juice"])
        layer = dynamic_lookup.DynamicLookup(
            vocabulary_size,
            initial_vocabulary=vocab,
            eviction_policy=eviction_policy,
        )
        input_1 = tf.constant(["apple", "banana", "cherry"])
        layer(input_1, learn_vocab=False)
        vocabulary_output = layer.get_vocabulary()
        self.assertTrue(tf.reduce_all(tf.equal(vocabulary_output, vocab)))

    def test_default_vocab(self):
        # test default initial vocabulary tf.string
        vocabulary_size = 5
        eviction_policy = "LFU"
        layer1 = dynamic_lookup.DynamicLookup(
            vocabulary_size,
            initial_vocabulary=tf.string,
            eviction_policy=eviction_policy,
        )
        input_1 = tf.constant(["apple", "banana", "cherry"])
        layer1(input_1, learn_vocab=False)
        vocabulary_output = layer1.get_vocabulary()
        self.assertEqual(vocabulary_output.dtype, tf.string)
        self.assertEqual(tf.shape(vocabulary_output)[0], vocabulary_size)

        # test default initial vocabulary tf.int32
        layer2 = dynamic_lookup.DynamicLookup(
            vocabulary_size,
            initial_vocabulary=tf.int32,
            eviction_policy=eviction_policy,
        )
        input_2 = tf.constant([1, 2, 3], dtype=tf.int32)
        layer2(input_2, learn_vocab=False)
        vocabulary_output = layer2.get_vocabulary()
        self.assertEqual(vocabulary_output.dtype, tf.int32)
        self.assertEqual(tf.shape(vocabulary_output)[0], vocabulary_size)

        # test default initial vocabulary tf.int64
        layer3 = dynamic_lookup.DynamicLookup(
            vocabulary_size,
            initial_vocabulary=tf.int64,
            eviction_policy=eviction_policy,
        )
        input_3 = tf.constant([1, 2, 3], dtype=tf.int64)
        layer3(input_3, learn_vocab=False)
        vocabulary_output = layer3.get_vocabulary()
        self.assertEqual(vocabulary_output.dtype, tf.int64)
        self.assertEqual(tf.shape(vocabulary_output)[0], vocabulary_size)

        # test value error when default initial vocabulary is tf.float32
        with self.assertRaises(ValueError):
            layer3 = dynamic_lookup.DynamicLookup(
                vocabulary_size,
                initial_vocabulary=tf.float32,
                eviction_policy=eviction_policy,
            )


if __name__ == "__main__":
    tf.test.main()

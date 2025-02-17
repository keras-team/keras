import numpy as np

from keras.src import testing
from keras.src.datasets import reuters


class ReutersLoadDataTest(testing.TestCase):
    def test_load_data_default(self):
        (x_train, y_train), (x_test, y_test) = reuters.load_data()
        # Check types
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(x_test, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)

        # Check shapes
        self.assertGreater(len(x_train), 0)
        self.assertEqual(len(x_train), len(y_train))
        self.assertGreater(len(x_test), 0)
        self.assertEqual(len(x_test), len(y_test))

    def test_num_words(self):
        # Only consider the top 1000 words
        (x_train, _), _ = reuters.load_data(num_words=1000)
        # Ensure no word index exceeds 999 (0-based indexing)
        max_index = max(max(sequence) for sequence in x_train if sequence)
        self.assertLessEqual(max_index, 999)

    def test_skip_top(self):
        # Skip the top 10 most frequent words
        (x_train, _), _ = reuters.load_data(skip_top=10, num_words=1000)
        # Assuming 1 is among top 10, check if it's skipped
        self.assertNotIn(1, x_train[0])

    def test_maxlen(self):
        # Only consider sequences shorter than 50
        (x_train, _), _ = reuters.load_data(maxlen=50)
        self.assertTrue(all(len(seq) <= 50 for seq in x_train))

    def test_get_word_index(self):
        word_index = reuters.get_word_index()
        self.assertIsInstance(word_index, dict)
        # Check if word_index contains specific known words
        self.assertIn("the", word_index)

    def test_get_label_names(self):
        label_names = reuters.get_label_names()
        self.assertIsInstance(label_names, tuple)
        # Check if the tuple contains specific known labels
        self.assertIn("earn", label_names)
        self.assertIn("acq", label_names)

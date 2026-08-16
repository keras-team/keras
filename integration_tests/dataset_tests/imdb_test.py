import numpy as np

from keras.src import testing
from keras.src.datasets import imdb


class ImdbLoadDataTest(testing.TestCase):
    def test_load_data_default(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(x_test, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)

        # Check lengths
        self.assertEqual(len(x_train), 25000)
        self.assertEqual(len(y_train), 25000)
        self.assertEqual(len(x_test), 25000)
        self.assertEqual(len(y_test), 25000)

        # Check types within lists for x
        self.assertIsInstance(x_train[0], list)
        self.assertIsInstance(x_test[0], list)

    def test_num_words(self):
        # Only consider the top 1000 words
        (x_train, _), _ = imdb.load_data(num_words=1000)
        # Ensure that no word index exceeds 999 (0-based indexing)
        max_index = max(max(sequence) for sequence in x_train if sequence)
        self.assertLessEqual(max_index, 999)

    def test_skip_top(self):
        # Skip the top 10 most frequent words
        (x_train, _), _ = imdb.load_data(skip_top=10, num_words=1000)
        # Check if top 10 words are skipped properly
        self.assertNotIn(1, x_train[0])  # Assuming 1 is among top 10

    def test_maxlen(self):
        # Only consider sequences shorter than 100
        (x_train, _), _ = imdb.load_data(maxlen=100)
        self.assertTrue(all(len(seq) <= 100 for seq in x_train))

    def test_get_word_index(self):
        word_index = imdb.get_word_index()
        self.assertIsInstance(word_index, dict)
        # Check if word_index contains specific known words
        self.assertIn("the", word_index)
        self.assertIn("and", word_index)

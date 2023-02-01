# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for text data preprocessing utils."""

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from keras.preprocessing import text


class TestText(tf.test.TestCase):
    def test_one_hot(self):
        sample_text = "The cat sat on the mat."
        encoded = text.one_hot(sample_text, 5)
        self.assertLen(encoded, 6)
        self.assertLessEqual(np.max(encoded), 4)
        self.assertGreaterEqual(np.min(encoded), 0)

        sample_text = "The-cat-sat-on-the-mat"
        encoded2 = text.one_hot(
            sample_text, 5, analyzer=lambda t: t.lower().split("-")
        )
        self.assertEqual(encoded, encoded2)
        self.assertLen(encoded, 6)
        self.assertLessEqual(np.max(encoded), 4)
        self.assertGreaterEqual(np.min(encoded), 0)

    def test_hashing_trick_hash(self):
        sample_text = "The cat sat on the mat."
        encoded = text.hashing_trick(sample_text, 5)
        self.assertLen(encoded, 6)
        self.assertLessEqual(np.max(encoded), 4)
        self.assertGreaterEqual(np.min(encoded), 1)

    def test_hashing_trick_md5(self):
        sample_text = "The cat sat on the mat."
        encoded = text.hashing_trick(sample_text, 5, hash_function="md5")
        self.assertLen(encoded, 6)
        self.assertLessEqual(np.max(encoded), 4)
        self.assertGreaterEqual(np.min(encoded), 1)

    def test_tokenizer(self):
        sample_texts = [
            "The cat sat on the mat.",
            "The dog sat on the log.",
            "Dogs and cats living together.",
        ]
        tokenizer = text.Tokenizer(num_words=10)
        tokenizer.fit_on_texts(sample_texts)

        sequences = []
        for seq in tokenizer.texts_to_sequences_generator(sample_texts):
            sequences.append(seq)
        self.assertLess(np.max(np.max(np.asarray(sequences, dtype=object))), 10)
        self.assertEqual(np.min(np.min(np.asarray(sequences, dtype=object))), 1)

        tokenizer.fit_on_sequences(sequences)

        for mode in ["binary", "count", "tfidf", "freq"]:
            tokenizer.texts_to_matrix(sample_texts, mode)

    def test_tokenizer_serde_no_fitting(self):
        tokenizer = text.Tokenizer(num_words=100)

        tokenizer_json = tokenizer.to_json()
        recovered = text.tokenizer_from_json(tokenizer_json)

        self.assertEqual(tokenizer.get_config(), recovered.get_config())

        self.assertEqual(tokenizer.word_docs, recovered.word_docs)
        self.assertEqual(tokenizer.word_counts, recovered.word_counts)
        self.assertEqual(tokenizer.word_index, recovered.word_index)
        self.assertEqual(tokenizer.index_word, recovered.index_word)
        self.assertEqual(tokenizer.index_docs, recovered.index_docs)

    def test_tokenizer_serde_fitting(self):
        sample_texts = [
            "There was a time that the pieces fit, but I watched "
            "them fall away",
            "Mildewed and smoldering, strangled by our coveting",
            "I've done the math enough to know the dangers of our second "
            "guessing",
        ]
        tokenizer = text.Tokenizer(num_words=100)
        tokenizer.fit_on_texts(sample_texts)

        seq_generator = tokenizer.texts_to_sequences_generator(sample_texts)
        sequences = [seq for seq in seq_generator]
        tokenizer.fit_on_sequences(sequences)

        tokenizer_json = tokenizer.to_json()
        recovered = text.tokenizer_from_json(tokenizer_json)

        self.assertEqual(tokenizer.char_level, recovered.char_level)
        self.assertEqual(tokenizer.document_count, recovered.document_count)
        self.assertEqual(tokenizer.filters, recovered.filters)
        self.assertEqual(tokenizer.lower, recovered.lower)
        self.assertEqual(tokenizer.num_words, recovered.num_words)
        self.assertEqual(tokenizer.oov_token, recovered.oov_token)

        self.assertEqual(tokenizer.word_docs, recovered.word_docs)
        self.assertEqual(tokenizer.word_counts, recovered.word_counts)
        self.assertEqual(tokenizer.word_index, recovered.word_index)
        self.assertEqual(tokenizer.index_word, recovered.index_word)
        self.assertEqual(tokenizer.index_docs, recovered.index_docs)

    def test_sequential_fit(self):
        texts = [
            "The cat sat on the mat.",
            "The dog sat on the log.",
            "Dogs and cats living together.",
        ]
        word_sequences = [
            ["The", "cat", "is", "sitting"],
            ["The", "dog", "is", "standing"],
        ]

        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(texts)
        tokenizer.fit_on_texts(word_sequences)

        self.assertEqual(tokenizer.document_count, 5)

        tokenizer.texts_to_matrix(texts)
        tokenizer.texts_to_matrix(word_sequences)

    def test_text_to_word_sequence(self):
        sample_text = "hello! ? world!"
        self.assertEqual(
            text.text_to_word_sequence(sample_text), ["hello", "world"]
        )

    def test_text_to_word_sequence_multichar_split(self):
        sample_text = "hello!stop?world!"
        self.assertEqual(
            text.text_to_word_sequence(sample_text, split="stop"),
            ["hello", "world"],
        )

    def test_text_to_word_sequence_unicode(self):
        sample_text = "ali! veli? kırk dokuz elli"
        self.assertEqual(
            text.text_to_word_sequence(sample_text),
            ["ali", "veli", "kırk", "dokuz", "elli"],
        )

    def test_text_to_word_sequence_unicode_multichar_split(self):
        sample_text = "ali!stopveli?stopkırkstopdokuzstopelli"
        self.assertEqual(
            text.text_to_word_sequence(sample_text, split="stop"),
            ["ali", "veli", "kırk", "dokuz", "elli"],
        )

    def test_tokenizer_unicode(self):
        sample_texts = [
            "ali veli kırk dokuz elli",
            "ali veli kırk dokuz elli veli kırk dokuz",
        ]
        tokenizer = text.Tokenizer(num_words=5)
        tokenizer.fit_on_texts(sample_texts)

        self.assertLen(tokenizer.word_counts, 5)

    def test_tokenizer_oov_flag(self):
        """Test of Out of Vocabulary (OOV) flag in text.Tokenizer."""
        x_train = ["This text has only known words"]
        x_test = ["This text has some unknown words"]  # 2 OOVs: some, unknown

        # Default, without OOV flag
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        self.assertLen(x_test_seq[0], 4)  # discards 2 OOVs

        # With OOV feature
        tokenizer = text.Tokenizer(oov_token="<unk>")
        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        self.assertLen(x_test_seq[0], 6)  # OOVs marked in place

    def test_tokenizer_oov_flag_and_num_words(self):
        x_train = ["This text has only known words this text"]
        x_test = ["This text has some unknown words"]

        tokenizer = text.Tokenizer(num_words=3, oov_token="<unk>")
        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        trans_text = " ".join(tokenizer.index_word[t] for t in x_test_seq[0])
        self.assertLen(x_test_seq[0], 6)
        self.assertEqual(trans_text, "this <unk> <unk> <unk> <unk> <unk>")

    def test_sequences_to_texts_with_num_words_and_oov_token(self):
        x_train = ["This text has only known words this text"]
        x_test = ["This text has some unknown words"]

        tokenizer = text.Tokenizer(num_words=3, oov_token="<unk>")

        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        trans_text = tokenizer.sequences_to_texts(x_test_seq)
        self.assertEqual(trans_text, ["this <unk> <unk> <unk> <unk> <unk>"])

    def test_sequences_to_texts_no_num_words(self):
        x_train = ["This text has only known words this text"]
        x_test = ["This text has some unknown words"]

        tokenizer = text.Tokenizer(oov_token="<unk>")

        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        trans_text = tokenizer.sequences_to_texts(x_test_seq)
        self.assertEqual(trans_text, ["this text has <unk> <unk> words"])

    def test_sequences_to_texts_no_oov_token(self):
        x_train = ["This text has only known words this text"]
        x_test = ["This text has some unknown words"]

        tokenizer = text.Tokenizer(num_words=3)

        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        trans_text = tokenizer.sequences_to_texts(x_test_seq)
        self.assertEqual(trans_text, ["this text"])

    def test_sequences_to_texts_no_num_words_no_oov_token(self):
        x_train = ["This text has only known words this text"]
        x_test = ["This text has some unknown words"]

        tokenizer = text.Tokenizer()

        tokenizer.fit_on_texts(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)
        trans_text = tokenizer.sequences_to_texts(x_test_seq)
        self.assertEqual(trans_text, ["this text has words"])

    def test_sequences_to_texts(self):
        texts = [
            "The cat sat on the mat.",
            "The dog sat on the log.",
            "Dogs and cats living together.",
        ]
        tokenizer = text.Tokenizer(num_words=10, oov_token="<unk>")
        tokenizer.fit_on_texts(texts)
        tokenized_text = tokenizer.texts_to_sequences(texts)
        trans_text = tokenizer.sequences_to_texts(tokenized_text)
        self.assertEqual(
            trans_text,
            [
                "the cat sat on the mat",
                "the dog sat on the log",
                "dogs <unk> <unk> <unk> <unk>",
            ],
        )

    def test_tokenizer_lower_flag(self):
        """Tests for `lower` flag in text.Tokenizer."""
        # word level tokenizer with sentences as texts
        word_tokenizer = text.Tokenizer(lower=True)
        texts = [
            "The cat sat on the mat.",
            "The dog sat on the log.",
            "Dog and Cat living Together.",
        ]
        word_tokenizer.fit_on_texts(texts)
        expected_word_counts = collections.OrderedDict(
            [
                ("the", 4),
                ("cat", 2),
                ("sat", 2),
                ("on", 2),
                ("mat", 1),
                ("dog", 2),
                ("log", 1),
                ("and", 1),
                ("living", 1),
                ("together", 1),
            ]
        )
        self.assertEqual(word_tokenizer.word_counts, expected_word_counts)

        # word level tokenizer with word_sequences as texts
        word_tokenizer = text.Tokenizer(lower=True)
        word_sequences = [
            ["The", "cat", "is", "sitting"],
            ["The", "dog", "is", "standing"],
        ]
        word_tokenizer.fit_on_texts(word_sequences)
        expected_word_counts = collections.OrderedDict(
            [
                ("the", 2),
                ("cat", 1),
                ("is", 2),
                ("sitting", 1),
                ("dog", 1),
                ("standing", 1),
            ]
        )
        self.assertEqual(word_tokenizer.word_counts, expected_word_counts)

        # char level tokenizer with sentences as texts
        char_tokenizer = text.Tokenizer(lower=True, char_level=True)
        texts = [
            "The cat sat on the mat.",
            "The dog sat on the log.",
            "Dog and Cat living Together.",
        ]
        char_tokenizer.fit_on_texts(texts)
        expected_word_counts = collections.OrderedDict(
            [
                ("t", 11),
                ("h", 5),
                ("e", 6),
                (" ", 14),
                ("c", 2),
                ("a", 6),
                ("s", 2),
                ("o", 6),
                ("n", 4),
                ("m", 1),
                (".", 3),
                ("d", 3),
                ("g", 5),
                ("l", 2),
                ("i", 2),
                ("v", 1),
                ("r", 1),
            ]
        )
        self.assertEqual(char_tokenizer.word_counts, expected_word_counts)


if __name__ == "__main__":
    tf.test.main()

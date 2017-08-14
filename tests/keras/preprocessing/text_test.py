# -*- coding: utf-8 -*-

import numpy as np
import pytest

from keras.preprocessing.text import Tokenizer, one_hot, hashing_trick, text_to_word_sequence


def test_one_hot():
    text = 'The cat sat on the mat.'
    encoded = one_hot(text, 5)
    assert len(encoded) == 6
    assert np.max(encoded) <= 4
    assert np.min(encoded) >= 0


def test_hashing_trick_hash():
    text = 'The cat sat on the mat.'
    encoded = hashing_trick(text, 5)
    assert len(encoded) == 6
    assert np.max(encoded) <= 4
    assert np.min(encoded) >= 1


def test_hashing_trick_md5():
    text = 'The cat sat on the mat.'
    encoded = hashing_trick(text, 5, hash_function='md5')
    assert len(encoded) == 6
    assert np.max(encoded) <= 4
    assert np.min(encoded) >= 1


def test_tokenizer():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenizer = Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)

    sequences = []
    for seq in tokenizer.texts_to_sequences_generator(texts):
        sequences.append(seq)
    assert np.max(np.max(sequences)) < 10
    assert np.min(np.min(sequences)) == 1

    tokenizer.fit_on_sequences(sequences)

    for mode in ['binary', 'count', 'tfidf', 'freq']:
        matrix = tokenizer.texts_to_matrix(texts, mode)


def test_text_to_word_sequence():
    text = 'hello! ? world!'
    assert text_to_word_sequence(text) == ['hello', 'world']


def test_text_to_word_sequence_unicode():
    text = u'ali! veli? kırk dokuz elli'
    assert text_to_word_sequence(text) == [u'ali', u'veli', u'kırk', u'dokuz', u'elli']


def test_tokenizer_unicode():
    texts = [u'ali veli kırk dokuz elli', u'ali veli kırk dokuz elli veli kırk dokuz']
    tokenizer = Tokenizer(num_words=5)
    tokenizer.fit_on_texts(texts)

    assert len(tokenizer.word_counts) == 5


if __name__ == '__main__':
    pytest.main([__file__])

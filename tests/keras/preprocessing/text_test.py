from keras.preprocessing.text import Tokenizer, hashing_trick
import pytest
import numpy as np


def test_hashing_trick():
    text = 'The cat sat on the mat.'
    encoded = hashing_trick(text, 5)
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


if __name__ == '__main__':
    pytest.main([__file__])

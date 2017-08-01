import numpy as np
import pytest

from keras.preprocessing.text import Tokenizer, one_hot, hashing_trick


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


if __name__ == '__main__':
    pytest.main([__file__])

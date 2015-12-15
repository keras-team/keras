import numpy as np
from numpy.testing import assert_allclose

from keras.preprocessing.sequences import (pad_sequences, make_sampling_table,
                                           skipgrams)


def test_pad_sequences():
    a = [np.arange(i) for i in np.arange(2, 5, 1)]
    b = pad_sequences(a, maxlen=3)
    assert_allclose(b, [[0, 0, 1], [0, 1, 2], [1, 2, 3]])


def test_make_sampling_table():
    a = make_sampling_table(3)
    assert_allclose(a, np.asarray([0.00315225,  0.00315225,  0.00547597]),
                    rtol=.1)


def test_skipgrams():
    couples, labels = skipgrams(np.arange(3), 3)
    for couple in couples:
        assert couple[0] in [0, 1, 2] and couple[1] in [0, 1, 2]

    assert 0 in labels and 1 in labels

import numpy as np
from numpy.testing import assert_allclose

import pytest

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import make_sampling_table
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.sequence import _remove_long_seq
from keras.preprocessing.sequence import TimeseriesGenerator


def test_pad_sequences():
    a = [[1], [1, 2], [1, 2, 3]]

    # test padding
    b = pad_sequences(a, maxlen=3, padding='pre')
    assert_allclose(b, [[0, 0, 1], [0, 1, 2], [1, 2, 3]])
    b = pad_sequences(a, maxlen=3, padding='post')
    assert_allclose(b, [[1, 0, 0], [1, 2, 0], [1, 2, 3]])

    # test truncating
    b = pad_sequences(a, maxlen=2, truncating='pre')
    assert_allclose(b, [[0, 1], [1, 2], [2, 3]])
    b = pad_sequences(a, maxlen=2, truncating='post')
    assert_allclose(b, [[0, 1], [1, 2], [1, 2]])

    # test value
    b = pad_sequences(a, maxlen=3, value=1)
    assert_allclose(b, [[1, 1, 1], [1, 1, 2], [1, 2, 3]])


def test_pad_sequences_vector():
    a = [[[1, 1]],
         [[2, 1], [2, 2]],
         [[3, 1], [3, 2], [3, 3]]]

    # test padding
    b = pad_sequences(a, maxlen=3, padding='pre')
    assert_allclose(b, [[[0, 0], [0, 0], [1, 1]],
                        [[0, 0], [2, 1], [2, 2]],
                        [[3, 1], [3, 2], [3, 3]]])
    b = pad_sequences(a, maxlen=3, padding='post')
    assert_allclose(b, [[[1, 1], [0, 0], [0, 0]],
                        [[2, 1], [2, 2], [0, 0]],
                        [[3, 1], [3, 2], [3, 3]]])

    # test truncating
    b = pad_sequences(a, maxlen=2, truncating='pre')
    assert_allclose(b, [[[0, 0], [1, 1]],
                        [[2, 1], [2, 2]],
                        [[3, 2], [3, 3]]])

    b = pad_sequences(a, maxlen=2, truncating='post')
    assert_allclose(b, [[[0, 0], [1, 1]],
                        [[2, 1], [2, 2]],
                        [[3, 1], [3, 2]]])

    # test value
    b = pad_sequences(a, maxlen=3, value=1)
    assert_allclose(b, [[[1, 1], [1, 1], [1, 1]],
                        [[1, 1], [2, 1], [2, 2]],
                        [[3, 1], [3, 2], [3, 3]]])


def test_make_sampling_table():
    a = make_sampling_table(3)
    assert_allclose(a, np.asarray([0.00315225, 0.00315225, 0.00547597]),
                    rtol=.1)


def test_skipgrams():
    # test with no window size and binary labels
    couples, labels = skipgrams(np.arange(3), vocabulary_size=3)
    for couple in couples:
        assert couple[0] in [0, 1, 2] and couple[1] in [0, 1, 2]

    # test window size and categorical labels
    couples, labels = skipgrams(np.arange(5), vocabulary_size=5, window_size=1,
                                categorical=True)
    for couple in couples:
        assert couple[0] - couple[1] <= 3
    for l in labels:
        assert len(l) == 2


def test_remove_long_seq():
    maxlen = 5
    seq = [
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6],
    ]
    label = ['a', 'b']
    new_seq, new_label = _remove_long_seq(maxlen, seq, label)
    assert new_seq == [[1, 2, 3]]
    assert new_label == ['a']


def test_TimeseriesGenerator():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    assert len(data_gen) == 20
    assert (np.allclose(data_gen[0][0],
                        np.array([[[0], [2], [4], [6], [8]],
                                  [[1], [3], [5], [7], [9]]])))
    assert (np.allclose(data_gen[0][1],
                        np.array([[10], [11]])))
    assert (np.allclose(data_gen[1][0],
                        np.array([[[2], [4], [6], [8], [10]],
                                  [[3], [5], [7], [9], [11]]])))
    assert (np.allclose(data_gen[1][1],
                        np.array([[12], [13]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, reverse=True,
                                   batch_size=2)
    assert len(data_gen) == 20
    assert (np.allclose(data_gen[0][0],
                        np.array([[[8], [6], [4], [2], [0]],
                                  [[9], [7], [5], [3], [1]]])))
    assert (np.allclose(data_gen[0][1],
                        np.array([[10], [11]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, shuffle=True,
                                   batch_size=1)
    batch = data_gen[0]
    r = batch[1][0][0]
    assert (np.allclose(batch[0],
                        np.array([[[r - 10],
                                   [r - 8],
                                   [r - 6],
                                   [r - 4],
                                   [r - 2]]])))
    assert (np.allclose(batch[1], np.array([[r], ])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, stride=2,
                                   batch_size=2)
    assert len(data_gen) == 10
    assert (np.allclose(data_gen[1][0],
                        np.array([[[4], [6], [8], [10], [12]],
                                  [[6], [8], [10], [12], [14]]])))
    assert (np.allclose(data_gen[1][1],
                        np.array([[14], [16]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2)
    assert len(data_gen) == 5
    assert (np.allclose(data_gen[0][0],
                        np.array([[[10], [12], [14], [16], [18]],
                                  [[11], [13], [15], [17], [19]]])))
    assert (np.allclose(data_gen[0][1],
                        np.array([[20], [21]])))

    data = np.array([np.random.random_sample((1, 2, 3, 4)) for i in range(50)])
    targets = np.array([np.random.random_sample((3, 2, 1)) for i in range(50)])
    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2)

    assert len(data_gen) == 5
    assert np.allclose(data_gen[0][0], np.array(
        [np.array(data[10:19:2]), np.array(data[11:20:2])]))
    assert (np.allclose(data_gen[0][1],
                        np.array([targets[20], targets[21]])))


if __name__ == '__main__':
    pytest.main([__file__])

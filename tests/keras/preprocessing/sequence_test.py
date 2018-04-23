from math import ceil

import numpy as np
from numpy.testing import assert_allclose, assert_raises
import warnings
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
                                   batch_size=2, gap=2)
    assert len(data_gen) == 20
    assert (np.array_equal(data_gen[0][0],
                           np.array([[[0], [2], [4], [6], [8]],
                                     [[1], [3], [5], [7], [9]]])))
    assert (np.array_equal(data_gen[0][1],
                           np.array([[10], [11]])))
    assert (np.array_equal(data_gen[1][0],
                           np.array([[[2], [4], [6], [8], [10]],
                                     [[3], [5], [7], [9], [11]]])))
    assert (np.array_equal(data_gen[1][1],
                           np.array([[12], [13]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, reverse=True,
                                   batch_size=2, gap=2)
    assert len(data_gen) == 20
    assert (np.array_equal(data_gen[0][0],
                           np.array([[[8], [6], [4], [2], [0]],
                                     [[9], [7], [5], [3], [1]]])))
    assert (np.array_equal(data_gen[0][1],
                           np.array([[10], [11]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, shuffle=True,
                                   batch_size=1, gap=2)
    batch = data_gen[0]
    r = batch[1][0][0]
    assert (np.array_equal(batch[0],
                           np.array([[[r - 10],
                                      [r - 8],
                                      [r - 6],
                                      [r - 4],
                                      [r - 2]]])))
    assert (np.array_equal(batch[1], np.array([[r], ])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, stride=2,
                                   batch_size=2, gap=2)
    assert len(data_gen) == 10
    assert (np.array_equal(data_gen[1][0],
                           np.array([[[4], [6], [8], [10], [12]],
                                     [[6], [8], [10], [12], [14]]])))
    assert (np.array_equal(data_gen[1][1],
                           np.array([[14], [16]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2, gap=2)
    assert len(data_gen) == 5
    assert (np.array_equal(data_gen[0][0],
                           np.array([[[10], [12], [14], [16], [18]],
                                     [[11], [13], [15], [17], [19]]])))
    assert (np.array_equal(data_gen[0][1],
                           np.array([[20], [21]])))

    data = np.array([np.random.random_sample((1, 2, 3, 4)) for i in range(50)])
    targets = np.array([np.random.random_sample((3, 2, 1)) for i in range(50)])
    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2, gap=2)
    assert len(data_gen) == 5
    assert np.array_equal(data_gen[0][0], np.array(
        [np.array(data[10:19:2]), np.array(data[11:20:2])]))
    assert (np.array_equal(data_gen[0][1],
                           np.array([targets[20], targets[21]])))


def test_TimeseriesGenerator_exceptions():

    data = np.array([[i] for i in range(50)])

    with assert_raises(ValueError) as context:
        TimeseriesGenerator(data, data, length=50, stride=0)
    error = str(context.exception)
    print(error)
    assert 'must be strictly positive.' in error

    with assert_raises(ValueError) as context:
        TimeseriesGenerator(data, data, length=50, sampling_rate=0)
    error = str(context.exception)
    print(error)
    assert 'must be strictly positive.' in error

    with assert_raises(ValueError) as context:
        TimeseriesGenerator(data, data, length=50, batch_size=0)
    error = str(context.exception)
    print(error)
    assert 'must be strictly positive.' in error

    with assert_raises(ValueError) as context:
        TimeseriesGenerator(data, data, length=50, start_index=50)
    error = str(context.exception)
    print(error)
    assert 'This configuration gives no output' in error

    with assert_raises(ValueError) as context:
        TimeseriesGenerator(data, data, length=50, sampling_rate=51)
    error = str(context.exception)
    print(error)
    assert '`length` has to be a multiple of `sampling_rate`. For instance, `length=102` would do.' in error

    with assert_raises(ValueError) as context:
        TimeseriesGenerator(data, data, length=10, sampling_rate=3)
    error = str(context.exception)
    print(error)
    assert '`length` has to be a multiple of `sampling_rate`. For instance, `length=6` would do.' in error


def test_TimeSeriesGenerator_doesnt_miss_any_sample1():
    x = np.array([[i] for i in range(10)])

    for gap in range(10):
        for length in range(1, 11-gap):

            expected = len(x) - length + 1 - gap

            if expected > 0:
                g = TimeseriesGenerator(x, x,
                                        length=length,
                                        batch_size=1, gap=gap)

                #print('gap: %i, hlength: %i, expected-len:%i, len: %i' % (g.gap, g.hlength, expected, g.len))
                # for i in range(len(g)):
                #    print(i,g[i])

                actual = len(g)
                assert expected == actual

    x = np.array([i for i in range(7)])

    g = TimeseriesGenerator(x, x, hlength=3, batch_size=2)

    expected_len = ceil((len(x) - g.hlength + 1.0) / g.batch_size)
    print('gap: %i, hlength: %i, expected-len:%i, len: %i' %
          (g.gap, g.hlength, expected_len, g.len))
    # for i in range(len(g)):
    #    print(i, g[i])

    assert len(g) == expected_len


def test_TimeSeriesGenerator_doesnt_miss_any_sample2():

    x = np.array([[i] for i in range(23)])

    strides = (1, 1, 5, 7, 3, 5, 3)
    lengths = (3, 3, 4, 3, 1, 3, 7)
    batch_sizes = (6, 6, 6, 5, 6, 6, 6)
    shuffles = (False, True, True, False, False, False, False)

    for stride, length, batch_size, shuffle in zip(strides,
                                                   lengths,
                                                   batch_sizes,
                                                   shuffles):
        g = TimeseriesGenerator(x, x,
                                length=length,
                                sampling_rate=1,
                                stride=stride,
                                start_index=0,
                                end_index=None,
                                shuffle=shuffle,
                                reverse=False,
                                batch_size=batch_size)

        # last batch will be different if `(samples - length) / stride`
        # is not a multiple of `batch_size`.
        expected_sequences = int(ceil((len(x) - length + 1.0) / stride))

        expected_batches = ceil(expected_sequences / float(batch_size))
        print('gap: %i, hlength: %i, expected-len:%i, len: %i' %
              (g.gap, g.hlength, expected_batches, g.len))
        for i in range(len(g)):
            print(i, g[i])

        y = [g[ix][1] for ix in range(len(g))]

        actual_sequences = sum(len(_y) for _y in y)
        actual_batches = len(y)

        assert expected_sequences == actual_sequences
        assert expected_batches == actual_batches


def test_TimeseriesGenerator_types():

    print("** test 0 (float types)")

    data = np.array([[i] for i in range(50)], dtype=np.float)
    targets = np.array([[float(i)] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   hlength=5, sampling_rate=2, gap=2,
                                   batch_size=2, shuffle=False)
    x, y = data_gen[0]

    assert np.allclose(x, np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.allclose(y, np.array([[10], [11]]))

    print("** test 1 (auto types)")

    data = np.array([[i] for i in range(50)], dtype=np.float)
    targets = np.array([[i] for i in range(50)], dtype=np.float)

    data_gen = TimeseriesGenerator(data, targets,
                                   hlength=5, sampling_rate=2, gap=2,
                                   batch_size=2, shuffle=False)
    x, y = data_gen[0]
    assert len(data_gen) == 20
    assert np.array_equal(x, np.array([[[0], [2], [4], [6], [8]],
                                       [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y, np.array([[10], [11]]))

    x, y = data_gen[-1]

    assert np.array_equal(x, np.array([[[38], [40], [42], [44], [46]],
                                       [[39], [41], [43], [45], [47]]]))
    assert np.array_equal(y, np.array([[48], [49]]))

    print("** test 2 (batch_size=4)")
    data_gen = TimeseriesGenerator(
        data, targets, hlength=10, batch_size=4, gap=1)
    assert len(data_gen) == 10
    x, y = data_gen[0]
    assert np.array_equal(x[1], np.array(
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))
    assert np.array_equal(y, np.array([[10], [11], [12], [13]]))

    data_gen = TimeseriesGenerator(
        data, targets, hlength=10, reverse=True, batch_size=2)
    x, y = data_gen[0]
    assert np.array_equal(x[1, 0], np.array([10]))

    print("** test 3 (when sampling_rate is not a multiple of hlength)")
    data_gen = TimeseriesGenerator(
        data, targets, hlength=10, sampling_rate=3, batch_size=2)

    # for i in range(len(data_gen)):
    #    print(i,data_gen[i])

    assert len(data_gen) == 12

    print("** test 4 (stateful)")
    data_gen = TimeseriesGenerator(
        data, targets, hlength=10, sampling_rate=2, batch_size=5, stateful=True, gap=2, stride=4)


def test_TimeseriesGenerator_on_text():

    txt = bytearray("Keras is simple.", 'utf-8')
    data_gen = TimeseriesGenerator(txt, txt, hlength=10, batch_size=1, gap=1)

    # for i in range(len(data_gen)):
    #    print(data_gen[i][0].tostring(), "->'%s'" % data_gen[i][1].tostring())

    assert data_gen[-1][0].shape == (1, 10) and data_gen[-1][1].shape == (1,)
    assert data_gen[-1][0].tostring() == u" is simple"
    assert data_gen[-1][1].tostring() == u"."

    data_gen = TimeseriesGenerator(
        txt, txt, hlength=10, target_seq=True, batch_size=1, gap=1)

    assert data_gen[-1][0].shape == (1,
                                     10) and data_gen[-1][1].shape == (1, 10, 1)
    # for i in range(len(data_gen)):
    #    print(data_gen[i][0].tostring(), "->'%s'" % data_gen[i][1].tostring())

    assert data_gen[0][1].tostring() == u"eras is si"


def test_TimeseriesGenerator_previous_tests():

    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, reverse=True,
                                   batch_size=2, gap=2)
    assert len(data_gen) == 20
    assert (np.allclose(data_gen[0][0],
                        np.array([[[8], [6], [4], [2], [0]],
                                  [[9], [7], [5], [3], [1]]])))
    assert (np.allclose(data_gen[0][1],
                        np.array([[10], [11]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2, shuffle=True,
                                   batch_size=1, gap=2)
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
                                   batch_size=2, gap=2)
    assert len(data_gen) == 10
    assert (np.allclose(data_gen[1][0],
                        np.array([[[4], [6], [8], [10], [12]],
                                  [[6], [8], [10], [12], [14]]])))
    assert (np.allclose(data_gen[1][1],
                        np.array([[14], [16]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2, gap=2)
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
                                   batch_size=2, gap=2)

    assert len(data_gen) == 5
    assert np.allclose(data_gen[0][0], np.array(
        [np.array(data[10:19:2]), np.array(data[11:20:2])]))
    assert (np.allclose(data_gen[0][1],
                        np.array([targets[20], targets[21]])))


if __name__ == '__main__':
    pytest.main([__file__])

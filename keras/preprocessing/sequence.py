# -*- coding: utf-8 -*-
"""Utilities for preprocessing sequence data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from six.moves import range
import warnings
from ..utils.data_utils import Sequence
from math import ceil


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def make_sampling_table(size, sampling_factor=1e-5):
    """Generates a word rank-based probabilistic sampling table.

    Used for generating the `sampling_table` argument for `skipgrams`.
    `sampling_table[i]` is the probability of sampling
    the word i-th most common word in a dataset
    (more common words should be sampled less frequently, for balance).

    The sampling probabilities are generated according
    to the sampling distribution used in word2vec:

    `p(word) = min(1, sqrt(word_frequency / sampling_factor) / (word_frequency / sampling_factor))`

    We assume that the word frequencies follow Zipf's law (s=1) to derive
    a numerical approximation of frequency(rank):

    `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
    where `gamma` is the Euler-Mascheroni constant.

    # Arguments
        size: Int, number of possible words to sample.
        sampling_factor: The sampling factor in the word2vec formula.

    # Returns
        A 1D Numpy array of length `size` where the ith entry
        is the probability that a word of rank i should be sampled.
    """
    gamma = 0.577
    rank = np.arange(size)
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
    f = sampling_factor * inv_fq

    return np.minimum(1., f / np.sqrt(f))


def skipgrams(sequence, vocabulary_size,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None, seed=None):
    """Generates skipgram word pairs.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Read more about Skipgram in this gnomic paper by Mikolov et al.:
    [Efficient Estimation of Word Representations in
    Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

    # Arguments
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        shuffle: Whether to shuffle the word couples before returning them.
        categorical: bool. if False, labels will be
            integers (eg. `[0, 1, 1 .. ]`),
            if `True`, labels will be categorical, e.g.
            `[[1,0],[0,1],[0,1] .. ]`.
        sampling_table: 1D array of size `vocabulary_size` where the entry i
            encodes the probability to sample a word of rank i.
        seed: Random seed.

    # Returns
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.

    # Note
        By convention, index 0 in the vocabulary is
        a non-word and will be skipped.
    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0, 1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[words[i % len(words)],
                     random.randint(1, vocabulary_size - 1)]
                    for i in range(num_negative_samples)]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels


def _remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


class TimeseriesGenerator(Sequence):
    """Utility class for generating batches of temporal data.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be convertible into a 1D numpy array,
            if 2D or more, axis 0 is expected to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have at least the same length as `data`.
        length: length of the output sub-sequence before sampling (depreciated, use hlength instead).
        sampling_rate: Period between successive individual timesteps
            within sequences, `length` has to be a multiple of `sampling_rate`.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index, end_index: Data points earlier than `start_index`
            or later than `end_index` will not be used in the output sequences.
            This is useful to reserve part of the data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `True`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch.
        hlength: Effective "history" length of the output sub-sequences after sampling (in number of timesteps).
        gap: prediction gap, i.e. numer of timesteps ahead (usually zero, or same as samplig_rate)
            `x=data[i - (hlength-1)*sampling_rate - gap:i-gap+1:sampling_rate]` and `y=targets[i]`
            are used respectively as sample sequence `x` and target value `y`.
        target_seq: Boolean: if 'True', produces full shifted sequence targets:
            If target_seq is set, for sampling rate `r`, timesteps
            `data[i - (hlength-1)*r - gap]`, ..., `data[i-r-gap]`, `data[i-gap]` and
            `targets[i - (hlength-1)*r]`, ..., `data[i-r]`, `data[i]`
            are used respectively as sample sequence `x` and target sequence `y`.
        dtype: force sample/target dtype (default is None)
        stateful: helper to check if parameters are valid for stateful learning (experimental).


    # Returns
        A [Sequence](/utils/#sequence) instance of tuples (x,y)
        where x is a numpy array of shape (batch_size, hlength, ...)
        and y is a numpy array of shape (batch_size, ...) if target_seq is `False`
        or (batch_size, hlength, ...) if target_seq is `True`.
        If not specified, output dtype is infered from data dtype.

    # Examples
    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np

    txt = bytearray("Keras is simple.", 'utf-8')
    data_gen = TimeseriesGenerator(txt, txt, hlength=10, batch_size=1, gap=1)

    for i in range(len(data_gen)):
        print(data_gen[i][0].tostring(), "->'%s'" % data_gen[i][1].tostring())

    assert data_gen[-1][0].shape == (1, 10) and data_gen[-1][1].shape == (1,)
    assert data_gen[-1][0].tostring() == u" is simple"
    assert data_gen[-1][1].tostring() == u"."

    t = np.linspace(0,20*np.pi, num=1000) # time
    x = np.sin(np.cos(3*t)) # input signa
    y = np.sin(np.cos(6*t+4)) # output signal

    # define recurrent model
    from keras.models import Model
    from keras.layers import Input, SimpleRNN, LSTM, GRU,Dense

    inputs = Input(batch_shape=(None, None, 1))
    l = SimpleRNN(100, return_sequences=True)(inputs)
    l = Dense(100, activation='tanh')(l)
    preds = Dense(1, activation='linear')(l)
    model = Model(inputs=inputs, outputs=preds)
    model.compile(loss='mean_squared_error', optimizer='Nadam')

    # fit model to sequence
    xx = np.expand_dims(x, axis=-1)
    g = TimeseriesGenerator(xx, y, hlength=100, target_seq=True, shuffle=True)
    model.fit_generator(g, steps_per_epoch=len(g), epochs=20, shuffle=True)

    # plot prediction
    x2 = np.reshape(x,(1,x.shape[0],1))
    z = model.predict(x2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,12))
    plt.title('Phase representation')
    plt.plot(x,y.flatten(), color='black')
    plt.plot(x,z.flatten(), dashes=[8,1], label='prediction', color='orange')
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.show()

    ```
    """

    def __init__(self, data, targets, length=None,
                 sampling_rate=1,
                 stride=1,
                 start_index=0, end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128,
                 hlength=None,
                 target_seq=False,
                 gap=0,
                 dtype=None,
                 stateful=False):

        # Sanity check

        if sampling_rate <= 0:
            raise ValueError('`sampling_rate` must be strictly positive.')
        if stride <= 0:
            raise ValueError('`stride` must be strictly positive.')
        if batch_size <= 0:
            raise ValueError('`batch_size` must be strictly positive.')
        if len(data) > len(targets):
            raise ValueError('`targets` has to be at least as long as `data`.')

        if hlength is None:
            if length % sampling_rate != 0:
                raise ValueError(
                    '`length` has to be a multiple of `sampling_rate`. For instance, `length=%i` would do.' % (2 * sampling_rate))
            hlength = length // sampling_rate

        if gap % sampling_rate != 0:
            warnings.warn(
                'Unless you know what you do, `gap` should be zero or a multiple of `sampling_rate`.', UserWarning)

        self.hlength = hlength
        assert self.hlength > 0

        self.data = np.asarray(data)
        self.targets = np.asarray(targets)

        # FIXME: targets must be 2D for sequences output
        if target_seq and len(self.targets.shape) < 2:
            self.targets = np.expand_dims(self.targets, axis=-1)

        if dtype is None:
            self.data_type = self.data.dtype
            self.targets_type = self.targets.dtype
        else:
            self.data_type = dtype
            self.targets_type = dtype

        # Check if parameters are stateful-compatible
        if stateful:
            if shuffle:
                raise ValueError('Do not shuffle for stateful learning.')
            if self.hlength % batch_size != 0:
                raise ValueError('For stateful learning, `hlength` has to be a multiple of `batch_size`.'
                                 'For instance, `hlength=%i` would do.' % (3 * batch_size))
            if stride != (self.hlength // batch_size) * sampling_rate:
                raise ValueError(
                    '`stride=%i`, for these parameters set `stride=%i`.' % (stride, (hlength // batch_size) * sampling_rate))

        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        assert stride > 0
        self.stride = stride
        self.gap = gap

        sliding_win_size = (self.hlength - 1) * sampling_rate + gap
        self.start_index = start_index + sliding_win_size
        if end_index is None:
            end_index = len(data)
        assert end_index <= len(data)
        self.end_index = end_index
        self.reverse = reverse
        self.target_seq = target_seq

        self.len = int(ceil(float(self.end_index - self.start_index) /
                            (self.batch_size * self.stride)))
        if self.len <= 0:
            err = 'This configuration gives no output, try with a longer input sequence or different parameters.'
            raise ValueError(err)

        assert self.len > 0

        self.perm = np.arange(self.start_index, self.end_index)
        if shuffle:
            np.random.shuffle(self.perm)

    def __len__(self):
        return self.len

    def _empty_batch(self, num_rows):
        samples_shape = [num_rows, self.hlength]
        samples_shape.extend(self.data.shape[1:])
        if self.target_seq:
            targets_shape = [num_rows, self.hlength]
        else:
            targets_shape = [num_rows]
        targets_shape.extend(self.targets.shape[1:])

        return np.empty(samples_shape, dtype=self.data_type), np.empty(targets_shape, dtype=self.targets_type)

    def __getitem__(self, index):
        while index < 0:
            index += self.len
        assert index < self.len
        batch_start = self.batch_size * self.stride * index
        rows = np.arange(batch_start, min(batch_start + self.batch_size *
                                          self.stride, self.end_index - self.start_index), self.stride)
        rows = self.perm[rows]

        samples, targets = self._empty_batch(len(rows))
        for j, row in enumerate(rows):
            indices = range(rows[j] - self.gap - (self.hlength - 1) * self.sampling_rate,
                            rows[j] - self.gap + 1, self.sampling_rate)
            samples[j] = self.data[indices]
            if self.target_seq:
                shifted_indices = range(rows[j] - (self.hlength - 1) * self.sampling_rate,
                                        rows[j] + 1, self.sampling_rate)
                targets[j] = self.targets[shifted_indices]
            else:
                targets[j] = self.targets[rows[j]]
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for preprocessing sequence data.

Deprecated: `tf.keras.preprocessing.sequence` APIs are not recommended for new
code. Prefer `tf.keras.utils.timeseries_dataset_from_array` and
the `tf.data` APIs which provide a much more flexible mechanisms for dealing
with sequences. See the [tf.data guide](https://www.tensorflow.org/guide/data)
for more details.
"""


import json
import random

import numpy as np

from keras.utils import data_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


def _remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    Args:
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    Returns:
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


@keras_export("keras.preprocessing.sequence.TimeseriesGenerator")
class TimeseriesGenerator(data_utils.Sequence):
    """Utility class for generating batches of temporal data.

    Deprecated: `tf.keras.preprocessing.sequence.TimeseriesGenerator` does not
    operate on tensors and is not recommended for new code. Prefer using a
    `tf.data.Dataset` which provides a more efficient and flexible mechanism for
    batching, shuffling, and windowing input. See the
    [tf.data guide](https://www.tensorflow.org/guide/data) for more details.

    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.

    Arguments:
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).

    Returns:
        A [Sequence](
        https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence)
        instance.

    Examples:
        ```python
        from keras.preprocessing.sequence import TimeseriesGenerator
        import numpy as np
        data = np.array([[i] for i in range(50)])
        targets = np.array([[i] for i in range(50)])
        data_gen = TimeseriesGenerator(data, targets,
                                       length=10, sampling_rate=2,
                                       batch_size=2)
        assert len(data_gen) == 20
        batch_0 = data_gen[0]
        x, y = batch_0
        assert np.array_equal(x,
                              np.array([[[0], [2], [4], [6], [8]],
                                        [[1], [3], [5], [7], [9]]]))
        assert np.array_equal(y,
                              np.array([[10], [11]]))
        ```
    """

    def __init__(
        self,
        data,
        targets,
        length,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=128,
    ):

        if len(data) != len(targets):
            raise ValueError(
                "Data and targets have to be" + " of same length. "
                "Data length is {}".format(len(data))
                + " while target length is {}".format(len(targets))
            )

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

        if self.start_index > self.end_index:
            raise ValueError(
                "`start_index+length=%i > end_index=%i` "
                "is disallowed, as no part of the sequence "
                "would be left to be used as current step."
                % (self.start_index, self.end_index)
            )

    def __len__(self):
        return (
            self.end_index - self.start_index + self.batch_size * self.stride
        ) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size
            )
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(
                i,
                min(i + self.batch_size * self.stride, self.end_index + 1),
                self.stride,
            )

        samples = np.array(
            [
                self.data[row - self.length : row : self.sampling_rate]
                for row in rows
            ]
        )
        targets = np.array([self.targets[row] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets

    def get_config(self):
        """Returns the TimeseriesGenerator configuration as Python dictionary.

        Returns:
            A Python dictionary with the TimeseriesGenerator configuration.
        """
        data = self.data
        if type(self.data).__module__ == np.__name__:
            data = self.data.tolist()
        try:
            json_data = json.dumps(data)
        except TypeError as e:
            raise TypeError("Data not JSON Serializable:", data) from e

        targets = self.targets
        if type(self.targets).__module__ == np.__name__:
            targets = self.targets.tolist()
        try:
            json_targets = json.dumps(targets)
        except TypeError as e:
            raise TypeError("Targets not JSON Serializable:", targets) from e

        return {
            "data": json_data,
            "targets": json_targets,
            "length": self.length,
            "sampling_rate": self.sampling_rate,
            "stride": self.stride,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "shuffle": self.shuffle,
            "reverse": self.reverse,
            "batch_size": self.batch_size,
        }

    def to_json(self, **kwargs):
        """Returns a JSON string containing the timeseries generator configuration.

        Args:
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.
        Returns:
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            "class_name": self.__class__.__name__,
            "config": config,
        }
        return json.dumps(timeseries_generator_config, **kwargs)


@keras_export("keras.preprocessing.sequence.make_sampling_table")
def make_sampling_table(size, sampling_factor=1e-5):
    """Generates a word rank-based probabilistic sampling table.

    Used for generating the `sampling_table` argument for `skipgrams`.
    `sampling_table[i]` is the probability of sampling
    the word i-th most common word in a dataset
    (more common words should be sampled less frequently, for balance).

    The sampling probabilities are generated according
    to the sampling distribution used in word2vec:

    ```
    p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
        (word_frequency / sampling_factor)))
    ```

    We assume that the word frequencies follow Zipf's law (s=1) to derive
    a numerical approximation of frequency(rank):

    `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
    where `gamma` is the Euler-Mascheroni constant.

    Args:
        size: Int, number of possible words to sample.
        sampling_factor: The sampling factor in the word2vec formula.

    Returns:
        A 1D Numpy array of length `size` where the ith entry
        is the probability that a word of rank i should be sampled.
    """
    gamma = 0.577
    rank = np.arange(size)
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1.0 / (12.0 * rank)
    f = sampling_factor * inv_fq

    return np.minimum(1.0, f / np.sqrt(f))


@keras_export("keras.preprocessing.sequence.skipgrams")
def skipgrams(
    sequence,
    vocabulary_size,
    window_size=4,
    negative_samples=1.0,
    shuffle=True,
    categorical=False,
    sampling_table=None,
    seed=None,
):
    """Generates skipgram word pairs.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Read more about Skipgram in this gnomic paper by Mikolov et al.:
    [Efficient Estimation of Word Representations in
    Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

    Args:
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

    Returns:
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.

    Note:
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

        couples += [
            [words[i % len(words)], random.randint(1, vocabulary_size - 1)]
            for i in range(num_negative_samples)
        ]
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

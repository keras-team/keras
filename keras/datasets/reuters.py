# -*- coding: utf-8 -*-
from __future__ import absolute_import
from ..utils.data_utils import get_file
from six.moves import cPickle
from six.moves import zip
import numpy as np
import sys


def load_data(path='reuters.pkl', nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    """Loads the Reuters newswire classification dataset.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        nb_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occuring words
            (which may not be informative).
        maxlen: truncate sequences after this length.
        test_split: Fraction of the dataset to be used as test data.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `nb_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `nb_words` cut here.
    Words that were not seen in the trining set but are in the test set
    have simply been skipped.
    """

    path = get_file(path, origin='https://s3.amazonaws.com/text-datasets/reuters.pkl')
    f = open(path, 'rb')
    xs, labels = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(xs)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        new_xs = []
        new_labels = []
        for x, y in zip(xs, labels):
            if len(x) < maxlen:
                new_xs.append(x)
                new_labels.append(y)
        xs = new_xs
        labels = new_labels

    if not nb_words:
        nb_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in xs]
    else:
        new_xs = []
        for x in xs:
            nx = []
            for w in x:
                if w >= nb_words or w < skip_top:
                    nx.append(w)
            new_xs.append(nx)
        xs = new_xs

    x_train = xs[:int(len(xs) * (1 - test_split))]
    y_train = labels[:int(len(xs) * (1 - test_split))]

    x_test = xs[int(len(xs) * (1 - test_split)):]
    y_test = labels[int(len(xs) * (1 - test_split)):]

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path='reuters_word_index.pkl'):
    """Retrieves the dictionary mapping word indices back to words.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    """
    path = get_file(path, origin='https://s3.amazonaws.com/text-datasets/reuters_word_index.pkl')
    f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='latin1')

    f.close()
    return data

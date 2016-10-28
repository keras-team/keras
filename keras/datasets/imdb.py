from __future__ import absolute_import
from six.moves import cPickle
import gzip
from ..utils.data_utils import get_file
from six.moves import zip
import numpy as np
import sys


def load_data(path='imdb_full.pkl', nb_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3):
    '''
    # Arguments
        path: where to store the data (in `/.keras/dataset`)
        nb_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occuring words
            (which may not be informative).
        maxlen: truncate sequences after this length.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `nb_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `nb_words` cut here.
    Words that were not seen in the trining set but are in the test set
    have simply been skipped.
    '''
    path = get_file(path,
                    origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                    md5_hash='d091312047c43cf9e4e38fef92437263')

    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    (x_train, labels_train), (x_test, labels_test) = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(labels_train)

    np.random.seed(seed * 2)
    np.random.shuffle(x_test)
    np.random.seed(seed * 2)
    np.random.shuffle(labels_test)

    X = x_train + x_test
    labels = labels_train + labels_test

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                        'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:len(x_train)])
    y_train = np.array(labels[:len(x_train)])

    X_test = np.array(X[len(x_train):])
    y_test = np.array(labels[len(x_train):])

    return (X_train, y_train), (X_test, y_test)


def get_word_index(path='imdb_word_index.pkl'):
    path = get_file(path,
                    origin='https://s3.amazonaws.com/text-datasets/imdb_word_index.pkl',
                    md5_hash='72d94b01291be4ff843198d3b0e1e4d7')
    f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='latin1')

    f.close()
    return data

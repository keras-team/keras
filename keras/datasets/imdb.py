from __future__ import absolute_import
import six.moves.cPickle
import gzip
from .data_utils import get_file
import random
from six.moves import zip

def load_data(path="imdb.pkl", nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113):
    path = get_file(path, origin="https://s3.amazonaws.com/text-datasets/imdb.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    X, labels = six.moves.cPickle.load(f)
    f.close()

    random.seed(seed)
    random.shuffle(X)
    random.seed(seed)
    random.shuffle(labels)

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels

    if not nb_words:
        nb_words = max([max(x) for x in X])

    X = [[0 if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)


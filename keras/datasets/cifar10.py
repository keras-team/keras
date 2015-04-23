# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .data_utils import get_file
import random
import sys, os
import numpy as np
import six.moves.cPickle
from six.moves import range

def load_data(test_split=0.1, seed=113):
    dirname = "cifar-10-batches-py"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)

    nb_samples = 50000
    X = np.zeros((nb_samples, 3, 32, 32), dtype="uint8")
    y = np.zeros((nb_samples,), dtype="uint8")
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        f = open(fpath, 'rb')
        if sys.version_info < (3,):
            d = six.moves.cPickle.load(f)
        else:
            d = six.moves.cPickle.load(f, encoding="bytes")
            # decode utf8
            for k, v in d.items():
                del(d[k])
                d[k.decode("utf8")] = v
        f.close()
        data = d["data"]
        labels = d["labels"]

        data = data.reshape(data.shape[0], 3, 32, 32)
        X[(i-1)*10000:i*10000, :, :, :] = data
        y[(i-1)*10000:i*10000] = labels

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y = np.reshape(y, (len(y), 1))

    X_train = X[:int(len(X)*(1-test_split))]
    y_train = y[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = y[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)

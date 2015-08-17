# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import six.moves.cPickle
from six.moves import range

def load_batch(fpath, label_key='labels'):
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
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

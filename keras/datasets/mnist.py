# -*- coding: utf-8 -*-
# modified from https://github.com/niboshi/mnist_loader.git
import os
import urllib
import urllib2
import StringIO
import gzip
import struct
import numpy as np

def load_data():
    (X_train, y_train) = load(set_name='train')
    (X_test, y_test) = load(set_name='test')

    return (X_train, y_train), (X_test, y_test)
    
    
def open_data_url(url, cache_dir=None):
    """
    Opens an URL as a file-like object.
    """

    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras', 'datasets','mnist'))

    cache_path = os.path.join(cache_dir, urllib.quote(url, safe=''))

    if os.path.isfile(cache_path):
        return open(cache_path, 'rb')
    else:
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        buf = StringIO.StringIO()
        block = 1024
        while True:
            data = response.read(block)
            if data is None:
                break
            if len(data) == 0:
                break

            buf.write(data)

        if not os.path.isdir(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        with open(cache_path, 'wb') as fo:
            fo.write(buf.getvalue())

        buf.seek(0)
        return buf

def read_data(file_in):
    """
    Parses the IDX file format.
    """

    # Magic code
    magic = file_in.read(4)
    magic = [ord(_) for _ in magic]
    if len(magic) != 4 or magic[0] != 0 or magic[1] != 0:
        raise RuntimeError("Invalid magic number: [{}]".format('-'.join(['{:02x}'.format(_) for _ in magic])))

    # Type code
    type_code = magic[2]
    dtype_map = {
        0x08: np.uint8,
        0x09: np.int8,
        0x0B: np.int16,
        0x0C: np.int32,
        0x0D: np.float32,
        0x0E: np.float64,
    }
    dtype = dtype_map[type_code]

    # Dimensions
    ndim = magic[3]

    dims = []
    for idim in range(ndim):
        dim, = struct.unpack('>I', file_in.read(4))
        dims.append(dim)

    # Data
    data = file_in.read()
    data = np.fromstring(data, dtype=dtype).reshape(tuple(dims))

    return data

def read_data_from_url(url, cache_dir=None):
    """
    Extracts multidimensional data from an URL.
    """
    return read_data(gzip.GzipFile(fileobj=open_data_url(url, cache_dir=cache_dir)))

def load(set_name='train', cache_dir=None):
    """
    Loads the MNIST data set.
    """

    if set_name == 'train':
        urls = (
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        )
    elif set_name == 'test':
        urls = (
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        )
    else:
        assert False, "Invalid set name: {}".format(set_name)

    data_url, labels_url = urls
    data   = read_data_from_url(data_url, cache_dir=cache_dir)
    labels = read_data_from_url(labels_url, cache_dir=cache_dir)

    return data, labels

